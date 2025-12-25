//! Frame management and synchronization.
//!
//! This module provides the [`FrameManager`] struct for managing per-frame
//! resources and coordinating the rendering loop. It handles:
//!
//! - Per-frame command buffers
//! - Synchronization primitives (semaphores and fences)
//! - Swapchain image acquisition and presentation
//! - Frame-in-flight management
//!
//! # Overview
//!
//! The frame manager implements a "frames in flight" pattern where multiple
//! frames can be processed concurrently:
//!
//! 1. While the GPU renders frame N, the CPU prepares frame N+1
//! 2. Each frame has its own set of resources to avoid contention
//! 3. Fences ensure the CPU doesn't overwrite resources still in use
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_rhi::device::Device;
//! use renderer_rhi::command::CommandPool;
//! use renderer_rhi::swapchain::Swapchain;
//! use renderer_renderer::frame_manager::FrameManager;
//!
//! # fn example(
//! #     device: Arc<Device>,
//! #     command_pool: &CommandPool,
//! #     swapchain: &Swapchain,
//! # ) -> Result<(), renderer_rhi::RhiError> {
//! let mut frame_manager = FrameManager::new(device, command_pool)?;
//!
//! // Main render loop
//! loop {
//!     // Wait for the previous frame using this slot to complete
//!     frame_manager.wait_for_frame()?;
//!
//!     // Acquire the next swapchain image
//!     let needs_resize = frame_manager.acquire_next_image(swapchain)?;
//!     if needs_resize {
//!         // Handle swapchain recreation
//!         break;
//!     }
//!
//!     // Begin recording commands for this frame
//!     frame_manager.begin_frame()?;
//!
//!     // Record rendering commands...
//!     let cmd = frame_manager.current_frame().command_buffer();
//!     // cmd.begin_rendering(...);
//!     // cmd.draw(...);
//!     // cmd.end_rendering();
//!
//!     // End command recording
//!     frame_manager.end_frame()?;
//!
//!     // Submit commands to the GPU
//!     frame_manager.submit()?;
//!
//!     // Present the rendered image
//!     let needs_resize = frame_manager.present(swapchain)?;
//!     if needs_resize {
//!         // Handle swapchain recreation
//!         break;
//!     }
//!
//!     // Advance to next frame slot
//!     frame_manager.next_frame();
//!     # break;
//! }
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;
use tracing::{debug, info};

use renderer_rhi::RhiResult;
use renderer_rhi::command::{CommandBuffer, CommandPool};
use renderer_rhi::device::Device;
use renderer_rhi::swapchain::Swapchain;
use renderer_rhi::sync::{Fence, Semaphore};

use crate::MAX_FRAMES_IN_FLIGHT;

/// Per-frame rendering data.
///
/// Each frame in flight has its own set of resources to avoid synchronization
/// issues between frames. This includes:
/// - A command buffer for recording rendering commands
/// - Semaphores for GPU-GPU synchronization
/// - A fence for CPU-GPU synchronization
///
/// # Synchronization Flow
///
/// ```text
/// 1. Wait on in_flight_fence (CPU waits for previous use of this slot)
/// 2. Acquire swapchain image (signals image_available_semaphore)
/// 3. Record commands to command_buffer
/// 4. Submit command_buffer:
///    - Wait on image_available_semaphore
///    - Signal render_finished_semaphore
///    - Signal in_flight_fence
/// 5. Present (waits on render_finished_semaphore)
/// ```
pub struct FrameData {
    /// Command buffer for recording rendering commands.
    command_buffer: CommandBuffer,
    /// Semaphore signaled when a swapchain image is available.
    image_available_semaphore: Semaphore,
    /// Semaphore signaled when rendering is complete.
    render_finished_semaphore: Semaphore,
    /// Fence used to wait for frame completion before reusing resources.
    in_flight_fence: Fence,
}

impl FrameData {
    /// Creates a new set of per-frame resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `command_pool` - The command pool to allocate from
    ///
    /// # Errors
    ///
    /// Returns an error if any resource creation fails.
    fn new(device: Arc<Device>, command_pool: &CommandPool) -> RhiResult<Self> {
        let command_buffer = CommandBuffer::new(device.clone(), command_pool)?;
        let image_available_semaphore = Semaphore::new(device.clone())?;
        let render_finished_semaphore = Semaphore::new(device.clone())?;
        // Create fence in signaled state so the first wait doesn't block forever
        let in_flight_fence = Fence::new(device, true)?;

        Ok(Self {
            command_buffer,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        })
    }

    /// Returns a reference to the command buffer.
    #[inline]
    pub fn command_buffer(&self) -> &CommandBuffer {
        &self.command_buffer
    }

    /// Returns a reference to the image available semaphore.
    #[inline]
    pub fn image_available_semaphore(&self) -> &Semaphore {
        &self.image_available_semaphore
    }

    /// Returns a reference to the render finished semaphore.
    #[inline]
    pub fn render_finished_semaphore(&self) -> &Semaphore {
        &self.render_finished_semaphore
    }

    /// Returns a reference to the in-flight fence.
    #[inline]
    pub fn in_flight_fence(&self) -> &Fence {
        &self.in_flight_fence
    }
}

/// Manages per-frame resources and the frame rendering loop.
///
/// The frame manager coordinates the rendering pipeline by:
/// - Managing multiple sets of per-frame resources
/// - Handling swapchain image acquisition and presentation
/// - Synchronizing CPU and GPU work
///
/// # Frames in Flight
///
/// The manager maintains [`MAX_FRAMES_IN_FLIGHT`] sets of resources,
/// typically 2 or 3. This allows the CPU to prepare frame N+1 while
/// the GPU is still rendering frame N.
///
/// # Thread Safety
///
/// The frame manager is not thread-safe. It should only be accessed
/// from a single thread (typically the main/render thread).
pub struct FrameManager {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Per-frame resources.
    frames: Vec<FrameData>,
    /// Current frame index (0 to MAX_FRAMES_IN_FLIGHT - 1).
    current_frame: usize,
    /// Current swapchain image index.
    image_index: u32,
}

impl FrameManager {
    /// Creates a new frame manager.
    ///
    /// This allocates [`MAX_FRAMES_IN_FLIGHT`] sets of per-frame resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `command_pool` - The command pool to allocate command buffers from
    ///
    /// # Errors
    ///
    /// Returns an error if any resource creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::command::CommandPool;
    /// use renderer_renderer::frame_manager::FrameManager;
    ///
    /// # fn example(device: Arc<Device>, command_pool: &CommandPool) -> Result<(), renderer_rhi::RhiError> {
    /// let frame_manager = FrameManager::new(device, command_pool)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: Arc<Device>, command_pool: &CommandPool) -> RhiResult<Self> {
        let mut frames = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let frame_data = FrameData::new(device.clone(), command_pool)?;
            debug!("Created frame data for frame {}", i);
            frames.push(frame_data);
        }

        info!(
            "Frame manager created with {} frames in flight",
            MAX_FRAMES_IN_FLIGHT
        );

        Ok(Self {
            device,
            frames,
            current_frame: 0,
            image_index: 0,
        })
    }

    /// Returns a reference to the current frame's data.
    ///
    /// This provides access to the command buffer and synchronization
    /// primitives for the current frame slot.
    #[inline]
    pub fn current_frame(&self) -> &FrameData {
        &self.frames[self.current_frame]
    }

    /// Returns the current frame index.
    ///
    /// This is the index into the frames in flight (0 to MAX_FRAMES_IN_FLIGHT - 1).
    #[inline]
    pub fn current_frame_index(&self) -> usize {
        self.current_frame
    }

    /// Returns the current swapchain image index.
    ///
    /// This is set by [`acquire_next_image`](Self::acquire_next_image) and
    /// used for rendering to the correct swapchain image.
    #[inline]
    pub fn image_index(&self) -> u32 {
        self.image_index
    }

    /// Waits for the current frame's previous work to complete.
    ///
    /// This blocks until the GPU has finished processing the last submission
    /// that used this frame slot. Must be called before recording new commands
    /// to this frame's command buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use renderer_rhi::device::Device;
    /// # use renderer_rhi::command::CommandPool;
    /// # use renderer_renderer::frame_manager::FrameManager;
    /// # fn example(frame_manager: &FrameManager) -> Result<(), renderer_rhi::RhiError> {
    /// // Wait before reusing this frame's resources
    /// frame_manager.wait_for_frame()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn wait_for_frame(&self) -> RhiResult<()> {
        self.frames[self.current_frame]
            .in_flight_fence
            .wait(u64::MAX)?;
        Ok(())
    }

    /// Acquires the next swapchain image for rendering.
    ///
    /// This signals the current frame's image available semaphore when
    /// the image is ready to be rendered to.
    ///
    /// # Arguments
    ///
    /// * `swapchain` - The swapchain to acquire an image from
    ///
    /// # Returns
    ///
    /// Returns `true` if the swapchain is out of date or suboptimal and
    /// should be recreated. Returns `false` if acquisition succeeded normally.
    ///
    /// # Errors
    ///
    /// Returns an error if image acquisition fails for reasons other than
    /// an out-of-date swapchain.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use renderer_rhi::device::Device;
    /// # use renderer_rhi::command::CommandPool;
    /// # use renderer_rhi::swapchain::Swapchain;
    /// # use renderer_renderer::frame_manager::FrameManager;
    /// # fn example(frame_manager: &mut FrameManager, swapchain: &Swapchain) -> Result<(), renderer_rhi::RhiError> {
    /// let needs_resize = frame_manager.acquire_next_image(swapchain)?;
    /// if needs_resize {
    ///     // Recreate swapchain and retry
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn acquire_next_image(&mut self, swapchain: &Swapchain) -> RhiResult<bool> {
        let frame = &self.frames[self.current_frame];

        match swapchain.acquire_next_image(frame.image_available_semaphore.handle()) {
            Ok((index, suboptimal)) => {
                self.image_index = index;
                Ok(suboptimal)
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                debug!("Swapchain out of date during acquire");
                Ok(true)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Begins recording commands for the current frame.
    ///
    /// This resets and begins the current frame's command buffer.
    /// Must be called after [`wait_for_frame`](Self::wait_for_frame).
    ///
    /// # Errors
    ///
    /// Returns an error if resetting or beginning the command buffer fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use renderer_rhi::device::Device;
    /// # use renderer_rhi::command::CommandPool;
    /// # use renderer_renderer::frame_manager::FrameManager;
    /// # fn example(frame_manager: &FrameManager) -> Result<(), renderer_rhi::RhiError> {
    /// frame_manager.wait_for_frame()?;
    /// frame_manager.begin_frame()?;
    /// // Record rendering commands...
    /// # Ok(())
    /// # }
    /// ```
    pub fn begin_frame(&self) -> RhiResult<()> {
        let frame = &self.frames[self.current_frame];
        frame.in_flight_fence.reset()?;
        frame.command_buffer.reset()?;
        frame.command_buffer.begin()?;
        Ok(())
    }

    /// Ends recording commands for the current frame.
    ///
    /// This finalizes the current frame's command buffer.
    /// Must be called after all rendering commands have been recorded.
    ///
    /// # Errors
    ///
    /// Returns an error if ending the command buffer fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use renderer_rhi::device::Device;
    /// # use renderer_rhi::command::CommandPool;
    /// # use renderer_renderer::frame_manager::FrameManager;
    /// # fn example(frame_manager: &FrameManager) -> Result<(), renderer_rhi::RhiError> {
    /// // After recording all rendering commands...
    /// frame_manager.end_frame()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn end_frame(&self) -> RhiResult<()> {
        self.frames[self.current_frame].command_buffer.end()?;
        Ok(())
    }

    /// Submits the current frame's commands to the graphics queue.
    ///
    /// This submits the command buffer with proper synchronization:
    /// - Waits on the image available semaphore (image ready for rendering)
    /// - Signals the render finished semaphore (rendering complete)
    /// - Signals the in-flight fence (for CPU synchronization)
    ///
    /// # Errors
    ///
    /// Returns an error if queue submission fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use renderer_rhi::device::Device;
    /// # use renderer_rhi::command::CommandPool;
    /// # use renderer_renderer::frame_manager::FrameManager;
    /// # fn example(frame_manager: &FrameManager) -> Result<(), renderer_rhi::RhiError> {
    /// frame_manager.end_frame()?;
    /// frame_manager.submit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn submit(&self) -> RhiResult<()> {
        let frame = &self.frames[self.current_frame];

        let wait_semaphores = [frame.image_available_semaphore.handle()];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [frame.render_finished_semaphore.handle()];
        let command_buffers = [frame.command_buffer.handle()];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device.handle().queue_submit(
                self.device.graphics_queue(),
                &[submit_info],
                frame.in_flight_fence.handle(),
            )?;
        }

        Ok(())
    }

    /// Presents the rendered image to the screen.
    ///
    /// This queues the current swapchain image for presentation, waiting
    /// for rendering to complete first.
    ///
    /// # Arguments
    ///
    /// * `swapchain` - The swapchain to present to
    ///
    /// # Returns
    ///
    /// Returns `true` if the swapchain is out of date or suboptimal and
    /// should be recreated. Returns `false` if presentation succeeded normally.
    ///
    /// # Errors
    ///
    /// Returns an error if presentation fails for reasons other than
    /// an out-of-date swapchain.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use renderer_rhi::device::Device;
    /// # use renderer_rhi::command::CommandPool;
    /// # use renderer_rhi::swapchain::Swapchain;
    /// # use renderer_renderer::frame_manager::FrameManager;
    /// # fn example(frame_manager: &FrameManager, swapchain: &Swapchain) -> Result<(), renderer_rhi::RhiError> {
    /// let needs_resize = frame_manager.present(swapchain)?;
    /// if needs_resize {
    ///     // Recreate swapchain
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn present(&self, swapchain: &Swapchain) -> RhiResult<bool> {
        let frame = &self.frames[self.current_frame];

        match swapchain.present(
            self.device.present_queue(),
            self.image_index,
            frame.render_finished_semaphore.handle(),
        ) {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                debug!("Swapchain out of date during present");
                Ok(true)
            }
            Err(vk::Result::SUBOPTIMAL_KHR) => {
                debug!("Swapchain suboptimal during present");
                Ok(true)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Advances to the next frame slot.
    ///
    /// This should be called at the end of each frame to cycle through
    /// the available frame slots.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use renderer_rhi::device::Device;
    /// # use renderer_rhi::command::CommandPool;
    /// # use renderer_renderer::frame_manager::FrameManager;
    /// # fn example(frame_manager: &mut FrameManager) {
    /// // After presenting...
    /// frame_manager.next_frame();
    /// # }
    /// ```
    pub fn next_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    /// Waits for all in-flight frames to complete.
    ///
    /// This is useful before destroying resources or recreating the swapchain
    /// to ensure all GPU work has finished.
    ///
    /// # Errors
    ///
    /// Returns an error if any wait fails.
    pub fn wait_for_all_frames(&self) -> RhiResult<()> {
        let fences: Vec<vk::Fence> = self
            .frames
            .iter()
            .map(|f| f.in_flight_fence.handle())
            .collect();

        unsafe {
            self.device
                .handle()
                .wait_for_fences(&fences, true, u64::MAX)?;
        }

        Ok(())
    }

    /// Resets all semaphores to ensure clean state after swapchain recreation.
    ///
    /// This creates new semaphores for all frame slots. Should be called
    /// after swapchain recreation to ensure semaphores are in a known state.
    ///
    /// # Errors
    ///
    /// Returns an error if semaphore creation fails.
    pub fn reset_semaphores(&mut self) -> RhiResult<()> {
        for (i, frame) in self.frames.iter_mut().enumerate() {
            // Create new semaphores
            let new_image_available = Semaphore::new(self.device.clone())?;
            let new_render_finished = Semaphore::new(self.device.clone())?;

            // Replace old semaphores (old ones are dropped automatically)
            frame.image_available_semaphore = new_image_available;
            frame.render_finished_semaphore = new_render_finished;

            debug!("Reset semaphores for frame {}", i);
        }

        info!("Reset all frame semaphores");
        Ok(())
    }

    /// Returns a reference to the device.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the number of frames in flight.
    #[inline]
    pub fn frames_in_flight(&self) -> usize {
        self.frames.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_frames_in_flight_constant() {
        // Verify the constant is a reasonable value for rendering
        assert!(MAX_FRAMES_IN_FLIGHT >= 1);
        assert!(MAX_FRAMES_IN_FLIGHT <= 4);
    }

    #[test]
    fn test_frame_manager_is_send() {
        // Compile-time check that FrameManager is Send
        fn assert_send<T: Send>() {}
        assert_send::<FrameManager>();
    }

    #[test]
    fn test_frame_data_is_send() {
        // Compile-time check that FrameData is Send
        fn assert_send<T: Send>() {}
        assert_send::<FrameData>();
    }
}

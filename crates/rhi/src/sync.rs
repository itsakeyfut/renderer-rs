//! Synchronization primitives for Vulkan.
//!
//! This module provides wrappers for Vulkan synchronization objects:
//! - [`Semaphore`] - GPU-to-GPU synchronization (between queue operations)
//! - [`Fence`] - GPU-to-CPU synchronization (for host waiting)
//! - [`FrameSync`] - Per-frame synchronization primitives for rendering
//!
//! # Overview
//!
//! Vulkan requires explicit synchronization between operations:
//!
//! - **Semaphores** are used to synchronize operations within or across queues.
//!   For example, waiting for image acquisition before rendering, or waiting for
//!   rendering to complete before presentation.
//!
//! - **Fences** are used to synchronize the CPU with GPU operations. The CPU can
//!   wait for a fence to be signaled, allowing it to know when GPU work is complete.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_rhi::device::Device;
//! use renderer_rhi::sync::{Semaphore, Fence};
//!
//! # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
//! // Create a semaphore for GPU-to-GPU synchronization
//! let image_available = Semaphore::new(device.clone())?;
//!
//! // Create a fence for GPU-to-CPU synchronization (signaled initially)
//! let in_flight_fence = Fence::new(device.clone(), true)?;
//!
//! // Wait for the fence before starting a new frame
//! in_flight_fence.wait(u64::MAX)?;
//! in_flight_fence.reset()?;
//!
//! // Use semaphores and fences with swapchain and command submissions...
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;
use tracing::{debug, info};

use crate::device::Device;
use crate::error::{RhiError, RhiResult};

/// Vulkan semaphore wrapper.
///
/// Semaphores are used for GPU-to-GPU synchronization between queue operations.
/// Common use cases include:
/// - Image available semaphore: signaled when a swapchain image is ready
/// - Render finished semaphore: signaled when rendering is complete
///
/// # Thread Safety
///
/// The semaphore is immutable after creation and can be safely shared between
/// threads. The Vulkan specification allows semaphore operations to be submitted
/// from multiple threads.
pub struct Semaphore {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan semaphore handle.
    semaphore: vk::Semaphore,
}

impl Semaphore {
    /// Creates a new semaphore.
    ///
    /// The semaphore is created in the unsignaled state.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    ///
    /// # Errors
    ///
    /// Returns an error if semaphore creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::sync::Semaphore;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let semaphore = Semaphore::new(device)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: Arc<Device>) -> RhiResult<Self> {
        let create_info = vk::SemaphoreCreateInfo::default();

        let semaphore = unsafe { device.handle().create_semaphore(&create_info, None)? };

        debug!("Created semaphore");

        Ok(Self { device, semaphore })
    }

    /// Returns the Vulkan semaphore handle.
    ///
    /// This handle can be used directly with Vulkan API calls.
    #[inline]
    pub fn handle(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_semaphore(self.semaphore, None);
        }
        debug!("Destroyed semaphore");
    }
}

/// Vulkan fence wrapper.
///
/// Fences are used for GPU-to-CPU synchronization, allowing the host to wait
/// for GPU operations to complete. Common use cases include:
/// - Frame-in-flight fence: wait before reusing command buffers
/// - Transfer completion fence: wait for data upload to complete
///
/// # Thread Safety
///
/// The fence is immutable after creation. Wait and reset operations can be
/// called from any thread, but proper synchronization is the caller's
/// responsibility when accessing fence state.
pub struct Fence {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan fence handle.
    fence: vk::Fence,
}

impl Fence {
    /// Creates a new fence.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `signaled` - If true, creates the fence in the signaled state.
    ///   This is useful for fences that are waited on before the first
    ///   GPU operation that would signal them.
    ///
    /// # Errors
    ///
    /// Returns an error if fence creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::sync::Fence;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// // Create a fence that starts signaled (for first frame)
    /// let fence = Fence::new(device, true)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: Arc<Device>, signaled: bool) -> RhiResult<Self> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };

        let create_info = vk::FenceCreateInfo::default().flags(flags);

        let fence = unsafe { device.handle().create_fence(&create_info, None)? };

        debug!(
            "Created fence ({})",
            if signaled { "signaled" } else { "unsignaled" }
        );

        Ok(Self { device, fence })
    }

    /// Returns the Vulkan fence handle.
    ///
    /// This handle can be used directly with Vulkan API calls.
    #[inline]
    pub fn handle(&self) -> vk::Fence {
        self.fence
    }

    /// Waits for the fence to become signaled.
    ///
    /// This function blocks until the fence is signaled or the timeout expires.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Timeout in nanoseconds. Use `u64::MAX` for infinite wait.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The wait times out (`vk::Result::TIMEOUT`)
    /// - The wait fails for another reason
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::sync::Fence;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let fence = Fence::new(device, false)?;
    /// // ... submit GPU work that signals the fence ...
    ///
    /// // Wait indefinitely
    /// fence.wait(u64::MAX)?;
    ///
    /// // Or wait with a timeout (1 second = 1_000_000_000 nanoseconds)
    /// // fence.wait(1_000_000_000)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn wait(&self, timeout: u64) -> Result<(), RhiError> {
        let fences = [self.fence];
        unsafe {
            self.device
                .handle()
                .wait_for_fences(&fences, true, timeout)?
        };
        Ok(())
    }

    /// Resets the fence to the unsignaled state.
    ///
    /// The fence must not be in use by any queue operation when this is called.
    ///
    /// # Errors
    ///
    /// Returns an error if the reset operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::sync::Fence;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let fence = Fence::new(device, true)?;
    ///
    /// // Wait for previous frame
    /// fence.wait(u64::MAX)?;
    /// fence.reset()?;
    ///
    /// // ... submit new GPU work ...
    /// # Ok(())
    /// # }
    /// ```
    pub fn reset(&self) -> Result<(), RhiError> {
        let fences = [self.fence];
        unsafe { self.device.handle().reset_fences(&fences)? };
        Ok(())
    }

    /// Checks if the fence is currently signaled.
    ///
    /// This is a non-blocking operation that returns immediately.
    ///
    /// # Returns
    ///
    /// Returns `true` if the fence is signaled, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::sync::Fence;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let fence = Fence::new(device, true)?;
    ///
    /// if fence.is_signaled() {
    ///     // Fence is ready, can proceed
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_signaled(&self) -> bool {
        let result = unsafe { self.device.handle().get_fence_status(self.fence) };
        matches!(result, Ok(true))
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_fence(self.fence, None);
        }
        debug!("Destroyed fence");
    }
}

/// Maximum number of frames that can be processed concurrently.
///
/// This is a common value for double or triple buffering strategies.
/// Using 2 allows the CPU to prepare the next frame while the GPU
/// renders the current one.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Per-frame synchronization primitives.
///
/// This struct groups all synchronization objects needed for frame rendering:
/// - Image available semaphore: signaled when swapchain image is acquired
/// - Render finished semaphore: signaled when rendering is complete
/// - In-flight fence: used to wait before reusing frame resources
///
/// # Usage Pattern
///
/// ```text
/// 1. Wait for in_flight_fence (CPU waits for GPU to finish previous frame)
/// 2. Reset in_flight_fence
/// 3. Acquire swapchain image (signals image_available_semaphore)
/// 4. Submit command buffer:
///    - Wait on image_available_semaphore
///    - Signal render_finished_semaphore
///    - Signal in_flight_fence on completion
/// 5. Present (waits on render_finished_semaphore)
/// ```
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use renderer_rhi::device::Device;
/// use renderer_rhi::sync::FrameSync;
///
/// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
/// // Create synchronization for multiple frames
/// let frame_syncs: Vec<FrameSync> = (0..2)
///     .map(|_| FrameSync::new(device.clone()))
///     .collect::<Result<_, _>>()?;
///
/// let mut current_frame = 0;
///
/// // In render loop:
/// let frame_sync = &frame_syncs[current_frame];
///
/// // Wait for this frame's resources to be available
/// frame_sync.in_flight_fence().wait(u64::MAX)?;
/// frame_sync.in_flight_fence().reset()?;
///
/// // Use frame_sync.image_available_semaphore() for swapchain acquire
/// // Use frame_sync.render_finished_semaphore() for present
/// // Use frame_sync.in_flight_fence() for command buffer submission
///
/// current_frame = (current_frame + 1) % 2;
/// # Ok(())
/// # }
/// ```
pub struct FrameSync {
    /// Semaphore signaled when a swapchain image is available.
    image_available_semaphore: Semaphore,
    /// Semaphore signaled when rendering is complete.
    render_finished_semaphore: Semaphore,
    /// Fence used to wait for frame completion before reusing resources.
    in_flight_fence: Fence,
}

impl FrameSync {
    /// Creates a new set of frame synchronization primitives.
    ///
    /// The in-flight fence is created in the signaled state so the first
    /// frame can proceed without waiting.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    ///
    /// # Errors
    ///
    /// Returns an error if any synchronization object creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::sync::FrameSync;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let frame_sync = FrameSync::new(device)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: Arc<Device>) -> RhiResult<Self> {
        let image_available_semaphore = Semaphore::new(device.clone())?;
        let render_finished_semaphore = Semaphore::new(device.clone())?;
        // Start signaled so the first wait doesn't block forever
        let in_flight_fence = Fence::new(device, true)?;

        info!("Created frame synchronization primitives");

        Ok(Self {
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        })
    }

    /// Returns a reference to the image available semaphore.
    ///
    /// This semaphore should be signaled by swapchain image acquisition
    /// and waited on before rendering to that image.
    #[inline]
    pub fn image_available_semaphore(&self) -> &Semaphore {
        &self.image_available_semaphore
    }

    /// Returns a reference to the render finished semaphore.
    ///
    /// This semaphore should be signaled when rendering is complete
    /// and waited on before presenting the image.
    #[inline]
    pub fn render_finished_semaphore(&self) -> &Semaphore {
        &self.render_finished_semaphore
    }

    /// Returns a reference to the in-flight fence.
    ///
    /// This fence should be signaled when command buffer execution completes
    /// and waited on before reusing frame resources.
    #[inline]
    pub fn in_flight_fence(&self) -> &Fence {
        &self.in_flight_fence
    }

    /// Returns the raw Vulkan handle for the image available semaphore.
    #[inline]
    pub fn image_available_handle(&self) -> vk::Semaphore {
        self.image_available_semaphore.handle()
    }

    /// Returns the raw Vulkan handle for the render finished semaphore.
    #[inline]
    pub fn render_finished_handle(&self) -> vk::Semaphore {
        self.render_finished_semaphore.handle()
    }

    /// Returns the raw Vulkan handle for the in-flight fence.
    #[inline]
    pub fn in_flight_fence_handle(&self) -> vk::Fence {
        self.in_flight_fence.handle()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_frames_in_flight_constant() {
        // Verify the constant is a reasonable value
        assert!(MAX_FRAMES_IN_FLIGHT >= 1);
        assert!(MAX_FRAMES_IN_FLIGHT <= 4);
    }

    #[test]
    fn test_semaphore_is_send_sync() {
        // Compile-time check that Semaphore is Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Semaphore>();
    }

    #[test]
    fn test_fence_is_send_sync() {
        // Compile-time check that Fence is Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fence>();
    }

    #[test]
    fn test_frame_sync_is_send_sync() {
        // Compile-time check that FrameSync is Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FrameSync>();
    }
}

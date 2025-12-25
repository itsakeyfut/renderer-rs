//! Main renderer orchestration.
//!
//! This module provides the main [`Renderer`] struct that coordinates
//! all Vulkan resources and rendering operations.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use ash::vk;
use tracing::{debug, error, info, warn};

use renderer_platform::{Surface, Window};
use renderer_rhi::device::Device;
use renderer_rhi::instance::Instance;
use renderer_rhi::physical_device::select_physical_device;
use renderer_rhi::swapchain::Swapchain;
use renderer_rhi::{RhiError, RhiResult};

use crate::MAX_FRAMES_IN_FLIGHT;

/// Per-frame synchronization resources.
struct FrameSync {
    /// Semaphore signaled when swapchain image is available.
    image_available: vk::Semaphore,
    /// Semaphore signaled when rendering is complete.
    render_finished: vk::Semaphore,
    /// Fence to wait for this frame's GPU work to complete.
    in_flight_fence: vk::Fence,
    /// Command pool for this frame.
    command_pool: vk::CommandPool,
    /// Command buffer for this frame.
    command_buffer: vk::CommandBuffer,
}

/// Main renderer that manages all Vulkan resources.
///
/// # Resource Destruction Order
///
/// Vulkan resources must be destroyed in the correct order:
/// 1. Wait for all GPU work to complete
/// 2. Destroy per-frame resources (semaphores, fences, command pools)
/// 3. Destroy swapchain
/// 4. Destroy surface
/// 5. Destroy device
/// 6. Destroy instance
///
/// ManuallyDrop is used to ensure correct destruction order.
pub struct Renderer {
    // Resources are wrapped in ManuallyDrop to control destruction order.
    // They are listed in REVERSE destruction order (last to be destroyed first in struct).
    /// Vulkan instance (destroyed last).
    instance: ManuallyDrop<Instance>,
    /// Logical device (destroyed after swapchain and surface).
    device: Arc<Device>,
    /// Window surface (destroyed after swapchain, before device).
    surface: ManuallyDrop<Surface>,
    /// Swapchain (destroyed first among these resources).
    swapchain: ManuallyDrop<Swapchain>,
    /// Per-frame synchronization resources.
    frame_sync: Vec<FrameSync>,
    /// Current frame index (0 to MAX_FRAMES_IN_FLIGHT - 1).
    current_frame: usize,
    /// Flag indicating swapchain needs recreation.
    framebuffer_resized: bool,
    /// Current window width.
    width: u32,
    /// Current window height.
    height: u32,
}

impl Renderer {
    /// Creates a new renderer for the given window.
    ///
    /// # Arguments
    ///
    /// * `window` - The window to render to
    ///
    /// # Errors
    ///
    /// Returns an error if any Vulkan resource creation fails.
    pub fn new(window: &Window) -> RhiResult<Self> {
        let width = window.width();
        let height = window.height();

        info!("Initializing Vulkan renderer ({}x{})", width, height);

        // Create Vulkan instance with validation in debug builds
        let enable_validation = cfg!(debug_assertions);
        let instance = Instance::new(enable_validation)?;

        // Create surface
        let surface = window
            .create_surface(instance.entry(), instance.handle())
            .map_err(|e| RhiError::SurfaceError(e.to_string()))?;

        // Select physical device
        let physical_device_info =
            select_physical_device(instance.handle(), surface.handle(), surface.loader())?;

        // Create logical device
        let device = Device::new(&instance, &physical_device_info)?;

        // Create swapchain
        let swapchain = Swapchain::new(&instance, device.clone(), surface.handle(), width, height)?;

        // Create per-frame synchronization resources
        let frame_sync = Self::create_frame_sync(&device, MAX_FRAMES_IN_FLIGHT)?;

        info!(
            "Renderer initialized: {} swapchain images, {} frames in flight",
            swapchain.image_count(),
            MAX_FRAMES_IN_FLIGHT
        );

        Ok(Self {
            instance: ManuallyDrop::new(instance),
            device,
            surface: ManuallyDrop::new(surface),
            swapchain: ManuallyDrop::new(swapchain),
            frame_sync,
            current_frame: 0,
            framebuffer_resized: false,
            width,
            height,
        })
    }

    /// Creates per-frame synchronization resources.
    fn create_frame_sync(device: &Device, count: usize) -> RhiResult<Vec<FrameSync>> {
        let mut frames = Vec::with_capacity(count);

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let graphics_family = device.queue_families().graphics_family.unwrap();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        for i in 0..count {
            let image_available =
                unsafe { device.handle().create_semaphore(&semaphore_info, None)? };
            let render_finished =
                unsafe { device.handle().create_semaphore(&semaphore_info, None)? };
            let in_flight_fence = unsafe { device.handle().create_fence(&fence_info, None)? };
            let command_pool = unsafe { device.handle().create_command_pool(&pool_info, None)? };

            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffers = unsafe { device.handle().allocate_command_buffers(&alloc_info)? };

            debug!("Created frame sync resources for frame {}", i);

            frames.push(FrameSync {
                image_available,
                render_finished,
                in_flight_fence,
                command_pool,
                command_buffer: command_buffers[0],
            });
        }

        Ok(frames)
    }

    /// Notifies the renderer that the window has been resized.
    ///
    /// The actual swapchain recreation will happen on the next frame.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            debug!("Ignoring resize to zero dimensions");
            return;
        }

        if width != self.width || height != self.height {
            info!(
                "Window resized: {}x{} -> {}x{}",
                self.width, self.height, width, height
            );
            self.width = width;
            self.height = height;
            self.framebuffer_resized = true;
        }
    }

    /// Waits for all in-flight frames to complete.
    ///
    /// This must be called before recreating the swapchain to ensure
    /// all semaphores are in a known state.
    fn wait_for_all_frames(&self) -> RhiResult<()> {
        let fences: Vec<vk::Fence> = self.frame_sync.iter().map(|f| f.in_flight_fence).collect();

        unsafe {
            self.device
                .handle()
                .wait_for_fences(&fences, true, u64::MAX)?;
        }

        Ok(())
    }

    /// Recreates the swapchain for the current window size.
    ///
    /// This also recreates all semaphores to ensure they are in a clean state.
    fn recreate_swapchain(&mut self) -> RhiResult<()> {
        // Wait for ALL frames to complete, not just the current one.
        // This ensures all semaphores are in a known state before recreation.
        self.wait_for_all_frames()?;

        self.swapchain.recreate(
            &self.instance,
            self.surface.handle(),
            self.width,
            self.height,
        )?;

        // Recreate semaphores to ensure they are in unsignaled state.
        // This is necessary because after swapchain recreation, old semaphores
        // may still be in a signaled state from previous acquire operations.
        self.recreate_semaphores()?;

        self.framebuffer_resized = false;
        Ok(())
    }

    /// Recreates all semaphores for frame synchronization.
    ///
    /// This is called during swapchain recreation to ensure all semaphores
    /// are in a clean, unsignaled state.
    fn recreate_semaphores(&mut self) -> RhiResult<()> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        for frame in &mut self.frame_sync {
            unsafe {
                // Destroy old semaphores
                self.device
                    .handle()
                    .destroy_semaphore(frame.image_available, None);
                self.device
                    .handle()
                    .destroy_semaphore(frame.render_finished, None);

                // Create new semaphores
                frame.image_available = self
                    .device
                    .handle()
                    .create_semaphore(&semaphore_info, None)?;
                frame.render_finished = self
                    .device
                    .handle()
                    .create_semaphore(&semaphore_info, None)?;
            }
        }

        debug!("Recreated {} semaphore pairs", self.frame_sync.len());
        Ok(())
    }

    /// Renders a frame.
    ///
    /// This method:
    /// 1. Checks if swapchain needs recreation (resize)
    /// 2. Waits for the previous frame using this index to complete
    /// 3. Acquires the next swapchain image
    /// 4. Records and submits rendering commands
    /// 5. Presents the image
    ///
    /// # Errors
    ///
    /// Returns an error if any Vulkan operation fails.
    pub fn render_frame(&mut self) -> RhiResult<()> {
        // Check if we need to recreate swapchain BEFORE acquiring an image.
        // This prevents acquiring with a semaphore that's already signaled.
        if self.framebuffer_resized {
            debug!("Resize requested, recreating swapchain before acquire");
            self.recreate_swapchain()?;
        }

        let frame = &self.frame_sync[self.current_frame];

        // Wait for this frame's previous work to complete
        unsafe {
            self.device
                .handle()
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
        }

        // Acquire next swapchain image
        let (image_index, _suboptimal) =
            match self.swapchain.acquire_next_image(frame.image_available) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    debug!("Swapchain out of date, recreating");
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(e) => return Err(RhiError::VulkanError(e)),
            };

        // Reset fence only after we're sure we'll submit work
        unsafe {
            self.device
                .handle()
                .reset_fences(&[frame.in_flight_fence])?;
        }

        // Record commands
        self.record_commands(image_index)?;

        // Submit
        let wait_semaphores = [frame.image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [frame.render_finished];
        let command_buffers = [frame.command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device.handle().queue_submit(
                self.device.graphics_queue(),
                &[submit_info],
                frame.in_flight_fence,
            )?;
        }

        // Present
        let present_result = self.swapchain.present(
            self.device.present_queue(),
            image_index,
            frame.render_finished,
        );

        let should_recreate = match present_result {
            Ok(suboptimal) => suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
            Err(e) => return Err(RhiError::VulkanError(e)),
        };

        // Advance to next frame first
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        // Recreate swapchain if needed (after advancing frame)
        if should_recreate {
            debug!("Swapchain suboptimal after present, recreating");
            self.recreate_swapchain()?;
        }

        Ok(())
    }

    /// Records rendering commands for a frame.
    fn record_commands(&self, image_index: u32) -> RhiResult<()> {
        let frame = &self.frame_sync[self.current_frame];
        let cmd = frame.command_buffer;

        // Reset and begin command buffer
        unsafe {
            self.device
                .handle()
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::empty());
            self.device
                .handle()
                .begin_command_buffer(cmd, &begin_info)?;
        }

        // Transition image to color attachment optimal
        let image = self.swapchain.image(image_index as usize);
        self.cmd_transition_image_layout(
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        // Begin rendering
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.image_view(image_index as usize))
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.15, 1.0], // Dark blue-gray
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent(),
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));

        unsafe {
            self.device
                .handle()
                .cmd_begin_rendering(cmd, &rendering_info);

            // Set viewport and scissor
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain.width() as f32,
                height: self.swapchain.height() as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.handle().cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent(),
            };
            self.device.handle().cmd_set_scissor(cmd, 0, &[scissor]);

            // TODO: Draw calls would go here

            self.device.handle().cmd_end_rendering(cmd);
        }

        // Transition image to present
        self.cmd_transition_image_layout(
            cmd,
            image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        // End command buffer
        unsafe {
            self.device.handle().end_command_buffer(cmd)?;
        }

        Ok(())
    }

    /// Records an image layout transition.
    fn cmd_transition_image_layout(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let (src_stage, src_access, dst_stage, dst_access) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::empty(),
            ),
            _ => {
                warn!(
                    "Unhandled layout transition: {:?} -> {:?}",
                    old_layout, new_layout
                );
                (
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                )
            }
        };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);

        unsafe {
            self.device.handle().cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Returns the current swapchain extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent()
    }

    /// Returns the swapchain format.
    pub fn format(&self) -> vk::Format {
        self.swapchain.format()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Wait for all GPU work to complete before destroying resources
        if let Err(e) = self.device.wait_idle() {
            error!(
                "Failed to wait for device idle during renderer drop: {:?}",
                e
            );
        }

        // Destroy per-frame resources first
        for frame in &self.frame_sync {
            unsafe {
                self.device
                    .handle()
                    .destroy_semaphore(frame.image_available, None);
                self.device
                    .handle()
                    .destroy_semaphore(frame.render_finished, None);
                self.device
                    .handle()
                    .destroy_fence(frame.in_flight_fence, None);
                self.device
                    .handle()
                    .destroy_command_pool(frame.command_pool, None);
            }
        }

        // Manually drop resources in correct order:
        // 1. Swapchain (uses device and surface)
        // 2. Surface (uses instance)
        // 3. Device is Arc, will be dropped when all refs are gone
        // 4. Instance (must be last)
        unsafe {
            ManuallyDrop::drop(&mut self.swapchain);
            ManuallyDrop::drop(&mut self.surface);
            // Device (Arc) will be dropped automatically
            ManuallyDrop::drop(&mut self.instance);
        }

        info!("Renderer destroyed");
    }
}

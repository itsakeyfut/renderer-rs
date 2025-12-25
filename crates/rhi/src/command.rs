//! Command pool and command buffer management.
//!
//! This module provides wrappers for VkCommandPool and VkCommandBuffer,
//! enabling safe recording and submission of Vulkan commands.
//!
//! # Overview
//!
//! - [`CommandPool`] manages VkCommandPool creation and command buffer allocation
//! - [`CommandBuffer`] wraps VkCommandBuffer with methods for recording commands
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_rhi::device::Device;
//! use renderer_rhi::command::{CommandPool, CommandBuffer};
//!
//! # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
//! // Create command pool for graphics queue
//! let queue_family = device.queue_families().graphics_family.unwrap();
//! let pool = CommandPool::new(device.clone(), queue_family)?;
//!
//! // Allocate command buffer
//! let cmd = CommandBuffer::new(device.clone(), &pool)?;
//!
//! // Record commands
//! cmd.begin()?;
//! // ... record rendering commands ...
//! cmd.end()?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;
use tracing::info;

use crate::device::Device;
use crate::error::RhiResult;

/// Vulkan command pool wrapper.
///
/// A command pool is used to allocate command buffers. Each pool is associated
/// with a specific queue family and can only allocate command buffers that
/// will be submitted to queues of that family.
///
/// # Thread Safety
///
/// Command pools are not thread-safe. For multi-threaded command recording,
/// create a separate pool per thread.
pub struct CommandPool {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan command pool handle.
    pool: vk::CommandPool,
    /// Queue family index this pool belongs to.
    queue_family_index: u32,
}

impl CommandPool {
    /// Creates a new command pool for the specified queue family.
    ///
    /// The pool is created with the `RESET_COMMAND_BUFFER` flag, allowing
    /// individual command buffers to be reset without resetting the entire pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `queue_family_index` - The queue family for command buffer submission
    ///
    /// # Errors
    ///
    /// Returns an error if command pool creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::command::CommandPool;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let graphics_family = device.queue_families().graphics_family.unwrap();
    /// let pool = CommandPool::new(device, graphics_family)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: Arc<Device>, queue_family_index: u32) -> RhiResult<Self> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let pool = unsafe { device.handle().create_command_pool(&create_info, None)? };

        info!(
            "Command pool created for queue family {}",
            queue_family_index
        );

        Ok(Self {
            device,
            pool,
            queue_family_index,
        })
    }

    /// Creates a transient command pool for short-lived command buffers.
    ///
    /// Transient pools are optimized for command buffers that are recorded
    /// once and submitted, then discarded. This is useful for one-time
    /// transfer operations.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `queue_family_index` - The queue family for command buffer submission
    ///
    /// # Errors
    ///
    /// Returns an error if command pool creation fails.
    pub fn new_transient(device: Arc<Device>, queue_family_index: u32) -> RhiResult<Self> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            );

        let pool = unsafe { device.handle().create_command_pool(&create_info, None)? };

        info!(
            "Transient command pool created for queue family {}",
            queue_family_index
        );

        Ok(Self {
            device,
            pool,
            queue_family_index,
        })
    }

    /// Returns the Vulkan command pool handle.
    #[inline]
    pub fn handle(&self) -> vk::CommandPool {
        self.pool
    }

    /// Returns the queue family index this pool belongs to.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Allocates a primary command buffer from this pool.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn allocate_command_buffer(&self) -> RhiResult<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe { self.device.handle().allocate_command_buffers(&alloc_info)? };
        Ok(buffers[0])
    }

    /// Allocates multiple primary command buffers from this pool.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of command buffers to allocate
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn allocate_command_buffers(&self, count: u32) -> RhiResult<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let buffers = unsafe { self.device.handle().allocate_command_buffers(&alloc_info)? };
        Ok(buffers)
    }

    /// Allocates a secondary command buffer from this pool.
    ///
    /// Secondary command buffers can be executed from primary command buffers
    /// and are useful for reusable command sequences.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn allocate_secondary_command_buffer(&self) -> RhiResult<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);

        let buffers = unsafe { self.device.handle().allocate_command_buffers(&alloc_info)? };
        Ok(buffers[0])
    }

    /// Resets the entire command pool, returning all allocated command buffers
    /// to their initial state.
    ///
    /// # Arguments
    ///
    /// * `release_resources` - If true, also releases memory back to the system
    ///
    /// # Errors
    ///
    /// Returns an error if the reset fails.
    pub fn reset(&self, release_resources: bool) -> RhiResult<()> {
        let flags = if release_resources {
            vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandPoolResetFlags::empty()
        };

        unsafe {
            self.device.handle().reset_command_pool(self.pool, flags)?;
        }

        Ok(())
    }

    /// Returns a reference to the device.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_command_pool(self.pool, None);
        }
        info!(
            "Command pool destroyed for queue family {}",
            self.queue_family_index
        );
    }
}

/// Vulkan command buffer wrapper.
///
/// Provides a safe interface for recording Vulkan commands. The command buffer
/// wraps the raw VkCommandBuffer handle and provides methods for common
/// rendering operations.
///
/// # Command Recording
///
/// Commands are recorded between `begin()` and `end()` calls:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use renderer_rhi::device::Device;
/// # use renderer_rhi::command::{CommandPool, CommandBuffer};
/// # fn example(device: Arc<Device>, pool: &CommandPool) -> Result<(), renderer_rhi::RhiError> {
/// let cmd = CommandBuffer::new(device, pool)?;
///
/// cmd.begin()?;
/// // Record commands here...
/// cmd.end()?;
/// # Ok(())
/// # }
/// ```
///
/// # Note
///
/// The command buffer does NOT own the underlying VkCommandBuffer handle.
/// The handle is freed when the owning CommandPool is destroyed.
pub struct CommandBuffer {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan command buffer handle.
    buffer: vk::CommandBuffer,
}

impl CommandBuffer {
    /// Creates a new command buffer from the given pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `pool` - The command pool to allocate from
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn new(device: Arc<Device>, pool: &CommandPool) -> RhiResult<Self> {
        let buffer = pool.allocate_command_buffer()?;
        Ok(Self { device, buffer })
    }

    /// Wraps an existing command buffer handle.
    ///
    /// This is useful when command buffers are allocated elsewhere
    /// but need the convenience methods of this wrapper.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `buffer` - The raw command buffer handle
    #[inline]
    pub fn from_handle(device: Arc<Device>, buffer: vk::CommandBuffer) -> Self {
        Self { device, buffer }
    }

    /// Returns the raw Vulkan command buffer handle.
    #[inline]
    pub fn handle(&self) -> vk::CommandBuffer {
        self.buffer
    }

    // =========================================================================
    // Recording Control
    // =========================================================================

    /// Begins recording commands to the buffer.
    ///
    /// The buffer is set up for one-time submission by default.
    ///
    /// # Errors
    ///
    /// Returns an error if beginning fails (e.g., if already recording).
    pub fn begin(&self) -> RhiResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .handle()
                .begin_command_buffer(self.buffer, &begin_info)?;
        }

        Ok(())
    }

    /// Begins recording commands that can be resubmitted.
    ///
    /// Use this when the command buffer will be submitted multiple times.
    ///
    /// # Errors
    ///
    /// Returns an error if beginning fails.
    pub fn begin_reusable(&self) -> RhiResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
                .handle()
                .begin_command_buffer(self.buffer, &begin_info)?;
        }

        Ok(())
    }

    /// Ends recording commands to the buffer.
    ///
    /// After this call, the command buffer is ready for submission.
    ///
    /// # Errors
    ///
    /// Returns an error if ending fails (e.g., if not recording).
    pub fn end(&self) -> RhiResult<()> {
        unsafe {
            self.device.handle().end_command_buffer(self.buffer)?;
        }

        Ok(())
    }

    /// Resets the command buffer to its initial state.
    ///
    /// This allows the buffer to be re-recorded without reallocating.
    ///
    /// # Errors
    ///
    /// Returns an error if the reset fails.
    pub fn reset(&self) -> RhiResult<()> {
        unsafe {
            self.device
                .handle()
                .reset_command_buffer(self.buffer, vk::CommandBufferResetFlags::empty())?;
        }

        Ok(())
    }

    // =========================================================================
    // Dynamic Rendering (Vulkan 1.3)
    // =========================================================================

    /// Begins dynamic rendering.
    ///
    /// This is the Vulkan 1.3 way to start rendering without a VkRenderPass.
    ///
    /// # Arguments
    ///
    /// * `rendering_info` - Configuration for the render pass
    pub fn begin_rendering(&self, rendering_info: &vk::RenderingInfo) {
        unsafe {
            self.device
                .handle()
                .cmd_begin_rendering(self.buffer, rendering_info);
        }
    }

    /// Ends dynamic rendering.
    pub fn end_rendering(&self) {
        unsafe {
            self.device.handle().cmd_end_rendering(self.buffer);
        }
    }

    // =========================================================================
    // Pipeline Binding
    // =========================================================================

    /// Binds a pipeline to the command buffer.
    ///
    /// # Arguments
    ///
    /// * `bind_point` - Whether this is a graphics or compute pipeline
    /// * `pipeline` - The pipeline to bind
    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .handle()
                .cmd_bind_pipeline(self.buffer, bind_point, pipeline);
        }
    }

    /// Binds vertex buffers to the command buffer.
    ///
    /// # Arguments
    ///
    /// * `first_binding` - First vertex input binding to update
    /// * `buffers` - Slice of buffer handles
    /// * `offsets` - Byte offsets into each buffer
    pub fn bind_vertex_buffers(
        &self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            self.device.handle().cmd_bind_vertex_buffers(
                self.buffer,
                first_binding,
                buffers,
                offsets,
            );
        }
    }

    /// Binds an index buffer to the command buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The index buffer
    /// * `offset` - Byte offset into the buffer
    /// * `index_type` - Type of indices (UINT16 or UINT32)
    pub fn bind_index_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            self.device
                .handle()
                .cmd_bind_index_buffer(self.buffer, buffer, offset, index_type);
        }
    }

    /// Binds descriptor sets to the command buffer.
    ///
    /// # Arguments
    ///
    /// * `bind_point` - Whether this is for graphics or compute
    /// * `layout` - The pipeline layout
    /// * `first_set` - First descriptor set to update
    /// * `descriptor_sets` - Slice of descriptor sets
    /// * `dynamic_offsets` - Dynamic offsets for dynamic descriptors
    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.device.handle().cmd_bind_descriptor_sets(
                self.buffer,
                bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }

    // =========================================================================
    // Dynamic State
    // =========================================================================

    /// Sets the viewport dynamically.
    ///
    /// # Arguments
    ///
    /// * `viewport` - The viewport configuration
    pub fn set_viewport(&self, viewport: &vk::Viewport) {
        unsafe {
            self.device
                .handle()
                .cmd_set_viewport(self.buffer, 0, std::slice::from_ref(viewport));
        }
    }

    /// Sets multiple viewports dynamically.
    ///
    /// # Arguments
    ///
    /// * `first_viewport` - First viewport index to update
    /// * `viewports` - Slice of viewport configurations
    pub fn set_viewports(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.device
                .handle()
                .cmd_set_viewport(self.buffer, first_viewport, viewports);
        }
    }

    /// Sets the scissor rectangle dynamically.
    ///
    /// # Arguments
    ///
    /// * `scissor` - The scissor rectangle
    pub fn set_scissor(&self, scissor: &vk::Rect2D) {
        unsafe {
            self.device
                .handle()
                .cmd_set_scissor(self.buffer, 0, std::slice::from_ref(scissor));
        }
    }

    /// Sets multiple scissor rectangles dynamically.
    ///
    /// # Arguments
    ///
    /// * `first_scissor` - First scissor index to update
    /// * `scissors` - Slice of scissor rectangles
    pub fn set_scissors(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.device
                .handle()
                .cmd_set_scissor(self.buffer, first_scissor, scissors);
        }
    }

    // =========================================================================
    // Drawing Commands
    // =========================================================================

    /// Issues a non-indexed draw command.
    ///
    /// # Arguments
    ///
    /// * `vertex_count` - Number of vertices to draw
    /// * `instance_count` - Number of instances to draw
    /// * `first_vertex` - Offset to the first vertex
    /// * `first_instance` - Offset to the first instance
    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.handle().cmd_draw(
                self.buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    /// Issues an indexed draw command.
    ///
    /// # Arguments
    ///
    /// * `index_count` - Number of indices to draw
    /// * `instance_count` - Number of instances to draw
    /// * `first_index` - Offset to the first index
    /// * `vertex_offset` - Constant added to each index
    /// * `first_instance` - Offset to the first instance
    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.handle().cmd_draw_indexed(
                self.buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    /// Issues an indirect draw command.
    ///
    /// Draw parameters are read from a buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer containing draw parameters
    /// * `offset` - Byte offset into the buffer
    /// * `draw_count` - Number of draws to execute
    /// * `stride` - Stride between draw parameter structures
    pub fn draw_indirect(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.device
                .handle()
                .cmd_draw_indirect(self.buffer, buffer, offset, draw_count, stride);
        }
    }

    /// Issues an indirect indexed draw command.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer containing draw parameters
    /// * `offset` - Byte offset into the buffer
    /// * `draw_count` - Number of draws to execute
    /// * `stride` - Stride between draw parameter structures
    pub fn draw_indexed_indirect(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.device.handle().cmd_draw_indexed_indirect(
                self.buffer,
                buffer,
                offset,
                draw_count,
                stride,
            );
        }
    }

    // =========================================================================
    // Compute Commands
    // =========================================================================

    /// Dispatches compute work.
    ///
    /// # Arguments
    ///
    /// * `group_count_x` - Number of workgroups in X dimension
    /// * `group_count_y` - Number of workgroups in Y dimension
    /// * `group_count_z` - Number of workgroups in Z dimension
    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device.handle().cmd_dispatch(
                self.buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }

    /// Dispatches compute work indirectly.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer containing dispatch parameters
    /// * `offset` - Byte offset into the buffer
    pub fn dispatch_indirect(&self, buffer: vk::Buffer, offset: vk::DeviceSize) {
        unsafe {
            self.device
                .handle()
                .cmd_dispatch_indirect(self.buffer, buffer, offset);
        }
    }

    // =========================================================================
    // Push Constants
    // =========================================================================

    /// Updates push constant data.
    ///
    /// # Arguments
    ///
    /// * `layout` - Pipeline layout containing push constant ranges
    /// * `stages` - Shader stages that will use the push constants
    /// * `offset` - Byte offset within push constant memory
    /// * `data` - Data to push
    ///
    /// # Type Parameters
    ///
    /// * `T` - The push constant data type (must be Copy)
    pub fn push_constants<T: Copy>(
        &self,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        data: &T,
    ) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of::<T>())
        };
        unsafe {
            self.device
                .handle()
                .cmd_push_constants(self.buffer, layout, stages, offset, bytes);
        }
    }

    /// Updates push constant data from a byte slice.
    ///
    /// # Arguments
    ///
    /// * `layout` - Pipeline layout containing push constant ranges
    /// * `stages` - Shader stages that will use the push constants
    /// * `offset` - Byte offset within push constant memory
    /// * `data` - Raw bytes to push
    pub fn push_constants_bytes(
        &self,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.device
                .handle()
                .cmd_push_constants(self.buffer, layout, stages, offset, data);
        }
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    /// Inserts a pipeline barrier for synchronization.
    ///
    /// # Arguments
    ///
    /// * `src_stage` - Source pipeline stages
    /// * `dst_stage` - Destination pipeline stages
    /// * `image_barriers` - Image memory barriers
    pub fn pipeline_barrier(
        &self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        image_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.handle().cmd_pipeline_barrier(
                self.buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                image_barriers,
            );
        }
    }

    /// Inserts a pipeline barrier with all barrier types.
    ///
    /// # Arguments
    ///
    /// * `src_stage` - Source pipeline stages
    /// * `dst_stage` - Destination pipeline stages
    /// * `dependency_flags` - Dependency flags
    /// * `memory_barriers` - Global memory barriers
    /// * `buffer_barriers` - Buffer memory barriers
    /// * `image_barriers` - Image memory barriers
    pub fn pipeline_barrier_full(
        &self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_barriers: &[vk::BufferMemoryBarrier],
        image_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.handle().cmd_pipeline_barrier(
                self.buffer,
                src_stage,
                dst_stage,
                dependency_flags,
                memory_barriers,
                buffer_barriers,
                image_barriers,
            );
        }
    }

    // =========================================================================
    // Copy Commands
    // =========================================================================

    /// Copies data between buffers.
    ///
    /// # Arguments
    ///
    /// * `src` - Source buffer
    /// * `dst` - Destination buffer
    /// * `regions` - Copy regions
    pub fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        unsafe {
            self.device
                .handle()
                .cmd_copy_buffer(self.buffer, src, dst, regions);
        }
    }

    /// Copies data from a buffer to an image.
    ///
    /// # Arguments
    ///
    /// * `src` - Source buffer
    /// * `dst` - Destination image
    /// * `dst_layout` - Current layout of destination image
    /// * `regions` - Copy regions
    pub fn copy_buffer_to_image(
        &self,
        src: vk::Buffer,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.handle().cmd_copy_buffer_to_image(
                self.buffer,
                src,
                dst,
                dst_layout,
                regions,
            );
        }
    }

    /// Copies data from an image to a buffer.
    ///
    /// # Arguments
    ///
    /// * `src` - Source image
    /// * `src_layout` - Current layout of source image
    /// * `dst` - Destination buffer
    /// * `regions` - Copy regions
    pub fn copy_image_to_buffer(
        &self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.handle().cmd_copy_image_to_buffer(
                self.buffer,
                src,
                src_layout,
                dst,
                regions,
            );
        }
    }

    /// Copies data between images.
    ///
    /// # Arguments
    ///
    /// * `src` - Source image
    /// * `src_layout` - Current layout of source image
    /// * `dst` - Destination image
    /// * `dst_layout` - Current layout of destination image
    /// * `regions` - Copy regions
    pub fn copy_image(
        &self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        unsafe {
            self.device.handle().cmd_copy_image(
                self.buffer,
                src,
                src_layout,
                dst,
                dst_layout,
                regions,
            );
        }
    }

    /// Blits (scaled copy) between images.
    ///
    /// # Arguments
    ///
    /// * `src` - Source image
    /// * `src_layout` - Current layout of source image
    /// * `dst` - Destination image
    /// * `dst_layout` - Current layout of destination image
    /// * `regions` - Blit regions
    /// * `filter` - Filtering to apply during scaling
    pub fn blit_image(
        &self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        unsafe {
            self.device.handle().cmd_blit_image(
                self.buffer,
                src,
                src_layout,
                dst,
                dst_layout,
                regions,
                filter,
            );
        }
    }

    // =========================================================================
    // Clear Commands
    // =========================================================================

    /// Clears regions of a color image.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to clear
    /// * `image_layout` - Current layout of the image
    /// * `color` - Clear color value
    /// * `ranges` - Subresource ranges to clear
    pub fn clear_color_image(
        &self,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        color: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            self.device.handle().cmd_clear_color_image(
                self.buffer,
                image,
                image_layout,
                color,
                ranges,
            );
        }
    }

    /// Clears regions of a depth/stencil image.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to clear
    /// * `image_layout` - Current layout of the image
    /// * `depth_stencil` - Clear value
    /// * `ranges` - Subresource ranges to clear
    pub fn clear_depth_stencil_image(
        &self,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        depth_stencil: &vk::ClearDepthStencilValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            self.device.handle().cmd_clear_depth_stencil_image(
                self.buffer,
                image,
                image_layout,
                depth_stencil,
                ranges,
            );
        }
    }

    // =========================================================================
    // Secondary Command Buffer Execution
    // =========================================================================

    /// Executes secondary command buffers.
    ///
    /// # Arguments
    ///
    /// * `command_buffers` - Secondary command buffers to execute
    pub fn execute_commands(&self, command_buffers: &[vk::CommandBuffer]) {
        unsafe {
            self.device
                .handle()
                .cmd_execute_commands(self.buffer, command_buffers);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_buffer_is_send() {
        // Compile-time check that CommandBuffer is Send
        fn assert_send<T: Send>() {}
        assert_send::<CommandBuffer>();
    }

    #[test]
    fn test_command_pool_is_send() {
        // Compile-time check that CommandPool is Send
        fn assert_send<T: Send>() {}
        assert_send::<CommandPool>();
    }
}

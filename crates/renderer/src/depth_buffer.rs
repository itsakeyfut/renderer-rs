//! Depth buffer management.
//!
//! This module handles depth buffer creation and management for depth testing
//! in 3D rendering. It creates a depth image with GPU-only memory and an
//! associated image view.
//!
//! # Overview
//!
//! - [`DepthBuffer`] wraps a VkImage and VkImageView for depth testing
//! - Uses D32_SFLOAT format by default (32-bit floating point)
//! - Memory is managed by gpu-allocator
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_rhi::device::Device;
//! use renderer_renderer::depth_buffer::DepthBuffer;
//! use ash::vk;
//!
//! # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
//! // Create a depth buffer with D32_SFLOAT format
//! let depth_buffer = DepthBuffer::new(
//!     device,
//!     1920,
//!     1080,
//!     vk::Format::D32_SFLOAT,
//! )?;
//!
//! // Use the depth buffer in rendering
//! let image_view = depth_buffer.image_view();
//! let format = depth_buffer.format();
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use tracing::{debug, info};

use renderer_rhi::device::Device;
use renderer_rhi::{RhiError, RhiResult};

/// Default depth buffer format (32-bit floating point).
pub const DEFAULT_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Depth buffer for depth testing.
///
/// This struct manages a Vulkan image and image view used for depth testing.
/// The depth buffer is created with GPU-only memory for optimal performance.
///
/// # Thread Safety
///
/// The depth buffer is immutable after creation and can be safely shared
/// between threads. However, the underlying Vulkan resources should be
/// synchronized with appropriate barriers during rendering.
///
/// # Resource Destruction
///
/// Resources are destroyed in the following order:
/// 1. Image view
/// 2. Image
/// 3. Memory allocation
pub struct DepthBuffer {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan image handle.
    image: vk::Image,
    /// Vulkan image view handle.
    image_view: vk::ImageView,
    /// GPU memory allocation.
    allocation: Option<Allocation>,
    /// Depth format.
    format: vk::Format,
    /// Depth buffer dimensions.
    extent: vk::Extent2D,
}

impl DepthBuffer {
    /// Creates a new depth buffer with the specified dimensions and format.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    /// * `format` - Depth format (D32_SFLOAT recommended)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image creation fails
    /// - Memory allocation fails
    /// - Image view creation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_renderer::depth_buffer::DepthBuffer;
    /// use ash::vk;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let depth_buffer = DepthBuffer::new(
    ///     device,
    ///     1920,
    ///     1080,
    ///     vk::Format::D32_SFLOAT,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        device: Arc<Device>,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> RhiResult<Self> {
        if width == 0 || height == 0 {
            return Err(RhiError::InvalidHandle(
                "Depth buffer dimensions must be greater than 0".to_string(),
            ));
        }

        let extent = vk::Extent2D { width, height };

        // Create depth image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.handle().create_image(&image_info, None)? };

        // Get memory requirements and allocate
        let requirements = unsafe { device.handle().get_image_memory_requirements(image) };

        let allocation = {
            let mut allocator = device.allocator().lock().unwrap();
            allocator.allocate(&AllocationCreateDesc {
                name: "depth_buffer",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false, // Optimal tiling is not linear
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?
        };

        // Bind memory to image
        unsafe {
            device
                .handle()
                .bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = unsafe { device.handle().create_image_view(&view_info, None)? };

        info!("Created depth buffer: {}x{} ({:?})", width, height, format);

        Ok(Self {
            device,
            image,
            image_view,
            allocation: Some(allocation),
            format,
            extent,
        })
    }

    /// Creates a depth buffer with the default format (D32_SFLOAT).
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    ///
    /// # Errors
    ///
    /// Returns an error if depth buffer creation fails.
    pub fn with_default_format(device: Arc<Device>, width: u32, height: u32) -> RhiResult<Self> {
        Self::new(device, width, height, DEFAULT_DEPTH_FORMAT)
    }

    /// Returns the Vulkan image handle.
    #[inline]
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Returns the Vulkan image view handle.
    #[inline]
    pub fn image_view(&self) -> vk::ImageView {
        self.image_view
    }

    /// Returns the depth format.
    #[inline]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Returns the depth buffer extent (width and height).
    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Returns the width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.extent.width
    }

    /// Returns the height in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.extent.height
    }
}

impl Drop for DepthBuffer {
    fn drop(&mut self) {
        // Destroy resources in correct order:
        // 1. Image view (depends on image)
        // 2. Image (depends on allocation)
        // 3. Allocation (frees memory)
        unsafe {
            self.device
                .handle()
                .destroy_image_view(self.image_view, None);
            self.device.handle().destroy_image(self.image, None);
        }

        // Free allocation
        if let Some(allocation) = self.allocation.take() {
            let mut allocator = self.device.allocator().lock().unwrap();
            if let Err(e) = allocator.free(allocation) {
                tracing::error!("Failed to free depth buffer allocation: {:?}", e);
            }
        }

        debug!(
            "Destroyed depth buffer: {}x{}",
            self.extent.width, self.extent.height
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_depth_format() {
        assert_eq!(DEFAULT_DEPTH_FORMAT, vk::Format::D32_SFLOAT);
    }

    #[test]
    fn test_depth_format_is_valid() {
        // Verify D32_SFLOAT is a depth format
        let format = DEFAULT_DEPTH_FORMAT;
        assert!(matches!(
            format,
            vk::Format::D32_SFLOAT
                | vk::Format::D32_SFLOAT_S8_UINT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D16_UNORM
        ));
    }
}

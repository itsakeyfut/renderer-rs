//! Swapchain management.
//!
//! This module handles VkSwapchainKHR creation, image acquisition, and presentation.
//!
//! # Overview
//!
//! The [`Swapchain`] struct provides a safe abstraction over the Vulkan swapchain,
//! including:
//! - Surface capability querying
//! - Format and present mode selection
//! - Image view creation and management
//! - Resize handling
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_rhi::instance::Instance;
//! use renderer_rhi::device::Device;
//! use renderer_rhi::swapchain::Swapchain;
//! use ash::vk;
//!
//! // Assume instance, device, and surface are already created
//! let instance = Instance::new(false).expect("Failed to create instance");
//! // ... create surface and device ...
//!
//! // Create swapchain
//! // let swapchain = Swapchain::new(&instance, device.clone(), surface, 800, 600)
//! //     .expect("Failed to create swapchain");
//!
//! // In render loop:
//! // let (image_index, suboptimal) = swapchain.acquire_next_image(semaphore)?;
//! // ... render to swapchain.image(image_index) ...
//! // let needs_resize = swapchain.present(queue, image_index, render_finished_semaphore)?;
//! ```

use std::sync::Arc;

use ash::vk;
use tracing::{debug, info, warn};

use crate::device::Device;
use crate::error::RhiError;
use crate::instance::Instance;

/// Swapchain surface support details.
///
/// Contains information about what the surface supports for swapchain creation.
#[derive(Debug, Clone)]
pub struct SwapchainSupportDetails {
    /// Surface capabilities (min/max image count, extents, transforms, etc.)
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    /// Supported surface formats (format and color space combinations)
    pub formats: Vec<vk::SurfaceFormatKHR>,
    /// Supported present modes (FIFO, MAILBOX, IMMEDIATE, etc.)
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    /// Queries swapchain support details for a physical device and surface.
    ///
    /// # Arguments
    ///
    /// * `physical_device` - The physical device to query
    /// * `surface` - The surface to query against
    /// * `surface_loader` - The surface extension loader
    ///
    /// # Errors
    ///
    /// Returns an error if any of the queries fail.
    pub fn query(
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &ash::khr::surface::Instance,
    ) -> Result<Self, RhiError> {
        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };

        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };

        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };

        debug!(
            "Swapchain support: {} formats, {} present modes, image count: {}-{}",
            formats.len(),
            present_modes.len(),
            capabilities.min_image_count,
            if capabilities.max_image_count == 0 {
                "unlimited".to_string()
            } else {
                capabilities.max_image_count.to_string()
            }
        );

        Ok(Self {
            capabilities,
            formats,
            present_modes,
        })
    }

    /// Checks if the swapchain support is adequate for rendering.
    ///
    /// Returns true if at least one format and one present mode are available.
    #[inline]
    pub fn is_adequate(&self) -> bool {
        !self.formats.is_empty() && !self.present_modes.is_empty()
    }
}

/// Vulkan swapchain wrapper.
///
/// This struct manages the swapchain and its associated resources:
/// - Swapchain images (owned by the swapchain, not explicitly managed)
/// - Image views (managed by this struct)
///
/// # Thread Safety
///
/// The swapchain is not thread-safe. Only one thread should interact with
/// it at a time. For multi-threaded rendering, proper synchronization is required.
pub struct Swapchain {
    /// Reference to the logical device
    device: Arc<Device>,
    /// Swapchain extension loader
    swapchain_loader: ash::khr::swapchain::Device,
    /// Swapchain handle
    swapchain: vk::SwapchainKHR,
    /// Swapchain images (owned by the swapchain)
    images: Vec<vk::Image>,
    /// Image views for the swapchain images
    image_views: Vec<vk::ImageView>,
    /// Swapchain image format
    format: vk::Format,
    /// Swapchain color space
    color_space: vk::ColorSpaceKHR,
    /// Swapchain extent (resolution)
    extent: vk::Extent2D,
    /// Present mode
    present_mode: vk::PresentModeKHR,
}

impl Swapchain {
    /// Creates a new swapchain.
    ///
    /// This function creates a swapchain with:
    /// - Preferred format: B8G8R8A8_SRGB with SRGB_NONLINEAR color space
    /// - Preferred present mode: MAILBOX (triple buffering), fallback to FIFO (vsync)
    /// - Image usage: COLOR_ATTACHMENT
    ///
    /// # Arguments
    ///
    /// * `instance` - The Vulkan instance
    /// * `device` - The logical device
    /// * `surface` - The window surface
    /// * `width` - Desired swapchain width
    /// * `height` - Desired swapchain height
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Surface queries fail
    /// - No suitable format or present mode is available
    /// - Swapchain creation fails
    /// - Image view creation fails
    pub fn new(
        instance: &Instance,
        device: Arc<Device>,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
    ) -> Result<Self, RhiError> {
        Self::create_internal(
            instance,
            device,
            surface,
            width,
            height,
            vk::SwapchainKHR::null(),
        )
    }

    /// Creates a new swapchain, optionally reusing resources from an old one.
    ///
    /// This is the internal creation function that supports both initial creation
    /// and recreation for resize operations.
    fn create_internal(
        instance: &Instance,
        device: Arc<Device>,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Self, RhiError> {
        let swapchain_loader = ash::khr::swapchain::Device::new(instance.handle(), device.handle());
        let surface_loader = ash::khr::surface::Instance::new(instance.entry(), instance.handle());

        // Query swapchain support
        let support =
            SwapchainSupportDetails::query(device.physical_device(), surface, &surface_loader)?;

        if !support.is_adequate() {
            return Err(RhiError::SwapchainError(
                "Inadequate swapchain support (no formats or present modes)".to_string(),
            ));
        }

        // Select optimal settings
        let surface_format = choose_surface_format(&support.formats);
        let present_mode = choose_present_mode(&support.present_modes);
        let extent = choose_extent(&support.capabilities, width, height);

        // Determine image count (prefer triple buffering)
        let image_count = determine_image_count(&support.capabilities);

        info!(
            "Creating swapchain: {}x{}, format {:?}, color space {:?}, present mode {:?}, {} images",
            extent.width,
            extent.height,
            surface_format.format,
            surface_format.color_space,
            present_mode,
            image_count
        );

        // Handle queue family sharing
        let queue_families = device.queue_families();
        let graphics_family = queue_families.graphics_family.unwrap();
        let present_family = queue_families.present_family.unwrap();
        let queue_family_indices = [graphics_family, present_family];

        let (sharing_mode, queue_family_indices_slice) = if graphics_family != present_family {
            debug!(
                "Using CONCURRENT sharing mode between graphics ({}) and present ({}) queues",
                graphics_family, present_family
            );
            (vk::SharingMode::CONCURRENT, queue_family_indices.as_slice())
        } else {
            debug!("Using EXCLUSIVE sharing mode (same queue family for graphics and present)");
            (vk::SharingMode::EXCLUSIVE, &[][..])
        };

        // Create swapchain
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(queue_family_indices_slice)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };

        // Get swapchain images
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        info!("Swapchain created with {} images", images.len());

        // Create image views
        let image_views = create_image_views(&device, &images, surface_format.format)?;

        Ok(Self {
            device,
            swapchain_loader,
            swapchain,
            images,
            image_views,
            format: surface_format.format,
            color_space: surface_format.color_space,
            extent,
            present_mode,
        })
    }

    /// Recreates the swapchain for a new window size.
    ///
    /// This should be called when the window is resized or when `acquire_next_image`
    /// or `present` return that the swapchain is suboptimal or out of date.
    ///
    /// # Arguments
    ///
    /// * `instance` - The Vulkan instance
    /// * `surface` - The window surface
    /// * `width` - New swapchain width
    /// * `height` - New swapchain height
    ///
    /// # Errors
    ///
    /// Returns an error if swapchain recreation fails.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - All operations using the old swapchain have completed
    /// - No command buffers referencing old swapchain images are in flight
    pub fn recreate(
        &mut self,
        instance: &Instance,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        // Wait for device to be idle before recreating
        self.device.wait_idle()?;

        info!("Recreating swapchain for new size: {}x{}", width, height);

        // Destroy old image views (images are owned by the swapchain and destroyed automatically)
        self.destroy_image_views();

        // Create new swapchain with old swapchain handle for resource reuse
        let old_swapchain = self.swapchain;
        let mut new_swapchain = Self::create_internal(
            instance,
            self.device.clone(),
            surface,
            width,
            height,
            old_swapchain,
        )?;

        // Destroy old swapchain
        unsafe {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }

        // Update self with new swapchain data using std::mem::take to move out of Drop type
        self.swapchain = new_swapchain.swapchain;
        self.images = std::mem::take(&mut new_swapchain.images);
        self.image_views = std::mem::take(&mut new_swapchain.image_views);
        self.format = new_swapchain.format;
        self.color_space = new_swapchain.color_space;
        self.extent = new_swapchain.extent;
        self.present_mode = new_swapchain.present_mode;

        // Clear the new_swapchain's swapchain handle to prevent double-free in its Drop impl
        new_swapchain.swapchain = vk::SwapchainKHR::null();
        // new_swapchain will now drop with empty vectors and null swapchain handle

        Ok(())
    }

    /// Acquires the next swapchain image for rendering.
    ///
    /// # Arguments
    ///
    /// * `semaphore` - Semaphore to signal when the image is available
    ///
    /// # Returns
    ///
    /// Returns a tuple of (image_index, suboptimal):
    /// - `image_index`: The index of the acquired image
    /// - `suboptimal`: True if the swapchain is suboptimal and should be recreated
    ///
    /// # Errors
    ///
    /// Returns an error if image acquisition fails. If the swapchain is out of date,
    /// `vk::Result::ERROR_OUT_OF_DATE_KHR` is returned and the caller should recreate
    /// the swapchain.
    pub fn acquire_next_image(&self, semaphore: vk::Semaphore) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                semaphore,
                vk::Fence::null(),
            )
        }
    }

    /// Acquires the next swapchain image with a fence.
    ///
    /// # Arguments
    ///
    /// * `semaphore` - Semaphore to signal when the image is available (can be null)
    /// * `fence` - Fence to signal when the image is available (can be null)
    /// * `timeout` - Timeout in nanoseconds (u64::MAX for infinite)
    ///
    /// # Returns
    ///
    /// Returns a tuple of (image_index, suboptimal).
    ///
    /// # Errors
    ///
    /// Returns an error if image acquisition fails.
    pub fn acquire_next_image_with_fence(
        &self,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
        timeout: u64,
    ) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.swapchain_loader
                .acquire_next_image(self.swapchain, timeout, semaphore, fence)
        }
    }

    /// Presents the rendered image to the screen.
    ///
    /// # Arguments
    ///
    /// * `queue` - The presentation queue
    /// * `image_index` - Index of the image to present (from `acquire_next_image`)
    /// * `wait_semaphore` - Semaphore to wait on before presenting
    ///
    /// # Returns
    ///
    /// Returns true if the swapchain is suboptimal and should be recreated.
    ///
    /// # Errors
    ///
    /// Returns an error if presentation fails. If the swapchain is out of date,
    /// `vk::Result::ERROR_OUT_OF_DATE_KHR` is returned.
    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
    ) -> Result<bool, vk::Result> {
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let wait_semaphores = [wait_semaphore];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { self.swapchain_loader.queue_present(queue, &present_info) }
    }

    /// Returns the swapchain handle.
    #[inline]
    pub fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    /// Returns the swapchain image format.
    #[inline]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Returns the swapchain color space.
    #[inline]
    pub fn color_space(&self) -> vk::ColorSpaceKHR {
        self.color_space
    }

    /// Returns the swapchain extent (resolution).
    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Returns the swapchain width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.extent.width
    }

    /// Returns the swapchain height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.extent.height
    }

    /// Returns the present mode.
    #[inline]
    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_mode
    }

    /// Returns the number of swapchain images.
    #[inline]
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Returns the swapchain image at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn image(&self, index: usize) -> vk::Image {
        self.images[index]
    }

    /// Returns the image view at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn image_view(&self, index: usize) -> vk::ImageView {
        self.image_views[index]
    }

    /// Returns all swapchain images.
    #[inline]
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// Returns all image views.
    #[inline]
    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    /// Destroys all image views.
    fn destroy_image_views(&mut self) {
        for &image_view in &self.image_views {
            unsafe {
                self.device.handle().destroy_image_view(image_view, None);
            }
        }
        self.image_views.clear();
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        // Destroy image views first
        self.destroy_image_views();

        // Destroy swapchain (images are destroyed automatically)
        // Skip if swapchain handle is null (e.g., after recreate moved resources)
        if self.swapchain != vk::SwapchainKHR::null() {
            unsafe {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }

            info!(
                "Swapchain destroyed (was {}x{}, {} images)",
                self.extent.width,
                self.extent.height,
                self.images.len()
            );
        }
    }
}

/// Chooses the best surface format from the available formats.
///
/// Prefers B8G8R8A8_SRGB with SRGB_NONLINEAR color space.
/// Falls back to the first available format if the preferred format is not available.
fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    // Prefer SRGB format for correct gamma handling
    let preferred = formats.iter().find(|f| {
        f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
    });

    if let Some(&format) = preferred {
        debug!("Selected preferred surface format: B8G8R8A8_SRGB with SRGB_NONLINEAR");
        return format;
    }

    // Second choice: B8G8R8A8_UNORM with SRGB color space
    let alternative = formats.iter().find(|f| {
        f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
    });

    if let Some(&format) = alternative {
        warn!("Using fallback surface format: B8G8R8A8_UNORM with SRGB_NONLINEAR");
        return format;
    }

    // Last resort: use the first available format
    warn!(
        "Using first available surface format: {:?}",
        formats[0].format
    );
    formats[0]
}

/// Chooses the best present mode from the available modes.
///
/// Prefers MAILBOX (triple buffering, no tearing, low latency).
/// Falls back to FIFO (vsync, guaranteed to be available).
fn choose_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    // MAILBOX: Triple buffering - no tearing, low latency
    if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        debug!("Selected MAILBOX present mode (triple buffering)");
        return vk::PresentModeKHR::MAILBOX;
    }

    // IMMEDIATE: No vsync, may tear, lowest latency
    // Could be preferred for some use cases but not default
    // if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
    //     return vk::PresentModeKHR::IMMEDIATE;
    // }

    // FIFO: VSync - no tearing, may have latency
    // Guaranteed to be available by the Vulkan spec
    debug!("Selected FIFO present mode (vsync)");
    vk::PresentModeKHR::FIFO
}

/// Chooses the swapchain extent (resolution).
///
/// If the current extent is not set (width/height are u32::MAX),
/// clamps the requested size to the surface's min/max extents.
fn choose_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    width: u32,
    height: u32,
) -> vk::Extent2D {
    // If current extent is defined, use it
    if capabilities.current_extent.width != u32::MAX {
        debug!(
            "Using current surface extent: {}x{}",
            capabilities.current_extent.width, capabilities.current_extent.height
        );
        return capabilities.current_extent;
    }

    // Otherwise, clamp the requested size to the surface's limits
    let extent = vk::Extent2D {
        width: width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    };

    debug!(
        "Calculated extent: {}x{} (requested: {}x{}, min: {}x{}, max: {}x{})",
        extent.width,
        extent.height,
        width,
        height,
        capabilities.min_image_extent.width,
        capabilities.min_image_extent.height,
        capabilities.max_image_extent.width,
        capabilities.max_image_extent.height
    );

    extent
}

/// Determines the optimal number of swapchain images.
///
/// Prefers one more than the minimum (for triple buffering),
/// but respects the maximum if set.
fn determine_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let preferred = capabilities.min_image_count + 1;

    // If max_image_count is 0, there's no maximum
    let image_count = if capabilities.max_image_count > 0 {
        preferred.min(capabilities.max_image_count)
    } else {
        preferred
    };

    debug!(
        "Image count: {} (min: {}, max: {})",
        image_count,
        capabilities.min_image_count,
        if capabilities.max_image_count == 0 {
            "unlimited".to_string()
        } else {
            capabilities.max_image_count.to_string()
        }
    );

    image_count
}

/// Creates image views for swapchain images.
fn create_image_views(
    device: &Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>, RhiError> {
    let mut image_views = Vec::with_capacity(images.len());

    for (i, &image) in images.iter().enumerate() {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = unsafe {
            device
                .handle()
                .create_image_view(&create_info, None)
                .map_err(|e| {
                    RhiError::SwapchainError(format!("Failed to create image view {}: {:?}", i, e))
                })?
        };

        image_views.push(image_view);
    }

    debug!("Created {} image views", image_views.len());
    Ok(image_views)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose_surface_format_prefers_srgb() {
        let formats = vec![
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];

        let selected = choose_surface_format(&formats);
        assert_eq!(selected.format, vk::Format::B8G8R8A8_SRGB);
        assert_eq!(selected.color_space, vk::ColorSpaceKHR::SRGB_NONLINEAR);
    }

    #[test]
    fn test_choose_surface_format_fallback() {
        let formats = vec![vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }];

        let selected = choose_surface_format(&formats);
        assert_eq!(selected.format, vk::Format::R8G8B8A8_UNORM);
    }

    #[test]
    fn test_choose_present_mode_prefers_mailbox() {
        let modes = vec![
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::IMMEDIATE,
        ];

        let selected = choose_present_mode(&modes);
        assert_eq!(selected, vk::PresentModeKHR::MAILBOX);
    }

    #[test]
    fn test_choose_present_mode_fallback_to_fifo() {
        let modes = vec![vk::PresentModeKHR::FIFO, vk::PresentModeKHR::IMMEDIATE];

        let selected = choose_present_mode(&modes);
        assert_eq!(selected, vk::PresentModeKHR::FIFO);
    }

    #[test]
    fn test_choose_extent_uses_current() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            ..Default::default()
        };

        let extent = choose_extent(&capabilities, 800, 600);
        assert_eq!(extent.width, 1920);
        assert_eq!(extent.height, 1080);
    }

    #[test]
    fn test_choose_extent_clamps_to_limits() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 100,
                height: 100,
            },
            max_image_extent: vk::Extent2D {
                width: 2000,
                height: 2000,
            },
            ..Default::default()
        };

        // Test clamping to max
        let extent = choose_extent(&capabilities, 3000, 3000);
        assert_eq!(extent.width, 2000);
        assert_eq!(extent.height, 2000);

        // Test clamping to min
        let extent = choose_extent(&capabilities, 50, 50);
        assert_eq!(extent.width, 100);
        assert_eq!(extent.height, 100);

        // Test within range
        let extent = choose_extent(&capabilities, 800, 600);
        assert_eq!(extent.width, 800);
        assert_eq!(extent.height, 600);
    }

    #[test]
    fn test_determine_image_count() {
        // Test with max limit
        let capabilities = vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: 3,
            ..Default::default()
        };
        assert_eq!(determine_image_count(&capabilities), 3);

        // Test with higher max limit
        let capabilities = vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: 8,
            ..Default::default()
        };
        assert_eq!(determine_image_count(&capabilities), 3);

        // Test with no max limit
        let capabilities = vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: 0, // 0 means no limit
            ..Default::default()
        };
        assert_eq!(determine_image_count(&capabilities), 3);
    }

    #[test]
    fn test_swapchain_support_details_is_adequate() {
        let adequate = SwapchainSupportDetails {
            capabilities: vk::SurfaceCapabilitiesKHR::default(),
            formats: vec![vk::SurfaceFormatKHR::default()],
            present_modes: vec![vk::PresentModeKHR::FIFO],
        };
        assert!(adequate.is_adequate());

        let no_formats = SwapchainSupportDetails {
            capabilities: vk::SurfaceCapabilitiesKHR::default(),
            formats: vec![],
            present_modes: vec![vk::PresentModeKHR::FIFO],
        };
        assert!(!no_formats.is_adequate());

        let no_modes = SwapchainSupportDetails {
            capabilities: vk::SurfaceCapabilitiesKHR::default(),
            formats: vec![vk::SurfaceFormatKHR::default()],
            present_modes: vec![],
        };
        assert!(!no_modes.is_adequate());
    }
}

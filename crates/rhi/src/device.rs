//! Vulkan logical device and queue management.
//!
//! This module handles VkDevice creation, queue retrieval, and gpu-allocator initialization.
//!
//! # Overview
//!
//! The [`Device`] struct provides a safe abstraction over the Vulkan logical device,
//! including:
//! - Logical device creation with required extensions and features
//! - Queue retrieval for graphics, presentation, and compute operations
//! - Memory allocation via gpu-allocator
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_rhi::instance::Instance;
//! use renderer_rhi::physical_device::select_physical_device;
//! use renderer_rhi::device::Device;
//! use ash::vk;
//!
//! let instance = Instance::new(false).expect("Failed to create instance");
//! let surface: vk::SurfaceKHR = vk::SurfaceKHR::null(); // placeholder
//! let surface_loader = ash::khr::surface::Instance::new(instance.entry(), instance.handle());
//!
//! let physical_device_info = select_physical_device(instance.handle(), surface, &surface_loader)
//!     .expect("No suitable GPU found");
//!
//! let device = Device::new(&instance, &physical_device_info)
//!     .expect("Failed to create logical device");
//!
//! // Access queues
//! let graphics_queue = device.graphics_queue();
//! let present_queue = device.present_queue();
//! ```

use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use tracing::{debug, info};

use crate::error::RhiError;
use crate::instance::Instance;
use crate::physical_device::{PhysicalDeviceInfo, QueueFamilyIndices};

/// Required device extensions.
const DEVICE_EXTENSIONS: &[&std::ffi::CStr] =
    &[ash::khr::swapchain::NAME, ash::khr::dynamic_rendering::NAME];

/// Vulkan logical device wrapper.
///
/// This struct manages the lifetime of the Vulkan logical device and its associated
/// resources including queues and the memory allocator.
///
/// # Thread Safety
///
/// The [`Device`] is designed to be shared across threads using `Arc`. The internal
/// allocator is protected by a `Mutex` for thread-safe memory allocation.
pub struct Device {
    /// Vulkan logical device handle.
    device: ash::Device,
    /// Physical device handle.
    physical_device: vk::PhysicalDevice,
    /// GPU memory allocator (thread-safe via Mutex).
    allocator: Mutex<Allocator>,
    /// Graphics queue handle.
    graphics_queue: vk::Queue,
    /// Presentation queue handle.
    present_queue: vk::Queue,
    /// Compute queue handle (may be the same as graphics queue).
    compute_queue: Option<vk::Queue>,
    /// Queue family indices.
    queue_families: QueueFamilyIndices,
}

impl Device {
    /// Creates a new logical device.
    ///
    /// This function creates a Vulkan logical device with:
    /// - Required extensions (swapchain, dynamic rendering)
    /// - Vulkan 1.2 features (descriptor indexing, buffer device address)
    /// - Vulkan 1.3 features (dynamic rendering, synchronization2)
    /// - Base features (sampler anisotropy, fill mode non-solid)
    ///
    /// It also initializes the gpu-allocator for memory management.
    ///
    /// # Arguments
    ///
    /// * `instance` - The Vulkan instance
    /// * `physical_device_info` - Information about the selected physical device
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Device creation fails
    /// - Allocator initialization fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::instance::Instance;
    /// use renderer_rhi::physical_device::select_physical_device;
    /// use renderer_rhi::device::Device;
    /// use ash::vk;
    ///
    /// let instance = Instance::new(false).expect("Failed to create instance");
    /// let surface: vk::SurfaceKHR = vk::SurfaceKHR::null();
    /// let surface_loader = ash::khr::surface::Instance::new(instance.entry(), instance.handle());
    ///
    /// let physical_device_info = select_physical_device(instance.handle(), surface, &surface_loader)
    ///     .expect("No suitable GPU found");
    ///
    /// let device = Device::new(&instance, &physical_device_info)
    ///     .expect("Failed to create logical device");
    /// ```
    pub fn new(
        instance: &Instance,
        physical_device_info: &PhysicalDeviceInfo,
    ) -> Result<Arc<Self>, RhiError> {
        let queue_families = &physical_device_info.queue_families;

        // Create queue create infos for unique queue families
        let unique_families = queue_families.unique_families();
        let queue_priorities = [1.0f32];

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priorities)
            })
            .collect();

        debug!(
            "Creating {} queue(s) for families: {:?}",
            queue_create_infos.len(),
            unique_families
        );

        // Enable Vulkan 1.2 features
        let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .buffer_device_address(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .shader_sampled_image_array_non_uniform_indexing(true);

        // Enable Vulkan 1.3 features
        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);

        // Enable base device features
        let features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(true)
            .fill_mode_non_solid(true)
            .wide_lines(true)
            .multi_draw_indirect(true);

        // Convert extension names to raw pointers
        let extension_names: Vec<*const i8> =
            DEVICE_EXTENSIONS.iter().map(|ext| ext.as_ptr()).collect();

        // Create device
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names)
            .enabled_features(&features)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_3);

        let device = unsafe {
            instance
                .handle()
                .create_device(physical_device_info.device, &create_info, None)?
        };

        info!(
            "Logical device created with {} extension(s)",
            DEVICE_EXTENSIONS.len()
        );

        // Retrieve queues
        let graphics_queue =
            unsafe { device.get_device_queue(queue_families.graphics_family.unwrap(), 0) };
        debug!(
            "Graphics queue retrieved from family {}",
            queue_families.graphics_family.unwrap()
        );

        let present_queue =
            unsafe { device.get_device_queue(queue_families.present_family.unwrap(), 0) };
        debug!(
            "Present queue retrieved from family {}",
            queue_families.present_family.unwrap()
        );

        // Compute queue may be the same as graphics queue
        let compute_queue = queue_families.compute_family.map(|family| {
            let queue = unsafe { device.get_device_queue(family, 0) };
            debug!("Compute queue retrieved from family {}", family);
            queue
        });

        // Initialize gpu-allocator
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.handle().clone(),
            device: device.clone(),
            physical_device: physical_device_info.device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        info!("GPU memory allocator initialized");

        Ok(Arc::new(Self {
            device,
            physical_device: physical_device_info.device,
            allocator: Mutex::new(allocator),
            graphics_queue,
            present_queue,
            compute_queue,
            queue_families: physical_device_info.queue_families,
        }))
    }

    /// Returns the Vulkan logical device handle.
    #[inline]
    pub fn handle(&self) -> &ash::Device {
        &self.device
    }

    /// Returns the physical device handle.
    #[inline]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Returns the graphics queue handle.
    #[inline]
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Returns the presentation queue handle.
    #[inline]
    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }

    /// Returns the compute queue handle.
    ///
    /// This may be the same queue as the graphics queue if no dedicated
    /// compute queue is available.
    #[inline]
    pub fn compute_queue(&self) -> Option<vk::Queue> {
        self.compute_queue
    }

    /// Returns the queue family indices.
    #[inline]
    pub fn queue_families(&self) -> &QueueFamilyIndices {
        &self.queue_families
    }

    /// Returns a reference to the GPU memory allocator.
    ///
    /// The allocator is protected by a Mutex for thread-safe access.
    #[inline]
    pub fn allocator(&self) -> &Mutex<Allocator> {
        &self.allocator
    }

    /// Waits for the device to become idle.
    ///
    /// This function blocks until all outstanding operations on all queues
    /// have completed. Useful before destroying resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait fails.
    pub fn wait_idle(&self) -> Result<(), RhiError> {
        unsafe { self.device.device_wait_idle()? };
        Ok(())
    }

    /// Submits command buffers to the graphics queue.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - All command buffers are valid and recorded
    /// - Synchronization is properly handled
    /// - The fence (if provided) is not in use
    ///
    /// # Arguments
    ///
    /// * `submit_infos` - Slice of submit info structures
    /// * `fence` - Optional fence to signal after completion
    ///
    /// # Errors
    ///
    /// Returns an error if the submission fails.
    pub unsafe fn submit_graphics(
        &self,
        submit_infos: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<(), RhiError> {
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, submit_infos, fence)?;
        }
        Ok(())
    }

    /// Submits command buffers to the compute queue.
    ///
    /// Falls back to the graphics queue if no dedicated compute queue is available.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - All command buffers are valid and recorded
    /// - Synchronization is properly handled
    /// - The fence (if provided) is not in use
    ///
    /// # Arguments
    ///
    /// * `submit_infos` - Slice of submit info structures
    /// * `fence` - Optional fence to signal after completion
    ///
    /// # Errors
    ///
    /// Returns an error if the submission fails.
    pub unsafe fn submit_compute(
        &self,
        submit_infos: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<(), RhiError> {
        let queue = self.compute_queue.unwrap_or(self.graphics_queue);
        unsafe {
            self.device.queue_submit(queue, submit_infos, fence)?;
        }
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // Wait for all operations to complete before cleanup
            if let Err(e) = self.device.device_wait_idle() {
                tracing::error!("Failed to wait for device idle during drop: {:?}", e);
            }

            // Allocator is dropped automatically when the Mutex is dropped
            // The allocator should be empty at this point (all allocations freed)

            self.device.destroy_device(None);
        }
        info!("Logical device destroyed");
    }
}

// Safety: Device is Send+Sync because:
// - ash::Device is Send+Sync
// - vk::PhysicalDevice and vk::Queue are Copy types (handles)
// - Allocator is protected by Mutex
// - QueueFamilyIndices is Copy
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_extensions_defined() {
        // Verify required extensions are defined
        assert!(!DEVICE_EXTENSIONS.is_empty());
        assert!(DEVICE_EXTENSIONS.contains(&ash::khr::swapchain::NAME));
        assert!(DEVICE_EXTENSIONS.contains(&ash::khr::dynamic_rendering::NAME));
    }

    #[test]
    fn test_device_is_send_sync() {
        // Compile-time check that Device is Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Device>();
    }
}

//! Physical device (GPU) selection.
//!
//! This module handles GPU enumeration and selection based on capabilities.
//!
//! # Overview
//!
//! The physical device selection process involves:
//! 1. Enumerating all available GPUs
//! 2. Checking each GPU for required queue families (Graphics, Present)
//! 3. Verifying required device features
//! 4. Selecting the most suitable GPU (preferring discrete GPUs)
//!
//! # Example
//!
//! ```no_run
//! use renderer_rhi::instance::Instance;
//! use renderer_rhi::physical_device::select_physical_device;
//! use ash::vk;
//!
//! let instance = Instance::new(false).expect("Failed to create instance");
//! // Assume surface is created from a window
//! let surface: vk::SurfaceKHR = vk::SurfaceKHR::null(); // placeholder
//! let surface_loader = ash::khr::surface::Instance::new(instance.entry(), instance.handle());
//!
//! let device_info = select_physical_device(instance.handle(), surface, &surface_loader)
//!     .expect("Failed to select physical device");
//!
//! println!("Selected GPU: {:?}", device_info.device_name());
//! ```

use std::ffi::CStr;

use ash::vk;
use tracing::{debug, info, warn};

use crate::error::RhiError;

/// Queue family indices for different queue types.
///
/// Vulkan devices can have multiple queue families, each supporting different
/// operations (graphics, compute, transfer, presentation).
#[derive(Clone, Copy, Debug, Default)]
pub struct QueueFamilyIndices {
    /// Index of the queue family that supports graphics operations.
    pub graphics_family: Option<u32>,
    /// Index of the queue family that supports presentation to a surface.
    pub present_family: Option<u32>,
    /// Index of the queue family that supports compute operations.
    pub compute_family: Option<u32>,
    /// Index of the queue family that supports transfer operations.
    pub transfer_family: Option<u32>,
}

impl QueueFamilyIndices {
    /// Checks if the minimum required queue families are available.
    ///
    /// For rendering, we need at least graphics and present queue families.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }

    /// Returns the unique queue family indices as a vector.
    ///
    /// This is useful when creating logical devices to avoid creating
    /// duplicate queues for the same family.
    pub fn unique_families(&self) -> Vec<u32> {
        let mut families = Vec::with_capacity(4);

        if let Some(graphics) = self.graphics_family {
            families.push(graphics);
        }
        if let Some(present) = self.present_family
            && !families.contains(&present)
        {
            families.push(present);
        }
        if let Some(compute) = self.compute_family
            && !families.contains(&compute)
        {
            families.push(compute);
        }
        if let Some(transfer) = self.transfer_family
            && !families.contains(&transfer)
        {
            families.push(transfer);
        }

        families
    }
}

/// Information about a physical device (GPU).
///
/// This struct contains all the information needed to create a logical device
/// and perform rendering operations.
#[derive(Clone)]
pub struct PhysicalDeviceInfo {
    /// Vulkan physical device handle.
    pub device: vk::PhysicalDevice,
    /// Device properties (name, limits, API version, etc.).
    pub properties: vk::PhysicalDeviceProperties,
    /// Supported device features.
    pub features: vk::PhysicalDeviceFeatures,
    /// Memory properties (heap sizes, memory types).
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Queue family indices for different operations.
    pub queue_families: QueueFamilyIndices,
}

impl PhysicalDeviceInfo {
    /// Returns the device name as a string.
    pub fn device_name(&self) -> &str {
        unsafe {
            CStr::from_ptr(self.properties.device_name.as_ptr())
                .to_str()
                .unwrap_or("Unknown Device")
        }
    }

    /// Returns the device type (Discrete, Integrated, etc.).
    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.properties.device_type
    }

    /// Returns a human-readable string for the device type.
    pub fn device_type_name(&self) -> &'static str {
        match self.properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
            vk::PhysicalDeviceType::CPU => "CPU",
            _ => "Other",
        }
    }

    /// Returns the Vulkan API version supported by the device.
    pub fn api_version(&self) -> (u32, u32, u32) {
        let version = self.properties.api_version;
        (
            vk::api_version_major(version),
            vk::api_version_minor(version),
            vk::api_version_patch(version),
        )
    }

    /// Returns the total device local memory in bytes.
    pub fn device_local_memory(&self) -> u64 {
        self.memory_properties
            .memory_heaps
            .iter()
            .take(self.memory_properties.memory_heap_count as usize)
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum()
    }
}

impl std::fmt::Debug for PhysicalDeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (major, minor, patch) = self.api_version();
        f.debug_struct("PhysicalDeviceInfo")
            .field("name", &self.device_name())
            .field("type", &self.device_type_name())
            .field("api_version", &format!("{}.{}.{}", major, minor, patch))
            .field("queue_families", &self.queue_families)
            .finish()
    }
}

/// Selects the most suitable physical device for rendering.
///
/// This function enumerates all available GPUs and selects one based on:
/// 1. Required queue family support (graphics and present)
/// 2. Required feature support (sampler anisotropy)
/// 3. Device type preference (discrete GPU preferred)
///
/// # Arguments
///
/// * `instance` - The Vulkan instance
/// * `surface` - The window surface for present support checking
/// * `surface_loader` - The surface extension loader
///
/// # Errors
///
/// Returns [`RhiError::NoSuitableGpu`] if no suitable GPU is found.
///
/// # Example
///
/// ```no_run
/// use renderer_rhi::instance::Instance;
/// use renderer_rhi::physical_device::select_physical_device;
/// use ash::vk;
///
/// let instance = Instance::new(false).expect("Failed to create instance");
/// let surface: vk::SurfaceKHR = vk::SurfaceKHR::null(); // placeholder
/// let surface_loader = ash::khr::surface::Instance::new(instance.entry(), instance.handle());
///
/// let device_info = select_physical_device(instance.handle(), surface, &surface_loader)
///     .expect("No suitable GPU found");
/// ```
pub fn select_physical_device(
    instance: &ash::Instance,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
) -> Result<PhysicalDeviceInfo, RhiError> {
    let devices = unsafe { instance.enumerate_physical_devices()? };

    if devices.is_empty() {
        warn!("No Vulkan-capable GPUs found");
        return Err(RhiError::NoSuitableGpu);
    }

    info!("Found {} GPU(s)", devices.len());

    // Collect all suitable devices with their scores
    let mut suitable_devices: Vec<(PhysicalDeviceInfo, u32)> = Vec::new();

    for device in devices {
        if let Some(info) = check_device_suitability(instance, device, surface, surface_loader) {
            let score = rate_device(&info);
            debug!(
                "GPU '{}' ({}) - Score: {}",
                info.device_name(),
                info.device_type_name(),
                score
            );
            suitable_devices.push((info, score));
        }
    }

    if suitable_devices.is_empty() {
        warn!("No suitable GPU found with required capabilities");
        return Err(RhiError::NoSuitableGpu);
    }

    // Sort by score (highest first) and pick the best one
    suitable_devices.sort_by(|a, b| b.1.cmp(&a.1));
    let (selected_device, score) = suitable_devices.remove(0);

    let (major, minor, patch) = selected_device.api_version();
    info!(
        "Selected GPU: '{}' ({}) - Vulkan {}.{}.{}, Score: {}",
        selected_device.device_name(),
        selected_device.device_type_name(),
        major,
        minor,
        patch,
        score
    );

    Ok(selected_device)
}

/// Checks if a physical device is suitable for rendering.
///
/// Returns `Some(PhysicalDeviceInfo)` if the device meets all requirements,
/// or `None` if it doesn't.
fn check_device_suitability(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
) -> Option<PhysicalDeviceInfo> {
    let properties = unsafe { instance.get_physical_device_properties(device) };
    let features = unsafe { instance.get_physical_device_features(device) };
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(device) };

    let device_name = unsafe {
        CStr::from_ptr(properties.device_name.as_ptr())
            .to_str()
            .unwrap_or("Unknown")
    };

    // Find queue families
    let queue_families = find_queue_families(instance, device, surface, surface_loader);

    // Check minimum requirements
    if !queue_families.is_complete() {
        debug!(
            "GPU '{}' skipped: missing required queue families (graphics={}, present={})",
            device_name,
            queue_families.graphics_family.is_some(),
            queue_families.present_family.is_some()
        );
        return None;
    }

    // Check required features
    if features.sampler_anisotropy == vk::FALSE {
        debug!(
            "GPU '{}' skipped: sampler anisotropy not supported",
            device_name
        );
        return None;
    }

    // Check Vulkan 1.3 support (required for dynamic rendering)
    if vk::api_version_major(properties.api_version) < 1
        || (vk::api_version_major(properties.api_version) == 1
            && vk::api_version_minor(properties.api_version) < 3)
    {
        debug!(
            "GPU '{}' skipped: Vulkan 1.3 not supported (version: {}.{})",
            device_name,
            vk::api_version_major(properties.api_version),
            vk::api_version_minor(properties.api_version)
        );
        return None;
    }

    Some(PhysicalDeviceInfo {
        device,
        properties,
        features,
        memory_properties,
        queue_families,
    })
}

/// Finds queue family indices for different operations.
fn find_queue_families(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
) -> QueueFamilyIndices {
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

    let mut indices = QueueFamilyIndices::default();

    // Track the best transfer-only queue for dedicated transfer operations
    let mut dedicated_transfer_family: Option<u32> = None;
    let mut dedicated_compute_family: Option<u32> = None;

    for (i, family) in queue_families.iter().enumerate() {
        let i = i as u32;

        // Skip queues with no queues available
        if family.queue_count == 0 {
            continue;
        }

        let has_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let has_compute = family.queue_flags.contains(vk::QueueFlags::COMPUTE);
        let has_transfer = family.queue_flags.contains(vk::QueueFlags::TRANSFER);

        // Graphics queue (also supports compute and transfer implicitly)
        if has_graphics && indices.graphics_family.is_none() {
            indices.graphics_family = Some(i);
        }

        // Compute queue - prefer dedicated compute queue
        if has_compute {
            if !has_graphics && dedicated_compute_family.is_none() {
                // Dedicated compute queue (no graphics)
                dedicated_compute_family = Some(i);
            } else if indices.compute_family.is_none() {
                indices.compute_family = Some(i);
            }
        }

        // Transfer queue - prefer dedicated transfer queue
        if has_transfer {
            if !has_graphics && !has_compute && dedicated_transfer_family.is_none() {
                // Dedicated transfer queue (no graphics or compute)
                dedicated_transfer_family = Some(i);
            } else if indices.transfer_family.is_none() {
                indices.transfer_family = Some(i);
            }
        }

        // Present queue - check surface support
        if indices.present_family.is_none() {
            let present_support = unsafe {
                surface_loader
                    .get_physical_device_surface_support(device, i, surface)
                    .unwrap_or(false)
            };

            if present_support {
                indices.present_family = Some(i);
            }
        }
    }

    // Use dedicated queues if available
    if let Some(dedicated) = dedicated_compute_family {
        indices.compute_family = Some(dedicated);
    }
    if let Some(dedicated) = dedicated_transfer_family {
        indices.transfer_family = Some(dedicated);
    }

    // If no dedicated transfer queue, use graphics queue (which supports transfer)
    if indices.transfer_family.is_none() {
        indices.transfer_family = indices.graphics_family;
    }

    // If no dedicated compute queue, use graphics queue (which supports compute)
    if indices.compute_family.is_none() {
        indices.compute_family = indices.graphics_family;
    }

    indices
}

/// Rates a physical device based on its capabilities.
///
/// Higher scores indicate more desirable devices.
fn rate_device(info: &PhysicalDeviceInfo) -> u32 {
    let mut score = 0u32;

    // Discrete GPUs are strongly preferred
    match info.properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => score += 10000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => score += 1000,
        vk::PhysicalDeviceType::VIRTUAL_GPU => score += 100,
        vk::PhysicalDeviceType::CPU => score += 10,
        _ => score += 1,
    }

    // Add score based on max image dimension (indicates GPU capability)
    score += info.properties.limits.max_image_dimension2_d;

    // Add score based on available VRAM (in MB, capped)
    let vram_mb = (info.device_local_memory() / (1024 * 1024)) as u32;
    score += vram_mb.min(16000); // Cap at 16GB contribution

    // Bonus for separate graphics and present queues (can help with performance)
    if info.queue_families.graphics_family != info.queue_families.present_family {
        score += 100;
    }

    // Bonus for dedicated compute queue
    if info.queue_families.compute_family != info.queue_families.graphics_family {
        score += 100;
    }

    // Bonus for dedicated transfer queue
    if info.queue_families.transfer_family != info.queue_families.graphics_family
        && info.queue_families.transfer_family != info.queue_families.compute_family
    {
        score += 100;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_family_indices_default() {
        let indices = QueueFamilyIndices::default();
        assert!(indices.graphics_family.is_none());
        assert!(indices.present_family.is_none());
        assert!(indices.compute_family.is_none());
        assert!(indices.transfer_family.is_none());
        assert!(!indices.is_complete());
    }

    #[test]
    fn test_queue_family_indices_complete() {
        let indices = QueueFamilyIndices {
            graphics_family: Some(0),
            present_family: Some(0),
            compute_family: None,
            transfer_family: None,
        };
        assert!(indices.is_complete());
    }

    #[test]
    fn test_queue_family_indices_incomplete() {
        let indices = QueueFamilyIndices {
            graphics_family: Some(0),
            present_family: None,
            compute_family: None,
            transfer_family: None,
        };
        assert!(!indices.is_complete());

        let indices2 = QueueFamilyIndices {
            graphics_family: None,
            present_family: Some(0),
            compute_family: None,
            transfer_family: None,
        };
        assert!(!indices2.is_complete());
    }

    #[test]
    fn test_unique_families_no_duplicates() {
        let indices = QueueFamilyIndices {
            graphics_family: Some(0),
            present_family: Some(1),
            compute_family: Some(2),
            transfer_family: Some(3),
        };
        let unique = indices.unique_families();
        assert_eq!(unique.len(), 4);
        assert!(unique.contains(&0));
        assert!(unique.contains(&1));
        assert!(unique.contains(&2));
        assert!(unique.contains(&3));
    }

    #[test]
    fn test_unique_families_with_duplicates() {
        let indices = QueueFamilyIndices {
            graphics_family: Some(0),
            present_family: Some(0),
            compute_family: Some(0),
            transfer_family: Some(1),
        };
        let unique = indices.unique_families();
        assert_eq!(unique.len(), 2);
        assert!(unique.contains(&0));
        assert!(unique.contains(&1));
    }

    #[test]
    fn test_unique_families_all_same() {
        let indices = QueueFamilyIndices {
            graphics_family: Some(0),
            present_family: Some(0),
            compute_family: Some(0),
            transfer_family: Some(0),
        };
        let unique = indices.unique_families();
        assert_eq!(unique.len(), 1);
        assert_eq!(unique[0], 0);
    }
}

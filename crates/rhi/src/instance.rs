//! Vulkan instance management.
//!
//! This module handles VkInstance creation, validation layers, and debug messengers.
//!
//! # Overview
//!
//! The [`Instance`] struct provides a safe abstraction over the Vulkan instance,
//! including optional validation layer support for debugging purposes.
//!
//! # Example
//!
//! ```no_run
//! use renderer_rhi::instance::Instance;
//!
//! // Create an instance with validation layers enabled (debug build)
//! let instance = Instance::new(cfg!(debug_assertions)).expect("Failed to create Vulkan instance");
//!
//! // Access the underlying Vulkan handles
//! let vk_instance = instance.handle();
//! let entry = instance.entry();
//! ```

use std::ffi::CStr;

use ash::{Entry, vk};
use tracing::{error, info, warn};

use crate::error::RhiError;

/// The Khronos validation layer name.
const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Vulkan instance wrapper with optional validation layer support.
///
/// This struct manages the lifetime of the Vulkan instance and its associated
/// debug utilities. When dropped, it properly cleans up all Vulkan resources.
pub struct Instance {
    /// Vulkan entry point loader
    entry: Entry,
    /// Vulkan instance handle
    instance: ash::Instance,
    /// Debug utils extension loader (only present when validation is enabled)
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    /// Debug messenger handle (only present when validation is enabled)
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Instance {
    /// Creates a new Vulkan instance.
    ///
    /// # Arguments
    ///
    /// * `enable_validation` - If true, enables validation layers and debug messenger
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vulkan library cannot be loaded
    /// - Required extensions are not available
    /// - Instance creation fails
    /// - Debug messenger setup fails (when validation is enabled)
    pub fn new(enable_validation: bool) -> Result<Self, RhiError> {
        // Load the Vulkan library
        let entry = unsafe { Entry::load()? };

        // Check if validation layer is available when requested
        if enable_validation && !Self::is_validation_layer_available(&entry)? {
            warn!("Validation layer requested but not available, proceeding without it");
        }

        let validation_available =
            enable_validation && Self::is_validation_layer_available(&entry)?;

        // Set up application info
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan Renderer")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"No Engine")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        // Collect required extensions
        let mut extensions = Self::get_required_extensions();
        if validation_available {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        // Set up layers
        let layers = if validation_available {
            vec![VALIDATION_LAYER_NAME.as_ptr()]
        } else {
            vec![]
        };

        // Create instance
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(RhiError::from)?
        };

        info!("Vulkan instance created successfully (API version 1.3)");

        // Set up debug messenger if validation is enabled
        let (debug_utils, debug_messenger) = if validation_available {
            let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger = Self::setup_debug_messenger(&debug_utils)?;
            info!("Debug messenger created successfully");
            (Some(debug_utils), Some(messenger))
        } else {
            (None, None)
        };

        if validation_available {
            info!("Validation layers enabled");
        } else if enable_validation {
            warn!("Validation layers were requested but are not available");
        }

        Ok(Self {
            entry,
            instance,
            debug_utils,
            debug_messenger,
        })
    }

    /// Returns the Vulkan instance handle.
    #[inline]
    pub fn handle(&self) -> &ash::Instance {
        &self.instance
    }

    /// Returns the Vulkan entry point loader.
    #[inline]
    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    /// Returns whether validation layers are enabled.
    #[inline]
    pub fn has_validation(&self) -> bool {
        self.debug_messenger.is_some()
    }

    /// Gets the list of required instance extensions.
    ///
    /// This includes the surface extension and platform-specific surface extensions.
    fn get_required_extensions() -> Vec<*const i8> {
        let mut extensions = vec![
            // Base surface extension
            ash::khr::surface::NAME.as_ptr(),
        ];

        // Platform-specific surface extension
        #[cfg(target_os = "windows")]
        extensions.push(ash::khr::win32_surface::NAME.as_ptr());

        #[cfg(target_os = "linux")]
        {
            // Support both X11 and Wayland on Linux
            extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
            extensions.push(ash::khr::wayland_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "macos")]
        extensions.push(ash::ext::metal_surface::NAME.as_ptr());

        extensions
    }

    /// Checks if the Khronos validation layer is available.
    fn is_validation_layer_available(entry: &Entry) -> Result<bool, RhiError> {
        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };

        let validation_layer_name = VALIDATION_LAYER_NAME.to_bytes_with_nul();

        let found = available_layers.iter().any(|layer| {
            let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            layer_name.to_bytes_with_nul() == validation_layer_name
        });

        Ok(found)
    }

    /// Sets up the debug messenger for validation layer callbacks.
    fn setup_debug_messenger(
        debug_utils: &ash::ext::debug_utils::Instance,
    ) -> Result<vk::DebugUtilsMessengerEXT, RhiError> {
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        let messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&create_info, None)
                .map_err(RhiError::from)?
        };

        Ok(messenger)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            // Destroy debug messenger before instance
            if let (Some(debug_utils), Some(messenger)) = (&self.debug_utils, self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
        info!("Vulkan instance destroyed");
    }
}

/// Debug callback function for validation layer messages.
///
/// This function is called by the Vulkan validation layer when it detects
/// issues with API usage. Messages are logged using the tracing crate.
///
/// # Safety
///
/// This function is called from the Vulkan driver and must follow the
/// Vulkan specification for debug callbacks.
unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if p_callback_data.is_null() {
        return vk::FALSE;
    }

    let callback_data = unsafe { &*p_callback_data };
    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::Borrowed("(no message)")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    let type_str = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "General",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "Validation",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "Performance",
        _ => "Unknown",
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!("[Vulkan {}] {}", type_str, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!("[Vulkan {}] {}", type_str, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!("[Vulkan {}] {}", type_str, message);
        }
        _ => {
            // VERBOSE level - use info for now
            info!("[Vulkan {} Verbose] {}", type_str, message);
        }
    }

    // Returning VK_FALSE indicates the call should not be aborted
    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_creation_without_validation() {
        // This test requires Vulkan to be installed
        let result = Instance::new(false);
        match result {
            Ok(instance) => {
                assert!(!instance.has_validation());
            }
            Err(RhiError::LoadingError(_)) => {
                // Vulkan not available - skip test
                eprintln!("Skipping test: Vulkan not available");
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }

    #[test]
    fn test_instance_creation_with_validation() {
        // This test requires Vulkan SDK to be installed with validation layers
        let result = Instance::new(true);
        match result {
            Ok(instance) => {
                // Validation might or might not be available depending on the system
                if instance.has_validation() {
                    assert!(instance.debug_utils.is_some());
                    assert!(instance.debug_messenger.is_some());
                }
            }
            Err(RhiError::LoadingError(_)) => {
                // Vulkan not available - skip test
                eprintln!("Skipping test: Vulkan not available");
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }

    #[test]
    fn test_required_extensions() {
        let extensions = Instance::get_required_extensions();

        // Should always include the surface extension
        assert!(!extensions.is_empty());

        // At least surface extension should be present
        #[cfg(target_os = "windows")]
        assert!(extensions.len() >= 2); // surface + win32_surface

        #[cfg(target_os = "linux")]
        assert!(extensions.len() >= 3); // surface + xlib_surface + wayland_surface

        #[cfg(target_os = "macos")]
        assert!(extensions.len() >= 2); // surface + metal_surface
    }
}

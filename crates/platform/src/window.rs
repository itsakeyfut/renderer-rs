//! Window management using winit.
//!
//! This module provides window creation and Vulkan surface creation functionality.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window as WinitWindow, WindowAttributes};

use renderer_core::{Error, Result};

/// RAII wrapper for a Vulkan surface.
///
/// This struct owns a `vk::SurfaceKHR` handle and ensures it is properly destroyed
/// when dropped. The surface loader is stored internally to perform cleanup.
///
/// # Ownership
/// The surface is destroyed automatically when this struct is dropped.
/// The caller must ensure that the Vulkan instance outlives this surface.
pub struct Surface {
    handle: vk::SurfaceKHR,
    surface_loader: ash::khr::surface::Instance,
}

impl Surface {
    /// Get the raw Vulkan surface handle.
    ///
    /// # Note
    /// The returned handle is valid only as long as this `Surface` instance exists.
    /// Do not store this handle beyond the lifetime of the `Surface`.
    #[inline]
    pub fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    /// Get a reference to the surface loader.
    ///
    /// This is useful for querying surface capabilities, formats, and present modes.
    #[inline]
    pub fn loader(&self) -> &ash::khr::surface::Instance {
        &self.surface_loader
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: The surface handle is valid and was created by ash_window::create_surface.
        // The surface loader is valid and was created from the same instance.
        // This is the only place where the surface is destroyed.
        unsafe {
            self.surface_loader.destroy_surface(self.handle, None);
        }
        tracing::debug!("Vulkan surface destroyed");
    }
}

/// A window wrapper that provides access to the underlying winit window
/// and raw handles for Vulkan surface creation.
pub struct Window {
    window: Arc<WinitWindow>,
    width: u32,
    height: u32,
}

impl Window {
    /// Create a new window with the given dimensions and title.
    pub fn new(event_loop: &ActiveEventLoop, width: u32, height: u32, title: &str) -> Result<Self> {
        let attrs = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(width, height))
            .with_resizable(true);

        let window = event_loop
            .create_window(attrs)
            .map_err(|e| Error::Window(e.to_string()))?;

        tracing::info!("Window created: {}x{}", width, height);

        Ok(Self {
            window: Arc::new(window),
            width,
            height,
        })
    }

    /// Get a reference to the underlying winit window.
    pub fn inner(&self) -> &WinitWindow {
        &self.window
    }

    /// Get an Arc reference to the underlying winit window.
    pub fn inner_arc(&self) -> Arc<WinitWindow> {
        self.window.clone()
    }

    /// Get the current width of the window.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the current height of the window.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Update the stored dimensions (call this when handling resize events).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        tracing::debug!("Window resized: {}x{}", width, height);
    }

    /// Get the aspect ratio of the window.
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    /// Get the display handle for Vulkan surface creation.
    pub fn display_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError>
    {
        self.window.display_handle()
    }

    /// Get the window handle for Vulkan surface creation.
    pub fn window_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError>
    {
        self.window.window_handle()
    }

    /// Request a redraw of the window.
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Create a Vulkan surface for this window.
    ///
    /// Returns a RAII [`Surface`] wrapper that automatically destroys the surface when dropped.
    ///
    /// # Arguments
    /// * `entry` - The Vulkan entry point
    /// * `instance` - The Vulkan instance (must outlive the returned `Surface`)
    ///
    /// # Errors
    /// Returns an error if surface creation fails due to:
    /// - Invalid window or display handles
    /// - Vulkan surface creation failure
    pub fn create_surface(&self, entry: &ash::Entry, instance: &ash::Instance) -> Result<Surface> {
        let display_handle = self
            .window
            .display_handle()
            .map_err(|e| Error::Window(format!("Failed to get display handle: {}", e)))?;

        let window_handle = self
            .window
            .window_handle()
            .map_err(|e| Error::Window(format!("Failed to get window handle: {}", e)))?;

        // SAFETY: The entry and instance are valid references provided by the caller.
        // The display and window handles are valid as they come from the winit window.
        // The surface will be destroyed in the Surface::drop implementation.
        let handle = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
            .map_err(|e| Error::Vulkan(format!("Failed to create Vulkan surface: {}", e)))?
        };

        let surface_loader = ash::khr::surface::Instance::new(entry, instance);

        tracing::info!("Vulkan surface created successfully");

        Ok(Surface {
            handle,
            surface_loader,
        })
    }
}

/// Get the required Vulkan extensions for surface creation on the current platform.
///
/// This function returns the extension names needed to create a Vulkan surface
/// for the given display handle.
///
/// # Return Value
/// Returns a vector of pointers to null-terminated C strings (extension names).
/// These pointers are valid for the lifetime of the program as they point to
/// static strings provided by the Vulkan loader. The caller must not free these
/// pointers or use them after the Vulkan loader is unloaded.
///
/// # Errors
/// Returns an error if the required extensions cannot be enumerated.
pub fn get_required_extensions(
    display_handle: raw_window_handle::RawDisplayHandle,
) -> Result<Vec<*const i8>> {
    let extensions = ash_window::enumerate_required_extensions(display_handle)
        .map_err(|e| Error::Vulkan(format!("Failed to enumerate required extensions: {}", e)))?;

    tracing::debug!(
        "Required Vulkan extensions for surface: {:?}",
        extensions
            .iter()
            // SAFETY: ash_window guarantees these pointers are valid, null-terminated
            // C strings that point to static data provided by the Vulkan loader.
            .map(|&ext| unsafe { std::ffi::CStr::from_ptr(ext) })
            .collect::<Vec<_>>()
    );

    Ok(extensions.to_vec())
}

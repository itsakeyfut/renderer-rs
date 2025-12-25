//! Window management using winit.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window as WinitWindow, WindowAttributes};

use renderer_core::{Error, Result};

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
}

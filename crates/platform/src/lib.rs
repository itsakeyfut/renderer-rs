//! Platform abstraction layer for the Vulkan renderer.
//!
//! This crate provides platform-specific functionality:
//! - Window management via winit
//! - Input handling (keyboard, mouse)
//! - Raw window handles for Vulkan surface creation
//! - Vulkan surface creation via ash-window

mod input;
mod window;

pub use input::{InputState, KeyCode, MouseButton};
pub use window::{Window, get_required_extensions};

// Re-export winit types that users might need
pub use winit::event::{Event, WindowEvent};
pub use winit::event_loop::EventLoop;

// Re-export raw_window_handle types for surface creation
pub use raw_window_handle::RawDisplayHandle;

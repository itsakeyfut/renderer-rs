//! Main rendering pipeline.
//!
//! This crate orchestrates the rendering process:
//! - Frame management
//! - Render pass execution
//! - Resource binding

pub mod frame;
pub mod renderer;

pub use frame::FrameManager;
pub use renderer::Renderer;

/// Maximum number of frames that can be in flight simultaneously.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

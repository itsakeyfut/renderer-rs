//! Main rendering pipeline.
//!
//! This crate orchestrates the rendering process:
//! - Frame management
//! - Render pass execution
//! - Resource binding

pub mod frame;

pub use frame::FrameManager;

/// Maximum number of frames that can be in flight simultaneously.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

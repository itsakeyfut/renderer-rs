//! Main rendering pipeline.
//!
//! This crate orchestrates the rendering process:
//! - Frame management
//! - Render pass execution
//! - Resource binding
//!
//! # Frame Management
//!
//! The [`FrameManager`] provides a high-level interface for managing
//! per-frame resources and coordinating the rendering loop. It handles:
//!
//! - Per-frame command buffers
//! - Synchronization primitives (semaphores and fences)
//! - Swapchain image acquisition and presentation
//! - Frame-in-flight management
//!
//! # Example
//!
//! ```no_run
//! use renderer_renderer::MAX_FRAMES_IN_FLIGHT;
//!
//! // The maximum number of frames that can be processed concurrently
//! assert_eq!(MAX_FRAMES_IN_FLIGHT, 2);
//! ```

pub mod frame_manager;
pub mod renderer;

pub use frame_manager::{FrameData, FrameManager};
pub use renderer::Renderer;

/// Maximum number of frames that can be in flight simultaneously.
///
/// This value determines how many frames can be processed concurrently:
/// - Value of 2: Double buffering (most common)
/// - Value of 3: Triple buffering (lower latency, more memory)
///
/// Each frame in flight has its own set of resources (command buffer,
/// semaphores, fence) to avoid synchronization issues.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

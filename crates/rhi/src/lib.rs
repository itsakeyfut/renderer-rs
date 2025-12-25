//! Vulkan abstraction layer (Render Hardware Interface).
//!
//! This crate provides a safe abstraction over Vulkan using the `ash` crate.
//! It handles:
//! - Instance and device creation
//! - Swapchain management
//! - Command buffer recording
//! - Buffer and image management
//! - Pipeline creation
//! - Synchronization primitives

mod error;

pub mod buffer;
pub mod command;
pub mod device;
pub mod image;
pub mod instance;
pub mod physical_device;
pub mod pipeline;
pub mod rendering;
pub mod sampler;
pub mod shader;
pub mod swapchain;
pub mod sync;
pub mod texture;
pub mod vertex;

pub use error::{RhiError, RhiResult};

// Re-export ash types that users might need
pub use ash::vk;

//! Core utilities for the Vulkan renderer.
//!
//! This crate provides foundational types and utilities used across the renderer:
//! - Error types and result aliases
//! - Logging initialization
//! - Timer utilities
//! - Configuration management

mod error;
mod logging;
mod timer;

pub use error::{Error, Result};
pub use logging::init_logging;
pub use timer::Timer;

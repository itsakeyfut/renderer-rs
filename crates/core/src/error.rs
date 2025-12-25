//! Error types for the renderer.

use thiserror::Error;

/// Main error type for the renderer.
#[derive(Error, Debug)]
pub enum Error {
    /// Vulkan-related errors
    #[error("Vulkan error: {0}")]
    Vulkan(String),

    /// Window creation or management errors
    #[error("Window error: {0}")]
    Window(String),

    /// Resource loading errors
    #[error("Resource error: {0}")]
    Resource(String),

    /// Shader compilation errors
    #[error("Shader error: {0}")]
    Shader(String),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration errors
    #[error("Config error: {0}")]
    Config(String),

    /// Generic internal errors
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias using the renderer's Error type.
pub type Result<T> = std::result::Result<T, Error>;

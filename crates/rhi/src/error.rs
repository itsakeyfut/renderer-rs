//! RHI-specific error types.

use thiserror::Error;

/// RHI-specific error type.
#[derive(Error, Debug)]
pub enum RhiError {
    /// Vulkan API error
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] ash::vk::Result),

    /// Failed to load Vulkan library
    #[error("Failed to load Vulkan: {0}")]
    LoadingError(#[from] ash::LoadingError),

    /// GPU allocator error
    #[error("Allocator error: {0}")]
    AllocatorError(#[from] gpu_allocator::AllocationError),

    /// No suitable GPU found
    #[error("No suitable GPU found")]
    NoSuitableGpu,

    /// Shader compilation error
    #[error("Shader error: {0}")]
    ShaderError(String),

    /// Surface creation error
    #[error("Surface error: {0}")]
    SurfaceError(String),

    /// Swapchain error
    #[error("Swapchain error: {0}")]
    SwapchainError(String),

    /// Invalid handle error
    #[error("Invalid handle: {0}")]
    InvalidHandle(String),

    /// Pipeline creation error
    #[error("Pipeline error: {0}")]
    PipelineError(String),
}

/// Result type alias for RHI operations.
pub type RhiResult<T> = std::result::Result<T, RhiError>;

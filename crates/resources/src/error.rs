//! Error types for resource loading.

use std::path::PathBuf;
use thiserror::Error;

/// Error type for resource loading operations.
#[derive(Error, Debug)]
pub enum ResourceError {
    /// Failed to load a glTF file.
    #[error("Failed to load glTF file '{path}': {message}")]
    GltfLoad {
        /// Path to the file that failed to load.
        path: PathBuf,
        /// Error message.
        message: String,
    },

    /// glTF file contains no meshes.
    #[error("glTF file '{0}' contains no meshes")]
    NoMeshes(PathBuf),

    /// A mesh primitive has no position data.
    #[error("Mesh primitive has no position data")]
    NoPositionData,

    /// IO error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Image loading error.
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),
}

/// Result type alias for resource operations.
pub type ResourceResult<T> = Result<T, ResourceError>;

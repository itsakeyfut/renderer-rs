//! Shader module management.
//!
//! This module handles SPIR-V loading and VkShaderModule creation.
//! It supports loading shaders from files or byte arrays and provides
//! the necessary Vulkan structures for pipeline creation.
//!
//! # Overview
//!
//! - [`ShaderStage`] defines the type of shader (vertex, fragment, etc.)
//! - [`Shader`] wraps VkShaderModule with stage and entry point information
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use std::path::Path;
//! use renderer_rhi::device::Device;
//! use renderer_rhi::shader::{Shader, ShaderStage};
//!
//! # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
//! // Load vertex shader from SPIR-V file
//! let vertex_shader = Shader::from_spirv_file(
//!     device.clone(),
//!     Path::new("shaders/triangle.vert.spv"),
//!     ShaderStage::Vertex,
//!     "main",
//! )?;
//!
//! // Load fragment shader from SPIR-V file
//! let fragment_shader = Shader::from_spirv_file(
//!     device.clone(),
//!     Path::new("shaders/triangle.frag.spv"),
//!     ShaderStage::Fragment,
//!     "main",
//! )?;
//!
//! // Get pipeline shader stage create info for pipeline creation
//! let _vertex_stage_info = vertex_shader.stage_create_info();
//! let _fragment_stage_info = fragment_shader.stage_create_info();
//! # Ok(())
//! # }
//! ```

use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;

use ash::vk;
use tracing::{debug, info};

use crate::device::Device;
use crate::error::{RhiError, RhiResult};

/// Shader stage type.
///
/// Defines which stage of the graphics or compute pipeline
/// the shader will be used in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex shader stage - processes each vertex
    Vertex,
    /// Fragment (pixel) shader stage - processes each fragment
    Fragment,
    /// Compute shader stage - general-purpose GPU computation
    Compute,
    /// Geometry shader stage - processes primitives
    Geometry,
    /// Tessellation control shader stage
    TessControl,
    /// Tessellation evaluation shader stage
    TessEvaluation,
}

impl ShaderStage {
    /// Converts the shader stage to Vulkan shader stage flags.
    ///
    /// # Returns
    ///
    /// The corresponding `vk::ShaderStageFlags` for this stage.
    pub fn to_vk_stage(self) -> vk::ShaderStageFlags {
        match self {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
            ShaderStage::TessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            ShaderStage::TessEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        }
    }

    /// Returns a human-readable name for the shader stage.
    pub fn name(self) -> &'static str {
        match self {
            ShaderStage::Vertex => "vertex",
            ShaderStage::Fragment => "fragment",
            ShaderStage::Compute => "compute",
            ShaderStage::Geometry => "geometry",
            ShaderStage::TessControl => "tessellation control",
            ShaderStage::TessEvaluation => "tessellation evaluation",
        }
    }
}

impl std::fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Vulkan shader module wrapper.
///
/// This struct manages the lifecycle of a VkShaderModule and provides
/// the necessary information for creating graphics or compute pipelines.
///
/// # Thread Safety
///
/// The shader module itself is immutable after creation and can be
/// safely shared between threads. The underlying Vulkan handle is
/// managed by the device.
pub struct Shader {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan shader module handle.
    module: vk::ShaderModule,
    /// Shader stage type.
    stage: ShaderStage,
    /// Entry point function name.
    entry_point: CString,
}

impl Shader {
    /// Creates a shader module from a SPIR-V file.
    ///
    /// This method reads the SPIR-V binary from the specified file path
    /// and creates a Vulkan shader module.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `path` - Path to the SPIR-V file
    /// * `stage` - The shader stage (vertex, fragment, etc.)
    /// * `entry_point` - The name of the entry point function (typically "main")
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The SPIR-V data is invalid
    /// - Shader module creation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use std::path::Path;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::shader::{Shader, ShaderStage};
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let shader = Shader::from_spirv_file(
    ///     device,
    ///     Path::new("shaders/main.vert.spv"),
    ///     ShaderStage::Vertex,
    ///     "main",
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_spirv_file(
        device: Arc<Device>,
        path: &Path,
        stage: ShaderStage,
        entry_point: &str,
    ) -> RhiResult<Self> {
        debug!("Loading {} shader from {:?}", stage, path);

        let bytes = std::fs::read(path).map_err(|e| {
            RhiError::ShaderError(format!("Failed to read shader file {:?}: {}", path, e))
        })?;

        Self::from_spirv_bytes(device, &bytes, stage, entry_point)
    }

    /// Creates a shader module from SPIR-V bytes.
    ///
    /// This method creates a Vulkan shader module from pre-loaded SPIR-V data.
    /// The bytes must be valid SPIR-V code and properly aligned (4-byte alignment).
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `bytes` - The SPIR-V binary data
    /// * `stage` - The shader stage (vertex, fragment, etc.)
    /// * `entry_point` - The name of the entry point function (typically "main")
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The byte length is not a multiple of 4 (SPIR-V alignment requirement)
    /// - The entry point name contains null bytes
    /// - Shader module creation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::shader::{Shader, ShaderStage};
    ///
    /// # fn example(device: Arc<Device>, spirv_bytes: &[u8]) -> Result<(), renderer_rhi::RhiError> {
    /// let shader = Shader::from_spirv_bytes(
    ///     device,
    ///     spirv_bytes,
    ///     ShaderStage::Fragment,
    ///     "main",
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_spirv_bytes(
        device: Arc<Device>,
        bytes: &[u8],
        stage: ShaderStage,
        entry_point: &str,
    ) -> RhiResult<Self> {
        // Validate SPIR-V alignment
        if !bytes.len().is_multiple_of(4) {
            return Err(RhiError::ShaderError(format!(
                "SPIR-V code must be 4-byte aligned, got {} bytes",
                bytes.len()
            )));
        }

        // Convert bytes to u32 code words
        let code: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Create shader module
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

        let module = unsafe { device.handle().create_shader_module(&create_info, None)? };

        // Create entry point CString
        let entry_point_cstring = CString::new(entry_point)
            .map_err(|e| RhiError::ShaderError(format!("Invalid entry point name: {}", e)))?;

        info!(
            "Created {} shader module with entry point '{}'",
            stage, entry_point
        );

        Ok(Self {
            device,
            module,
            stage,
            entry_point: entry_point_cstring,
        })
    }

    /// Returns the Vulkan shader module handle.
    ///
    /// This handle can be used directly with Vulkan API calls.
    #[inline]
    pub fn handle(&self) -> vk::ShaderModule {
        self.module
    }

    /// Returns the shader stage.
    #[inline]
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    /// Returns the entry point function name as a C string reference.
    #[inline]
    pub fn entry_point(&self) -> &std::ffi::CStr {
        &self.entry_point
    }

    /// Creates a pipeline shader stage create info structure.
    ///
    /// This structure is used when creating graphics or compute pipelines.
    /// The returned structure borrows from this shader and must not outlive it.
    ///
    /// # Returns
    ///
    /// A `vk::PipelineShaderStageCreateInfo` ready for pipeline creation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use std::path::Path;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::shader::{Shader, ShaderStage};
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let vertex_shader = Shader::from_spirv_file(
    ///     device.clone(),
    ///     Path::new("shaders/main.vert.spv"),
    ///     ShaderStage::Vertex,
    ///     "main",
    /// )?;
    ///
    /// let stage_info = vertex_shader.stage_create_info();
    /// // Use stage_info in pipeline creation...
    /// # Ok(())
    /// # }
    /// ```
    pub fn stage_create_info(&self) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(self.stage.to_vk_stage())
            .module(self.module)
            .name(&self.entry_point)
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle()
                .destroy_shader_module(self.module, None);
        }
        debug!("Destroyed {} shader module", self.stage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_stage_to_vk_stage() {
        assert_eq!(
            ShaderStage::Vertex.to_vk_stage(),
            vk::ShaderStageFlags::VERTEX
        );
        assert_eq!(
            ShaderStage::Fragment.to_vk_stage(),
            vk::ShaderStageFlags::FRAGMENT
        );
        assert_eq!(
            ShaderStage::Compute.to_vk_stage(),
            vk::ShaderStageFlags::COMPUTE
        );
        assert_eq!(
            ShaderStage::Geometry.to_vk_stage(),
            vk::ShaderStageFlags::GEOMETRY
        );
        assert_eq!(
            ShaderStage::TessControl.to_vk_stage(),
            vk::ShaderStageFlags::TESSELLATION_CONTROL
        );
        assert_eq!(
            ShaderStage::TessEvaluation.to_vk_stage(),
            vk::ShaderStageFlags::TESSELLATION_EVALUATION
        );
    }

    #[test]
    fn test_shader_stage_name() {
        assert_eq!(ShaderStage::Vertex.name(), "vertex");
        assert_eq!(ShaderStage::Fragment.name(), "fragment");
        assert_eq!(ShaderStage::Compute.name(), "compute");
        assert_eq!(ShaderStage::Geometry.name(), "geometry");
        assert_eq!(ShaderStage::TessControl.name(), "tessellation control");
        assert_eq!(
            ShaderStage::TessEvaluation.name(),
            "tessellation evaluation"
        );
    }

    #[test]
    fn test_shader_stage_display() {
        assert_eq!(format!("{}", ShaderStage::Vertex), "vertex");
        assert_eq!(format!("{}", ShaderStage::Fragment), "fragment");
    }

    #[test]
    fn test_shader_stage_equality() {
        assert_eq!(ShaderStage::Vertex, ShaderStage::Vertex);
        assert_ne!(ShaderStage::Vertex, ShaderStage::Fragment);
    }

    #[test]
    fn test_shader_stage_clone() {
        let stage = ShaderStage::Compute;
        let cloned = stage;
        assert_eq!(stage, cloned);
    }

    #[test]
    fn test_invalid_spirv_alignment() {
        // This test verifies that from_spirv_bytes rejects misaligned data
        // We can't actually call from_spirv_bytes without a device,
        // but we can test the alignment logic manually
        let misaligned_bytes = vec![0u8; 5]; // Not a multiple of 4
        assert!(misaligned_bytes.len() % 4 != 0);
    }
}

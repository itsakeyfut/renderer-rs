//! Uniform buffer object definitions for shaders.
//!
//! These structures must match the HLSL shader uniform buffer layouts exactly.
//! All structures use `#[repr(C)]` for predictable memory layout and implement
//! `Pod` and `Zeroable` for safe byte casting.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Camera uniform buffer data.
///
/// This structure matches the HLSL `CameraData` cbuffer (register b0).
///
/// # Memory Layout
///
/// - Offset 0: view matrix (64 bytes)
/// - Offset 64: projection matrix (64 bytes)
/// - Offset 128: viewProjection matrix (64 bytes)
/// - Offset 192: camera position (12 bytes)
/// - Offset 204: padding (4 bytes)
/// - Total size: 208 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CameraUBO {
    /// View matrix (world to view space).
    pub view: Mat4,
    /// Projection matrix (view to clip space).
    pub projection: Mat4,
    /// Combined view-projection matrix.
    pub view_projection: Mat4,
    /// Camera world position.
    pub camera_position: Vec3,
    /// Padding for 16-byte alignment.
    pub _padding: f32,
}

impl CameraUBO {
    /// Size of the struct in bytes.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Creates a new camera UBO from matrices and position.
    pub fn new(view: Mat4, projection: Mat4, camera_position: Vec3) -> Self {
        Self {
            view,
            projection,
            view_projection: projection * view,
            camera_position,
            _padding: 0.0,
        }
    }
}

/// Object uniform buffer data.
///
/// This structure matches the HLSL `ObjectData` cbuffer (register b1).
///
/// # Memory Layout
///
/// - Offset 0: model matrix (64 bytes)
/// - Offset 64: normal matrix (64 bytes)
/// - Total size: 128 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ObjectUBO {
    /// Model matrix (object to world space).
    pub model: Mat4,
    /// Normal matrix for transforming normals.
    /// This is typically the transpose of the inverse of the model matrix.
    pub normal_matrix: Mat4,
}

impl ObjectUBO {
    /// Size of the struct in bytes.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Creates a new object UBO from a model matrix.
    ///
    /// The normal matrix is computed as the transpose of the inverse of the
    /// upper-left 3x3 of the model matrix, extended to a 4x4 matrix.
    pub fn new(model: Mat4) -> Self {
        // Calculate the normal matrix (transpose of inverse of upper-left 3x3)
        let normal_matrix = model.inverse().transpose();
        Self {
            model,
            normal_matrix,
        }
    }

    /// Creates an identity object UBO.
    pub fn identity() -> Self {
        Self {
            model: Mat4::IDENTITY,
            normal_matrix: Mat4::IDENTITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_ubo_size() {
        // 3 Mat4 (3 * 64) + Vec3 (12) + padding (4) = 208 bytes
        assert_eq!(CameraUBO::SIZE, 208);
    }

    #[test]
    fn test_camera_ubo_alignment() {
        // Verify proper alignment for GPU (Mat4 requires 16-byte alignment)
        assert_eq!(std::mem::align_of::<CameraUBO>(), 16);
    }

    #[test]
    fn test_object_ubo_size() {
        // 2 Mat4 (2 * 64) = 128 bytes
        assert_eq!(ObjectUBO::SIZE, 128);
    }

    #[test]
    fn test_object_ubo_alignment() {
        // Verify proper alignment for GPU (Mat4 requires 16-byte alignment)
        assert_eq!(std::mem::align_of::<ObjectUBO>(), 16);
    }

    #[test]
    fn test_camera_ubo_new() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        let camera_pos = Vec3::new(0.0, 0.0, 5.0);

        let ubo = CameraUBO::new(view, projection, camera_pos);

        assert_eq!(ubo.view, view);
        assert_eq!(ubo.projection, projection);
        assert_eq!(ubo.view_projection, projection * view);
        assert_eq!(ubo.camera_position, camera_pos);
    }

    #[test]
    fn test_object_ubo_identity() {
        let ubo = ObjectUBO::identity();
        assert_eq!(ubo.model, Mat4::IDENTITY);
        assert_eq!(ubo.normal_matrix, Mat4::IDENTITY);
    }

    #[test]
    fn test_object_ubo_new() {
        let model = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
        let ubo = ObjectUBO::new(model);

        assert_eq!(ubo.model, model);
        // Normal matrix should be transpose(inverse(model))
        let expected_normal = model.inverse().transpose();
        assert_eq!(ubo.normal_matrix, expected_normal);
    }

    #[test]
    fn test_ubo_pod_zeroable() {
        // Verify Pod and Zeroable traits work
        let camera = CameraUBO::default();
        let bytes: &[u8] = bytemuck::bytes_of(&camera);
        assert_eq!(bytes.len(), CameraUBO::SIZE);

        let object = ObjectUBO::identity();
        let bytes: &[u8] = bytemuck::bytes_of(&object);
        assert_eq!(bytes.len(), ObjectUBO::SIZE);
    }
}

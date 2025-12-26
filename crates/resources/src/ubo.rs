//! Uniform Buffer Object (UBO) structures for shader data.
//!
//! This module defines the data structures used to pass uniform data to shaders.
//! All structures use `#[repr(C)]` for correct memory layout and implement
//! `bytemuck::Pod` and `bytemuck::Zeroable` for safe byte-level operations.
//!
//! # Overview
//!
//! - [`CameraUbo`] contains camera-related matrices and position
//! - [`ObjectUbo`] contains per-object transformation matrices
//!
//! # GPU Memory Layout
//!
//! All structures follow std140 layout rules for uniform buffers:
//! - `Mat4` is 64 bytes (16 floats)
//! - `Vec3` is 12 bytes but must be aligned to 16 bytes
//! - Padding is added where necessary
//!
//! # Example
//!
//! ```
//! use renderer_resources::ubo::{CameraUbo, ObjectUbo};
//! use glam::{Mat4, Vec3};
//!
//! // Create camera UBO data
//! let camera = CameraUbo::new(
//!     Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y),
//!     Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0),
//!     Vec3::new(0.0, 0.0, 5.0),
//! );
//!
//! // Create object UBO data
//! let object = ObjectUbo::new(Mat4::IDENTITY);
//!
//! // Convert to bytes for GPU upload
//! let camera_bytes: &[u8] = bytemuck::bytes_of(&camera);
//! let object_bytes: &[u8] = bytemuck::bytes_of(&object);
//! ```

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Camera uniform buffer object.
///
/// Contains all camera-related data needed by shaders:
/// - View matrix (world to camera space)
/// - Projection matrix (camera to clip space)
/// - Combined view-projection matrix (optimization to avoid per-vertex multiply)
/// - Camera world position (for lighting calculations)
///
/// # Memory Layout (std140)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 64   | view |
/// | 64     | 64   | projection |
/// | 128    | 64   | view_projection |
/// | 192    | 12   | camera_position |
/// | 204    | 4    | _padding |
///
/// Total size: 208 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CameraUbo {
    /// View matrix (world space to camera space transformation).
    pub view: Mat4,
    /// Projection matrix (camera space to clip space transformation).
    pub projection: Mat4,
    /// Combined view-projection matrix.
    ///
    /// Pre-computed as `projection * view` to reduce per-vertex calculations.
    pub view_projection: Mat4,
    /// Camera position in world space.
    ///
    /// Used for lighting calculations (e.g., specular reflection).
    pub camera_position: Vec3,
    /// Padding to align structure to 16 bytes.
    pub _padding: f32,
}

impl CameraUbo {
    /// Creates a new camera UBO with the given matrices and position.
    ///
    /// The view-projection matrix is automatically computed from the
    /// view and projection matrices.
    ///
    /// # Arguments
    ///
    /// * `view` - The view matrix (world to camera space)
    /// * `projection` - The projection matrix (camera to clip space)
    /// * `camera_position` - The camera position in world space
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_resources::ubo::CameraUbo;
    /// use glam::{Mat4, Vec3};
    ///
    /// let eye = Vec3::new(0.0, 2.0, 5.0);
    /// let target = Vec3::ZERO;
    /// let up = Vec3::Y;
    ///
    /// let view = Mat4::look_at_rh(eye, target, up);
    /// let projection = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
    ///
    /// let camera = CameraUbo::new(view, projection, eye);
    /// ```
    #[inline]
    pub fn new(view: Mat4, projection: Mat4, camera_position: Vec3) -> Self {
        Self {
            view,
            projection,
            view_projection: projection * view,
            camera_position,
            _padding: 0.0,
        }
    }

    /// Updates the view matrix and recomputes the view-projection matrix.
    ///
    /// # Arguments
    ///
    /// * `view` - The new view matrix
    /// * `camera_position` - The new camera position
    #[inline]
    pub fn update_view(&mut self, view: Mat4, camera_position: Vec3) {
        self.view = view;
        self.camera_position = camera_position;
        self.view_projection = self.projection * self.view;
    }

    /// Updates the projection matrix and recomputes the view-projection matrix.
    ///
    /// # Arguments
    ///
    /// * `projection` - The new projection matrix
    #[inline]
    pub fn update_projection(&mut self, projection: Mat4) {
        self.projection = projection;
        self.view_projection = self.projection * self.view;
    }

    /// Returns the size of this structure in bytes.
    ///
    /// This is useful when creating uniform buffers.
    #[inline]
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Object uniform buffer object.
///
/// Contains per-object transformation data:
/// - Model matrix (object space to world space)
/// - Normal matrix (for transforming normals correctly)
///
/// # Normal Matrix
///
/// The normal matrix is the transpose of the inverse of the upper-left 3x3
/// portion of the model matrix. This is necessary to correctly transform
/// normal vectors when the model matrix contains non-uniform scaling.
///
/// # Memory Layout (std140)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 64   | model |
/// | 64     | 64   | normal_matrix |
///
/// Total size: 128 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ObjectUbo {
    /// Model matrix (object space to world space transformation).
    pub model: Mat4,
    /// Normal matrix for correct normal vector transformation.
    ///
    /// Computed as the transpose of the inverse of the model matrix.
    /// Stored as Mat4 for alignment, but only the upper-left 3x3 is used.
    pub normal_matrix: Mat4,
}

impl ObjectUbo {
    /// Creates a new object UBO with the given model matrix.
    ///
    /// The normal matrix is automatically computed from the model matrix.
    ///
    /// # Arguments
    ///
    /// * `model` - The model matrix (object to world space)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_resources::ubo::ObjectUbo;
    /// use glam::{Mat4, Vec3};
    ///
    /// // Create object at position (1, 2, 3) with 2x scale
    /// let model = Mat4::from_scale_rotation_translation(
    ///     Vec3::splat(2.0),
    ///     glam::Quat::IDENTITY,
    ///     Vec3::new(1.0, 2.0, 3.0),
    /// );
    ///
    /// let object = ObjectUbo::new(model);
    /// ```
    #[inline]
    pub fn new(model: Mat4) -> Self {
        Self {
            model,
            normal_matrix: Self::compute_normal_matrix(model),
        }
    }

    /// Updates the model matrix and recomputes the normal matrix.
    ///
    /// # Arguments
    ///
    /// * `model` - The new model matrix
    #[inline]
    pub fn update_model(&mut self, model: Mat4) {
        self.model = model;
        self.normal_matrix = Self::compute_normal_matrix(model);
    }

    /// Computes the normal matrix from a model matrix.
    ///
    /// The normal matrix is the transpose of the inverse of the model matrix.
    /// This ensures normals are transformed correctly even with non-uniform scaling.
    ///
    /// # Non-invertible matrices
    ///
    /// If the model matrix is not invertible (e.g., contains zero scale),
    /// the identity matrix is returned as a fallback to avoid NaN/Inf values
    /// propagating to shaders.
    ///
    /// # Arguments
    ///
    /// * `model` - The model matrix
    #[inline]
    pub fn compute_normal_matrix(model: Mat4) -> Mat4 {
        // For correct normal transformation, we need transpose(inverse(model))
        // If the model matrix is orthogonal (no scaling or uniform scaling),
        // the normal matrix equals the model matrix.

        // Check if matrix is invertible by checking determinant
        // Use a small epsilon to handle floating-point precision issues
        const EPSILON: f32 = 1e-6;
        let det = model.determinant();

        if det.abs() < EPSILON {
            // Matrix is not invertible (e.g., zero scale)
            // Return identity as a safe fallback
            Mat4::IDENTITY
        } else {
            model.inverse().transpose()
        }
    }

    /// Returns the size of this structure in bytes.
    ///
    /// This is useful when creating uniform buffers.
    #[inline]
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Light uniform buffer object.
///
/// Contains data for a single directional light source.
/// For multiple lights, use an array of this structure or a storage buffer.
///
/// # Memory Layout (std140)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 12   | direction |
/// | 12     | 4    | _padding1 |
/// | 16     | 12   | color |
/// | 28     | 4    | intensity |
///
/// Total size: 32 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DirectionalLightUbo {
    /// Light direction (normalized, pointing toward the light source).
    pub direction: Vec3,
    /// Padding for 16-byte alignment.
    pub _padding1: f32,
    /// Light color (RGB, typically in range [0, 1]).
    pub color: Vec3,
    /// Light intensity multiplier.
    pub intensity: f32,
}

impl DirectionalLightUbo {
    /// Creates a new directional light UBO.
    ///
    /// # Arguments
    ///
    /// * `direction` - The light direction (will be normalized)
    /// * `color` - The light color (RGB)
    /// * `intensity` - The light intensity multiplier
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_resources::ubo::DirectionalLightUbo;
    /// use glam::Vec3;
    ///
    /// // Create a white sun light coming from above
    /// let sun = DirectionalLightUbo::new(
    ///     Vec3::new(0.5, -1.0, 0.3),
    ///     Vec3::ONE,
    ///     1.0,
    /// );
    /// ```
    #[inline]
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            // Use normalize_or_zero to handle zero-length vectors safely
            // (avoids NaN propagation to shaders)
            direction: direction.normalize_or_zero(),
            _padding1: 0.0,
            color,
            intensity,
        }
    }

    /// Returns the size of this structure in bytes.
    #[inline]
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Scene-level uniform buffer object.
///
/// Contains global scene data such as ambient lighting and time.
///
/// # Memory Layout (std140)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 12   | ambient_color |
/// | 12     | 4    | time |
/// | 16     | 4    | delta_time |
/// | 20     | 12   | _padding |
///
/// Total size: 32 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SceneUbo {
    /// Ambient light color (RGB).
    pub ambient_color: Vec3,
    /// Total elapsed time in seconds.
    pub time: f32,
    /// Time since last frame in seconds.
    pub delta_time: f32,
    /// Padding for 16-byte alignment.
    pub _padding: [f32; 3],
}

impl SceneUbo {
    /// Creates a new scene UBO.
    ///
    /// # Arguments
    ///
    /// * `ambient_color` - The ambient light color (RGB)
    /// * `time` - The current time in seconds
    /// * `delta_time` - The time since last frame in seconds
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_resources::ubo::SceneUbo;
    /// use glam::Vec3;
    ///
    /// let scene = SceneUbo::new(
    ///     Vec3::splat(0.1), // Dim ambient light
    ///     0.0,              // Start time
    ///     0.016,            // ~60 FPS
    /// );
    /// ```
    #[inline]
    pub fn new(ambient_color: Vec3, time: f32, delta_time: f32) -> Self {
        Self {
            ambient_color,
            time,
            delta_time,
            _padding: [0.0; 3],
        }
    }

    /// Updates the time values.
    ///
    /// # Arguments
    ///
    /// * `time` - The new total time
    /// * `delta_time` - The new delta time
    #[inline]
    pub fn update_time(&mut self, time: f32, delta_time: f32) {
        self.time = time;
        self.delta_time = delta_time;
    }

    /// Returns the size of this structure in bytes.
    #[inline]
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_camera_ubo_size() {
        // Mat4 = 64 bytes, Vec3 = 12 bytes, f32 = 4 bytes
        // 64 + 64 + 64 + 12 + 4 = 208 bytes
        assert_eq!(size_of::<CameraUbo>(), 208);
        assert_eq!(CameraUbo::size(), 208);
    }

    #[test]
    fn test_camera_ubo_alignment() {
        // Must be 4-byte aligned at minimum (float alignment)
        assert!(align_of::<CameraUbo>() >= 4);
    }

    #[test]
    fn test_camera_ubo_new() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let position = Vec3::new(0.0, 0.0, 5.0);

        let ubo = CameraUbo::new(view, projection, position);

        assert_eq!(ubo.view, view);
        assert_eq!(ubo.projection, projection);
        assert_eq!(ubo.view_projection, projection * view);
        assert_eq!(ubo.camera_position, position);
        assert_eq!(ubo._padding, 0.0);
    }

    #[test]
    fn test_camera_ubo_update_view() {
        let view1 = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let view2 = Mat4::look_at_rh(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let position1 = Vec3::new(0.0, 0.0, 5.0);
        let position2 = Vec3::new(5.0, 0.0, 0.0);

        let mut ubo = CameraUbo::new(view1, projection, position1);
        ubo.update_view(view2, position2);

        assert_eq!(ubo.view, view2);
        assert_eq!(ubo.camera_position, position2);
        assert_eq!(ubo.view_projection, projection * view2);
    }

    #[test]
    fn test_object_ubo_size() {
        // Mat4 = 64 bytes each, 2 Mat4s = 128 bytes
        assert_eq!(size_of::<ObjectUbo>(), 128);
        assert_eq!(ObjectUbo::size(), 128);
    }

    #[test]
    fn test_object_ubo_alignment() {
        assert!(align_of::<ObjectUbo>() >= 4);
    }

    #[test]
    fn test_object_ubo_new() {
        let model = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let ubo = ObjectUbo::new(model);

        assert_eq!(ubo.model, model);
        // For pure translation, normal matrix equals model matrix
        let expected_normal = model.inverse().transpose();
        assert_eq!(ubo.normal_matrix, expected_normal);
    }

    #[test]
    fn test_object_ubo_normal_matrix_with_scale() {
        // Non-uniform scale should produce different normal matrix
        let scale = Vec3::new(1.0, 2.0, 1.0);
        let model = Mat4::from_scale(scale);
        let ubo = ObjectUbo::new(model);

        // Normal matrix should be inverse transpose
        let expected_normal = model.inverse().transpose();
        assert_eq!(ubo.normal_matrix, expected_normal);
    }

    #[test]
    fn test_object_ubo_non_invertible_matrix() {
        // Zero scale makes the matrix non-invertible
        let model = Mat4::from_scale(Vec3::ZERO);
        let ubo = ObjectUbo::new(model);

        // Should return identity matrix as fallback, not NaN
        assert_eq!(ubo.normal_matrix, Mat4::IDENTITY);

        // Ensure no NaN values in the normal matrix
        let cols = [
            ubo.normal_matrix.x_axis,
            ubo.normal_matrix.y_axis,
            ubo.normal_matrix.z_axis,
            ubo.normal_matrix.w_axis,
        ];
        for col in cols {
            assert!(!col.x.is_nan());
            assert!(!col.y.is_nan());
            assert!(!col.z.is_nan());
            assert!(!col.w.is_nan());
        }
    }

    #[test]
    fn test_directional_light_ubo_size() {
        // Vec3 + padding + Vec3 + f32 = 12 + 4 + 12 + 4 = 32 bytes
        assert_eq!(size_of::<DirectionalLightUbo>(), 32);
        assert_eq!(DirectionalLightUbo::size(), 32);
    }

    #[test]
    fn test_directional_light_ubo_new() {
        let direction = Vec3::new(0.0, -1.0, 0.0);
        let color = Vec3::ONE;
        let intensity = 1.5;

        let light = DirectionalLightUbo::new(direction, color, intensity);

        assert_eq!(light.direction, direction.normalize());
        assert_eq!(light.color, color);
        assert_eq!(light.intensity, intensity);
    }

    #[test]
    fn test_directional_light_ubo_zero_direction() {
        // Zero-length direction should not produce NaN
        let light = DirectionalLightUbo::new(Vec3::ZERO, Vec3::ONE, 1.0);

        // normalize_or_zero returns zero vector for zero-length input
        assert_eq!(light.direction, Vec3::ZERO);
        // Ensure no NaN values
        assert!(!light.direction.x.is_nan());
        assert!(!light.direction.y.is_nan());
        assert!(!light.direction.z.is_nan());
    }

    #[test]
    fn test_scene_ubo_size() {
        // Vec3 + f32 + f32 + [f32; 3] = 12 + 4 + 4 + 12 = 32 bytes
        assert_eq!(size_of::<SceneUbo>(), 32);
        assert_eq!(SceneUbo::size(), 32);
    }

    #[test]
    fn test_scene_ubo_new() {
        let ambient = Vec3::splat(0.1);
        let time = 1.5;
        let delta = 0.016;

        let scene = SceneUbo::new(ambient, time, delta);

        assert_eq!(scene.ambient_color, ambient);
        assert_eq!(scene.time, time);
        assert_eq!(scene.delta_time, delta);
    }

    #[test]
    fn test_bytemuck_cast() {
        // Verify that bytemuck can safely cast these types
        let camera = CameraUbo::default();
        let bytes: &[u8] = bytemuck::bytes_of(&camera);
        assert_eq!(bytes.len(), CameraUbo::size());

        let object = ObjectUbo::default();
        let bytes: &[u8] = bytemuck::bytes_of(&object);
        assert_eq!(bytes.len(), ObjectUbo::size());

        let light = DirectionalLightUbo::default();
        let bytes: &[u8] = bytemuck::bytes_of(&light);
        assert_eq!(bytes.len(), DirectionalLightUbo::size());

        let scene = SceneUbo::default();
        let bytes: &[u8] = bytemuck::bytes_of(&scene);
        assert_eq!(bytes.len(), SceneUbo::size());
    }
}

//! Transform component for scene objects.

use glam::{Mat4, Quat, Vec3};

/// A transform representing position, rotation, and scale.
#[derive(Clone, Debug)]
pub struct Transform {
    /// Position in world space
    pub position: Vec3,
    /// Rotation as a quaternion
    pub rotation: Quat,
    /// Scale factor
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// Create a new transform at the origin.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a transform with the given position.
    pub fn with_position(mut self, position: Vec3) -> Self {
        self.position = position;
        self
    }

    /// Create a transform with the given rotation.
    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Create a transform with the given scale.
    pub fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    /// Get the local transformation matrix.
    pub fn local_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Get the normal matrix (inverse transpose of model matrix).
    ///
    /// # Non-invertible transforms
    ///
    /// If the transform is not invertible (e.g., contains zero scale),
    /// the identity matrix is returned as a fallback to avoid NaN/Inf values.
    pub fn normal_matrix(&self) -> Mat4 {
        let model = self.local_matrix();

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

    /// Get the forward direction vector.
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Get the right direction vector.
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Get the up direction vector.
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_default() {
        let t = Transform::default();
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn test_transform_builder() {
        let t = Transform::new()
            .with_position(Vec3::new(1.0, 2.0, 3.0))
            .with_scale(Vec3::splat(2.0));

        assert_eq!(t.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.scale, Vec3::splat(2.0));
    }

    #[test]
    fn test_normal_matrix_identity() {
        let t = Transform::default();
        let normal = t.normal_matrix();

        // For identity transform, normal matrix should be identity
        assert_eq!(normal, Mat4::IDENTITY);
    }

    #[test]
    fn test_normal_matrix_with_scale() {
        let t = Transform::new().with_scale(Vec3::new(1.0, 2.0, 1.0));
        let normal = t.normal_matrix();
        let model = t.local_matrix();

        // Normal matrix should be inverse transpose
        let expected = model.inverse().transpose();
        assert_eq!(normal, expected);
    }

    #[test]
    fn test_normal_matrix_non_invertible() {
        // Zero scale makes the transform non-invertible
        let t = Transform::new().with_scale(Vec3::ZERO);
        let normal = t.normal_matrix();

        // Should return identity matrix as fallback, not NaN
        assert_eq!(normal, Mat4::IDENTITY);

        // Ensure no NaN values
        let cols = [normal.x_axis, normal.y_axis, normal.z_axis, normal.w_axis];
        for col in cols {
            assert!(!col.x.is_nan());
            assert!(!col.y.is_nan());
            assert!(!col.z.is_nan());
            assert!(!col.w.is_nan());
        }
    }

    #[test]
    fn test_direction_vectors() {
        let t = Transform::default();

        // Default orientation: -Z forward, +X right, +Y up
        assert_eq!(t.forward(), Vec3::NEG_Z);
        assert_eq!(t.right(), Vec3::X);
        assert_eq!(t.up(), Vec3::Y);
    }
}

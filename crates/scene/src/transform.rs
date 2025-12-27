//! Transform component for scene objects.
//!
//! This module provides the [`Transform`] struct for representing position,
//! rotation, and scale of scene objects. It supports hierarchical transforms
//! through parent-child relationships.
//!
//! # Example
//!
//! ```
//! use renderer_scene::Transform;
//! use glam::{Vec3, Quat};
//!
//! // Create a parent transform
//! let parent = Transform::new()
//!     .with_position(Vec3::new(1.0, 0.0, 0.0));
//!
//! // Create a child transform with parent
//! let child = Transform::new()
//!     .with_position(Vec3::new(0.0, 1.0, 0.0))
//!     .with_parent(parent);
//!
//! // The world position of the child is (1.0, 1.0, 0.0)
//! let world_matrix = child.world_matrix();
//! ```

use glam::{Mat4, Quat, Vec3};

/// A transform representing position, rotation, and scale.
///
/// Transforms can optionally have a parent, enabling hierarchical
/// transformations where child transforms are relative to their parent.
#[derive(Clone, Debug)]
pub struct Transform {
    /// Position in local space (relative to parent if any)
    pub position: Vec3,
    /// Rotation as a quaternion
    pub rotation: Quat,
    /// Scale factor
    pub scale: Vec3,
    /// Optional parent transform for hierarchical transformations
    parent: Option<Box<Transform>>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            parent: None,
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

    /// Create a transform with the given parent.
    ///
    /// The parent transform is cloned and stored, enabling hierarchical
    /// transformations where this transform's position, rotation, and scale
    /// are relative to the parent.
    pub fn with_parent(mut self, parent: Transform) -> Self {
        self.parent = Some(Box::new(parent));
        self
    }

    /// Set the parent transform.
    ///
    /// # Arguments
    /// * `parent` - The parent transform to set
    pub fn set_parent(&mut self, parent: Transform) {
        self.parent = Some(Box::new(parent));
    }

    /// Clear the parent transform.
    ///
    /// After calling this, the transform will no longer have a parent
    /// and its local and world matrices will be the same.
    pub fn clear_parent(&mut self) {
        self.parent = None;
    }

    /// Check if this transform has a parent.
    pub fn has_parent(&self) -> bool {
        self.parent.is_some()
    }

    /// Get a reference to the parent transform, if any.
    pub fn parent(&self) -> Option<&Transform> {
        self.parent.as_deref()
    }

    /// Get the local transformation matrix.
    ///
    /// This returns the transformation matrix in local space,
    /// not accounting for any parent transforms.
    pub fn local_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Get the world transformation matrix.
    ///
    /// This returns the transformation matrix in world space,
    /// accounting for the entire parent hierarchy. If there is no parent,
    /// this is equivalent to `local_matrix()`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_scene::Transform;
    /// use glam::Vec3;
    ///
    /// let parent = Transform::new()
    ///     .with_position(Vec3::new(10.0, 0.0, 0.0));
    ///
    /// let child = Transform::new()
    ///     .with_position(Vec3::new(0.0, 5.0, 0.0))
    ///     .with_parent(parent);
    ///
    /// // World matrix includes parent translation
    /// let world = child.world_matrix();
    /// let world_pos = world.transform_point3(Vec3::ZERO);
    /// assert!((world_pos - Vec3::new(10.0, 5.0, 0.0)).length() < 0.001);
    /// ```
    pub fn world_matrix(&self) -> Mat4 {
        let local = self.local_matrix();
        match &self.parent {
            Some(parent) => parent.world_matrix() * local,
            None => local,
        }
    }

    /// Get the normal matrix (inverse transpose of world matrix).
    ///
    /// The normal matrix is used for transforming normal vectors correctly
    /// when the model matrix contains non-uniform scaling.
    ///
    /// # Non-invertible transforms
    ///
    /// If the transform is not invertible (e.g., contains zero scale),
    /// the identity matrix is returned as a fallback to avoid NaN/Inf values.
    pub fn normal_matrix(&self) -> Mat4 {
        let model = self.world_matrix();

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

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_vec3(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    #[test]
    fn test_transform_default() {
        let t = Transform::default();
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
        assert!(!t.has_parent());
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
        let model = t.world_matrix();

        // Normal matrix should be inverse transpose of world matrix
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

    // Hierarchical transform tests

    #[test]
    fn test_with_parent() {
        let parent = Transform::new().with_position(Vec3::new(1.0, 0.0, 0.0));
        let child = Transform::new()
            .with_position(Vec3::new(0.0, 2.0, 0.0))
            .with_parent(parent);

        assert!(child.has_parent());
        assert!(child.parent().is_some());
    }

    #[test]
    fn test_set_parent() {
        let parent = Transform::new().with_position(Vec3::new(1.0, 0.0, 0.0));
        let mut child = Transform::new().with_position(Vec3::new(0.0, 2.0, 0.0));

        assert!(!child.has_parent());
        child.set_parent(parent);
        assert!(child.has_parent());
    }

    #[test]
    fn test_clear_parent() {
        let parent = Transform::new().with_position(Vec3::new(1.0, 0.0, 0.0));
        let mut child = Transform::new()
            .with_position(Vec3::new(0.0, 2.0, 0.0))
            .with_parent(parent);

        assert!(child.has_parent());
        child.clear_parent();
        assert!(!child.has_parent());
    }

    #[test]
    fn test_world_matrix_without_parent() {
        let t = Transform::new()
            .with_position(Vec3::new(1.0, 2.0, 3.0))
            .with_scale(Vec3::splat(2.0));

        // Without parent, world matrix equals local matrix
        assert_eq!(t.world_matrix(), t.local_matrix());
    }

    #[test]
    fn test_world_matrix_with_parent_translation() {
        let parent = Transform::new().with_position(Vec3::new(10.0, 0.0, 0.0));
        let child = Transform::new()
            .with_position(Vec3::new(0.0, 5.0, 0.0))
            .with_parent(parent);

        let world = child.world_matrix();
        let world_pos = world.transform_point3(Vec3::ZERO);

        // Child at (0, 5, 0) relative to parent at (10, 0, 0)
        // World position should be (10, 5, 0)
        assert!(
            approx_eq_vec3(world_pos, Vec3::new(10.0, 5.0, 0.0)),
            "Expected (10, 5, 0), got {:?}",
            world_pos
        );
    }

    #[test]
    fn test_world_matrix_with_parent_scale() {
        let parent = Transform::new().with_scale(Vec3::splat(2.0));
        let child = Transform::new()
            .with_position(Vec3::new(1.0, 0.0, 0.0))
            .with_parent(parent);

        let world = child.world_matrix();
        let world_pos = world.transform_point3(Vec3::ZERO);

        // Child at (1, 0, 0) with parent scale of 2
        // World position should be (2, 0, 0)
        assert!(
            approx_eq_vec3(world_pos, Vec3::new(2.0, 0.0, 0.0)),
            "Expected (2, 0, 0), got {:?}",
            world_pos
        );
    }

    #[test]
    fn test_world_matrix_with_parent_rotation() {
        // Parent rotated 90 degrees around Y axis
        let parent =
            Transform::new().with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
        let child = Transform::new()
            .with_position(Vec3::new(1.0, 0.0, 0.0))
            .with_parent(parent);

        let world = child.world_matrix();
        let world_pos = world.transform_point3(Vec3::ZERO);

        // Child at (1, 0, 0) with parent rotated 90 degrees around Y
        // World position should be approximately (0, 0, -1)
        assert!(
            approx_eq_vec3(world_pos, Vec3::new(0.0, 0.0, -1.0)),
            "Expected (0, 0, -1), got {:?}",
            world_pos
        );
    }

    #[test]
    fn test_world_matrix_nested_hierarchy() {
        // Create a 3-level hierarchy: grandparent -> parent -> child
        let grandparent = Transform::new().with_position(Vec3::new(100.0, 0.0, 0.0));
        let parent = Transform::new()
            .with_position(Vec3::new(10.0, 0.0, 0.0))
            .with_parent(grandparent);
        let child = Transform::new()
            .with_position(Vec3::new(1.0, 0.0, 0.0))
            .with_parent(parent);

        let world = child.world_matrix();
        let world_pos = world.transform_point3(Vec3::ZERO);

        // Child at (1, 0, 0) relative to parent at (10, 0, 0) relative to grandparent at (100, 0, 0)
        // World position should be (111, 0, 0)
        assert!(
            approx_eq_vec3(world_pos, Vec3::new(111.0, 0.0, 0.0)),
            "Expected (111, 0, 0), got {:?}",
            world_pos
        );
    }

    #[test]
    fn test_world_matrix_combined_transforms() {
        // Parent with position, rotation, and scale
        let parent = Transform::new()
            .with_position(Vec3::new(10.0, 0.0, 0.0))
            .with_scale(Vec3::splat(2.0));

        let child = Transform::new()
            .with_position(Vec3::new(5.0, 0.0, 0.0))
            .with_parent(parent);

        let world = child.world_matrix();
        let world_pos = world.transform_point3(Vec3::ZERO);

        // Child at (5, 0, 0) with parent scale 2 at (10, 0, 0)
        // World position: 10 + (5 * 2) = 20
        assert!(
            approx_eq_vec3(world_pos, Vec3::new(20.0, 0.0, 0.0)),
            "Expected (20, 0, 0), got {:?}",
            world_pos
        );
    }

    #[test]
    fn test_normal_matrix_with_parent() {
        let parent = Transform::new().with_scale(Vec3::new(2.0, 1.0, 1.0));
        let child = Transform::new()
            .with_scale(Vec3::new(1.0, 3.0, 1.0))
            .with_parent(parent);

        let normal = child.normal_matrix();
        let world = child.world_matrix();

        // Normal matrix should be inverse transpose of world matrix
        let expected = world.inverse().transpose();
        assert_eq!(normal, expected);
    }

    #[test]
    fn test_parent_accessor() {
        let parent = Transform::new().with_position(Vec3::new(1.0, 2.0, 3.0));
        let child = Transform::new().with_parent(parent.clone());

        let retrieved_parent = child.parent().expect("Should have parent");
        assert_eq!(retrieved_parent.position, parent.position);
        assert_eq!(retrieved_parent.rotation, parent.rotation);
        assert_eq!(retrieved_parent.scale, parent.scale);
    }
}

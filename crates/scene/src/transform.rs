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
    pub fn normal_matrix(&self) -> Mat4 {
        self.local_matrix().inverse().transpose()
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

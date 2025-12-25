//! Camera systems for rendering.

use glam::{Mat4, Quat, Vec3};

/// Projection type for the camera.
#[derive(Clone, Debug)]
pub enum Projection {
    /// Perspective projection
    Perspective {
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    },
    /// Orthographic projection
    Orthographic {
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    },
}

/// A camera for rendering the scene.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Camera position in world space
    pub position: Vec3,
    /// Camera rotation
    pub rotation: Quat,
    /// Projection settings
    pub projection: Projection,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            rotation: Quat::IDENTITY,
            projection: Projection::Perspective {
                fov_y: 45.0_f32.to_radians(),
                aspect: 16.0 / 9.0,
                near: 0.1,
                far: 1000.0,
            },
        }
    }
}

impl Camera {
    /// Create a new camera with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the perspective projection.
    pub fn set_perspective(&mut self, fov_y: f32, aspect: f32, near: f32, far: f32) {
        self.projection = Projection::Perspective {
            fov_y,
            aspect,
            near,
            far,
        };
    }

    /// Set the orthographic projection.
    pub fn set_orthographic(
        &mut self,
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) {
        self.projection = Projection::Orthographic {
            left,
            right,
            bottom,
            top,
            near,
            far,
        };
    }

    /// Update the aspect ratio (for perspective projection).
    pub fn set_aspect(&mut self, aspect: f32) {
        if let Projection::Perspective {
            fov_y, near, far, ..
        } = self.projection
        {
            self.projection = Projection::Perspective {
                fov_y,
                aspect,
                near,
                far,
            };
        }
    }

    /// Get the view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        let forward = self.rotation * Vec3::NEG_Z;
        let target = self.position + forward;
        Mat4::look_at_rh(self.position, target, Vec3::Y)
    }

    /// Get the projection matrix (with Vulkan Y-flip).
    pub fn projection_matrix(&self) -> Mat4 {
        let mut proj = match self.projection {
            Projection::Perspective {
                fov_y,
                aspect,
                near,
                far,
            } => Mat4::perspective_rh(fov_y, aspect, near, far),
            Projection::Orthographic {
                left,
                right,
                bottom,
                top,
                near,
                far,
            } => Mat4::orthographic_rh(left, right, bottom, top, near, far),
        };
        // Flip Y for Vulkan coordinate system
        proj.y_axis.y *= -1.0;
        proj
    }

    /// Get the view-projection matrix.
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
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

    /// Look at a target position.
    pub fn look_at(&mut self, target: Vec3) {
        let forward = (target - self.position).normalize();
        if forward.length_squared() > 0.0 {
            self.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, forward);
        }
    }
}

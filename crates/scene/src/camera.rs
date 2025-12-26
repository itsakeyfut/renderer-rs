//! Camera systems for rendering.
//!
//! This module provides camera implementations with support for:
//! - Perspective and orthographic projections
//! - View matrix calculation
//! - FPS-style camera controller (WASD movement, mouse look)
//! - Orbit-style camera controller (rotate around target)

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

    /// Set the camera rotation using pitch and yaw angles (Euler angles).
    ///
    /// # Arguments
    /// * `pitch` - Rotation around the X axis (up/down), in radians. Clamped to [-89°, 89°].
    /// * `yaw` - Rotation around the Y axis (left/right), in radians.
    pub fn set_rotation(&mut self, pitch: f32, yaw: f32) {
        // Clamp pitch to prevent gimbal lock (avoid looking straight up/down)
        let max_pitch = 89.0_f32.to_radians();
        let clamped_pitch = pitch.clamp(-max_pitch, max_pitch);

        // Create rotation from Euler angles (YXZ order: yaw first, then pitch)
        self.rotation = Quat::from_euler(glam::EulerRot::YXZ, yaw, clamped_pitch, 0.0);
    }

    /// Translate the camera by a given offset in world space.
    pub fn translate(&mut self, offset: Vec3) {
        self.position += offset;
    }

    /// Move the camera forward/backward relative to its current orientation.
    pub fn move_forward(&mut self, distance: f32) {
        self.position += self.forward() * distance;
    }

    /// Move the camera right/left relative to its current orientation.
    pub fn move_right(&mut self, distance: f32) {
        self.position += self.right() * distance;
    }

    /// Move the camera up/down relative to its current orientation.
    pub fn move_up(&mut self, distance: f32) {
        self.position += self.up() * distance;
    }
}

/// FPS-style camera controller.
///
/// Provides first-person shooter style camera movement:
/// - WASD keys for forward/backward/strafe movement
/// - Mouse movement for look rotation (pitch/yaw)
///
/// # Example
/// ```
/// use scene::camera::{Camera, FpsController};
///
/// let mut camera = Camera::new();
/// let mut controller = FpsController::new();
///
/// // In your update loop:
/// controller.process_mouse_movement(mouse_delta_x, mouse_delta_y);
/// controller.update_camera(&mut camera, delta_time);
/// ```
#[derive(Clone, Debug)]
pub struct FpsController {
    /// Pitch angle in radians (up/down rotation)
    pitch: f32,
    /// Yaw angle in radians (left/right rotation)
    yaw: f32,
    /// Movement speed in units per second
    pub move_speed: f32,
    /// Mouse sensitivity for rotation
    pub mouse_sensitivity: f32,
    /// Movement input (forward/backward, right/left, up/down)
    movement_input: Vec3,
}

impl Default for FpsController {
    fn default() -> Self {
        Self {
            pitch: 0.0,
            yaw: 0.0,
            move_speed: 5.0,
            mouse_sensitivity: 0.002,
            movement_input: Vec3::ZERO,
        }
    }
}

impl FpsController {
    /// Create a new FPS controller with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new FPS controller with custom speed and sensitivity.
    pub fn with_settings(move_speed: f32, mouse_sensitivity: f32) -> Self {
        Self {
            move_speed,
            mouse_sensitivity,
            ..Default::default()
        }
    }

    /// Get the current pitch angle in radians.
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Get the current yaw angle in radians.
    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Set the pitch angle directly (clamped to valid range).
    pub fn set_pitch(&mut self, pitch: f32) {
        let max_pitch = 89.0_f32.to_radians();
        self.pitch = pitch.clamp(-max_pitch, max_pitch);
    }

    /// Set the yaw angle directly.
    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = yaw;
    }

    /// Process mouse movement delta to update rotation.
    ///
    /// # Arguments
    /// * `delta_x` - Mouse movement in X direction (horizontal)
    /// * `delta_y` - Mouse movement in Y direction (vertical)
    pub fn process_mouse_movement(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * self.mouse_sensitivity;
        self.pitch -= delta_y * self.mouse_sensitivity;

        // Clamp pitch to prevent gimbal lock
        let max_pitch = 89.0_f32.to_radians();
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

        // Wrap yaw to [-PI, PI] to prevent floating point precision issues
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    }

    /// Set movement input for the controller.
    ///
    /// # Arguments
    /// * `forward` - Forward/backward input (-1.0 to 1.0)
    /// * `right` - Right/left input (-1.0 to 1.0)
    /// * `up` - Up/down input (-1.0 to 1.0)
    pub fn set_movement_input(&mut self, forward: f32, right: f32, up: f32) {
        self.movement_input = Vec3::new(right, up, -forward);
    }

    /// Update the camera based on current controller state.
    ///
    /// # Arguments
    /// * `camera` - The camera to update
    /// * `delta_time` - Time elapsed since last update in seconds
    pub fn update_camera(&self, camera: &mut Camera, delta_time: f32) {
        // Apply rotation
        camera.set_rotation(self.pitch, self.yaw);

        // Apply movement relative to camera orientation
        if self.movement_input.length_squared() > 0.0 {
            let movement = self.movement_input.normalize() * self.move_speed * delta_time;

            camera.position += camera.right() * movement.x;
            camera.position += camera.up() * movement.y;
            camera.position += camera.forward() * -movement.z;
        }
    }

    /// Reset the controller to match the current camera orientation.
    ///
    /// Useful when taking control of an existing camera.
    pub fn sync_with_camera(&mut self, camera: &Camera) {
        // Extract pitch and yaw from the camera's rotation quaternion
        let (yaw, pitch, _roll) = camera.rotation.to_euler(glam::EulerRot::YXZ);
        self.yaw = yaw;
        self.pitch = pitch;
    }
}

/// Orbit-style camera controller.
///
/// Provides orbit camera control that rotates around a target point:
/// - Mouse drag to orbit around the target
/// - Scroll to zoom in/out
/// - Optional pan with middle mouse button
///
/// # Example
/// ```
/// use scene::camera::{Camera, OrbitController};
/// use glam::Vec3;
///
/// let mut camera = Camera::new();
/// let mut controller = OrbitController::new(Vec3::ZERO, 5.0);
///
/// // In your update loop:
/// controller.process_mouse_movement(mouse_delta_x, mouse_delta_y);
/// controller.update_camera(&mut camera);
/// ```
#[derive(Clone, Debug)]
pub struct OrbitController {
    /// Target point to orbit around
    pub target: Vec3,
    /// Distance from target
    distance: f32,
    /// Azimuth angle (horizontal rotation around Y axis) in radians
    azimuth: f32,
    /// Polar angle (vertical rotation) in radians, 0 = looking from above, PI/2 = side
    polar: f32,
    /// Minimum distance from target
    pub min_distance: f32,
    /// Maximum distance from target
    pub max_distance: f32,
    /// Minimum polar angle (prevents looking directly from above)
    pub min_polar: f32,
    /// Maximum polar angle (prevents looking directly from below)
    pub max_polar: f32,
    /// Mouse sensitivity for rotation
    pub mouse_sensitivity: f32,
    /// Zoom sensitivity
    pub zoom_sensitivity: f32,
}

impl Default for OrbitController {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 5.0,
            azimuth: 0.0,
            polar: std::f32::consts::FRAC_PI_4, // 45 degrees
            min_distance: 0.5,
            max_distance: 100.0,
            min_polar: 0.1,
            max_polar: std::f32::consts::PI - 0.1,
            mouse_sensitivity: 0.005,
            zoom_sensitivity: 0.5,
        }
    }
}

impl OrbitController {
    /// Create a new orbit controller.
    ///
    /// # Arguments
    /// * `target` - The point to orbit around
    /// * `distance` - Initial distance from the target
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            ..Default::default()
        }
    }

    /// Get the current distance from target.
    pub fn distance(&self) -> f32 {
        self.distance
    }

    /// Set the distance from target (clamped to min/max).
    pub fn set_distance(&mut self, distance: f32) {
        self.distance = distance.clamp(self.min_distance, self.max_distance);
    }

    /// Get the current azimuth angle in radians.
    pub fn azimuth(&self) -> f32 {
        self.azimuth
    }

    /// Set the azimuth angle directly.
    pub fn set_azimuth(&mut self, azimuth: f32) {
        self.azimuth = azimuth;
    }

    /// Get the current polar angle in radians.
    pub fn polar(&self) -> f32 {
        self.polar
    }

    /// Set the polar angle directly (clamped to valid range).
    pub fn set_polar(&mut self, polar: f32) {
        self.polar = polar.clamp(self.min_polar, self.max_polar);
    }

    /// Process mouse movement delta to update orbit rotation.
    ///
    /// # Arguments
    /// * `delta_x` - Mouse movement in X direction (horizontal)
    /// * `delta_y` - Mouse movement in Y direction (vertical)
    pub fn process_mouse_movement(&mut self, delta_x: f32, delta_y: f32) {
        self.azimuth -= delta_x * self.mouse_sensitivity;
        self.polar += delta_y * self.mouse_sensitivity;

        // Clamp polar angle to prevent flipping
        self.polar = self.polar.clamp(self.min_polar, self.max_polar);

        // Wrap azimuth to [-PI, PI]
        self.azimuth = self.azimuth.rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    }

    /// Process zoom input (mouse scroll).
    ///
    /// # Arguments
    /// * `delta` - Zoom delta (positive = zoom in, negative = zoom out)
    pub fn process_zoom(&mut self, delta: f32) {
        self.distance -= delta * self.zoom_sensitivity;
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }

    /// Pan the target position relative to the current view.
    ///
    /// # Arguments
    /// * `delta_x` - Pan amount in X (right/left)
    /// * `delta_y` - Pan amount in Y (up/down)
    /// * `camera` - Reference camera for orientation
    pub fn pan(&mut self, delta_x: f32, delta_y: f32, camera: &Camera) {
        let pan_speed = self.distance * 0.002;
        self.target += camera.right() * (-delta_x * pan_speed);
        self.target += camera.up() * (delta_y * pan_speed);
    }

    /// Calculate the camera position based on orbit parameters.
    fn calculate_position(&self) -> Vec3 {
        // Convert spherical coordinates to Cartesian
        let sin_polar = self.polar.sin();
        let cos_polar = self.polar.cos();
        let sin_azimuth = self.azimuth.sin();
        let cos_azimuth = self.azimuth.cos();

        let offset = Vec3::new(
            self.distance * sin_polar * sin_azimuth,
            self.distance * cos_polar,
            self.distance * sin_polar * cos_azimuth,
        );

        self.target + offset
    }

    /// Update the camera based on current orbit state.
    ///
    /// # Arguments
    /// * `camera` - The camera to update
    pub fn update_camera(&self, camera: &mut Camera) {
        camera.position = self.calculate_position();
        camera.look_at(self.target);
    }

    /// Reset the controller to view a camera's current target.
    ///
    /// # Arguments
    /// * `camera` - The camera to sync from
    /// * `target` - The new target to orbit around
    pub fn sync_with_camera(&mut self, camera: &Camera, target: Vec3) {
        self.target = target;
        self.distance = (camera.position - target).length();

        // Calculate azimuth and polar from camera position
        let offset = camera.position - target;
        if offset.length_squared() > 0.0 {
            let offset_normalized = offset.normalize();
            self.polar = offset_normalized.y.acos();
            self.azimuth =
                offset_normalized.z.atan2(offset_normalized.x) - std::f32::consts::FRAC_PI_2;
        }
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
    fn test_camera_default() {
        let camera = Camera::default();
        assert_eq!(camera.position, Vec3::new(0.0, 0.0, 5.0));
        assert_eq!(camera.rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_camera_view_matrix() {
        let camera = Camera::default();
        let view = camera.view_matrix();

        // Default camera looks along -Z from position (0, 0, 5)
        // View matrix should transform world origin to (0, 0, -5) in view space
        let origin = view.transform_point3(Vec3::ZERO);
        assert!(approx_eq_vec3(origin, Vec3::new(0.0, 0.0, -5.0)));
    }

    #[test]
    fn test_camera_projection_matrix_perspective() {
        let camera = Camera::default();
        let proj = camera.projection_matrix();

        // Projection matrix should have negative Y scale for Vulkan
        assert!(proj.y_axis.y < 0.0, "Y axis should be flipped for Vulkan");
    }

    #[test]
    fn test_camera_set_rotation() {
        let mut camera = Camera::new();

        // Set rotation to look slightly down and to the right
        let pitch = -0.2;
        let yaw = 0.5;
        camera.set_rotation(pitch, yaw);

        // Camera should now be rotated
        assert_ne!(camera.rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_camera_set_rotation_clamps_pitch() {
        let mut camera = Camera::new();

        // Try to set pitch beyond limits
        camera.set_rotation(2.0, 0.0); // > 90 degrees

        // Pitch should be clamped, but we can verify rotation is valid
        assert!(!camera.rotation.x.is_nan());
        assert!(!camera.rotation.y.is_nan());
        assert!(!camera.rotation.z.is_nan());
        assert!(!camera.rotation.w.is_nan());
    }

    #[test]
    fn test_camera_movement() {
        let mut camera = Camera::new();
        let initial_pos = camera.position;

        camera.move_forward(1.0);
        assert_ne!(camera.position, initial_pos);
        assert!(camera.position.z < initial_pos.z); // Moving forward in -Z direction

        camera.position = initial_pos;
        camera.move_right(1.0);
        assert!(camera.position.x > initial_pos.x);

        camera.position = initial_pos;
        camera.move_up(1.0);
        assert!(camera.position.y > initial_pos.y);
    }

    #[test]
    fn test_fps_controller_default() {
        let controller = FpsController::default();
        assert_eq!(controller.pitch, 0.0);
        assert_eq!(controller.yaw, 0.0);
        assert!(controller.move_speed > 0.0);
        assert!(controller.mouse_sensitivity > 0.0);
    }

    #[test]
    fn test_fps_controller_mouse_movement() {
        let mut controller = FpsController::new();
        let initial_yaw = controller.yaw;
        let initial_pitch = controller.pitch;

        controller.process_mouse_movement(100.0, 50.0);

        // Yaw and pitch should change with mouse movement
        assert_ne!(controller.yaw, initial_yaw);
        assert_ne!(controller.pitch, initial_pitch);
        // Pitch should decrease (moving mouse down looks up in typical FPS)
        assert!(controller.pitch < initial_pitch);
    }

    #[test]
    fn test_fps_controller_pitch_clamp() {
        let mut controller = FpsController::new();

        // Large vertical movement should be clamped
        controller.process_mouse_movement(0.0, 100000.0);

        let max_pitch = 89.0_f32.to_radians();
        assert!(controller.pitch >= -max_pitch);
        assert!(controller.pitch <= max_pitch);
    }

    #[test]
    fn test_fps_controller_update_camera() {
        let mut camera = Camera::new();
        let mut controller = FpsController::new();

        // Set some input
        controller.process_mouse_movement(100.0, 0.0);
        controller.set_movement_input(1.0, 0.0, 0.0);

        let initial_pos = camera.position;
        controller.update_camera(&mut camera, 0.1);

        // Camera should have moved forward
        assert_ne!(camera.position, initial_pos);
        // Camera rotation should have changed
        assert_ne!(camera.rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_orbit_controller_default() {
        let controller = OrbitController::default();
        assert_eq!(controller.target, Vec3::ZERO);
        assert!(controller.distance > 0.0);
    }

    #[test]
    fn test_orbit_controller_new() {
        let target = Vec3::new(1.0, 2.0, 3.0);
        let controller = OrbitController::new(target, 10.0);
        assert_eq!(controller.target, target);
        assert_eq!(controller.distance, 10.0);
    }

    #[test]
    fn test_orbit_controller_mouse_movement() {
        let mut controller = OrbitController::new(Vec3::ZERO, 5.0);
        let initial_azimuth = controller.azimuth;
        let initial_polar = controller.polar;

        controller.process_mouse_movement(100.0, 50.0);

        assert_ne!(controller.azimuth, initial_azimuth);
        assert_ne!(controller.polar, initial_polar);
    }

    #[test]
    fn test_orbit_controller_polar_clamp() {
        let mut controller = OrbitController::new(Vec3::ZERO, 5.0);

        // Large vertical movement should be clamped
        controller.process_mouse_movement(0.0, 100000.0);

        assert!(controller.polar >= controller.min_polar);
        assert!(controller.polar <= controller.max_polar);
    }

    #[test]
    fn test_orbit_controller_zoom() {
        let mut controller = OrbitController::new(Vec3::ZERO, 5.0);
        let initial_distance = controller.distance;

        controller.process_zoom(1.0); // Zoom in
        assert!(controller.distance < initial_distance);

        controller.process_zoom(-2.0); // Zoom out
        assert!(controller.distance > initial_distance);
    }

    #[test]
    fn test_orbit_controller_zoom_clamp() {
        let mut controller = OrbitController::new(Vec3::ZERO, 5.0);

        controller.process_zoom(1000.0);
        assert!(controller.distance >= controller.min_distance);

        controller.process_zoom(-1000.0);
        assert!(controller.distance <= controller.max_distance);
    }

    #[test]
    fn test_orbit_controller_update_camera() {
        let mut camera = Camera::new();
        let controller = OrbitController::new(Vec3::ZERO, 5.0);

        controller.update_camera(&mut camera);

        // Camera should be positioned at orbit distance from target
        let distance = (camera.position - controller.target).length();
        assert!(approx_eq(distance, 5.0));
    }

    #[test]
    fn test_orbit_controller_update_camera_look_at_target() {
        let mut camera = Camera::new();
        let target = Vec3::new(1.0, 0.0, 0.0);
        let controller = OrbitController::new(target, 5.0);

        controller.update_camera(&mut camera);

        // Camera forward direction should point towards target
        let to_target = (target - camera.position).normalize();
        let forward = camera.forward();

        // Should be roughly pointing at target
        let dot = forward.dot(to_target);
        assert!(dot > 0.9, "Camera should be looking at target, dot={}", dot);
    }
}

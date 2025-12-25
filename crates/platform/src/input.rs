//! Input handling for keyboard and mouse.

use std::collections::HashSet;

pub use winit::keyboard::KeyCode;

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

impl From<winit::event::MouseButton> for MouseButton {
    fn from(button: winit::event::MouseButton) -> Self {
        match button {
            winit::event::MouseButton::Left => MouseButton::Left,
            winit::event::MouseButton::Right => MouseButton::Right,
            winit::event::MouseButton::Middle => MouseButton::Middle,
            _ => MouseButton::Left,
        }
    }
}

/// Tracks the current state of keyboard and mouse input.
#[derive(Debug, Default)]
pub struct InputState {
    /// Currently pressed keys
    pressed_keys: HashSet<KeyCode>,
    /// Keys that were just pressed this frame
    just_pressed_keys: HashSet<KeyCode>,
    /// Keys that were just released this frame
    just_released_keys: HashSet<KeyCode>,

    /// Currently pressed mouse buttons
    pressed_buttons: HashSet<MouseButton>,
    /// Mouse buttons that were just pressed this frame
    just_pressed_buttons: HashSet<MouseButton>,
    /// Mouse buttons that were just released this frame
    just_released_buttons: HashSet<MouseButton>,

    /// Current mouse position
    mouse_position: (f32, f32),
    /// Mouse movement delta since last frame
    mouse_delta: (f32, f32),
    /// Mouse scroll delta since last frame
    scroll_delta: (f32, f32),
}

impl InputState {
    /// Create a new input state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Call at the beginning of each frame to clear per-frame state.
    pub fn begin_frame(&mut self) {
        self.just_pressed_keys.clear();
        self.just_released_keys.clear();
        self.just_pressed_buttons.clear();
        self.just_released_buttons.clear();
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = (0.0, 0.0);
    }

    /// Handle a key press event.
    pub fn on_key_pressed(&mut self, key: KeyCode) {
        if self.pressed_keys.insert(key) {
            self.just_pressed_keys.insert(key);
        }
    }

    /// Handle a key release event.
    pub fn on_key_released(&mut self, key: KeyCode) {
        if self.pressed_keys.remove(&key) {
            self.just_released_keys.insert(key);
        }
    }

    /// Handle a mouse button press event.
    pub fn on_mouse_pressed(&mut self, button: MouseButton) {
        if self.pressed_buttons.insert(button) {
            self.just_pressed_buttons.insert(button);
        }
    }

    /// Handle a mouse button release event.
    pub fn on_mouse_released(&mut self, button: MouseButton) {
        if self.pressed_buttons.remove(&button) {
            self.just_released_buttons.insert(button);
        }
    }

    /// Handle mouse movement.
    pub fn on_mouse_moved(&mut self, x: f32, y: f32) {
        let old = self.mouse_position;
        self.mouse_position = (x, y);
        self.mouse_delta = (x - old.0, y - old.1);
    }

    /// Handle mouse scroll.
    pub fn on_scroll(&mut self, delta_x: f32, delta_y: f32) {
        self.scroll_delta = (delta_x, delta_y);
    }

    /// Check if a key is currently pressed.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    /// Check if a key was just pressed this frame.
    pub fn is_key_just_pressed(&self, key: KeyCode) -> bool {
        self.just_pressed_keys.contains(&key)
    }

    /// Check if a key was just released this frame.
    pub fn is_key_just_released(&self, key: KeyCode) -> bool {
        self.just_released_keys.contains(&key)
    }

    /// Check if a mouse button is currently pressed.
    pub fn is_mouse_pressed(&self, button: MouseButton) -> bool {
        self.pressed_buttons.contains(&button)
    }

    /// Check if a mouse button was just pressed this frame.
    pub fn is_mouse_just_pressed(&self, button: MouseButton) -> bool {
        self.just_pressed_buttons.contains(&button)
    }

    /// Get the current mouse position.
    pub fn mouse_position(&self) -> (f32, f32) {
        self.mouse_position
    }

    /// Get the mouse movement delta since last frame.
    pub fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

    /// Get the scroll delta since last frame.
    pub fn scroll_delta(&self) -> (f32, f32) {
        self.scroll_delta
    }
}

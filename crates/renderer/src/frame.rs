//! Frame management and synchronization.

use crate::MAX_FRAMES_IN_FLIGHT;

/// Manages per-frame resources and synchronization.
pub struct FrameManager {
    /// Current frame index (0 to MAX_FRAMES_IN_FLIGHT - 1)
    current_frame: usize,
    /// Swapchain image index for the current frame
    image_index: u32,
}

impl FrameManager {
    /// Create a new frame manager.
    pub fn new() -> Self {
        Self {
            current_frame: 0,
            image_index: 0,
        }
    }

    /// Get the current frame index.
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Get the current swapchain image index.
    pub fn image_index(&self) -> u32 {
        self.image_index
    }

    /// Set the current swapchain image index.
    pub fn set_image_index(&mut self, index: u32) {
        self.image_index = index;
    }

    /// Advance to the next frame.
    pub fn next_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}

impl Default for FrameManager {
    fn default() -> Self {
        Self::new()
    }
}

//! High-resolution timer for frame timing and profiling.

use std::time::{Duration, Instant};

/// High-resolution timer for measuring elapsed time.
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    last_tick: Instant,
}

impl Timer {
    /// Create a new timer, starting from now.
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_tick: now,
        }
    }

    /// Get the total elapsed time since the timer was created.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get the elapsed time in seconds since the timer was created.
    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed().as_secs_f32()
    }

    /// Get the time elapsed since the last call to `tick()`.
    /// This is useful for calculating delta time in a game loop.
    pub fn tick(&mut self) -> Duration {
        let now = Instant::now();
        let delta = now - self.last_tick;
        self.last_tick = now;
        delta
    }

    /// Get the delta time in seconds since the last tick.
    pub fn delta_secs(&mut self) -> f32 {
        self.tick().as_secs_f32()
    }

    /// Reset the timer to the current time.
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.start = now;
        self.last_tick = now;
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

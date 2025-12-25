//! Light definitions for the scene.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// A directional light (sun-like).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DirectionalLight {
    /// Light direction (normalized)
    pub direction: Vec3,
    pub _pad0: f32,
    /// Light color
    pub color: Vec3,
    /// Light intensity
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, -1.0, 0.0),
            _pad0: 0.0,
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }
}

/// A point light (omnidirectional).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointLight {
    /// Light position in world space
    pub position: Vec3,
    /// Attenuation radius
    pub radius: f32,
    /// Light color
    pub color: Vec3,
    /// Light intensity
    pub intensity: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            radius: 10.0,
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }
}

/// A spot light (cone-shaped).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpotLight {
    /// Light position in world space
    pub position: Vec3,
    pub _pad0: f32,
    /// Light direction (normalized)
    pub direction: Vec3,
    pub _pad1: f32,
    /// Light color
    pub color: Vec3,
    /// Light intensity
    pub intensity: f32,
    /// Inner cone angle cosine
    pub inner_cutoff: f32,
    /// Outer cone angle cosine
    pub outer_cutoff: f32,
    pub _pad2: [f32; 2],
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            _pad0: 0.0,
            direction: Vec3::new(0.0, -1.0, 0.0),
            _pad1: 0.0,
            color: Vec3::ONE,
            intensity: 1.0,
            inner_cutoff: 0.9, // ~25 degrees
            outer_cutoff: 0.8, // ~37 degrees
            _pad2: [0.0; 2],
        }
    }
}

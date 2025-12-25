//! Material definitions and loading.

use glam::Vec4;

/// PBR material properties.
#[derive(Debug, Clone)]
pub struct Material {
    /// Base color (albedo)
    pub base_color: Vec4,
    /// Metallic factor (0.0 = dielectric, 1.0 = metal)
    pub metallic: f32,
    /// Roughness factor (0.0 = smooth, 1.0 = rough)
    pub roughness: f32,
    /// Ambient occlusion factor
    pub ao: f32,
    /// Emissive color
    pub emissive: Vec4,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            emissive: Vec4::ZERO,
        }
    }
}

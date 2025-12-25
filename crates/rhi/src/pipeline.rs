//! Graphics and compute pipeline management.
//!
//! This module handles VkPipeline and VkPipelineLayout creation for graphics
//! and compute pipelines.
//!
//! # Overview
//!
//! - [`PipelineLayout`] wraps VkPipelineLayout for descriptor set and push constant configuration
//! - [`Pipeline`] wraps VkPipeline for graphics or compute pipeline state
//! - [`GraphicsPipelineBuilder`] provides a flexible builder for graphics pipeline creation
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use std::path::Path;
//! use renderer_rhi::device::Device;
//! use renderer_rhi::shader::{Shader, ShaderStage};
//! use renderer_rhi::pipeline::{Pipeline, PipelineLayout, GraphicsPipelineBuilder};
//! use renderer_rhi::vertex::Vertex;
//! use ash::vk;
//!
//! # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
//! // Load shaders
//! let vertex_shader = Shader::from_spirv_file(
//!     device.clone(),
//!     Path::new("shaders/main.vert.spv"),
//!     ShaderStage::Vertex,
//!     "main",
//! )?;
//!
//! let fragment_shader = Shader::from_spirv_file(
//!     device.clone(),
//!     Path::new("shaders/main.frag.spv"),
//!     ShaderStage::Fragment,
//!     "main",
//! )?;
//!
//! // Create pipeline layout
//! let layout = PipelineLayout::new(device.clone(), &[], &[])?;
//!
//! // Build graphics pipeline
//! let pipeline = GraphicsPipelineBuilder::new()
//!     .vertex_shader(&vertex_shader)
//!     .fragment_shader(&fragment_shader)
//!     .vertex_binding(Vertex::binding_description())
//!     .vertex_attributes(&Vertex::attribute_descriptions())
//!     .color_attachment_format(vk::Format::B8G8R8A8_SRGB)
//!     .build(device.clone(), &layout)?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;
use tracing::{debug, info};

use crate::device::Device;
use crate::error::{RhiError, RhiResult};
use crate::shader::Shader;

/// Vulkan pipeline layout wrapper.
///
/// A pipeline layout describes the complete set of resources that can be
/// accessed by a pipeline. This includes descriptor set layouts and push
/// constant ranges.
///
/// # Thread Safety
///
/// The pipeline layout is immutable after creation and can be safely shared
/// between threads.
pub struct PipelineLayout {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan pipeline layout handle.
    layout: vk::PipelineLayout,
}

impl PipelineLayout {
    /// Creates a new pipeline layout.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `descriptor_set_layouts` - Slice of descriptor set layout handles
    /// * `push_constant_ranges` - Slice of push constant ranges
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline layout creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::pipeline::PipelineLayout;
    /// use ash::vk;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// // Create a simple layout with no descriptors and one push constant range
    /// let push_constant_range = vk::PushConstantRange {
    ///     stage_flags: vk::ShaderStageFlags::VERTEX,
    ///     offset: 0,
    ///     size: 64,
    /// };
    ///
    /// let layout = PipelineLayout::new(device, &[], &[push_constant_range])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        device: Arc<Device>,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> RhiResult<Self> {
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let layout = unsafe { device.handle().create_pipeline_layout(&create_info, None)? };

        debug!(
            "Created pipeline layout with {} descriptor set layout(s) and {} push constant range(s)",
            descriptor_set_layouts.len(),
            push_constant_ranges.len()
        );

        Ok(Self { device, layout })
    }

    /// Returns the Vulkan pipeline layout handle.
    #[inline]
    pub fn handle(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle()
                .destroy_pipeline_layout(self.layout, None);
        }
        debug!("Pipeline layout destroyed");
    }
}

/// Vulkan pipeline wrapper.
///
/// A pipeline encapsulates all the shader stages and fixed-function state
/// needed to process vertices and generate fragments. This struct manages
/// both graphics and compute pipelines.
///
/// # Thread Safety
///
/// The pipeline is immutable after creation and can be safely shared
/// between threads.
pub struct Pipeline {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan pipeline handle.
    pipeline: vk::Pipeline,
    /// Pipeline bind point (graphics or compute).
    bind_point: vk::PipelineBindPoint,
}

impl Pipeline {
    /// Creates a graphics pipeline from a builder configuration.
    ///
    /// This is the internal constructor used by [`GraphicsPipelineBuilder`].
    /// For most use cases, prefer using the builder pattern.
    fn create_graphics_internal(
        device: Arc<Device>,
        create_info: &vk::GraphicsPipelineCreateInfo,
    ) -> RhiResult<Self> {
        let pipeline = unsafe {
            device
                .handle()
                .create_graphics_pipelines(vk::PipelineCache::null(), &[*create_info], None)
                .map_err(|(_, result)| result)?[0]
        };

        info!("Graphics pipeline created");

        Ok(Self {
            device,
            pipeline,
            bind_point: vk::PipelineBindPoint::GRAPHICS,
        })
    }

    /// Creates a graphics pipeline with common defaults.
    ///
    /// This is a convenience method that creates a graphics pipeline with:
    /// - Triangle list topology
    /// - Dynamic viewport and scissor
    /// - Back-face culling
    /// - Counter-clockwise front face
    /// - No blending
    ///
    /// For more control, use [`GraphicsPipelineBuilder`].
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `vertex_shader` - The vertex shader module
    /// * `fragment_shader` - The fragment shader module
    /// * `vertex_bindings` - Vertex input binding descriptions
    /// * `vertex_attributes` - Vertex input attribute descriptions
    /// * `color_attachment_formats` - Color attachment formats for dynamic rendering
    /// * `depth_attachment_format` - Optional depth attachment format
    /// * `layout` - The pipeline layout
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline creation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn create_graphics(
        device: Arc<Device>,
        vertex_shader: &Shader,
        fragment_shader: &Shader,
        vertex_bindings: &[vk::VertexInputBindingDescription],
        vertex_attributes: &[vk::VertexInputAttributeDescription],
        color_attachment_formats: &[vk::Format],
        depth_attachment_format: Option<vk::Format>,
        layout: &PipelineLayout,
    ) -> RhiResult<Self> {
        GraphicsPipelineBuilder::new()
            .vertex_shader(vertex_shader)
            .fragment_shader(fragment_shader)
            .vertex_bindings(vertex_bindings)
            .vertex_attributes(vertex_attributes)
            .color_attachment_formats(color_attachment_formats)
            .depth_attachment_format_opt(depth_attachment_format)
            .build(device, layout)
    }

    /// Returns the Vulkan pipeline handle.
    #[inline]
    pub fn handle(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Returns the pipeline bind point (graphics or compute).
    #[inline]
    pub fn bind_point(&self) -> vk::PipelineBindPoint {
        self.bind_point
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_pipeline(self.pipeline, None);
        }
        info!(
            "{} pipeline destroyed",
            if self.bind_point == vk::PipelineBindPoint::GRAPHICS {
                "Graphics"
            } else {
                "Compute"
            }
        );
    }
}

/// Primitive topology for input assembly.
///
/// Defines how vertices are assembled into primitives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrimitiveTopology {
    /// Independent points.
    PointList,
    /// Independent lines.
    LineList,
    /// Connected lines with each vertex after the first starting a new line.
    LineStrip,
    /// Independent triangles.
    #[default]
    TriangleList,
    /// Connected triangles with shared edges.
    TriangleStrip,
    /// Triangles with shared first vertex (fan).
    TriangleFan,
}

impl PrimitiveTopology {
    /// Converts to Vulkan primitive topology.
    pub fn to_vk(self) -> vk::PrimitiveTopology {
        match self {
            PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
            PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            PrimitiveTopology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
        }
    }
}

/// Polygon rasterization mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PolygonMode {
    /// Fill the polygon interior.
    #[default]
    Fill,
    /// Draw polygon edges as lines.
    Line,
    /// Draw polygon vertices as points.
    Point,
}

impl PolygonMode {
    /// Converts to Vulkan polygon mode.
    pub fn to_vk(self) -> vk::PolygonMode {
        match self {
            PolygonMode::Fill => vk::PolygonMode::FILL,
            PolygonMode::Line => vk::PolygonMode::LINE,
            PolygonMode::Point => vk::PolygonMode::POINT,
        }
    }
}

/// Face culling mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CullMode {
    /// Do not cull any faces.
    None,
    /// Cull front-facing triangles.
    Front,
    /// Cull back-facing triangles.
    #[default]
    Back,
    /// Cull both front and back faces.
    FrontAndBack,
}

impl CullMode {
    /// Converts to Vulkan cull mode flags.
    pub fn to_vk(self) -> vk::CullModeFlags {
        match self {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::Back => vk::CullModeFlags::BACK,
            CullMode::FrontAndBack => vk::CullModeFlags::FRONT_AND_BACK,
        }
    }
}

/// Front face winding order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrontFace {
    /// Counter-clockwise winding is front-facing.
    #[default]
    CounterClockwise,
    /// Clockwise winding is front-facing.
    Clockwise,
}

impl FrontFace {
    /// Converts to Vulkan front face.
    pub fn to_vk(self) -> vk::FrontFace {
        match self {
            FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        }
    }
}

/// Depth comparison operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompareOp {
    /// Never passes.
    Never,
    /// Passes if less than.
    #[default]
    Less,
    /// Passes if equal.
    Equal,
    /// Passes if less than or equal.
    LessOrEqual,
    /// Passes if greater than.
    Greater,
    /// Passes if not equal.
    NotEqual,
    /// Passes if greater than or equal.
    GreaterOrEqual,
    /// Always passes.
    Always,
}

impl CompareOp {
    /// Converts to Vulkan compare op.
    pub fn to_vk(self) -> vk::CompareOp {
        match self {
            CompareOp::Never => vk::CompareOp::NEVER,
            CompareOp::Less => vk::CompareOp::LESS,
            CompareOp::Equal => vk::CompareOp::EQUAL,
            CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareOp::Greater => vk::CompareOp::GREATER,
            CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareOp::Always => vk::CompareOp::ALWAYS,
        }
    }
}

/// Blend factor for color blending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
}

impl BlendFactor {
    /// Converts to Vulkan blend factor.
    pub fn to_vk(self) -> vk::BlendFactor {
        match self {
            BlendFactor::Zero => vk::BlendFactor::ZERO,
            BlendFactor::One => vk::BlendFactor::ONE,
            BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
            BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
            BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            BlendFactor::ConstantColor => vk::BlendFactor::CONSTANT_COLOR,
            BlendFactor::OneMinusConstantColor => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
            BlendFactor::ConstantAlpha => vk::BlendFactor::CONSTANT_ALPHA,
            BlendFactor::OneMinusConstantAlpha => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
            BlendFactor::SrcAlphaSaturate => vk::BlendFactor::SRC_ALPHA_SATURATE,
        }
    }
}

/// Blend operation for color blending.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendOp {
    #[default]
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl BlendOp {
    /// Converts to Vulkan blend op.
    pub fn to_vk(self) -> vk::BlendOp {
        match self {
            BlendOp::Add => vk::BlendOp::ADD,
            BlendOp::Subtract => vk::BlendOp::SUBTRACT,
            BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            BlendOp::Min => vk::BlendOp::MIN,
            BlendOp::Max => vk::BlendOp::MAX,
        }
    }
}

/// Color blend attachment configuration.
#[derive(Clone, Copy, Debug)]
pub struct ColorBlendAttachment {
    /// Enable blending for this attachment.
    pub blend_enable: bool,
    /// Source color blend factor.
    pub src_color_blend_factor: BlendFactor,
    /// Destination color blend factor.
    pub dst_color_blend_factor: BlendFactor,
    /// Color blend operation.
    pub color_blend_op: BlendOp,
    /// Source alpha blend factor.
    pub src_alpha_blend_factor: BlendFactor,
    /// Destination alpha blend factor.
    pub dst_alpha_blend_factor: BlendFactor,
    /// Alpha blend operation.
    pub alpha_blend_op: BlendOp,
    /// Color write mask.
    pub color_write_mask: vk::ColorComponentFlags,
}

impl Default for ColorBlendAttachment {
    fn default() -> Self {
        Self {
            blend_enable: false,
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::Zero,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::Zero,
            alpha_blend_op: BlendOp::Add,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }
    }
}

impl ColorBlendAttachment {
    /// Creates a blend attachment with alpha blending enabled.
    ///
    /// Uses standard alpha blending: `src * src_alpha + dst * (1 - src_alpha)`
    pub fn alpha_blend() -> Self {
        Self {
            blend_enable: true,
            src_color_blend_factor: BlendFactor::SrcAlpha,
            dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::Zero,
            alpha_blend_op: BlendOp::Add,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }
    }

    /// Converts to Vulkan pipeline color blend attachment state.
    pub fn to_vk(&self) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState {
            blend_enable: self.blend_enable.into(),
            src_color_blend_factor: self.src_color_blend_factor.to_vk(),
            dst_color_blend_factor: self.dst_color_blend_factor.to_vk(),
            color_blend_op: self.color_blend_op.to_vk(),
            src_alpha_blend_factor: self.src_alpha_blend_factor.to_vk(),
            dst_alpha_blend_factor: self.dst_alpha_blend_factor.to_vk(),
            alpha_blend_op: self.alpha_blend_op.to_vk(),
            color_write_mask: self.color_write_mask,
        }
    }
}

/// Builder for creating graphics pipelines.
///
/// This builder provides a flexible way to configure all aspects of a
/// graphics pipeline. It uses sensible defaults for most settings:
///
/// - Primitive topology: Triangle list
/// - Polygon mode: Fill
/// - Cull mode: Back-face culling
/// - Front face: Counter-clockwise
/// - Depth test: Enabled (if depth format is set)
/// - Depth write: Enabled (if depth format is set)
/// - Depth compare op: Less
/// - Multisampling: 1 sample (no MSAA)
/// - Dynamic states: Viewport and Scissor
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use std::path::Path;
/// use renderer_rhi::device::Device;
/// use renderer_rhi::shader::{Shader, ShaderStage};
/// use renderer_rhi::pipeline::{GraphicsPipelineBuilder, PipelineLayout, CullMode};
/// use renderer_rhi::vertex::Vertex;
/// use ash::vk;
///
/// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
/// # let vertex_shader = Shader::from_spirv_file(device.clone(), Path::new("a.spv"), ShaderStage::Vertex, "main")?;
/// # let fragment_shader = Shader::from_spirv_file(device.clone(), Path::new("b.spv"), ShaderStage::Fragment, "main")?;
/// let layout = PipelineLayout::new(device.clone(), &[], &[])?;
///
/// let pipeline = GraphicsPipelineBuilder::new()
///     .vertex_shader(&vertex_shader)
///     .fragment_shader(&fragment_shader)
///     .vertex_binding(Vertex::binding_description())
///     .vertex_attributes(&Vertex::attribute_descriptions())
///     .color_attachment_format(vk::Format::B8G8R8A8_SRGB)
///     .depth_attachment_format(vk::Format::D32_SFLOAT)
///     .cull_mode(CullMode::Back)
///     .build(device, &layout)?;
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct GraphicsPipelineBuilder<'a> {
    // Shader stages
    vertex_shader: Option<&'a Shader>,
    fragment_shader: Option<&'a Shader>,

    // Vertex input state
    vertex_bindings: Vec<vk::VertexInputBindingDescription>,
    vertex_attributes: Vec<vk::VertexInputAttributeDescription>,

    // Input assembly state
    topology: PrimitiveTopology,
    primitive_restart_enable: bool,

    // Rasterization state
    polygon_mode: PolygonMode,
    cull_mode: CullMode,
    front_face: FrontFace,
    depth_clamp_enable: bool,
    rasterizer_discard_enable: bool,
    depth_bias_enable: bool,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
    line_width: f32,

    // Multisampling state
    rasterization_samples: vk::SampleCountFlags,
    sample_shading_enable: bool,
    min_sample_shading: f32,

    // Depth/stencil state
    depth_test_enable: bool,
    depth_write_enable: bool,
    depth_compare_op: CompareOp,
    depth_bounds_test_enable: bool,
    stencil_test_enable: bool,
    min_depth_bounds: f32,
    max_depth_bounds: f32,

    // Color blend state
    color_blend_attachments: Vec<ColorBlendAttachment>,
    logic_op_enable: bool,
    blend_constants: [f32; 4],

    // Dynamic rendering
    color_attachment_formats: Vec<vk::Format>,
    depth_attachment_format: Option<vk::Format>,
    stencil_attachment_format: Option<vk::Format>,

    // Dynamic state
    dynamic_states: Vec<vk::DynamicState>,
}

impl<'a> GraphicsPipelineBuilder<'a> {
    /// Creates a new graphics pipeline builder with default settings.
    pub fn new() -> Self {
        Self {
            // Shader stages
            vertex_shader: None,
            fragment_shader: None,

            // Vertex input state
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),

            // Input assembly state
            topology: PrimitiveTopology::TriangleList,
            primitive_restart_enable: false,

            // Rasterization state
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,

            // Multisampling state
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: false,
            min_sample_shading: 1.0,

            // Depth/stencil state
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare_op: CompareOp::Less,
            depth_bounds_test_enable: false,
            stencil_test_enable: false,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,

            // Color blend state
            color_blend_attachments: Vec::new(),
            logic_op_enable: false,
            blend_constants: [0.0, 0.0, 0.0, 0.0],

            // Dynamic rendering
            color_attachment_formats: Vec::new(),
            depth_attachment_format: None,
            stencil_attachment_format: None,

            // Dynamic state (viewport and scissor by default)
            dynamic_states: vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
        }
    }

    /// Sets the vertex shader.
    ///
    /// # Panics
    ///
    /// The build will fail if no vertex shader is set.
    pub fn vertex_shader(mut self, shader: &'a Shader) -> Self {
        self.vertex_shader = Some(shader);
        self
    }

    /// Sets the fragment shader.
    ///
    /// # Panics
    ///
    /// The build will fail if no fragment shader is set.
    pub fn fragment_shader(mut self, shader: &'a Shader) -> Self {
        self.fragment_shader = Some(shader);
        self
    }

    /// Adds a vertex input binding description.
    pub fn vertex_binding(mut self, binding: vk::VertexInputBindingDescription) -> Self {
        self.vertex_bindings.push(binding);
        self
    }

    /// Sets all vertex input binding descriptions.
    pub fn vertex_bindings(mut self, bindings: &[vk::VertexInputBindingDescription]) -> Self {
        self.vertex_bindings = bindings.to_vec();
        self
    }

    /// Adds vertex input attribute descriptions.
    pub fn vertex_attributes(mut self, attributes: &[vk::VertexInputAttributeDescription]) -> Self {
        self.vertex_attributes.extend_from_slice(attributes);
        self
    }

    /// Sets the primitive topology.
    pub fn topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Enables or disables primitive restart.
    pub fn primitive_restart_enable(mut self, enable: bool) -> Self {
        self.primitive_restart_enable = enable;
        self
    }

    /// Sets the polygon rasterization mode.
    pub fn polygon_mode(mut self, mode: PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    /// Sets the face culling mode.
    pub fn cull_mode(mut self, mode: CullMode) -> Self {
        self.cull_mode = mode;
        self
    }

    /// Sets the front face winding order.
    pub fn front_face(mut self, face: FrontFace) -> Self {
        self.front_face = face;
        self
    }

    /// Enables or disables depth clamping.
    pub fn depth_clamp_enable(mut self, enable: bool) -> Self {
        self.depth_clamp_enable = enable;
        self
    }

    /// Enables or disables rasterizer discard.
    pub fn rasterizer_discard_enable(mut self, enable: bool) -> Self {
        self.rasterizer_discard_enable = enable;
        self
    }

    /// Enables depth bias with the specified parameters.
    pub fn depth_bias(mut self, constant_factor: f32, clamp: f32, slope_factor: f32) -> Self {
        self.depth_bias_enable = true;
        self.depth_bias_constant_factor = constant_factor;
        self.depth_bias_clamp = clamp;
        self.depth_bias_slope_factor = slope_factor;
        self
    }

    /// Sets the line width for line primitives.
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = width;
        self
    }

    /// Sets the number of rasterization samples (MSAA).
    pub fn rasterization_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.rasterization_samples = samples;
        self
    }

    /// Enables sample shading with the specified minimum fraction.
    pub fn sample_shading(mut self, min_sample_shading: f32) -> Self {
        self.sample_shading_enable = true;
        self.min_sample_shading = min_sample_shading;
        self
    }

    /// Enables or disables depth testing.
    pub fn depth_test_enable(mut self, enable: bool) -> Self {
        self.depth_test_enable = enable;
        self
    }

    /// Enables or disables depth writing.
    pub fn depth_write_enable(mut self, enable: bool) -> Self {
        self.depth_write_enable = enable;
        self
    }

    /// Sets the depth comparison operation.
    pub fn depth_compare_op(mut self, op: CompareOp) -> Self {
        self.depth_compare_op = op;
        self
    }

    /// Enables depth bounds testing with the specified range.
    pub fn depth_bounds(mut self, min: f32, max: f32) -> Self {
        self.depth_bounds_test_enable = true;
        self.min_depth_bounds = min;
        self.max_depth_bounds = max;
        self
    }

    /// Enables stencil testing.
    pub fn stencil_test_enable(mut self, enable: bool) -> Self {
        self.stencil_test_enable = enable;
        self
    }

    /// Adds a color blend attachment configuration.
    pub fn color_blend_attachment(mut self, attachment: ColorBlendAttachment) -> Self {
        self.color_blend_attachments.push(attachment);
        self
    }

    /// Enables logic operations for color blending.
    pub fn logic_op_enable(mut self, enable: bool) -> Self {
        self.logic_op_enable = enable;
        self
    }

    /// Sets the blend constants.
    pub fn blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend_constants = constants;
        self
    }

    /// Adds a color attachment format for dynamic rendering.
    pub fn color_attachment_format(mut self, format: vk::Format) -> Self {
        self.color_attachment_formats.push(format);
        self
    }

    /// Sets all color attachment formats for dynamic rendering.
    pub fn color_attachment_formats(mut self, formats: &[vk::Format]) -> Self {
        self.color_attachment_formats = formats.to_vec();
        self
    }

    /// Sets the depth attachment format for dynamic rendering.
    pub fn depth_attachment_format(mut self, format: vk::Format) -> Self {
        self.depth_attachment_format = Some(format);
        self
    }

    /// Sets the depth attachment format (optional version).
    pub fn depth_attachment_format_opt(mut self, format: Option<vk::Format>) -> Self {
        self.depth_attachment_format = format;
        self
    }

    /// Sets the stencil attachment format for dynamic rendering.
    pub fn stencil_attachment_format(mut self, format: vk::Format) -> Self {
        self.stencil_attachment_format = Some(format);
        self
    }

    /// Adds a dynamic state.
    pub fn dynamic_state(mut self, state: vk::DynamicState) -> Self {
        if !self.dynamic_states.contains(&state) {
            self.dynamic_states.push(state);
        }
        self
    }

    /// Sets all dynamic states (replaces defaults).
    pub fn dynamic_states(mut self, states: &[vk::DynamicState]) -> Self {
        self.dynamic_states = states.to_vec();
        self
    }

    /// Builds the graphics pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `layout` - The pipeline layout
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vertex shader is not set
    /// - Fragment shader is not set
    /// - No color attachment formats are specified
    /// - Pipeline creation fails
    pub fn build(self, device: Arc<Device>, layout: &PipelineLayout) -> RhiResult<Pipeline> {
        // Validate required fields
        let vertex_shader = self
            .vertex_shader
            .ok_or_else(|| RhiError::PipelineError("Vertex shader is required".to_string()))?;

        let fragment_shader = self
            .fragment_shader
            .ok_or_else(|| RhiError::PipelineError("Fragment shader is required".to_string()))?;

        if self.color_attachment_formats.is_empty() {
            return Err(RhiError::PipelineError(
                "At least one color attachment format is required".to_string(),
            ));
        }

        // Create shader stage infos
        let shader_stages = [
            vertex_shader.stage_create_info(),
            fragment_shader.stage_create_info(),
        ];

        // Vertex input state
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&self.vertex_bindings)
            .vertex_attribute_descriptions(&self.vertex_attributes);

        // Input assembly state
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(self.topology.to_vk())
            .primitive_restart_enable(self.primitive_restart_enable);

        // Viewport state (dynamic)
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Rasterization state
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(self.depth_clamp_enable)
            .rasterizer_discard_enable(self.rasterizer_discard_enable)
            .polygon_mode(self.polygon_mode.to_vk())
            .line_width(self.line_width)
            .cull_mode(self.cull_mode.to_vk())
            .front_face(self.front_face.to_vk())
            .depth_bias_enable(self.depth_bias_enable)
            .depth_bias_constant_factor(self.depth_bias_constant_factor)
            .depth_bias_clamp(self.depth_bias_clamp)
            .depth_bias_slope_factor(self.depth_bias_slope_factor);

        // Multisample state
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(self.sample_shading_enable)
            .rasterization_samples(self.rasterization_samples)
            .min_sample_shading(self.min_sample_shading);

        // Depth/stencil state
        let has_depth = self.depth_attachment_format.is_some();
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(has_depth && self.depth_test_enable)
            .depth_write_enable(has_depth && self.depth_write_enable)
            .depth_compare_op(self.depth_compare_op.to_vk())
            .depth_bounds_test_enable(self.depth_bounds_test_enable)
            .min_depth_bounds(self.min_depth_bounds)
            .max_depth_bounds(self.max_depth_bounds)
            .stencil_test_enable(self.stencil_test_enable);

        // Color blend attachments
        let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> =
            if self.color_blend_attachments.is_empty() {
                // Create default attachment for each color format
                self.color_attachment_formats
                    .iter()
                    .map(|_| ColorBlendAttachment::default().to_vk())
                    .collect()
            } else {
                self.color_blend_attachments
                    .iter()
                    .map(|a| a.to_vk())
                    .collect()
            };

        // Color blend state
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(self.logic_op_enable)
            .attachments(&color_blend_attachments)
            .blend_constants(self.blend_constants);

        // Dynamic state
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&self.dynamic_states);

        // Dynamic rendering info (Vulkan 1.3)
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&self.color_attachment_formats);

        if let Some(depth_format) = self.depth_attachment_format {
            rendering_info = rendering_info.depth_attachment_format(depth_format);
        }

        if let Some(stencil_format) = self.stencil_attachment_format {
            rendering_info = rendering_info.stencil_attachment_format(stencil_format);
        }

        // Create pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(layout.handle())
            .push_next(&mut rendering_info);

        Pipeline::create_graphics_internal(device, &pipeline_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_topology_to_vk() {
        assert_eq!(
            PrimitiveTopology::PointList.to_vk(),
            vk::PrimitiveTopology::POINT_LIST
        );
        assert_eq!(
            PrimitiveTopology::LineList.to_vk(),
            vk::PrimitiveTopology::LINE_LIST
        );
        assert_eq!(
            PrimitiveTopology::LineStrip.to_vk(),
            vk::PrimitiveTopology::LINE_STRIP
        );
        assert_eq!(
            PrimitiveTopology::TriangleList.to_vk(),
            vk::PrimitiveTopology::TRIANGLE_LIST
        );
        assert_eq!(
            PrimitiveTopology::TriangleStrip.to_vk(),
            vk::PrimitiveTopology::TRIANGLE_STRIP
        );
        assert_eq!(
            PrimitiveTopology::TriangleFan.to_vk(),
            vk::PrimitiveTopology::TRIANGLE_FAN
        );
    }

    #[test]
    fn test_polygon_mode_to_vk() {
        assert_eq!(PolygonMode::Fill.to_vk(), vk::PolygonMode::FILL);
        assert_eq!(PolygonMode::Line.to_vk(), vk::PolygonMode::LINE);
        assert_eq!(PolygonMode::Point.to_vk(), vk::PolygonMode::POINT);
    }

    #[test]
    fn test_cull_mode_to_vk() {
        assert_eq!(CullMode::None.to_vk(), vk::CullModeFlags::NONE);
        assert_eq!(CullMode::Front.to_vk(), vk::CullModeFlags::FRONT);
        assert_eq!(CullMode::Back.to_vk(), vk::CullModeFlags::BACK);
        assert_eq!(
            CullMode::FrontAndBack.to_vk(),
            vk::CullModeFlags::FRONT_AND_BACK
        );
    }

    #[test]
    fn test_front_face_to_vk() {
        assert_eq!(
            FrontFace::CounterClockwise.to_vk(),
            vk::FrontFace::COUNTER_CLOCKWISE
        );
        assert_eq!(FrontFace::Clockwise.to_vk(), vk::FrontFace::CLOCKWISE);
    }

    #[test]
    fn test_compare_op_to_vk() {
        assert_eq!(CompareOp::Never.to_vk(), vk::CompareOp::NEVER);
        assert_eq!(CompareOp::Less.to_vk(), vk::CompareOp::LESS);
        assert_eq!(CompareOp::Equal.to_vk(), vk::CompareOp::EQUAL);
        assert_eq!(CompareOp::LessOrEqual.to_vk(), vk::CompareOp::LESS_OR_EQUAL);
        assert_eq!(CompareOp::Greater.to_vk(), vk::CompareOp::GREATER);
        assert_eq!(CompareOp::NotEqual.to_vk(), vk::CompareOp::NOT_EQUAL);
        assert_eq!(
            CompareOp::GreaterOrEqual.to_vk(),
            vk::CompareOp::GREATER_OR_EQUAL
        );
        assert_eq!(CompareOp::Always.to_vk(), vk::CompareOp::ALWAYS);
    }

    #[test]
    fn test_blend_factor_to_vk() {
        assert_eq!(BlendFactor::Zero.to_vk(), vk::BlendFactor::ZERO);
        assert_eq!(BlendFactor::One.to_vk(), vk::BlendFactor::ONE);
        assert_eq!(BlendFactor::SrcAlpha.to_vk(), vk::BlendFactor::SRC_ALPHA);
        assert_eq!(
            BlendFactor::OneMinusSrcAlpha.to_vk(),
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA
        );
    }

    #[test]
    fn test_blend_op_to_vk() {
        assert_eq!(BlendOp::Add.to_vk(), vk::BlendOp::ADD);
        assert_eq!(BlendOp::Subtract.to_vk(), vk::BlendOp::SUBTRACT);
        assert_eq!(
            BlendOp::ReverseSubtract.to_vk(),
            vk::BlendOp::REVERSE_SUBTRACT
        );
        assert_eq!(BlendOp::Min.to_vk(), vk::BlendOp::MIN);
        assert_eq!(BlendOp::Max.to_vk(), vk::BlendOp::MAX);
    }

    #[test]
    fn test_color_blend_attachment_default() {
        let attachment = ColorBlendAttachment::default();
        assert!(!attachment.blend_enable);
        assert_eq!(attachment.color_write_mask, vk::ColorComponentFlags::RGBA);
    }

    #[test]
    fn test_color_blend_attachment_alpha_blend() {
        let attachment = ColorBlendAttachment::alpha_blend();
        assert!(attachment.blend_enable);
        assert_eq!(attachment.src_color_blend_factor, BlendFactor::SrcAlpha);
        assert_eq!(
            attachment.dst_color_blend_factor,
            BlendFactor::OneMinusSrcAlpha
        );
    }

    #[test]
    fn test_graphics_pipeline_builder_default() {
        let builder = GraphicsPipelineBuilder::new();
        assert!(builder.vertex_shader.is_none());
        assert!(builder.fragment_shader.is_none());
        assert!(builder.vertex_bindings.is_empty());
        assert!(builder.vertex_attributes.is_empty());
        assert_eq!(builder.topology, PrimitiveTopology::TriangleList);
        assert_eq!(builder.cull_mode, CullMode::Back);
        assert_eq!(builder.front_face, FrontFace::CounterClockwise);
        assert!(builder.depth_test_enable);
        assert!(builder.depth_write_enable);
        assert_eq!(builder.dynamic_states.len(), 2);
    }

    #[test]
    fn test_graphics_pipeline_builder_topology() {
        let builder = GraphicsPipelineBuilder::new().topology(PrimitiveTopology::LineList);
        assert_eq!(builder.topology, PrimitiveTopology::LineList);
    }

    #[test]
    fn test_graphics_pipeline_builder_cull_mode() {
        let builder = GraphicsPipelineBuilder::new().cull_mode(CullMode::None);
        assert_eq!(builder.cull_mode, CullMode::None);
    }

    #[test]
    fn test_graphics_pipeline_builder_depth_settings() {
        let builder = GraphicsPipelineBuilder::new()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(CompareOp::Always);
        assert!(!builder.depth_test_enable);
        assert!(!builder.depth_write_enable);
        assert_eq!(builder.depth_compare_op, CompareOp::Always);
    }

    #[test]
    fn test_graphics_pipeline_builder_dynamic_state() {
        let builder = GraphicsPipelineBuilder::new()
            .dynamic_state(vk::DynamicState::LINE_WIDTH)
            .dynamic_state(vk::DynamicState::LINE_WIDTH); // duplicate should not add twice
        assert!(builder.dynamic_states.contains(&vk::DynamicState::VIEWPORT));
        assert!(builder.dynamic_states.contains(&vk::DynamicState::SCISSOR));
        assert!(
            builder
                .dynamic_states
                .contains(&vk::DynamicState::LINE_WIDTH)
        );
        assert_eq!(builder.dynamic_states.len(), 3);
    }

    #[test]
    fn test_default_trait_implementations() {
        // Test Default for PrimitiveTopology
        let topo: PrimitiveTopology = Default::default();
        assert_eq!(topo, PrimitiveTopology::TriangleList);

        // Test Default for PolygonMode
        let mode: PolygonMode = Default::default();
        assert_eq!(mode, PolygonMode::Fill);

        // Test Default for CullMode
        let cull: CullMode = Default::default();
        assert_eq!(cull, CullMode::Back);

        // Test Default for FrontFace
        let face: FrontFace = Default::default();
        assert_eq!(face, FrontFace::CounterClockwise);

        // Test Default for CompareOp
        let op: CompareOp = Default::default();
        assert_eq!(op, CompareOp::Less);

        // Test Default for BlendOp
        let blend: BlendOp = Default::default();
        assert_eq!(blend, BlendOp::Add);
    }
}

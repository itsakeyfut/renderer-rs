//! Dynamic rendering helpers (Vulkan 1.3).
//!
//! This module provides utilities for setting up dynamic rendering without
//! using traditional VkRenderPass objects. Dynamic rendering (introduced in
//! Vulkan 1.3) offers more flexible attachment management.
//!
//! # Overview
//!
//! The main types in this module are:
//!
//! - [`ColorAttachment`] - Configuration for a color attachment
//! - [`DepthAttachment`] - Configuration for a depth attachment
//! - [`RenderingConfig`] - Complete rendering configuration
//!
//! # Example
//!
//! ```no_run
//! use ash::vk;
//! use renderer_rhi::rendering::{ColorAttachment, DepthAttachment, RenderingConfig};
//! use renderer_rhi::command::CommandBuffer;
//!
//! # fn example(
//! #     swapchain_image_view: vk::ImageView,
//! #     depth_image_view: vk::ImageView,
//! #     cmd: &CommandBuffer,
//! # ) {
//! // Set up color attachment with clear
//! let color_attachment = ColorAttachment::new(swapchain_image_view)
//!     .with_clear_color([0.1, 0.1, 0.1, 1.0]);
//!
//! // Set up depth attachment
//! let depth_attachment = DepthAttachment::new(depth_image_view)
//!     .with_clear_depth(1.0);
//!
//! // Create rendering config
//! let config = RenderingConfig::new(800, 600)
//!     .with_color_attachment(color_attachment)
//!     .with_depth_attachment(depth_attachment);
//!
//! // Build rendering info bundle with proper lifetime management
//! let bundle = config.build();
//! cmd.begin_rendering(bundle.info());
//! // ... draw commands ...
//! cmd.end_rendering();
//! # }
//! ```

use ash::vk;

/// Configuration for a color attachment in dynamic rendering.
///
/// This struct wraps the configuration needed to create a
/// `VkRenderingAttachmentInfo` for a color attachment.
///
/// # Default Values
///
/// - `layout`: `COLOR_ATTACHMENT_OPTIMAL`
/// - `load_op`: `CLEAR`
/// - `store_op`: `STORE`
/// - `clear_value`: Black (0.0, 0.0, 0.0, 1.0)
/// - `resolve_image_view`: `null`
/// - `resolve_image_layout`: `UNDEFINED`
/// - `resolve_mode`: `NONE`
#[derive(Clone)]
pub struct ColorAttachment {
    /// The image view to render to.
    pub image_view: vk::ImageView,
    /// The image layout during rendering.
    pub layout: vk::ImageLayout,
    /// How to load the attachment contents at the start of rendering.
    pub load_op: vk::AttachmentLoadOp,
    /// How to store the attachment contents at the end of rendering.
    pub store_op: vk::AttachmentStoreOp,
    /// Clear value when load_op is CLEAR.
    pub clear_value: vk::ClearColorValue,
    /// Optional resolve image view for MSAA.
    pub resolve_image_view: vk::ImageView,
    /// Layout of the resolve image during rendering.
    pub resolve_image_layout: vk::ImageLayout,
    /// Resolve mode for MSAA.
    pub resolve_mode: vk::ResolveModeFlags,
}

impl ColorAttachment {
    /// Creates a new color attachment with default settings.
    ///
    /// # Arguments
    ///
    /// * `image_view` - The image view to render to
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ash::vk;
    /// use renderer_rhi::rendering::ColorAttachment;
    ///
    /// # fn example(swapchain_image_view: vk::ImageView) {
    /// let attachment = ColorAttachment::new(swapchain_image_view);
    /// # }
    /// ```
    #[inline]
    pub fn new(image_view: vk::ImageView) -> Self {
        Self {
            image_view,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            resolve_mode: vk::ResolveModeFlags::NONE,
        }
    }

    /// Sets the image layout for this attachment.
    ///
    /// # Arguments
    ///
    /// * `layout` - The image layout to use during rendering
    #[inline]
    pub fn with_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Sets the load operation for this attachment.
    ///
    /// # Arguments
    ///
    /// * `load_op` - How to initialize attachment contents
    #[inline]
    pub fn with_load_op(mut self, load_op: vk::AttachmentLoadOp) -> Self {
        self.load_op = load_op;
        self
    }

    /// Sets the store operation for this attachment.
    ///
    /// # Arguments
    ///
    /// * `store_op` - How to handle attachment contents after rendering
    #[inline]
    pub fn with_store_op(mut self, store_op: vk::AttachmentStoreOp) -> Self {
        self.store_op = store_op;
        self
    }

    /// Sets the clear color as RGBA float values.
    ///
    /// # Arguments
    ///
    /// * `color` - Clear color as [R, G, B, A] floats in range [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ash::vk;
    /// use renderer_rhi::rendering::ColorAttachment;
    ///
    /// # fn example(image_view: vk::ImageView) {
    /// // Set a cornflower blue clear color
    /// let attachment = ColorAttachment::new(image_view)
    ///     .with_clear_color([0.392, 0.584, 0.929, 1.0]);
    /// # }
    /// ```
    #[inline]
    pub fn with_clear_color(mut self, color: [f32; 4]) -> Self {
        self.clear_value = vk::ClearColorValue { float32: color };
        self
    }

    /// Sets the clear color using integer RGBA values.
    ///
    /// # Arguments
    ///
    /// * `color` - Clear color as [R, G, B, A] integers
    #[inline]
    pub fn with_clear_color_int(mut self, color: [i32; 4]) -> Self {
        self.clear_value = vk::ClearColorValue { int32: color };
        self
    }

    /// Sets the clear color using unsigned integer RGBA values.
    ///
    /// # Arguments
    ///
    /// * `color` - Clear color as [R, G, B, A] unsigned integers
    #[inline]
    pub fn with_clear_color_uint(mut self, color: [u32; 4]) -> Self {
        self.clear_value = vk::ClearColorValue { uint32: color };
        self
    }

    /// Configures this attachment to load existing contents.
    ///
    /// Sets `load_op` to `LOAD`, which preserves existing image contents.
    #[inline]
    pub fn load(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::LOAD;
        self
    }

    /// Configures this attachment to not store results.
    ///
    /// Sets `store_op` to `DONT_CARE`, useful for transient attachments.
    #[inline]
    pub fn dont_store(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    /// Configures MSAA resolve for this attachment.
    ///
    /// # Arguments
    ///
    /// * `resolve_view` - Image view to resolve to
    /// * `resolve_layout` - Layout of resolve image during rendering
    /// * `resolve_mode` - How to resolve multi-sampled data
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ash::vk;
    /// use renderer_rhi::rendering::ColorAttachment;
    ///
    /// # fn example(msaa_view: vk::ImageView, resolve_view: vk::ImageView) {
    /// let attachment = ColorAttachment::new(msaa_view)
    ///     .with_resolve(
    ///         resolve_view,
    ///         vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    ///         vk::ResolveModeFlags::AVERAGE,
    ///     );
    /// # }
    /// ```
    #[inline]
    pub fn with_resolve(
        mut self,
        resolve_view: vk::ImageView,
        resolve_layout: vk::ImageLayout,
        resolve_mode: vk::ResolveModeFlags,
    ) -> Self {
        self.resolve_image_view = resolve_view;
        self.resolve_image_layout = resolve_layout;
        self.resolve_mode = resolve_mode;
        self
    }

    /// Converts this attachment to a `VkRenderingAttachmentInfo`.
    ///
    /// # Returns
    ///
    /// A `VkRenderingAttachmentInfo` suitable for use with `vkCmdBeginRendering`.
    #[inline]
    pub fn to_rendering_attachment_info(&self) -> vk::RenderingAttachmentInfo<'static> {
        let mut info = vk::RenderingAttachmentInfo::default()
            .image_view(self.image_view)
            .image_layout(self.layout)
            .load_op(self.load_op)
            .store_op(self.store_op)
            .clear_value(vk::ClearValue {
                color: self.clear_value,
            });

        if self.resolve_image_view != vk::ImageView::null() {
            info = info
                .resolve_image_view(self.resolve_image_view)
                .resolve_image_layout(self.resolve_image_layout)
                .resolve_mode(self.resolve_mode);
        }

        info
    }
}

impl std::fmt::Debug for ColorAttachment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ClearColorValue is a union, so we format the float32 variant by default
        let clear_color = unsafe { self.clear_value.float32 };
        f.debug_struct("ColorAttachment")
            .field("image_view", &self.image_view)
            .field("layout", &self.layout)
            .field("load_op", &self.load_op)
            .field("store_op", &self.store_op)
            .field("clear_value", &clear_color)
            .field("resolve_image_view", &self.resolve_image_view)
            .field("resolve_image_layout", &self.resolve_image_layout)
            .field("resolve_mode", &self.resolve_mode)
            .finish()
    }
}

impl Default for ColorAttachment {
    /// Creates a default color attachment with null image view.
    ///
    /// Note: You must set a valid `image_view` before use.
    fn default() -> Self {
        Self::new(vk::ImageView::null())
    }
}

/// Configuration for a depth attachment in dynamic rendering.
///
/// This struct wraps the configuration needed to create a
/// `VkRenderingAttachmentInfo` for a depth attachment.
///
/// # Default Values
///
/// - `layout`: `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
/// - `load_op`: `CLEAR`
/// - `store_op`: `DONT_CARE`
/// - `clear_value`: depth=1.0, stencil=0
/// - `resolve_image_view`: `null`
/// - `resolve_image_layout`: `UNDEFINED`
/// - `resolve_mode`: `NONE`
#[derive(Clone, Debug)]
pub struct DepthAttachment {
    /// The image view to render to.
    pub image_view: vk::ImageView,
    /// The image layout during rendering.
    pub layout: vk::ImageLayout,
    /// How to load the attachment contents at the start of rendering.
    pub load_op: vk::AttachmentLoadOp,
    /// How to store the attachment contents at the end of rendering.
    pub store_op: vk::AttachmentStoreOp,
    /// Clear value when load_op is CLEAR.
    pub clear_value: vk::ClearDepthStencilValue,
    /// Optional resolve image view for MSAA.
    pub resolve_image_view: vk::ImageView,
    /// Layout of the resolve image during rendering.
    pub resolve_image_layout: vk::ImageLayout,
    /// Resolve mode for MSAA.
    pub resolve_mode: vk::ResolveModeFlags,
}

impl DepthAttachment {
    /// Creates a new depth attachment with default settings.
    ///
    /// # Arguments
    ///
    /// * `image_view` - The depth image view to render to
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ash::vk;
    /// use renderer_rhi::rendering::DepthAttachment;
    ///
    /// # fn example(depth_image_view: vk::ImageView) {
    /// let attachment = DepthAttachment::new(depth_image_view);
    /// # }
    /// ```
    #[inline]
    pub fn new(image_view: vk::ImageView) -> Self {
        Self {
            image_view,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            clear_value: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            resolve_mode: vk::ResolveModeFlags::NONE,
        }
    }

    /// Sets the image layout for this attachment.
    ///
    /// # Arguments
    ///
    /// * `layout` - The image layout to use during rendering
    #[inline]
    pub fn with_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Sets the load operation for this attachment.
    ///
    /// # Arguments
    ///
    /// * `load_op` - How to initialize attachment contents
    #[inline]
    pub fn with_load_op(mut self, load_op: vk::AttachmentLoadOp) -> Self {
        self.load_op = load_op;
        self
    }

    /// Sets the store operation for this attachment.
    ///
    /// # Arguments
    ///
    /// * `store_op` - How to handle attachment contents after rendering
    #[inline]
    pub fn with_store_op(mut self, store_op: vk::AttachmentStoreOp) -> Self {
        self.store_op = store_op;
        self
    }

    /// Sets the clear depth value.
    ///
    /// # Arguments
    ///
    /// * `depth` - Clear depth value, typically 1.0 for far plane
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ash::vk;
    /// use renderer_rhi::rendering::DepthAttachment;
    ///
    /// # fn example(image_view: vk::ImageView) {
    /// let attachment = DepthAttachment::new(image_view)
    ///     .with_clear_depth(1.0);
    /// # }
    /// ```
    #[inline]
    pub fn with_clear_depth(mut self, depth: f32) -> Self {
        self.clear_value.depth = depth;
        self
    }

    /// Sets the clear stencil value.
    ///
    /// # Arguments
    ///
    /// * `stencil` - Clear stencil value
    #[inline]
    pub fn with_clear_stencil(mut self, stencil: u32) -> Self {
        self.clear_value.stencil = stencil;
        self
    }

    /// Sets both clear depth and stencil values.
    ///
    /// # Arguments
    ///
    /// * `depth` - Clear depth value
    /// * `stencil` - Clear stencil value
    #[inline]
    pub fn with_clear_depth_stencil(mut self, depth: f32, stencil: u32) -> Self {
        self.clear_value = vk::ClearDepthStencilValue { depth, stencil };
        self
    }

    /// Configures this attachment to load existing contents.
    ///
    /// Sets `load_op` to `LOAD`, which preserves existing depth values.
    #[inline]
    pub fn load(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::LOAD;
        self
    }

    /// Configures this attachment to store results.
    ///
    /// Sets `store_op` to `STORE`, useful when depth is needed later.
    #[inline]
    pub fn store(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::STORE;
        self
    }

    /// Configures MSAA resolve for this attachment.
    ///
    /// # Arguments
    ///
    /// * `resolve_view` - Image view to resolve to
    /// * `resolve_layout` - Layout of resolve image during rendering
    /// * `resolve_mode` - How to resolve multi-sampled data
    #[inline]
    pub fn with_resolve(
        mut self,
        resolve_view: vk::ImageView,
        resolve_layout: vk::ImageLayout,
        resolve_mode: vk::ResolveModeFlags,
    ) -> Self {
        self.resolve_image_view = resolve_view;
        self.resolve_image_layout = resolve_layout;
        self.resolve_mode = resolve_mode;
        self
    }

    /// Converts this attachment to a `VkRenderingAttachmentInfo`.
    ///
    /// # Returns
    ///
    /// A `VkRenderingAttachmentInfo` suitable for use with `vkCmdBeginRendering`.
    #[inline]
    pub fn to_rendering_attachment_info(&self) -> vk::RenderingAttachmentInfo<'static> {
        let mut info = vk::RenderingAttachmentInfo::default()
            .image_view(self.image_view)
            .image_layout(self.layout)
            .load_op(self.load_op)
            .store_op(self.store_op)
            .clear_value(vk::ClearValue {
                depth_stencil: self.clear_value,
            });

        if self.resolve_image_view != vk::ImageView::null() {
            info = info
                .resolve_image_view(self.resolve_image_view)
                .resolve_image_layout(self.resolve_image_layout)
                .resolve_mode(self.resolve_mode);
        }

        info
    }
}

impl Default for DepthAttachment {
    /// Creates a default depth attachment with null image view.
    ///
    /// Note: You must set a valid `image_view` before use.
    fn default() -> Self {
        Self::new(vk::ImageView::null())
    }
}

/// Configuration for stencil attachment in dynamic rendering.
///
/// This is similar to [`DepthAttachment`] but specifically for stencil-only
/// attachments. For combined depth-stencil, use [`DepthAttachment`].
///
/// # Default Values
///
/// - `layout`: `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
/// - `load_op`: `CLEAR`
/// - `store_op`: `DONT_CARE`
/// - `clear_value`: stencil=0
#[derive(Clone, Debug)]
pub struct StencilAttachment {
    /// The image view to render to.
    pub image_view: vk::ImageView,
    /// The image layout during rendering.
    pub layout: vk::ImageLayout,
    /// How to load the attachment contents at the start of rendering.
    pub load_op: vk::AttachmentLoadOp,
    /// How to store the attachment contents at the end of rendering.
    pub store_op: vk::AttachmentStoreOp,
    /// Clear stencil value when load_op is CLEAR.
    pub clear_stencil: u32,
    /// Optional resolve image view for MSAA.
    pub resolve_image_view: vk::ImageView,
    /// Layout of the resolve image during rendering.
    pub resolve_image_layout: vk::ImageLayout,
    /// Resolve mode for MSAA.
    pub resolve_mode: vk::ResolveModeFlags,
}

impl StencilAttachment {
    /// Creates a new stencil attachment with default settings.
    ///
    /// # Arguments
    ///
    /// * `image_view` - The stencil image view to render to
    #[inline]
    pub fn new(image_view: vk::ImageView) -> Self {
        Self {
            image_view,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            clear_stencil: 0,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            resolve_mode: vk::ResolveModeFlags::NONE,
        }
    }

    /// Sets the image layout for this attachment.
    #[inline]
    pub fn with_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Sets the load operation for this attachment.
    #[inline]
    pub fn with_load_op(mut self, load_op: vk::AttachmentLoadOp) -> Self {
        self.load_op = load_op;
        self
    }

    /// Sets the store operation for this attachment.
    #[inline]
    pub fn with_store_op(mut self, store_op: vk::AttachmentStoreOp) -> Self {
        self.store_op = store_op;
        self
    }

    /// Sets the clear stencil value.
    #[inline]
    pub fn with_clear_stencil(mut self, stencil: u32) -> Self {
        self.clear_stencil = stencil;
        self
    }

    /// Configures this attachment to load existing contents.
    #[inline]
    pub fn load(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::LOAD;
        self
    }

    /// Configures this attachment to store results.
    #[inline]
    pub fn store(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::STORE;
        self
    }

    /// Converts this attachment to a `VkRenderingAttachmentInfo`.
    #[inline]
    pub fn to_rendering_attachment_info(&self) -> vk::RenderingAttachmentInfo<'static> {
        let mut info = vk::RenderingAttachmentInfo::default()
            .image_view(self.image_view)
            .image_layout(self.layout)
            .load_op(self.load_op)
            .store_op(self.store_op)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: self.clear_stencil,
                },
            });

        if self.resolve_image_view != vk::ImageView::null() {
            info = info
                .resolve_image_view(self.resolve_image_view)
                .resolve_image_layout(self.resolve_image_layout)
                .resolve_mode(self.resolve_mode);
        }

        info
    }
}

impl Default for StencilAttachment {
    fn default() -> Self {
        Self::new(vk::ImageView::null())
    }
}

/// Complete rendering configuration for dynamic rendering.
///
/// This struct holds all the information needed to construct a
/// `VkRenderingInfo` for use with `vkCmdBeginRendering`.
///
/// # Example
///
/// ```no_run
/// use ash::vk;
/// use renderer_rhi::rendering::{ColorAttachment, DepthAttachment, RenderingConfig};
///
/// # fn example(color_view: vk::ImageView, depth_view: vk::ImageView) {
/// let config = RenderingConfig::new(1920, 1080)
///     .with_color_attachment(
///         ColorAttachment::new(color_view)
///             .with_clear_color([0.0, 0.0, 0.0, 1.0])
///     )
///     .with_depth_attachment(
///         DepthAttachment::new(depth_view)
///             .with_clear_depth(1.0)
///     );
///
/// let (rendering_info, color_attachments, depth_attachment) = config.build_rendering_info();
/// // Use rendering_info with cmd.begin_rendering(&rendering_info)
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub struct RenderingConfig {
    /// Color attachments for this rendering operation.
    pub color_attachments: Vec<ColorAttachment>,
    /// Optional depth attachment.
    pub depth_attachment: Option<DepthAttachment>,
    /// Optional stencil attachment (separate from depth).
    pub stencil_attachment: Option<StencilAttachment>,
    /// Render area (region to render to).
    pub render_area: vk::Rect2D,
    /// Number of layers to render.
    pub layer_count: u32,
    /// View mask for multiview rendering.
    pub view_mask: u32,
    /// Rendering flags.
    pub flags: vk::RenderingFlags,
}

impl RenderingConfig {
    /// Creates a new rendering configuration with the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `width` - Render area width in pixels
    /// * `height` - Render area height in pixels
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_rhi::rendering::RenderingConfig;
    ///
    /// let config = RenderingConfig::new(800, 600);
    /// ```
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            color_attachments: Vec::new(),
            depth_attachment: None,
            stencil_attachment: None,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            },
            layer_count: 1,
            view_mask: 0,
            flags: vk::RenderingFlags::empty(),
        }
    }

    /// Creates a new rendering configuration from an extent.
    ///
    /// # Arguments
    ///
    /// * `extent` - Render area extent
    #[inline]
    pub fn from_extent(extent: vk::Extent2D) -> Self {
        Self::new(extent.width, extent.height)
    }

    /// Adds a color attachment to this configuration.
    ///
    /// # Arguments
    ///
    /// * `attachment` - The color attachment to add
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ash::vk;
    /// use renderer_rhi::rendering::{ColorAttachment, RenderingConfig};
    ///
    /// # fn example(view: vk::ImageView) {
    /// let config = RenderingConfig::new(800, 600)
    ///     .with_color_attachment(ColorAttachment::new(view));
    /// # }
    /// ```
    #[inline]
    pub fn with_color_attachment(mut self, attachment: ColorAttachment) -> Self {
        self.color_attachments.push(attachment);
        self
    }

    /// Adds multiple color attachments to this configuration.
    ///
    /// # Arguments
    ///
    /// * `attachments` - The color attachments to add
    #[inline]
    pub fn with_color_attachments(
        mut self,
        attachments: impl IntoIterator<Item = ColorAttachment>,
    ) -> Self {
        self.color_attachments.extend(attachments);
        self
    }

    /// Sets the depth attachment for this configuration.
    ///
    /// # Arguments
    ///
    /// * `attachment` - The depth attachment
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ash::vk;
    /// use renderer_rhi::rendering::{DepthAttachment, RenderingConfig};
    ///
    /// # fn example(view: vk::ImageView) {
    /// let config = RenderingConfig::new(800, 600)
    ///     .with_depth_attachment(DepthAttachment::new(view));
    /// # }
    /// ```
    #[inline]
    pub fn with_depth_attachment(mut self, attachment: DepthAttachment) -> Self {
        self.depth_attachment = Some(attachment);
        self
    }

    /// Sets the stencil attachment for this configuration.
    ///
    /// For combined depth-stencil formats, you can use `with_depth_attachment`
    /// with a combined format instead.
    ///
    /// # Arguments
    ///
    /// * `attachment` - The stencil attachment
    #[inline]
    pub fn with_stencil_attachment(mut self, attachment: StencilAttachment) -> Self {
        self.stencil_attachment = Some(attachment);
        self
    }

    /// Sets the render area for this configuration.
    ///
    /// # Arguments
    ///
    /// * `area` - The render area rectangle
    #[inline]
    pub fn with_render_area(mut self, area: vk::Rect2D) -> Self {
        self.render_area = area;
        self
    }

    /// Sets the render area offset.
    ///
    /// # Arguments
    ///
    /// * `x` - X offset in pixels
    /// * `y` - Y offset in pixels
    #[inline]
    pub fn with_offset(mut self, x: i32, y: i32) -> Self {
        self.render_area.offset = vk::Offset2D { x, y };
        self
    }

    /// Sets the layer count for layered rendering.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of layers to render
    #[inline]
    pub fn with_layer_count(mut self, count: u32) -> Self {
        self.layer_count = count;
        self
    }

    /// Sets the view mask for multiview rendering.
    ///
    /// # Arguments
    ///
    /// * `mask` - Bitmask of views to render
    #[inline]
    pub fn with_view_mask(mut self, mask: u32) -> Self {
        self.view_mask = mask;
        self
    }

    /// Sets rendering flags.
    ///
    /// # Arguments
    ///
    /// * `flags` - Rendering flags
    #[inline]
    pub fn with_flags(mut self, flags: vk::RenderingFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Enables suspending rendering (for split rendering).
    #[inline]
    pub fn suspending(mut self) -> Self {
        self.flags |= vk::RenderingFlags::SUSPENDING;
        self
    }

    /// Enables resuming rendering (for split rendering).
    #[inline]
    pub fn resuming(mut self) -> Self {
        self.flags |= vk::RenderingFlags::RESUMING;
        self
    }

    /// Returns the render area extent.
    #[inline]
    pub fn extent(&self) -> vk::Extent2D {
        self.render_area.extent
    }

    /// Returns the width of the render area.
    #[inline]
    pub fn width(&self) -> u32 {
        self.render_area.extent.width
    }

    /// Returns the height of the render area.
    #[inline]
    pub fn height(&self) -> u32 {
        self.render_area.extent.height
    }

    /// Builds the complete `VkRenderingInfo` with proper lifetimes.
    ///
    /// This is a convenience method that handles the lifetime complexity
    /// by taking ownership of the attachment info storage.
    ///
    /// # Returns
    ///
    /// A `RenderingInfoBundle` that contains all necessary data with proper lifetimes.
    pub fn build(&self) -> RenderingInfoBundle {
        RenderingInfoBundle::new(self)
    }
}

/// A bundle containing `VkRenderingInfo` and its backing data.
///
/// This struct ensures that the attachment info arrays outlive the
/// `VkRenderingInfo` that references them.
///
/// # Example
///
/// ```no_run
/// use ash::vk;
/// use renderer_rhi::rendering::{ColorAttachment, RenderingConfig};
/// use renderer_rhi::command::CommandBuffer;
///
/// # fn example(color_view: vk::ImageView, cmd: &CommandBuffer) {
/// let config = RenderingConfig::new(800, 600)
///     .with_color_attachment(ColorAttachment::new(color_view));
///
/// let bundle = config.build();
/// cmd.begin_rendering(bundle.info());
/// // ... draw commands ...
/// cmd.end_rendering();
/// # }
/// ```
pub struct RenderingInfoBundle {
    color_attachments: Vec<vk::RenderingAttachmentInfo<'static>>,
    depth_attachment: Option<vk::RenderingAttachmentInfo<'static>>,
    stencil_attachment: Option<vk::RenderingAttachmentInfo<'static>>,
    render_area: vk::Rect2D,
    layer_count: u32,
    view_mask: u32,
    flags: vk::RenderingFlags,
}

impl RenderingInfoBundle {
    /// Creates a new bundle from a rendering configuration.
    pub fn new(config: &RenderingConfig) -> Self {
        let color_attachments: Vec<vk::RenderingAttachmentInfo> = config
            .color_attachments
            .iter()
            .map(|a| a.to_rendering_attachment_info())
            .collect();

        let depth_attachment = config
            .depth_attachment
            .as_ref()
            .map(|a| a.to_rendering_attachment_info());

        let stencil_attachment = config
            .stencil_attachment
            .as_ref()
            .map(|a| a.to_rendering_attachment_info());

        Self {
            color_attachments,
            depth_attachment,
            stencil_attachment,
            render_area: config.render_area,
            layer_count: config.layer_count,
            view_mask: config.view_mask,
            flags: config.flags,
        }
    }

    /// Returns the `VkRenderingInfo` referencing this bundle's data.
    ///
    /// The returned reference is valid as long as this bundle exists.
    pub fn info(&self) -> vk::RenderingInfo<'_> {
        let mut info = vk::RenderingInfo::default()
            .render_area(self.render_area)
            .layer_count(self.layer_count)
            .view_mask(self.view_mask)
            .flags(self.flags)
            .color_attachments(&self.color_attachments);

        if let Some(ref depth) = self.depth_attachment {
            info = info.depth_attachment(depth);
        }

        if let Some(ref stencil) = self.stencil_attachment {
            info = info.stencil_attachment(stencil);
        }

        info
    }

    /// Returns the color attachments.
    #[inline]
    pub fn color_attachments(&self) -> &[vk::RenderingAttachmentInfo<'static>] {
        &self.color_attachments
    }

    /// Returns the depth attachment if present.
    #[inline]
    pub fn depth_attachment(&self) -> Option<&vk::RenderingAttachmentInfo<'static>> {
        self.depth_attachment.as_ref()
    }

    /// Returns the stencil attachment if present.
    #[inline]
    pub fn stencil_attachment(&self) -> Option<&vk::RenderingAttachmentInfo<'static>> {
        self.stencil_attachment.as_ref()
    }

    /// Returns the render area.
    #[inline]
    pub fn render_area(&self) -> vk::Rect2D {
        self.render_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_attachment_default() {
        let attachment = ColorAttachment::default();
        assert_eq!(attachment.layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(attachment.load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(attachment.store_op, vk::AttachmentStoreOp::STORE);
        // Check clear value is black
        let clear = unsafe { attachment.clear_value.float32 };
        assert_eq!(clear, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_color_attachment_builder() {
        let attachment = ColorAttachment::new(vk::ImageView::null())
            .with_clear_color([1.0, 0.0, 0.0, 1.0])
            .with_load_op(vk::AttachmentLoadOp::LOAD)
            .with_store_op(vk::AttachmentStoreOp::DONT_CARE);

        assert_eq!(attachment.load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(attachment.store_op, vk::AttachmentStoreOp::DONT_CARE);
        let clear = unsafe { attachment.clear_value.float32 };
        assert_eq!(clear, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_color_attachment_load_helper() {
        let attachment = ColorAttachment::new(vk::ImageView::null()).load();
        assert_eq!(attachment.load_op, vk::AttachmentLoadOp::LOAD);
    }

    #[test]
    fn test_color_attachment_dont_store_helper() {
        let attachment = ColorAttachment::new(vk::ImageView::null()).dont_store();
        assert_eq!(attachment.store_op, vk::AttachmentStoreOp::DONT_CARE);
    }

    #[test]
    fn test_depth_attachment_default() {
        let attachment = DepthAttachment::default();
        assert_eq!(
            attachment.layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(attachment.load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(attachment.store_op, vk::AttachmentStoreOp::DONT_CARE);
        assert_eq!(attachment.clear_value.depth, 1.0);
        assert_eq!(attachment.clear_value.stencil, 0);
    }

    #[test]
    fn test_depth_attachment_builder() {
        let attachment = DepthAttachment::new(vk::ImageView::null())
            .with_clear_depth(0.5)
            .with_clear_stencil(128)
            .store();

        assert_eq!(attachment.clear_value.depth, 0.5);
        assert_eq!(attachment.clear_value.stencil, 128);
        assert_eq!(attachment.store_op, vk::AttachmentStoreOp::STORE);
    }

    #[test]
    fn test_depth_attachment_clear_depth_stencil() {
        let attachment =
            DepthAttachment::new(vk::ImageView::null()).with_clear_depth_stencil(0.0, 255);

        assert_eq!(attachment.clear_value.depth, 0.0);
        assert_eq!(attachment.clear_value.stencil, 255);
    }

    #[test]
    fn test_stencil_attachment_default() {
        let attachment = StencilAttachment::default();
        assert_eq!(
            attachment.layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(attachment.load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(attachment.store_op, vk::AttachmentStoreOp::DONT_CARE);
        assert_eq!(attachment.clear_stencil, 0);
    }

    #[test]
    fn test_rendering_config_new() {
        let config = RenderingConfig::new(1920, 1080);
        assert_eq!(config.render_area.extent.width, 1920);
        assert_eq!(config.render_area.extent.height, 1080);
        assert_eq!(config.render_area.offset.x, 0);
        assert_eq!(config.render_area.offset.y, 0);
        assert_eq!(config.layer_count, 1);
        assert_eq!(config.view_mask, 0);
        assert!(config.color_attachments.is_empty());
        assert!(config.depth_attachment.is_none());
    }

    #[test]
    fn test_rendering_config_from_extent() {
        let extent = vk::Extent2D {
            width: 800,
            height: 600,
        };
        let config = RenderingConfig::from_extent(extent);
        assert_eq!(config.width(), 800);
        assert_eq!(config.height(), 600);
    }

    #[test]
    fn test_rendering_config_builder() {
        let color = ColorAttachment::new(vk::ImageView::null());
        let depth = DepthAttachment::new(vk::ImageView::null());

        let config = RenderingConfig::new(800, 600)
            .with_color_attachment(color)
            .with_depth_attachment(depth)
            .with_offset(10, 20)
            .with_layer_count(2)
            .with_view_mask(0b11);

        assert_eq!(config.color_attachments.len(), 1);
        assert!(config.depth_attachment.is_some());
        assert_eq!(config.render_area.offset.x, 10);
        assert_eq!(config.render_area.offset.y, 20);
        assert_eq!(config.layer_count, 2);
        assert_eq!(config.view_mask, 0b11);
    }

    #[test]
    fn test_rendering_config_multiple_color_attachments() {
        let attachments = vec![
            ColorAttachment::new(vk::ImageView::null()),
            ColorAttachment::new(vk::ImageView::null()),
            ColorAttachment::new(vk::ImageView::null()),
        ];

        let config = RenderingConfig::new(800, 600).with_color_attachments(attachments);

        assert_eq!(config.color_attachments.len(), 3);
    }

    #[test]
    fn test_rendering_config_flags() {
        let config = RenderingConfig::new(800, 600).suspending().resuming();

        assert!(config.flags.contains(vk::RenderingFlags::SUSPENDING));
        assert!(config.flags.contains(vk::RenderingFlags::RESUMING));
    }

    #[test]
    fn test_rendering_info_bundle() {
        let color =
            ColorAttachment::new(vk::ImageView::null()).with_clear_color([0.1, 0.2, 0.3, 1.0]);
        let depth = DepthAttachment::new(vk::ImageView::null()).with_clear_depth(1.0);

        let config = RenderingConfig::new(1920, 1080)
            .with_color_attachment(color)
            .with_depth_attachment(depth);

        let bundle = config.build();

        assert_eq!(bundle.color_attachments().len(), 1);
        assert!(bundle.depth_attachment().is_some());
        assert_eq!(bundle.render_area().extent.width, 1920);
        assert_eq!(bundle.render_area().extent.height, 1080);
    }

    #[test]
    fn test_rendering_info_bundle_info_method() {
        let config = RenderingConfig::new(800, 600)
            .with_color_attachment(ColorAttachment::new(vk::ImageView::null()));

        let bundle = config.build();
        let info = bundle.info();

        assert_eq!(info.render_area.extent.width, 800);
        assert_eq!(info.render_area.extent.height, 600);
        assert_eq!(info.layer_count, 1);
    }

    #[test]
    fn test_color_attachment_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ColorAttachment>();
    }

    #[test]
    fn test_depth_attachment_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DepthAttachment>();
    }

    #[test]
    fn test_rendering_config_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RenderingConfig>();
    }

    #[test]
    fn test_rendering_info_bundle_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RenderingInfoBundle>();
    }
}

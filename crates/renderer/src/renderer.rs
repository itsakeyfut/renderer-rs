//! Main renderer orchestration.
//!
//! This module provides the main [`Renderer`] struct that coordinates
//! all Vulkan resources and rendering operations.

use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;

use ash::vk;
use glam::{Mat4, Vec3};
use tracing::{debug, error, info, warn};

use renderer_platform::{InputState, KeyCode, MouseButton, Surface, Window};
use renderer_resources::Model;
use renderer_rhi::buffer::{Buffer, BufferUsage};
use renderer_rhi::descriptor::{
    DescriptorBindingBuilder, DescriptorPool, DescriptorSetLayout, update_descriptor_sets,
};
use renderer_rhi::device::Device;
use renderer_rhi::instance::Instance;
use renderer_rhi::physical_device::select_physical_device;
use renderer_rhi::pipeline::{
    CullMode, FrontFace, GraphicsPipelineBuilder, Pipeline, PipelineLayout,
};
use renderer_rhi::shader::{Shader, ShaderStage};
use renderer_rhi::swapchain::Swapchain;
use renderer_rhi::vertex::Vertex;
use renderer_rhi::{RhiError, RhiResult};
use renderer_scene::camera::{Camera, FpsController};

use crate::MAX_FRAMES_IN_FLIGHT;
use crate::depth_buffer::{DEFAULT_DEPTH_FORMAT, DepthBuffer};
use crate::ubo::{CameraUBO, ObjectUBO};

/// Per-frame synchronization and per-frame resources.
struct FrameData {
    /// Fence to wait for this frame's GPU work to complete.
    in_flight_fence: vk::Fence,
    /// Command pool for this frame.
    command_pool: vk::CommandPool,
    /// Command buffer for this frame.
    command_buffer: vk::CommandBuffer,
    /// Camera uniform buffer for this frame.
    camera_ubo: Buffer,
    /// Object uniform buffer for this frame.
    object_ubo: Buffer,
    /// Descriptor set for this frame.
    descriptor_set: vk::DescriptorSet,
}

/// Per-swapchain-image synchronization resources.
struct ImageSyncData {
    /// Semaphore signaled when swapchain image is available.
    image_available: vk::Semaphore,
    /// Semaphore signaled when rendering is complete.
    render_finished: vk::Semaphore,
}

/// Mesh GPU resources.
struct MeshGpuData {
    /// Vertex buffer for this mesh.
    vertex_buffer: Buffer,
    /// Index buffer for this mesh.
    index_buffer: Buffer,
    /// Number of indices.
    index_count: u32,
}

/// Main renderer that manages all Vulkan resources.
///
/// # Phase 2 Features
///
/// - Camera system with FPS controls (WASD + mouse)
/// - glTF model loading and rendering
/// - Depth buffer for correct 3D rendering
/// - Uniform buffers for MVP matrices
///
/// # Resource Destruction Order
///
/// Vulkan resources must be destroyed in the correct order:
/// 1. Wait for all GPU work to complete
/// 2. Destroy per-frame resources (semaphores, fences, command pools, UBOs)
/// 3. Destroy model resources (vertex/index buffers)
/// 4. Destroy pipeline resources
/// 5. Destroy descriptor resources
/// 6. Destroy depth buffer
/// 7. Destroy swapchain
/// 8. Destroy surface
/// 9. Destroy device
/// 10. Destroy instance
///
/// ManuallyDrop is used to ensure correct destruction order.
pub struct Renderer {
    // Core Vulkan resources (in reverse destruction order)
    /// Vulkan instance (destroyed last).
    instance: ManuallyDrop<Instance>,
    /// Logical device (destroyed after most resources).
    device: Arc<Device>,
    /// Window surface (destroyed after swapchain, before device).
    surface: ManuallyDrop<Surface>,
    /// Swapchain (destroyed after depth buffer).
    swapchain: ManuallyDrop<Swapchain>,
    /// Depth buffer for depth testing.
    depth_buffer: ManuallyDrop<DepthBuffer>,

    // Descriptor resources
    /// Descriptor set layout for camera and object UBOs.
    descriptor_set_layout: ManuallyDrop<DescriptorSetLayout>,
    /// Descriptor pool for allocating descriptor sets.
    descriptor_pool: ManuallyDrop<DescriptorPool>,

    // Pipeline resources
    /// Model graphics pipeline with depth testing.
    model_pipeline: ManuallyDrop<Pipeline>,
    /// Model pipeline layout.
    model_pipeline_layout: ManuallyDrop<PipelineLayout>,

    // Model resources
    /// GPU data for each mesh in the model.
    mesh_gpu_data: Vec<MeshGpuData>,
    /// Model transform (position, rotation, scale).
    model_transform: Mat4,

    // Per-frame resources
    /// Per-frame synchronization and uniform buffer data.
    frame_data: Vec<FrameData>,
    /// Per-swapchain-image semaphores.
    image_sync_data: Vec<ImageSyncData>,
    /// Current frame index (0 to MAX_FRAMES_IN_FLIGHT - 1).
    current_frame: usize,
    /// Current semaphore index (cycles through swapchain images).
    current_semaphore: usize,

    // Camera system
    /// Main camera.
    camera: Camera,
    /// FPS-style camera controller.
    fps_controller: FpsController,

    // State
    /// Flag indicating swapchain needs recreation.
    framebuffer_resized: bool,
    /// Current window width.
    width: u32,
    /// Current window height.
    height: u32,
}

impl Renderer {
    /// Creates a new renderer for the given window.
    ///
    /// This initializes all Vulkan resources and loads a test model.
    ///
    /// # Arguments
    ///
    /// * `window` - The window to render to
    ///
    /// # Errors
    ///
    /// Returns an error if any Vulkan resource creation fails.
    pub fn new(window: &Window) -> RhiResult<Self> {
        let width = window.width();
        let height = window.height();

        info!("Initializing Vulkan renderer ({}x{})", width, height);

        // Create Vulkan instance with validation in debug builds
        let enable_validation = cfg!(debug_assertions);
        let instance = Instance::new(enable_validation)?;

        // Create surface
        let surface = window
            .create_surface(instance.entry(), instance.handle())
            .map_err(|e| RhiError::SurfaceError(e.to_string()))?;

        // Select physical device
        let physical_device_info =
            select_physical_device(instance.handle(), surface.handle(), surface.loader())?;

        // Create logical device
        let device = Device::new(&instance, &physical_device_info)?;

        // Create swapchain
        let swapchain = Swapchain::new(&instance, device.clone(), surface.handle(), width, height)?;

        // Create depth buffer
        let depth_buffer = DepthBuffer::with_default_format(device.clone(), width, height)?;

        // Create descriptor set layout for UBOs (binding 0: CameraUBO, binding 1: ObjectUBO)
        let camera_binding = DescriptorBindingBuilder::uniform_buffer(
            0,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        );
        let object_binding =
            DescriptorBindingBuilder::uniform_buffer(1, vk::ShaderStageFlags::VERTEX);
        let descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), &[camera_binding, object_binding])?;

        // Create descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count((MAX_FRAMES_IN_FLIGHT * 2) as u32)]; // 2 UBOs per frame
        let descriptor_pool =
            DescriptorPool::new(device.clone(), MAX_FRAMES_IN_FLIGHT as u32, &pool_sizes)?;

        // Create per-frame data
        let frame_data = Self::create_frame_data(
            &device,
            &descriptor_pool,
            &descriptor_set_layout,
            MAX_FRAMES_IN_FLIGHT,
        )?;

        // Create per-swapchain-image semaphores
        let image_sync_data =
            Self::create_image_sync_data(&device, swapchain.image_count() as usize)?;

        // Create model pipeline
        let (model_pipeline, model_pipeline_layout) = Self::create_model_pipeline(
            device.clone(),
            &descriptor_set_layout,
            swapchain.format(),
        )?;

        // Load model
        let model_path = Path::new("assets/models/a_contortionist_dancer/scene.gltf");
        let (mesh_gpu_data, model_center, model_size) =
            Self::load_model(device.clone(), model_path)?;

        // Calculate model transform to center the model at origin
        // and scale it to fit in a reasonable viewing area
        let scale_factor = 2.0 / model_size.max_element().max(0.001);
        let model_transform =
            Mat4::from_scale(Vec3::splat(scale_factor)) * Mat4::from_translation(-model_center);

        info!(
            "Model: center={:?}, size={:?}, scale={}",
            model_center, model_size, scale_factor
        );

        // Initialize camera - position based on model scale
        let mut camera = Camera::new();
        camera.position = Vec3::new(0.0, 0.0, 5.0);
        camera.set_perspective(
            45.0_f32.to_radians(),
            width as f32 / height as f32,
            0.01,
            1000.0,
        );

        let fps_controller = FpsController::with_settings(3.0, 0.002);

        info!(
            "Renderer initialized: {} swapchain images, {} frames in flight, {} meshes loaded",
            swapchain.image_count(),
            MAX_FRAMES_IN_FLIGHT,
            mesh_gpu_data.len()
        );

        Ok(Self {
            instance: ManuallyDrop::new(instance),
            device,
            surface: ManuallyDrop::new(surface),
            swapchain: ManuallyDrop::new(swapchain),
            depth_buffer: ManuallyDrop::new(depth_buffer),
            descriptor_set_layout: ManuallyDrop::new(descriptor_set_layout),
            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            model_pipeline: ManuallyDrop::new(model_pipeline),
            model_pipeline_layout: ManuallyDrop::new(model_pipeline_layout),
            mesh_gpu_data,
            model_transform,
            frame_data,
            image_sync_data,
            current_frame: 0,
            current_semaphore: 0,
            camera,
            fps_controller,
            framebuffer_resized: false,
            width,
            height,
        })
    }

    /// Creates per-frame synchronization and uniform buffer resources.
    fn create_frame_data(
        device: &Arc<Device>,
        descriptor_pool: &DescriptorPool,
        descriptor_set_layout: &DescriptorSetLayout,
        count: usize,
    ) -> RhiResult<Vec<FrameData>> {
        let mut frames = Vec::with_capacity(count);

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let graphics_family = device.queue_families().graphics_family.unwrap();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // Allocate all descriptor sets at once
        let layouts: Vec<_> = (0..count).map(|_| descriptor_set_layout.handle()).collect();
        let descriptor_sets = descriptor_pool.allocate(&layouts)?;

        for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let in_flight_fence = unsafe { device.handle().create_fence(&fence_info, None)? };
            let command_pool = unsafe { device.handle().create_command_pool(&pool_info, None)? };

            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffers = unsafe { device.handle().allocate_command_buffers(&alloc_info)? };

            // Create uniform buffers
            let camera_ubo =
                Buffer::new(device.clone(), BufferUsage::Uniform, CameraUBO::SIZE as u64)?;
            let object_ubo =
                Buffer::new(device.clone(), BufferUsage::Uniform, ObjectUBO::SIZE as u64)?;

            // Update descriptor set to point to the UBOs
            let camera_buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(camera_ubo.handle())
                .offset(0)
                .range(CameraUBO::SIZE as u64);
            let object_buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(object_ubo.handle())
                .offset(0)
                .range(ObjectUBO::SIZE as u64);

            let camera_buffer_infos = [camera_buffer_info];
            let object_buffer_infos = [object_buffer_info];

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&camera_buffer_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&object_buffer_infos),
            ];
            update_descriptor_sets(device, &writes);

            debug!("Created frame data for frame {}", i);

            frames.push(FrameData {
                in_flight_fence,
                command_pool,
                command_buffer: command_buffers[0],
                camera_ubo,
                object_ubo,
                descriptor_set,
            });
        }

        Ok(frames)
    }

    /// Creates per-swapchain-image semaphores.
    fn create_image_sync_data(device: &Arc<Device>, count: usize) -> RhiResult<Vec<ImageSyncData>> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let mut sync_data = Vec::with_capacity(count);

        for i in 0..count {
            let image_available =
                unsafe { device.handle().create_semaphore(&semaphore_info, None)? };
            let render_finished =
                unsafe { device.handle().create_semaphore(&semaphore_info, None)? };

            debug!("Created image sync data for image {}", i);

            sync_data.push(ImageSyncData {
                image_available,
                render_finished,
            });
        }

        Ok(sync_data)
    }

    /// Creates the model rendering pipeline with depth testing.
    fn create_model_pipeline(
        device: Arc<Device>,
        descriptor_set_layout: &DescriptorSetLayout,
        swapchain_format: vk::Format,
    ) -> RhiResult<(Pipeline, PipelineLayout)> {
        // Load shaders
        let vertex_shader = Shader::from_spirv_file(
            device.clone(),
            Path::new("shaders/spirv/model.vert.spv"),
            ShaderStage::Vertex,
            "main",
        )?;

        let fragment_shader = Shader::from_spirv_file(
            device.clone(),
            Path::new("shaders/spirv/model.frag.spv"),
            ShaderStage::Fragment,
            "main",
        )?;

        // Create pipeline layout with descriptor set
        let pipeline_layout =
            PipelineLayout::new(device.clone(), &[descriptor_set_layout.handle()], &[])?;

        // Create graphics pipeline with depth testing enabled
        let pipeline = GraphicsPipelineBuilder::new()
            .vertex_shader(&vertex_shader)
            .fragment_shader(&fragment_shader)
            .vertex_binding(Vertex::binding_description())
            .vertex_attributes(&Vertex::attribute_descriptions())
            .color_attachment_format(swapchain_format)
            .depth_attachment_format(DEFAULT_DEPTH_FORMAT)
            .cull_mode(CullMode::None)
            .front_face(FrontFace::CounterClockwise)
            .depth_test_enable(true)
            .depth_write_enable(true)
            .build(device.clone(), &pipeline_layout)?;

        info!("Model pipeline created with depth testing");

        Ok((pipeline, pipeline_layout))
    }

    /// Loads a glTF model and creates GPU buffers for its meshes.
    /// Returns the mesh GPU data, model center, and model size.
    fn load_model(device: Arc<Device>, path: &Path) -> RhiResult<(Vec<MeshGpuData>, Vec3, Vec3)> {
        info!("Loading model: {}", path.display());

        let model = Model::load(path).map_err(|e| RhiError::InvalidHandle(e.to_string()))?;

        let center = model.center();
        let size = model.size();

        info!(
            "Model loaded: {} meshes, {} total vertices, {} total triangles",
            model.meshes.len(),
            model.total_vertex_count(),
            model.total_triangle_count()
        );
        info!(
            "Model bounds: min={:?}, max={:?}, center={:?}, size={:?}",
            model.aabb_min, model.aabb_max, center, size
        );

        let mut mesh_gpu_data = Vec::with_capacity(model.meshes.len());

        for (i, mesh) in model.meshes.iter().enumerate() {
            // Convert mesh data to Vertex format
            let vertices: Vec<Vertex> = (0..mesh.positions.len())
                .map(|j| {
                    Vertex::new(
                        mesh.positions[j],
                        mesh.normals[j],
                        mesh.tex_coords[j],
                        mesh.tangents[j],
                    )
                })
                .collect();

            // Create vertex buffer
            let vertex_buffer = Buffer::new_with_data(
                device.clone(),
                BufferUsage::Vertex,
                bytemuck::cast_slice(&vertices),
            )?;

            // Create index buffer
            let index_buffer = Buffer::new_with_data(
                device.clone(),
                BufferUsage::Index,
                bytemuck::cast_slice(&mesh.indices),
            )?;

            debug!(
                "Mesh {}: {} vertices, {} indices",
                i,
                vertices.len(),
                mesh.indices.len()
            );

            mesh_gpu_data.push(MeshGpuData {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
            });
        }

        Ok((mesh_gpu_data, center, size))
    }

    /// Notifies the renderer that the window has been resized.
    ///
    /// The actual swapchain recreation will happen on the next frame.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            debug!("Ignoring resize to zero dimensions");
            return;
        }

        if width != self.width || height != self.height {
            debug!(
                "Resize triggered: {}x{} -> {}x{}",
                self.width, self.height, width, height
            );
            self.width = width;
            self.height = height;
            self.framebuffer_resized = true;

            // Update camera aspect ratio
            self.camera
                .set_aspect(self.width as f32 / self.height as f32);
        }
    }

    /// Waits for all GPU work to complete including presentation.
    fn wait_for_device_idle(&self) -> RhiResult<()> {
        // Wait for all GPU work to complete, including presentation
        self.device.wait_idle()?;
        Ok(())
    }

    /// Recreates the swapchain and depth buffer for the current window size.
    fn recreate_swapchain(&mut self) -> RhiResult<()> {
        self.wait_for_device_idle()?;

        self.swapchain.recreate(
            &self.instance,
            self.surface.handle(),
            self.width,
            self.height,
        )?;

        // Recreate depth buffer with new dimensions
        let new_depth_buffer =
            DepthBuffer::with_default_format(self.device.clone(), self.width, self.height)?;

        // Replace old depth buffer (ManuallyDrop handles cleanup)
        unsafe {
            ManuallyDrop::drop(&mut self.depth_buffer);
        }
        self.depth_buffer = ManuallyDrop::new(new_depth_buffer);

        // Recreate image sync data (semaphores per swapchain image)
        self.recreate_image_sync_data()?;

        self.framebuffer_resized = false;
        Ok(())
    }

    /// Recreates all semaphores for swapchain image synchronization.
    fn recreate_image_sync_data(&mut self) -> RhiResult<()> {
        // Destroy old semaphores
        for sync in &self.image_sync_data {
            unsafe {
                self.device
                    .handle()
                    .destroy_semaphore(sync.image_available, None);
                self.device
                    .handle()
                    .destroy_semaphore(sync.render_finished, None);
            }
        }

        // Create new semaphores for the new swapchain image count
        self.image_sync_data =
            Self::create_image_sync_data(&self.device, self.swapchain.image_count() as usize)?;
        self.current_semaphore = 0;

        debug!(
            "Recreated {} image sync data entries",
            self.image_sync_data.len()
        );
        Ok(())
    }

    /// Updates the camera based on input state.
    ///
    /// # Arguments
    ///
    /// * `input` - Current input state
    /// * `delta_time` - Time elapsed since last frame in seconds
    pub fn update(&mut self, input: &InputState, delta_time: f32) {
        let (dx, dy) = input.mouse_delta();
        let is_pressed = input.is_mouse_pressed(MouseButton::Right);
        let just_pressed = input.is_mouse_just_pressed(MouseButton::Right);

        // Process mouse movement when right mouse button is held
        // Skip the first frame when button is pressed to avoid jump from accumulated delta
        if is_pressed && !just_pressed {
            // Clamp delta to prevent extreme camera movements
            let max_delta = 100.0;
            let dx = dx.clamp(-max_delta, max_delta);
            let dy = dy.clamp(-max_delta, max_delta);

            self.fps_controller.process_mouse_movement(dx, dy);
        }

        // Process keyboard movement
        let forward = if input.is_key_pressed(KeyCode::KeyW) {
            1.0
        } else if input.is_key_pressed(KeyCode::KeyS) {
            -1.0
        } else {
            0.0
        };

        let right = if input.is_key_pressed(KeyCode::KeyD) {
            1.0
        } else if input.is_key_pressed(KeyCode::KeyA) {
            -1.0
        } else {
            0.0
        };

        let up = if input.is_key_pressed(KeyCode::KeyQ) {
            1.0
        } else if input.is_key_pressed(KeyCode::KeyE) {
            -1.0
        } else {
            0.0
        };

        self.fps_controller.set_movement_input(forward, right, up);
        self.fps_controller
            .update_camera(&mut self.camera, delta_time);
    }

    /// Renders a frame.
    ///
    /// # Errors
    ///
    /// Returns an error if any Vulkan operation fails.
    pub fn render_frame(&mut self) -> RhiResult<()> {
        // Check if we need to recreate swapchain
        if self.framebuffer_resized {
            debug!("Resize requested, recreating swapchain before acquire");
            self.recreate_swapchain()?;
        }

        let frame = &self.frame_data[self.current_frame];

        // Wait for this frame's previous work to complete
        unsafe {
            self.device
                .handle()
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
        }

        // Get semaphore for acquiring (use cycling index to avoid reusing in-flight semaphore)
        let acquire_semaphore = self.image_sync_data[self.current_semaphore].image_available;

        // Acquire next swapchain image
        let (image_index, _suboptimal) = match self.swapchain.acquire_next_image(acquire_semaphore)
        {
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                debug!("Swapchain out of date, recreating");
                self.recreate_swapchain()?;
                return Ok(());
            }
            Err(e) => return Err(RhiError::VulkanError(e)),
        };

        // Use the image index to get the correct sync data for this image
        let image_sync = &self.image_sync_data[image_index as usize];

        // Reset fence only after we're sure we'll submit work
        unsafe {
            self.device
                .handle()
                .reset_fences(&[frame.in_flight_fence])?;
        }

        // Update uniform buffers
        self.update_uniform_buffers()?;

        // Record commands
        self.record_commands(image_index)?;

        // Submit - wait on acquire semaphore, signal render finished semaphore for this image
        let wait_semaphores = [acquire_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [image_sync.render_finished];
        let command_buffers = [frame.command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device.handle().queue_submit(
                self.device.graphics_queue(),
                &[submit_info],
                frame.in_flight_fence,
            )?;
        }

        // Present - wait on render finished semaphore for this image
        let present_result = self.swapchain.present(
            self.device.present_queue(),
            image_index,
            image_sync.render_finished,
        );

        // Advance semaphore index (cycle through all swapchain images)
        self.current_semaphore = (self.current_semaphore + 1) % self.image_sync_data.len();

        let should_recreate = match present_result {
            Ok(suboptimal) => {
                if suboptimal {
                    debug!("Present returned suboptimal=true");
                }
                suboptimal
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                debug!("Present returned ERROR_OUT_OF_DATE_KHR");
                true
            }
            Err(vk::Result::SUBOPTIMAL_KHR) => {
                debug!("Present returned SUBOPTIMAL_KHR");
                true
            }
            Err(e) => return Err(RhiError::VulkanError(e)),
        };

        // Advance to next frame
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        if should_recreate {
            debug!("Swapchain needs recreation, recreating");
            self.recreate_swapchain()?;
        }

        Ok(())
    }

    /// Updates uniform buffers with current camera and object data.
    fn update_uniform_buffers(&self) -> RhiResult<()> {
        let frame = &self.frame_data[self.current_frame];

        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix();

        // Update camera UBO
        let camera_data = CameraUBO::new(view, proj, self.camera.position);
        frame.camera_ubo.upload(bytemuck::bytes_of(&camera_data))?;

        // Update object UBO
        let object_data = ObjectUBO::new(self.model_transform);
        frame.object_ubo.upload(bytemuck::bytes_of(&object_data))?;

        Ok(())
    }

    /// Records rendering commands for a frame.
    fn record_commands(&self, image_index: u32) -> RhiResult<()> {
        let frame = &self.frame_data[self.current_frame];
        let cmd = frame.command_buffer;

        // Reset and begin command buffer
        unsafe {
            self.device
                .handle()
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::empty());
            self.device
                .handle()
                .begin_command_buffer(cmd, &begin_info)?;
        }

        // Transition color image to color attachment optimal
        let color_image = self.swapchain.image(image_index as usize);
        self.cmd_transition_image_layout(
            cmd,
            color_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );

        // Transition depth image to depth attachment optimal
        self.cmd_transition_image_layout(
            cmd,
            self.depth_buffer.image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::ImageAspectFlags::DEPTH,
        );

        // Begin rendering with color and depth attachments
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.image_view(image_index as usize))
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.15, 1.0], // Dark blue-gray
                },
            });

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_buffer.image_view())
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent(),
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);

        unsafe {
            self.device
                .handle()
                .cmd_begin_rendering(cmd, &rendering_info);

            // Set viewport and scissor
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain.width() as f32,
                height: self.swapchain.height() as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.handle().cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent(),
            };
            self.device.handle().cmd_set_scissor(cmd, 0, &[scissor]);

            // Bind model pipeline
            self.device.handle().cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.model_pipeline.handle(),
            );

            // Bind descriptor set
            self.device.handle().cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.model_pipeline_layout.handle(),
                0,
                &[frame.descriptor_set],
                &[],
            );

            // Draw all meshes
            for mesh in &self.mesh_gpu_data {
                // Bind vertex buffer
                self.device.handle().cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[mesh.vertex_buffer.handle()],
                    &[0],
                );

                // Bind index buffer
                self.device.handle().cmd_bind_index_buffer(
                    cmd,
                    mesh.index_buffer.handle(),
                    0,
                    vk::IndexType::UINT32,
                );

                // Draw indexed
                self.device
                    .handle()
                    .cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
            }

            self.device.handle().cmd_end_rendering(cmd);
        }

        // Transition color image to present
        self.cmd_transition_image_layout(
            cmd,
            color_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageAspectFlags::COLOR,
        );

        // End command buffer
        unsafe {
            self.device.handle().end_command_buffer(cmd)?;
        }

        Ok(())
    }

    /// Records an image layout transition.
    fn cmd_transition_image_layout(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        let (src_stage, src_access, dst_stage, dst_access) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL) => (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::empty(),
            ),
            _ => {
                warn!(
                    "Unhandled layout transition: {:?} -> {:?}",
                    old_layout, new_layout
                );
                (
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                )
            }
        };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);

        unsafe {
            self.device.handle().cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Returns the current swapchain extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent()
    }

    /// Returns the swapchain format.
    pub fn format(&self) -> vk::Format {
        self.swapchain.format()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Wait for all GPU work to complete before destroying resources
        if let Err(e) = self.device.wait_idle() {
            error!(
                "Failed to wait for device idle during renderer drop: {:?}",
                e
            );
        }

        // Destroy per-swapchain-image semaphores
        for sync in &self.image_sync_data {
            unsafe {
                self.device
                    .handle()
                    .destroy_semaphore(sync.image_available, None);
                self.device
                    .handle()
                    .destroy_semaphore(sync.render_finished, None);
            }
        }

        // Destroy per-frame resources
        for frame in &self.frame_data {
            unsafe {
                self.device
                    .handle()
                    .destroy_fence(frame.in_flight_fence, None);
                self.device
                    .handle()
                    .destroy_command_pool(frame.command_pool, None);
            }
            // UBOs are dropped automatically by Buffer's Drop impl
        }

        // Drop mesh GPU data
        self.mesh_gpu_data.clear();

        // Manually drop resources in correct order
        unsafe {
            ManuallyDrop::drop(&mut self.model_pipeline);
            ManuallyDrop::drop(&mut self.model_pipeline_layout);
            ManuallyDrop::drop(&mut self.descriptor_pool);
            ManuallyDrop::drop(&mut self.descriptor_set_layout);
            ManuallyDrop::drop(&mut self.depth_buffer);
            ManuallyDrop::drop(&mut self.swapchain);
            ManuallyDrop::drop(&mut self.surface);
            ManuallyDrop::drop(&mut self.instance);
        }

        info!("Renderer destroyed");
    }
}

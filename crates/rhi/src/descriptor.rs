//! Descriptor set management for shader resource binding.
//!
//! This module provides abstractions for Vulkan descriptor management:
//! - [`DescriptorSetLayout`] defines the layout of shader bindings
//! - [`DescriptorPool`] manages allocation of descriptor sets
//! - Helper functions for updating descriptor sets
//!
//! # Overview
//!
//! Descriptors in Vulkan are used to connect shader uniform buffers, textures,
//! and other resources to shaders. This module provides a safe Rust API for:
//! 1. Creating descriptor set layouts
//! 2. Allocating descriptor sets from pools
//! 3. Updating descriptor sets with resource bindings
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use ash::vk;
//! use renderer_rhi::device::Device;
//! use renderer_rhi::descriptor::{DescriptorSetLayout, DescriptorPool};
//!
//! # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
//! // Create layout with a uniform buffer binding
//! let binding = vk::DescriptorSetLayoutBinding::default()
//!     .binding(0)
//!     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
//!     .descriptor_count(1)
//!     .stage_flags(vk::ShaderStageFlags::VERTEX);
//!
//! let layout = DescriptorSetLayout::new(device.clone(), &[binding])?;
//!
//! // Create pool
//! let pool_size = vk::DescriptorPoolSize::default()
//!     .ty(vk::DescriptorType::UNIFORM_BUFFER)
//!     .descriptor_count(10);
//!
//! let pool = DescriptorPool::new(device.clone(), 10, &[pool_size])?;
//!
//! // Allocate descriptor sets
//! let sets = pool.allocate(&[layout.handle()])?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;
use tracing::debug;

use crate::device::Device;
use crate::error::RhiResult;

/// Descriptor set layout wrapper.
///
/// A descriptor set layout defines the structure of resources that can be
/// bound to a shader. It specifies the binding points, descriptor types,
/// and shader stages that can access each resource.
///
/// # Thread Safety
///
/// The layout itself is immutable after creation. It can be shared between
/// threads when wrapped in `Arc`.
pub struct DescriptorSetLayout {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan descriptor set layout handle.
    layout: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    /// Creates a new descriptor set layout.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `bindings` - Array of binding descriptions
    ///
    /// # Errors
    ///
    /// Returns an error if layout creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use ash::vk;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::descriptor::DescriptorSetLayout;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let binding = vk::DescriptorSetLayoutBinding::default()
    ///     .binding(0)
    ///     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
    ///     .descriptor_count(1)
    ///     .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
    ///
    /// let layout = DescriptorSetLayout::new(device, &[binding])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        device: Arc<Device>,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> RhiResult<Self> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

        let layout = unsafe {
            device
                .handle()
                .create_descriptor_set_layout(&create_info, None)?
        };

        debug!(
            "Created descriptor set layout with {} binding(s)",
            bindings.len()
        );

        Ok(Self { device, layout })
    }

    /// Returns the Vulkan descriptor set layout handle.
    #[inline]
    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle()
                .destroy_descriptor_set_layout(self.layout, None);
        }
        debug!("Destroyed descriptor set layout");
    }
}

/// Descriptor pool for allocating descriptor sets.
///
/// A descriptor pool manages a pool of descriptors from which descriptor
/// sets can be allocated. The pool must be created with enough capacity
/// for all descriptor types and sets that will be allocated from it.
///
/// # Thread Safety
///
/// Descriptor pool operations are not thread-safe. Synchronize access
/// externally when sharing between threads.
pub struct DescriptorPool {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan descriptor pool handle.
    pool: vk::DescriptorPool,
    /// Maximum number of sets that can be allocated.
    max_sets: u32,
}

impl DescriptorPool {
    /// Creates a new descriptor pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `max_sets` - Maximum number of descriptor sets that can be allocated
    /// * `pool_sizes` - Array of pool sizes for each descriptor type
    ///
    /// # Errors
    ///
    /// Returns an error if pool creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use ash::vk;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::descriptor::DescriptorPool;
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let pool_sizes = [
    ///     vk::DescriptorPoolSize::default()
    ///         .ty(vk::DescriptorType::UNIFORM_BUFFER)
    ///         .descriptor_count(100),
    ///     vk::DescriptorPoolSize::default()
    ///         .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    ///         .descriptor_count(100),
    /// ];
    ///
    /// let pool = DescriptorPool::new(device, 100, &pool_sizes)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        device: Arc<Device>,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> RhiResult<Self> {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let pool = unsafe { device.handle().create_descriptor_pool(&create_info, None)? };

        debug!(
            "Created descriptor pool: max_sets={}, pool_sizes={}",
            max_sets,
            pool_sizes.len()
        );

        Ok(Self {
            device,
            pool,
            max_sets,
        })
    }

    /// Allocates descriptor sets from the pool.
    ///
    /// # Arguments
    ///
    /// * `layouts` - Array of descriptor set layouts for each set to allocate
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails (e.g., pool exhausted).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use ash::vk;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::descriptor::{DescriptorSetLayout, DescriptorPool};
    ///
    /// # fn example(device: Arc<Device>, layout: &DescriptorSetLayout, pool: &DescriptorPool) -> Result<(), renderer_rhi::RhiError> {
    /// // Allocate 3 descriptor sets with the same layout
    /// let layouts = [layout.handle(); 3];
    /// let sets = pool.allocate(&layouts)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn allocate(
        &self,
        layouts: &[vk::DescriptorSetLayout],
    ) -> RhiResult<Vec<vk::DescriptorSet>> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        let sets = unsafe { self.device.handle().allocate_descriptor_sets(&alloc_info)? };

        debug!("Allocated {} descriptor set(s)", sets.len());

        Ok(sets)
    }

    /// Frees descriptor sets back to the pool.
    ///
    /// # Arguments
    ///
    /// * `sets` - The descriptor sets to free
    ///
    /// # Errors
    ///
    /// Returns an error if freeing fails.
    ///
    /// # Safety
    ///
    /// The caller must ensure the descriptor sets are not in use by the GPU.
    pub fn free(&self, sets: &[vk::DescriptorSet]) -> RhiResult<()> {
        unsafe {
            self.device.handle().free_descriptor_sets(self.pool, sets)?;
        }

        debug!("Freed {} descriptor set(s)", sets.len());

        Ok(())
    }

    /// Resets the descriptor pool, returning all allocated sets to the pool.
    ///
    /// This is more efficient than freeing individual sets when you want
    /// to reclaim all allocations at once.
    ///
    /// # Errors
    ///
    /// Returns an error if the reset fails.
    ///
    /// # Safety
    ///
    /// The caller must ensure no descriptor sets from this pool are in use
    /// by the GPU when this function is called.
    pub fn reset(&self) -> RhiResult<()> {
        unsafe {
            self.device
                .handle()
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())?;
        }

        debug!("Reset descriptor pool");

        Ok(())
    }

    /// Returns the Vulkan descriptor pool handle.
    #[inline]
    pub fn handle(&self) -> vk::DescriptorPool {
        self.pool
    }

    /// Returns the maximum number of sets that can be allocated from this pool.
    #[inline]
    pub fn max_sets(&self) -> u32 {
        self.max_sets
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle()
                .destroy_descriptor_pool(self.pool, None);
        }
        debug!("Destroyed descriptor pool");
    }
}

/// Updates descriptor sets with resource bindings.
///
/// This function writes resource bindings to one or more descriptor sets.
/// It's the primary way to connect buffers, images, and samplers to shaders.
///
/// # Arguments
///
/// * `device` - The logical device
/// * `writes` - Array of write descriptor set operations
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use ash::vk;
/// use renderer_rhi::device::Device;
/// use renderer_rhi::descriptor::update_descriptor_sets;
///
/// # fn example(device: &Device, descriptor_set: vk::DescriptorSet, buffer: vk::Buffer) {
/// let buffer_info = vk::DescriptorBufferInfo::default()
///     .buffer(buffer)
///     .offset(0)
///     .range(vk::WHOLE_SIZE);
///
/// let buffer_infos = [buffer_info];
///
/// let write = vk::WriteDescriptorSet::default()
///     .dst_set(descriptor_set)
///     .dst_binding(0)
///     .dst_array_element(0)
///     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
///     .buffer_info(&buffer_infos);
///
/// update_descriptor_sets(device, &[write]);
/// # }
/// ```
pub fn update_descriptor_sets(device: &Device, writes: &[vk::WriteDescriptorSet]) {
    if writes.is_empty() {
        return;
    }

    unsafe {
        device.handle().update_descriptor_sets(writes, &[]);
    }

    debug!("Updated {} descriptor set(s)", writes.len());
}

/// Creates a buffer info for descriptor set updates.
///
/// Convenience function for creating a `DescriptorBufferInfo` structure.
///
/// # Arguments
///
/// * `buffer` - The buffer handle
/// * `offset` - Offset into the buffer in bytes
/// * `range` - Size of the buffer range to bind, or `vk::WHOLE_SIZE` for the entire buffer
#[inline]
pub fn buffer_info(
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    range: vk::DeviceSize,
) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(offset)
        .range(range)
}

/// Creates an image info for descriptor set updates.
///
/// Convenience function for creating a `DescriptorImageInfo` structure.
///
/// # Arguments
///
/// * `sampler` - The sampler handle
/// * `image_view` - The image view handle
/// * `image_layout` - The layout of the image
#[inline]
pub fn image_info(
    sampler: vk::Sampler,
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(image_view)
        .image_layout(image_layout)
}

/// Builder for creating descriptor set layout bindings.
///
/// Provides a convenient way to construct descriptor set layout bindings
/// with a fluent API.
///
/// # Example
///
/// ```no_run
/// use ash::vk;
/// use renderer_rhi::descriptor::DescriptorBindingBuilder;
///
/// let bindings = [
///     DescriptorBindingBuilder::uniform_buffer(0, vk::ShaderStageFlags::VERTEX),
///     DescriptorBindingBuilder::combined_image_sampler(1, vk::ShaderStageFlags::FRAGMENT),
/// ];
/// ```
pub struct DescriptorBindingBuilder;

impl DescriptorBindingBuilder {
    /// Creates a uniform buffer binding.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding index
    /// * `stage_flags` - The shader stages that can access this binding
    #[inline]
    pub fn uniform_buffer(
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(stage_flags)
    }

    /// Creates a storage buffer binding.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding index
    /// * `stage_flags` - The shader stages that can access this binding
    #[inline]
    pub fn storage_buffer(
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(stage_flags)
    }

    /// Creates a combined image sampler binding.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding index
    /// * `stage_flags` - The shader stages that can access this binding
    #[inline]
    pub fn combined_image_sampler(
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(stage_flags)
    }

    /// Creates a sampled image binding.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding index
    /// * `stage_flags` - The shader stages that can access this binding
    #[inline]
    pub fn sampled_image(
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(stage_flags)
    }

    /// Creates a sampler binding.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding index
    /// * `stage_flags` - The shader stages that can access this binding
    #[inline]
    pub fn sampler(
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(stage_flags)
    }

    /// Creates a storage image binding.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding index
    /// * `stage_flags` - The shader stages that can access this binding
    #[inline]
    pub fn storage_image(
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(stage_flags)
    }

    /// Creates a dynamic uniform buffer binding.
    ///
    /// Dynamic uniform buffers allow specifying the buffer offset at bind time.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding index
    /// * `stage_flags` - The shader stages that can access this binding
    #[inline]
    pub fn uniform_buffer_dynamic(
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(stage_flags)
    }

    /// Creates a dynamic storage buffer binding.
    ///
    /// Dynamic storage buffers allow specifying the buffer offset at bind time.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding index
    /// * `stage_flags` - The shader stages that can access this binding
    #[inline]
    pub fn storage_buffer_dynamic(
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(stage_flags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor_binding_builder_uniform_buffer() {
        let binding = DescriptorBindingBuilder::uniform_buffer(0, vk::ShaderStageFlags::VERTEX);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(binding.descriptor_count, 1);
        assert_eq!(binding.stage_flags, vk::ShaderStageFlags::VERTEX);
    }

    #[test]
    fn test_descriptor_binding_builder_storage_buffer() {
        let binding = DescriptorBindingBuilder::storage_buffer(1, vk::ShaderStageFlags::COMPUTE);
        assert_eq!(binding.binding, 1);
        assert_eq!(binding.descriptor_type, vk::DescriptorType::STORAGE_BUFFER);
        assert_eq!(binding.descriptor_count, 1);
        assert_eq!(binding.stage_flags, vk::ShaderStageFlags::COMPUTE);
    }

    #[test]
    fn test_descriptor_binding_builder_combined_image_sampler() {
        let binding =
            DescriptorBindingBuilder::combined_image_sampler(2, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(binding.binding, 2);
        assert_eq!(
            binding.descriptor_type,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        );
        assert_eq!(binding.descriptor_count, 1);
        assert_eq!(binding.stage_flags, vk::ShaderStageFlags::FRAGMENT);
    }

    #[test]
    fn test_descriptor_binding_builder_dynamic_uniform_buffer() {
        let binding = DescriptorBindingBuilder::uniform_buffer_dynamic(
            0,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        );
        assert_eq!(binding.binding, 0);
        assert_eq!(
            binding.descriptor_type,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
        );
        assert_eq!(binding.descriptor_count, 1);
        assert!(binding.stage_flags.contains(vk::ShaderStageFlags::VERTEX));
        assert!(binding.stage_flags.contains(vk::ShaderStageFlags::FRAGMENT));
    }

    #[test]
    fn test_buffer_info_helper() {
        let info = buffer_info(vk::Buffer::null(), 64, 128);
        assert_eq!(info.buffer, vk::Buffer::null());
        assert_eq!(info.offset, 64);
        assert_eq!(info.range, 128);
    }

    #[test]
    fn test_image_info_helper() {
        let info = image_info(
            vk::Sampler::null(),
            vk::ImageView::null(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        assert_eq!(info.sampler, vk::Sampler::null());
        assert_eq!(info.image_view, vk::ImageView::null());
        assert_eq!(info.image_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
}

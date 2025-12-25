//! GPU buffer management.
//!
//! This module handles vertex, index, uniform, and staging buffers.
//! It uses gpu-allocator for memory management and provides safe abstractions
//! for buffer creation and data transfer.
//!
//! # Overview
//!
//! - [`BufferUsage`] defines how a buffer will be used (vertex, index, uniform, etc.)
//! - [`Buffer`] wraps VkBuffer with gpu-allocator managed memory
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_rhi::device::Device;
//! use renderer_rhi::buffer::{Buffer, BufferUsage};
//!
//! # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
//! // Create a vertex buffer with initial data
//! let vertices: [f32; 6] = [0.0, 0.5, -0.5, -0.5, 0.5, -0.5];
//! let vertex_buffer = Buffer::new_with_data(
//!     device,
//!     BufferUsage::Vertex,
//!     bytemuck::cast_slice(&vertices),
//! )?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use tracing::debug;

use crate::device::Device;
use crate::error::{RhiError, RhiResult};

/// Buffer usage type.
///
/// Defines the intended use of the buffer, which affects
/// Vulkan usage flags and memory allocation strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferUsage {
    /// Vertex buffer - stores vertex data
    Vertex,
    /// Index buffer - stores index data
    Index,
    /// Uniform buffer - stores shader uniform data
    Uniform,
    /// Storage buffer - general-purpose GPU storage
    Storage,
    /// Staging buffer - CPU-writable for data upload
    Staging,
}

impl BufferUsage {
    /// Converts to Vulkan buffer usage flags.
    pub fn to_vk_usage(self) -> vk::BufferUsageFlags {
        match self {
            BufferUsage::Vertex => {
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Index => {
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Uniform => {
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Storage => {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Staging => vk::BufferUsageFlags::TRANSFER_SRC,
        }
    }

    /// Returns the preferred memory location for this buffer type.
    pub fn memory_location(self) -> MemoryLocation {
        match self {
            // CPU-visible for easy upload (host coherent)
            BufferUsage::Vertex | BufferUsage::Index => MemoryLocation::CpuToGpu,
            // Uniform buffers need frequent CPU updates
            BufferUsage::Uniform => MemoryLocation::CpuToGpu,
            // Storage buffers typically GPU-only
            BufferUsage::Storage => MemoryLocation::GpuOnly,
            // Staging buffers are CPU-writable
            BufferUsage::Staging => MemoryLocation::CpuToGpu,
        }
    }

    /// Returns a human-readable name for the buffer type.
    pub fn name(self) -> &'static str {
        match self {
            BufferUsage::Vertex => "vertex",
            BufferUsage::Index => "index",
            BufferUsage::Uniform => "uniform",
            BufferUsage::Storage => "storage",
            BufferUsage::Staging => "staging",
        }
    }
}

/// GPU buffer wrapper with managed memory.
///
/// This struct wraps a Vulkan buffer and its associated memory allocation.
/// Memory is managed by gpu-allocator, which handles suballocation and
/// memory type selection.
///
/// # Thread Safety
///
/// The buffer itself is not thread-safe. Synchronize access externally
/// when sharing between threads.
pub struct Buffer {
    /// Reference to the logical device.
    device: Arc<Device>,
    /// Vulkan buffer handle.
    buffer: vk::Buffer,
    /// GPU memory allocation.
    allocation: Option<Allocation>,
    /// Buffer size in bytes.
    size: vk::DeviceSize,
    /// Buffer usage type.
    usage: BufferUsage,
}

impl Buffer {
    /// Creates a new buffer with the specified size.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `usage` - The intended buffer usage
    /// * `size` - Buffer size in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if buffer or memory allocation fails.
    pub fn new(device: Arc<Device>, usage: BufferUsage, size: vk::DeviceSize) -> RhiResult<Self> {
        if size == 0 {
            return Err(RhiError::InvalidHandle(
                "Buffer size must be greater than 0".to_string(),
            ));
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage.to_vk_usage())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.handle().create_buffer(&buffer_info, None)? };

        let requirements = unsafe { device.handle().get_buffer_memory_requirements(buffer) };

        // Allocate memory
        let allocation = {
            let mut allocator = device.allocator().lock().unwrap();
            allocator.allocate(&AllocationCreateDesc {
                name: usage.name(),
                requirements,
                location: usage.memory_location(),
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?
        };

        // Bind memory to buffer
        unsafe {
            device
                .handle()
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        debug!("Created {} buffer: {} bytes", usage.name(), size);

        Ok(Self {
            device,
            buffer,
            allocation: Some(allocation),
            size,
            usage,
        })
    }

    /// Creates a new buffer and initializes it with data.
    ///
    /// This is a convenience method that creates a buffer and immediately
    /// uploads data to it. The buffer must use CPU-visible memory.
    ///
    /// # Arguments
    ///
    /// * `device` - The logical device
    /// * `usage` - The intended buffer usage
    /// * `data` - Initial data to upload
    ///
    /// # Errors
    ///
    /// Returns an error if buffer creation or data upload fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_rhi::device::Device;
    /// use renderer_rhi::buffer::{Buffer, BufferUsage};
    ///
    /// # fn example(device: Arc<Device>) -> Result<(), renderer_rhi::RhiError> {
    /// let vertices: [f32; 6] = [0.0, 0.5, -0.5, -0.5, 0.5, -0.5];
    /// let buffer = Buffer::new_with_data(
    ///     device,
    ///     BufferUsage::Vertex,
    ///     bytemuck::cast_slice(&vertices),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_data(device: Arc<Device>, usage: BufferUsage, data: &[u8]) -> RhiResult<Self> {
        let buffer = Self::new(device, usage, data.len() as vk::DeviceSize)?;
        buffer.write_data(0, data)?;
        Ok(buffer)
    }

    /// Writes data to the buffer at the specified offset.
    ///
    /// The buffer must use CPU-visible memory (CpuToGpu or similar).
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset into the buffer
    /// * `data` - Data to write
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The buffer memory is not mapped
    /// - The write would exceed the buffer size
    pub fn write_data(&self, offset: vk::DeviceSize, data: &[u8]) -> RhiResult<()> {
        if data.is_empty() {
            return Ok(());
        }

        let end = offset + data.len() as vk::DeviceSize;
        if end > self.size {
            return Err(RhiError::InvalidHandle(format!(
                "Write exceeds buffer size: offset {} + data {} > buffer {}",
                offset,
                data.len(),
                self.size
            )));
        }

        let allocation = self.allocation.as_ref().ok_or_else(|| {
            RhiError::InvalidHandle("Buffer allocation is not available".to_string())
        })?;

        let mapped_ptr = allocation
            .mapped_ptr()
            .ok_or_else(|| RhiError::InvalidHandle("Buffer memory is not mapped".to_string()))?;

        unsafe {
            let dst = mapped_ptr.as_ptr().add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut u8, data.len());
        }

        Ok(())
    }

    /// Returns the Vulkan buffer handle.
    #[inline]
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Returns the buffer size in bytes.
    #[inline]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Returns the buffer usage type.
    #[inline]
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Free allocation first, then destroy buffer
        if let Some(allocation) = self.allocation.take() {
            let mut allocator = self.device.allocator().lock().unwrap();
            if let Err(e) = allocator.free(allocation) {
                tracing::error!("Failed to free buffer allocation: {:?}", e);
            }
        }

        unsafe {
            self.device.handle().destroy_buffer(self.buffer, None);
        }

        debug!("Destroyed {} buffer", self.usage.name());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_usage_to_vk_usage() {
        assert!(
            BufferUsage::Vertex
                .to_vk_usage()
                .contains(vk::BufferUsageFlags::VERTEX_BUFFER)
        );
        assert!(
            BufferUsage::Index
                .to_vk_usage()
                .contains(vk::BufferUsageFlags::INDEX_BUFFER)
        );
        assert!(
            BufferUsage::Uniform
                .to_vk_usage()
                .contains(vk::BufferUsageFlags::UNIFORM_BUFFER)
        );
        assert!(
            BufferUsage::Storage
                .to_vk_usage()
                .contains(vk::BufferUsageFlags::STORAGE_BUFFER)
        );
        assert!(
            BufferUsage::Staging
                .to_vk_usage()
                .contains(vk::BufferUsageFlags::TRANSFER_SRC)
        );
    }

    #[test]
    fn test_buffer_usage_memory_location() {
        assert_eq!(
            BufferUsage::Vertex.memory_location(),
            MemoryLocation::CpuToGpu
        );
        assert_eq!(
            BufferUsage::Index.memory_location(),
            MemoryLocation::CpuToGpu
        );
        assert_eq!(
            BufferUsage::Uniform.memory_location(),
            MemoryLocation::CpuToGpu
        );
        assert_eq!(
            BufferUsage::Storage.memory_location(),
            MemoryLocation::GpuOnly
        );
        assert_eq!(
            BufferUsage::Staging.memory_location(),
            MemoryLocation::CpuToGpu
        );
    }

    #[test]
    fn test_buffer_usage_name() {
        assert_eq!(BufferUsage::Vertex.name(), "vertex");
        assert_eq!(BufferUsage::Index.name(), "index");
        assert_eq!(BufferUsage::Uniform.name(), "uniform");
        assert_eq!(BufferUsage::Storage.name(), "storage");
        assert_eq!(BufferUsage::Staging.name(), "staging");
    }
}

#pragma once

#include "device.hpp"

namespace ud {

    // UDBuffer is a wrapper around VkBuffer and VkDeviceMemory
    // It also provides utility functions to map and copy memory between CPU and GPU
    class UDBuffer {
        public:
            UDBuffer(UDDevice& device,
                VkDeviceSize instanceSize,
                uint32_t instanceCount,
                VkBufferUsageFlags usageFlags,
                VkMemoryPropertyFlags memoryPropertyFlags,
                VkDeviceSize minOffsetAlignment = 1);
            ~UDBuffer();

            UDBuffer(const UDBuffer&) = delete;
            UDBuffer& operator=(const UDBuffer&) = delete;

            // Mapping memory
            VkResult map(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
            void unmap();

            // Copy memory to GPU
            void writeToBuffer(void* data, VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
            VkResult flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
            VkDescriptorBufferInfo descriptorInfo(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
            VkResult invalidate(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

            // Index variance for each operation
            // Useful for grouping multiple instances into a single buffer
            void writeToIndex(void* data, int index);
            VkResult flushIndex(int index);
            VkDescriptorBufferInfo descriptorInfoForIndex(int index);
            VkResult invalidateIndex(int index);

            // Getters
            VkBuffer getBuffer() const { return buffer; }
            void* getMappedMemory() const { return mapped; }
            uint32_t getInstanceCount() const { return instanceCount; }
            VkDeviceSize getInstanceSize() const { return instanceSize; }
            VkDeviceSize getAlignmentSize() const { return instanceSize; }
            VkBufferUsageFlags getUsageFlags() const { return usageFlags; }
            VkMemoryPropertyFlags getMemoryPropertyFlags() const { return memoryPropertyFlags; }
            VkDeviceSize getBufferSize() const { return bufferSize; }

        private:
            // Helper function to get the alignment of the buffer
            // The alignment guideline is that the instance of a uniform block must be at
            // an offset that is an integer multiple of the min uniform buffer offset alignment
            static VkDeviceSize getAlignment(VkDeviceSize instanceSize, VkDeviceSize minOffsetAlignment);

            UDDevice& udDevice;
            void* mapped = nullptr;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;

            VkDeviceSize bufferSize;
            uint32_t instanceCount;
            VkDeviceSize instanceSize;
            VkDeviceSize alignmentSize;
            VkBufferUsageFlags usageFlags;
            VkMemoryPropertyFlags memoryPropertyFlags;
    };
}  // namespace ud

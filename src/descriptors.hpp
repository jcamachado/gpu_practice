#pragma once

#include "device.hpp"

// std
#include <memory>
#include <unordered_map>
#include <vector>

namespace ud {

    class UDDescriptorSetLayout {
    public:
        class Builder {
        public:
            Builder(UDDevice& udDevice) : udDevice{ udDevice } {}

            Builder& addBinding(    // <--- This is the Builder pattern, tells vulkan what kind of data will be in the descriptor
                uint32_t binding,
                VkDescriptorType descriptorType,
                VkShaderStageFlags stageFlags,
                uint32_t count = 1);
            std::unique_ptr<UDDescriptorSetLayout> build() const;

        private:
            UDDevice& udDevice;
            std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings{};
        };

        UDDescriptorSetLayout(
            UDDevice& udDevice, std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings);
        ~UDDescriptorSetLayout();
        UDDescriptorSetLayout(const UDDescriptorSetLayout&) = delete;
        UDDescriptorSetLayout& operator=(const UDDescriptorSetLayout&) = delete;

        VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }

    private:
        UDDevice& udDevice;
        VkDescriptorSetLayout descriptorSetLayout;
        std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings;

        friend class UDDescriptorWriter;
    };

    class UDDescriptorPool {
    public:
        class Builder {
        public:
            Builder(UDDevice& udDevice) : udDevice{ udDevice } {}

            Builder& addPoolSize(VkDescriptorType descriptorType, uint32_t count);
            Builder& setPoolFlags(VkDescriptorPoolCreateFlags flags);
            Builder& setMaxSets(uint32_t count);
            std::unique_ptr<UDDescriptorPool> build() const;

        private:
            UDDevice& udDevice;
            std::vector<VkDescriptorPoolSize> poolSizes{};
            uint32_t maxSets = 1000;
            VkDescriptorPoolCreateFlags poolFlags = 0;
        };

        UDDescriptorPool(
            UDDevice& udDevice,
            uint32_t maxSets,
            VkDescriptorPoolCreateFlags poolFlags,
            const std::vector<VkDescriptorPoolSize>& poolSizes);
        ~UDDescriptorPool();
        UDDescriptorPool(const UDDescriptorPool&) = delete;
        UDDescriptorPool& operator=(const UDDescriptorPool&) = delete;

        bool allocateDescriptor(
            const VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet& descriptor) const;

        void freeDescriptors(std::vector<VkDescriptorSet>& descriptors) const;

        void resetPool();

    private:
        UDDevice& udDevice;
        VkDescriptorPool descriptorPool;

        friend class UDDescriptorWriter;
    };

    class UDDescriptorWriter {
    public:
        UDDescriptorWriter(UDDescriptorSetLayout& setLayout, UDDescriptorPool& pool);

        UDDescriptorWriter& writeBuffer(uint32_t binding, VkDescriptorBufferInfo* bufferInfo);
        UDDescriptorWriter& writeImage(uint32_t binding, VkDescriptorImageInfo* imageInfo);

        bool build(VkDescriptorSet& set);
        void overwrite(VkDescriptorSet& set);

    private:
        UDDescriptorSetLayout& setLayout;
        UDDescriptorPool& pool;
        std::vector<VkWriteDescriptorSet> writes;
    };

}  // namespace ud
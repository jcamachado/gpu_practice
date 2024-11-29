#pragma once

#include "device.hpp"
#include "descriptors.hpp"
#include "swap_chain.hpp"
#include <memory>
#include <unordered_map>

namespace ud {

    class DescriptorManager {
    public:
        static DescriptorManager& getInstance() {
            static DescriptorManager instance;
            return instance;
        }

        void initialize(UDDevice& device) {
            if (!initialized) {
                globalPool = UDDescriptorPool::Builder(device)
                    .setMaxSets(UDSwapChain::MAX_FRAMES_IN_FLIGHT)
                    .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, UDSwapChain::MAX_FRAMES_IN_FLIGHT)
                    .build();

                texturePool = UDDescriptorPool::Builder(device)
                    .setMaxSets(UDSwapChain::MAX_FRAMES_IN_FLIGHT)
                    .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, UDSwapChain::MAX_FRAMES_IN_FLIGHT)
                    .build();

                initialized = true;
            }
        }

        UDDescriptorPool& getGlobalPool() { return *globalPool; }
        UDDescriptorPool& getTexturePool() { return *texturePool; }

        UDDescriptorSetLayout& getGlobalSetLayout(UDDevice& device) {
            if (!globalSetLayout) {
                globalSetLayout = UDDescriptorSetLayout::Builder(device)
                    .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)
                    .build();
            }
            return *globalSetLayout;
        }

        UDDescriptorSetLayout& getTextureSetLayout(UDDevice& device) {
            if (!textureSetLayout) {
                textureSetLayout = UDDescriptorSetLayout::Builder(device)
                    .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
                    .build();
            }
            return *textureSetLayout;
        }

    private:
        DescriptorManager() = default;
        ~DescriptorManager() = default;

        DescriptorManager(const DescriptorManager&) = delete;
        DescriptorManager& operator=(const DescriptorManager&) = delete;

        std::unique_ptr<UDDescriptorPool> globalPool;
        std::unique_ptr<UDDescriptorPool> texturePool;
        std::unique_ptr<UDDescriptorSetLayout> globalSetLayout;
        std::unique_ptr<UDDescriptorSetLayout> textureSetLayout;
        bool initialized = false;
    };

} // namespace ud
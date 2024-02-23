#pragma once

#include "camera.hpp"

// libs
#include <vulkan/vulkan.h>

namespace ud {
    struct FrameInfo {
        int frameIndex;
        float frameTime;
        VkCommandBuffer commandBuffer;
        UDCamera &camera;
        VkDescriptorSet globalDescriptorSet; 
    };
}
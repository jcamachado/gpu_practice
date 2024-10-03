#pragma once

#include "camera.hpp"
#include "game_object.hpp"

// libs
#include <vulkan/vulkan.h>

namespace ud {

#define MAX_LIGHTS 10

    /**
        Specular light has better quality when the gaussian distribution is used
        but it is more computationally expensive.
        We will use Blinn-Phong shading model, which is a compromise between quality and performance for now.
    */
    struct PointLight {
        glm::vec4 position{}; // ignore w 
        glm::vec4 color{}; // w is intensity
    };

    struct GlobalUBO { // Uniform Buffer Object (needs alignment)
        glm::mat4 projection{ 1.0f };
        glm::mat4 view{ 1.0f };
        glm::mat4 inverseView{ 1.0f }; // get camera position from last column
        glm::vec4 ambientLightColor{ 1.0f, 1.0f, 1.0f, 0.02f };
        PointLight pointLights[MAX_LIGHTS];
        int numLights;
    };

    struct FrameInfo {
        int frameIndex;
        float frameTime;
        VkCommandBuffer commandBuffer;
        // UDCamera &camera;
        const UDCamera* camera;
        VkDescriptorSet globalDescriptorSet;
        UDGameObject::Map& gameObjects;
    };
}
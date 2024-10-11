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

    // for 1 camera
    // struct GlobalUBO { // Uniform Buffer Object (needs alignment) 
    //     glm::mat4 projection{ 1.0f };
    //     glm::mat4 view{ 1.0f };
    //     glm::mat4 inverseView{ 1.0f }; // get camera position from last column
    //     glm::vec4 ambientLightColor{ 1.0f, 1.0f, 1.0f, 0.02f };
    //     PointLight pointLights[MAX_LIGHTS];
    //     int numLights;
    // };

    // for 2 cameras
    struct GlobalUBO { // total of bits = 16 * 4 * 4 = 256 bits = 32 bytes
        glm::mat4 projection[2] = { glm::mat4(1.0f), glm::mat4(1.0f) }; // Projection matrices for left and right eyes
        glm::mat4 view[2] = { glm::mat4(1.0f), glm::mat4(1.0f) };       // View matrices for left and right eyes
        glm::mat4 inverseView[2] = { glm::mat4(1.0f), glm::mat4(1.0f) }; // Inverse view matrices for left and right eyes
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
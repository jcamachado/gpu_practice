#pragma once

#include "device.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// std
#include <vector>

namespace ud {
    class UDModel {
    public:
        struct Vertex {
            glm::vec3 position;
            glm::vec3 color;

            /*
                The binding description and attribute descriptions are static methods
                because they are not specific to a single vertex, but to the Vertex
                struct as a whole.

                The binding description is used to describe at which rate to load data
                from memory throughout the vertices. The attribute descriptions are
                used to describe how to extract a vertex attribute from a chunk of
                vertex data in memory.

                Example of attribute descriptions:
                - position: float32, 2 elements, offset 0
                - color:    float32, 3 elements, offset 8

                Example of binding descriptions:
                - The rate at which data is loaded for vertices is per-vertex
                - The rate at which data is loaded for instances is per-instance
            */
            static std::vector<VkVertexInputBindingDescription> getBindingDescriptions();
            static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
        };

        UDModel(UDDevice &device, const std::vector<Vertex> &vertices);
        ~UDModel();

        UDModel(const UDModel&) = delete;
        UDModel& operator=(const UDModel&) = delete;

        void bind(VkCommandBuffer commandBuffer);
        void draw(VkCommandBuffer commandBuffer);

    private:
        UDDevice &device;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        uint32_t vertexCount;

        void createVertexBuffers(const std::vector<Vertex> &vertices);
    };
}
#pragma once

#include "device.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// std
#include <memory>
#include <vector>

namespace ud {
    class UDModel {
    public:
        struct Vertex {
            glm::vec3 position{};
            glm::vec3 color{};
            glm::vec3 normal{};
            glm::vec2 uv{}; // 2D texture coordinates

            /*
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

            // Overloading the == operator to compare vertices using hashCombine
            bool operator==(const Vertex& other) const {
                return position == other.position && 
                    color == other.color && 
                    normal == other.normal && 
                    uv == other.uv;
            }
        };

        struct Builder { //This struct is used to load the model vertices and indices to be rendered
            std::vector<Vertex> vertices{};
            std::vector<uint32_t> indices{};

            void loadModel(const std::string &filepath);
        };

        UDModel(UDDevice &device, const UDModel::Builder &builder);
        ~UDModel();

        UDModel(const UDModel&) = delete;
        UDModel& operator=(const UDModel&) = delete;

        static std::unique_ptr<UDModel> createModelFromFile(UDDevice &device, const std::string &filepath);

        void bind(VkCommandBuffer commandBuffer);
        void draw(VkCommandBuffer commandBuffer);

    private:
        void createVertexBuffers(const std::vector<Vertex> &vertices);
        void createIndexBuffers(const std::vector<uint32_t> &indices);
        
        UDDevice &device;

        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        uint32_t vertexCount;

        /*
            Vulkan only allows one index buffer per model, so we cant use one index buffer 
            for each tipe of vertex attrib, such as vertex normal, texture and so one.
            All the vertex attribs must be stored in the same index buffer.
            To do this, we must have a way to know if a loaded vertex has already been loaded
            or if it is a new vertex. We will use a hash table to do this. Hence the ud_utils.hpp.
        */
        
        bool hasIndexBuffer{false};
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;
        uint32_t indexCount;


    };
}
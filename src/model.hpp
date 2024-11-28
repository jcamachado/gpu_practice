#pragma once

#include "buffer.hpp"
// #include "device.hpp"
#include "renderer.hpp"

// tinygltf
#include "lib/tinygltf/tiny_gltf.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>


// std
#include <memory>
#include <vector>

namespace ud {
    class UDModel {
    public:
        /*
            The binding description sets the rate to load data from memory throughout the vertices.
            The attrib descriptions sets how to extract a vertex attrib from a chunk of
            vertex data in memory.

            Example of attribute descriptions:
            - position: float32, 2 elements, offset 0
            - color:    float32, 3 elements, offset 8

            Example of binding descriptions:
            - The rate at which data is loaded for vertices is per-vertex
            - The rate at which data is loaded for instances is per-instance
        */
        struct Vertex {
            glm::vec3 position{};
            glm::vec3 color{};
            glm::vec3 normal{};
            glm::vec2 uv{}; // 2D texture coordinates

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

        struct Material {
            glm::vec4 baseColorFactor{ 1.0f };
            int baseColorTextureIndex{ -1 };
            // Add other material properties as needed
        };

        struct Builder { //This struct is used to load the model vertices and indices to be rendered
            UDDevice& device; // Reference to the device
            std::vector<Vertex> vertices{};
            std::vector<uint32_t> indices{};
            std::vector<Material> materials{};
            std::vector<tinygltf::Image> images{};
            std::vector<tinygltf::Texture> textures{};

            Builder(UDDevice& device) :
                device(device) {
            }

            void loadModelObj(const std::string& filepath);
            void loadModelGltf(const std::string& filepath);
            // Uses device, should it be here?
            // void loadTextureImage(const tinygltf::Image& image);
        };

        // UDModel(UDDevice& device, const UDModel::Builder& builder);
        UDModel(UDRenderer& renderer, const std::string& filepath);
        ~UDModel();

        UDModel(const UDModel&) = delete;
        UDModel& operator=(const UDModel&) = delete;

        // static std::unique_ptr<UDModel> createModelFromFile(UDDevice& device, const std::string& filepath);

        void bind(VkCommandBuffer commandBuffer);
        void draw(VkCommandBuffer commandBuffer);

        VkImageView getTextureImageView() const { return textureImageView; }
        VkSampler getTextureSampler() const { return textureSampler; }
        // void createTextureImage(const tinygltf::Image& image);
        void createTextureImage();
        // void createImage(
        //     uint32_t width,
        //     uint32_t height,
        //     VkFormat format,
        //     VkImageTiling tiling,
        //     VkImageUsageFlags usage,
        //     VkMemoryPropertyFlags properties,
        //     VkImage& image,
        //     VkDeviceMemory& imageMemory);
        void createTextureImageView();
        void createTextureSampler();


    private:
        void loadData();
        void createVertexBuffers(const std::vector<Vertex>& vertices);
        void createIndexBuffers(const std::vector<uint32_t>& indices);
        //here?

        // texture loading

        UDRenderer& renderer;
        UDDevice& device;
        std::string filepath;
        bool dataLoaded{ false };

        /*
            Vulkan only allows one index buffer per model, so we cant use one index buffer
            for each tipe of vertex attrib, such as vertex normal, texture and so on.
            All the vertex attribs must be stored in the same index buffer.
            To do this, we must have a way to know if a loaded vertex has already been loaded
            or if it is a new vertex. We will use a hash table to do this. Hence the ud_utils.hpp.
        */

        std::unique_ptr<UDBuffer> vertexBuffer;
        uint32_t vertexCount;

        bool hasIndexBuffer{ false };
        std::unique_ptr<UDBuffer> indexBuffer;
        uint32_t indexCount;

        VkImage textureImage;
        VkDeviceMemory textureImageMemory;
        VkImageView textureImageView;
        VkSampler textureSampler;
    };
}
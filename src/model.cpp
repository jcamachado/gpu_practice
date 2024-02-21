#include "model.hpp"

#include "ud_utils.hpp"

//libs
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* cpp file for the entire project
#include <tiny_obj_loader.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

// std
#include <cstring>
#include <cassert>
#include <unordered_map>

namespace std{
    /*
        With this we can take an instance of the Vertex struct and hash it to a
        single value of type size_t. This is useful for the unordered_map to use
        it as a key.

        It allows the use of the Vertex struct as a key in an unordered_map.
        So this is allowed:
        std::unordered_map<Vertex, int> uniqueVertices{};
    */
    template<> 
    struct hash<ud::UDModel::Vertex> { // Injections of the hash function for the Vertex struct
        size_t operator()(ud::UDModel::Vertex const& vertex) const {
            size_t seed = 0;
            ud::hashCombine(seed, vertex.position, vertex.color, vertex.normal, vertex.uv);
            return seed;
        }
    };
}

namespace ud {
    UDModel::UDModel(UDDevice &device, const UDModel::Builder &builder) : device{device} {
        createVertexBuffers(builder.vertices);
        createIndexBuffers(builder.indices);
    }

    UDModel::~UDModel() {
        vkDestroyBuffer(device.device(), vertexBuffer, nullptr);
        vkFreeMemory(device.device(), vertexBufferMemory, nullptr);

        if (hasIndexBuffer) {
            vkDestroyBuffer(device.device(), indexBuffer, nullptr);
            vkFreeMemory(device.device(), indexBufferMemory, nullptr);
        }
    }

    std::unique_ptr<UDModel> UDModel::createModelFromFile(
        // The createModelFromFile function is a static method that creates a new model from a file
        UDDevice &device, const std::string &filepath
    ) {
        Builder builder{};
        builder.loadModel(filepath);
        return std::make_unique<UDModel>(device, builder);
    }

    void UDModel::bind(VkCommandBuffer commandBuffer) {
        VkBuffer buffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);

        if (hasIndexBuffer) {
            // Index type must be the same as the indices vector type
            // For small projects, we could use 16 bits (uint16_t). But for general purposes, we will use 32 bits
            // 16 bits= 65535 vertices, 32 bits= 4,294,967,295 vertices
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }
    }

    void UDModel::draw(VkCommandBuffer commandBuffer) {
        if (hasIndexBuffer) {
            vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
        } else{
            vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
        }
    }
    /*
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            This means that the memory is mappable by the CPU and is coherent, so CPU
            writes are immediately visible to the GPU without having to flush the cache.
            This is not the as fast as it could be. It is for learning purposes.

            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT is the fastest memory type, but it is not mappable by the CPU.
            To use device local memory, we must have a staging buffer, which is a buffer in host visible memory
            that we copy the data to, and then copy the data to the device local memory.

            STAGING BUFFER IS RECOMMENDED FOR STATIC DATA, THOSE THAT ARE LOADED ONCE IN THE BEGINNING
            AND NEVER CHANGED
        */
        
    void UDModel::createVertexBuffers(const std::vector<Vertex> &vertices) {
        vertexCount = static_cast<uint32_t>(vertices.size());
        assert(vertexCount >= 3 && "Vertex count must be at least 3");

        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertexCount;

        // device.createBuffer(bufferSize, 
        //     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
        //     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
        //     vertexBuffer, 
        //     vertexBufferMemory
        // );
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        device.createBuffer(bufferSize,         // Creates the staging buffer
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // Buffer is used as the source in a memory transfer operation
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, 
            stagingBufferMemory
        );

        void* data;
        vkMapMemory(device.device(), stagingBufferMemory, 0, bufferSize, 0, &data); // maps the device memory to the host
        memcpy(data, vertices.data(), static_cast<size_t>(bufferSize)); // Copies Data from Host to Device
        vkUnmapMemory(device.device(), stagingBufferMemory);

        device.createBuffer(bufferSize,  // Creates a space in device memory for the vertex buffer
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            vertexBuffer, 
            vertexBufferMemory
        );

         // Copies the data from the staging buffer to the vertex buffer
        device.copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device.device(), stagingBuffer, nullptr);
        vkFreeMemory(device.device(), stagingBufferMemory, nullptr);
    }

    void UDModel::createIndexBuffers(const std::vector<uint32_t> &indices) {
        indexCount = static_cast<uint32_t>(indices.size());
        hasIndexBuffer = indexCount > 0;

        if (!hasIndexBuffer) return;

        VkDeviceSize bufferSize = sizeof(indices[0]) * indexCount;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        device.createBuffer(bufferSize,         // Creates the staging buffer
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // Buffer is used as the source in a memory transfer operation
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, 
            stagingBufferMemory
        );

        void* data;
        vkMapMemory(device.device(), stagingBufferMemory, 0, bufferSize, 0, &data); // maps the device memory to the host
        memcpy(data, indices.data(), static_cast<size_t>(bufferSize)); // Copies Data from Host to Device
        vkUnmapMemory(device.device(), stagingBufferMemory);

        device.createBuffer(bufferSize,  // Creates a space in device memory for the vertex buffer
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            indexBuffer, 
            indexBufferMemory
        );

         // Copies the data from the staging buffer to the vertex buffer
        device.copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device.device(), stagingBuffer, nullptr);
        vkFreeMemory(device.device(), stagingBufferMemory, nullptr);
    }

    std::vector<VkVertexInputBindingDescription> UDModel::Vertex::getBindingDescriptions() {
        std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
        bindingDescriptions[0].binding = 0;
        bindingDescriptions[0].stride = sizeof(Vertex);
        bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescriptions; // Same as {{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}}
    }
    
    std::vector<VkVertexInputAttributeDescription> UDModel::Vertex::getAttributeDescriptions() {
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);
        // position
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; // Only R32G32_SFLOAT because we only have position (x, y)
        attributeDescriptions[0].offset = offsetof(Vertex, position);
        // color
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; 
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions; 
    }

    void UDModel::Builder::loadModel(const std::string &filepath) {
        // Load the model from the file using tinyobjloader
        tinyobj::attrib_t attrib; // Vertex attributes
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str())) {
            throw std::runtime_error(warn + err);
        }

        vertices.clear();
        indices.clear();

        // Vertex as key (in hash format)
        std::unordered_map<Vertex, uint32_t> uniqueVertices{};
        for (const auto &shape : shapes) {
            for (const auto &index : shape.mesh.indices) {
                Vertex vertex{};

                if(index.vertex_index >= 0){
                    vertex.position = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                    };
                    // Unnofficial extension for vertex color but it is supported by tinyobjloader
                    // This is not a standard feature of the .obj file format
                    auto colorIndex = 3 * index.vertex_index + 2;
                    if (colorIndex < attrib.colors.size()) {
                        vertex.color = {
                            attrib.colors[colorIndex - 2],
                            attrib.colors[colorIndex - 1],
                            attrib.colors[colorIndex - 0]
                        };
                    } else {
                        vertex.color = {1.0f, 1.0f, 1.0f};
                    }

                }

                if(index.normal_index >= 0) {
                    vertex.normal = {
                        attrib.normals[3 * index.normal_index + 0],
                        attrib.normals[3 * index.normal_index + 1],
                        attrib.normals[3 * index.normal_index + 2]
                    };
                }

                if(index.texcoord_index >= 0) {
                    vertex.uv = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        attrib.texcoords[2 * index.texcoord_index + 1]
                    };
                }

                // Check if the vertex is already in the uniqueVertices map
                // Vertices.size() is the index of a new vertex that will be added to the vertices vector
                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }
                indices.push_back(uniqueVertices[vertex]);
            }
        }
    }

}
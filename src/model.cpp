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

#ifndef ENGINE_DIR
#define ENGINE_DIR "../"
#endif

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

    UDModel::~UDModel() { }

    std::unique_ptr<UDModel> UDModel::createModelFromFile(
        // The createModelFromFile function is a static method that creates a new model from a file
        UDDevice &device, const std::string &filepath
    ) {
        Builder builder{};
        builder.loadModel(filepath);
        return std::make_unique<UDModel>(device, builder);
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
        uint32_t vertexSize = sizeof(vertices[0]);
        UDBuffer stagingBuffer{
            device, 
            vertexSize,
            vertexCount,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        };
        stagingBuffer.map(); // Maps the buffer to the host memory
        stagingBuffer.writeToBuffer((void *)vertices.data()); // Copies the data to the buffer

        // Create space in device memory for the vertex buffer
        vertexBuffer = std::make_unique<UDBuffer>(
            device, 
            vertexSize, 
            vertexCount, 
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

         // Copies the data from the staging buffer to the vertex buffer
        device.copyBuffer(stagingBuffer.getBuffer(), vertexBuffer->getBuffer(), bufferSize);
        // No need to clear memory. stagingBuffer is a stack variable, so it will be cleaned up
        // when createVertexBuffers ends
    }

    void UDModel::createIndexBuffers(const std::vector<uint32_t> &indices) {
        // Same as createVertexBuffers but for the index buffer
        indexCount = static_cast<uint32_t>(indices.size());
        hasIndexBuffer = indexCount > 0;

        if (!hasIndexBuffer) return;

        VkDeviceSize bufferSize = sizeof(indices[0]) * indexCount;
        uint32_t indexSize = sizeof(indices[0]);
        UDBuffer stagingBuffer{
            device, 
            indexSize, 
            indexCount, 
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        };
        stagingBuffer.map();
        stagingBuffer.writeToBuffer((void *)indices.data());

        indexBuffer = std::make_unique<UDBuffer>(
            device, 
            indexSize, 
            indexCount, 
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        device.copyBuffer(stagingBuffer.getBuffer(), indexBuffer->getBuffer(), bufferSize);
    }

    std::vector<VkVertexInputBindingDescription> UDModel::Vertex::getBindingDescriptions() {
        std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
        bindingDescriptions[0].binding = 0;
        bindingDescriptions[0].stride = sizeof(Vertex);
        bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescriptions; // Same as {{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}}
    }
    
    // Attribute descriptions define how to extract a vertex attribute from a chunk of vertex data originating from a binding description
    // As an example of attribute description, we have the position and color of the vertex
    std::vector<VkVertexInputAttributeDescription> UDModel::Vertex::getAttributeDescriptions() {
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
        
        // params(location, binding, format, offset)
        // Similar to OpenGLs glVertexAttribPointer
        attributeDescriptions.push_back({0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)}); // rgb because 3 floats x, y, z
        attributeDescriptions.push_back({1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)});
        attributeDescriptions.push_back({2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)});
        attributeDescriptions.push_back({3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)});

        return attributeDescriptions; 
    }


    void UDModel::bind(VkCommandBuffer commandBuffer) {
        VkBuffer buffers[] = {vertexBuffer->getBuffer()};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);

        if (hasIndexBuffer) {
            // Index type must be the same as the indices vector type
            // For small projects, we could use 16 bits (uint16_t). But for general purposes, we will use 32 bits
            // 16 bits= 65535 vertices, 32 bits= 4,294,967,295 vertices
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer->getBuffer(), 0, VK_INDEX_TYPE_UINT32);
        }
    }

    void UDModel::draw(VkCommandBuffer commandBuffer) {
        if (hasIndexBuffer) {
            vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
        } else{
            vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
        }
    }


    void UDModel::Builder::loadModel(const std::string &filepath) {
        // Load the model from the file using tinyobjloader
        tinyobj::attrib_t attrib; // Vertex attributes
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, (ENGINE_DIR + filepath).c_str())) {
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
                    vertex.color = {
                        attrib.colors[3 * index.vertex_index + 0],
                        attrib.colors[3 * index.vertex_index + 1],
                        attrib.colors[3 * index.vertex_index + 2]
                    };

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
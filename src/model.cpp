#include "model.hpp"

#include "ud_utils.hpp"

//libs
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* cpp file for the entire project
#include <tinyobjloader/tiny_obj_loader.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

// std
#include <cstring>
#include <cassert>
#include <unordered_map>

#include <iostream>
using namespace tinygltf;


namespace std {
    /*
        With this we can take an instance of the Vertex struct and hash it to a value of type size_t.
        This is useful for the unordered_map to use it as a key. So this is allowed:
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
    UDModel::UDModel(UDRenderer& renderer, const std::string& filepath) :
        renderer{ renderer }, filepath{ filepath }, device{ renderer.getDevice() } {
    }

    UDModel::~UDModel() {
        vkDestroySampler(device.device(), textureSampler, nullptr);
        vkDestroyImageView(device.device(), textureImageView, nullptr);
        vkDestroyImage(device.device(), textureImage, nullptr);
        vkFreeMemory(device.device(), textureImageMemory, nullptr);
    }

    // std::unique_ptr<UDModel> UDModel::createModelFromFile(
    //     // The createModelFromFile function is a static method that creates a new model from a file
    //     UDDevice& device, const std::string& filepath
    // ) {
    //     Builder builder{};
    //     if (filepath.substr(filepath.find_last_of(".") + 1) == "obj") {
    //         builder.loadModelObj(filepath);
    //     }
    //     else if (filepath.substr(filepath.find_last_of(".") + 1) == "gltf") {
    //         builder.loadModelGltf(filepath);
    //     }
    //     else {
    //         throw std::runtime_error("Unsupported file format: " + filepath);
    //     }
    //     return std::make_unique<UDModel>(device, builder);
    // }
    void UDModel::loadData() {
        if (dataLoaded) return;

        Builder builder{ device, renderer.getSwapChain() };
        if (filepath.substr(filepath.find_last_of(".") + 1) == "obj") {
            builder.loadModelObj(filepath);
        }
        else if (filepath.substr(filepath.find_last_of(".") + 1) == "gltf"
            || filepath.substr(filepath.find_last_of(".") + 1) == "glb") {
            builder.loadModelGltf(filepath);
        }
        else {
            throw std::runtime_error("Unsupported file format: " + filepath);
        }

        createVertexBuffers(builder.vertices);
        createIndexBuffers(builder.indices);

        if (!builder.images.empty()) {
            createTextureImage(builder.images[0]);
            createTextureImageView();
            createTextureSampler();
        }

        dataLoaded = true;
    }

    /*
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        This means that the memory is mappable by the CPU and is coherent, so CPU
        writes are immediately visible to the GPU without having to flush the cache.

        -*> This is not the as fast as it could be. It is for learning purposes.<*-

        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT is the fastest memory type, but it is not mappable by the CPU.
        To use device local memory, we must have a staging buffer, which is a buffer in host visible memory
        that we copy the data to, and then copy the data to the device local memory.

        STAGING BUFFER IS RECOMMENDED FOR STATIC DATA, THOSE THAT ARE LOADED ONCE IN THE BEGINNING
        AND NEVER CHANGED
    */
    void UDModel::createVertexBuffers(const std::vector<Vertex>& vertices) {
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
        stagingBuffer.writeToBuffer((void*)vertices.data()); // Copies the data to the buffer

        // Create space in device memory for the vertex buffer
        vertexBuffer = std::make_unique<UDBuffer>(
            device,
            vertexSize,
            vertexCount,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        // Copies data: staging buffer -> vertex buffer
        device.copyBuffer(stagingBuffer.getBuffer(), vertexBuffer->getBuffer(), bufferSize);
        // StagingBuffer is a stack variable, so it will be cleaned up when createVertexBuffers ends
    }

    // Same as createVertexBuffers but for the index buffer
    void UDModel::createIndexBuffers(const std::vector<uint32_t>& indices) {
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
        stagingBuffer.writeToBuffer((void*)indices.data());

        indexBuffer = std::make_unique<UDBuffer>(
            device,
            indexSize,
            indexCount,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        device.copyBuffer(stagingBuffer.getBuffer(), indexBuffer->getBuffer(), bufferSize);
        stagingBuffer.unmap();
    }

    std::vector<VkVertexInputBindingDescription> UDModel::Vertex::getBindingDescriptions() {
        std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
        bindingDescriptions[0].binding = 0;
        bindingDescriptions[0].stride = sizeof(Vertex);
        bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescriptions; // Same as {{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}}
    }


    std::vector<VkVertexInputAttributeDescription> UDModel::Vertex::getAttributeDescriptions() {
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};

        /*
            params(location, binding, format, offset), Similar to OpenGLs glVertexAttribPointer
            rgb because 3 floats x, y, z
        */
        attributeDescriptions.push_back(
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) }
        );
        attributeDescriptions.push_back(
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) }
        );
        attributeDescriptions.push_back(
            { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) }
        );
        attributeDescriptions.push_back(
            { 3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) }
        );

        return attributeDescriptions;
    }


    void UDModel::bind(VkCommandBuffer commandBuffer) {
        loadData();

        VkBuffer buffers[] = { vertexBuffer->getBuffer() };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);

        if (hasIndexBuffer) {
            // Index type must be the same as the indices vector type
            // For general purposes, uses 32 bits. 16bits = 65535 vertices, 32bits= 4,294,967,295 vertices
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer->getBuffer(), 0, VK_INDEX_TYPE_UINT32);
        }
    }

    void UDModel::draw(VkCommandBuffer commandBuffer) {
        loadData(); // Load data on demand
        if (hasIndexBuffer) {
            vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
        }
        else {
            vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
        }
    }


    void UDModel::createTextureImage(const tinygltf::Image& image) {
        VkDeviceSize imageSize = image.width * image.height * 4; // Assuming 4 bytes per pixel (RGBA)

        UDBuffer stagingBuffer{
            device,
            imageSize,
            1,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        };
        stagingBuffer.map();
        stagingBuffer.writeToBuffer(reinterpret_cast<void*>(const_cast<unsigned char*>(image.image.data())));

        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.extent.width = static_cast<uint32_t>(image.width);
        imageCreateInfo.extent.height = static_cast<uint32_t>(image.height);
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        device.createImageWithInfo(imageCreateInfo,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            textureImage,
            textureImageMemory);

        renderer.getSwapChain().transitionImageLayout(
            device.device(),
            device.getCommandPool(),
            device.graphicsQueue(),
            textureImage,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );
        device.copyBufferToImage(
            stagingBuffer.getBuffer(),
            textureImage,
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            1
        );
        renderer.getSwapChain().transitionImageLayout(
            device.device(),
            device.getCommandPool(),
            device.graphicsQueue(),
            textureImage,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }

    void UDModel::createTextureImageView() {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = textureImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device.device(), &viewInfo, nullptr, &textureImageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }
    }

    void UDModel::createTextureSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device.device(), &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    void UDModel::Builder::loadModelObj(const std::string& filepath) {
        // Load the model from the file using tinyobjloader
        tinyobj::attrib_t attrib; // Vertex attributes
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, (filepath).c_str())) {
            throw std::runtime_error(warn + err);
        }

        vertices.clear();
        indices.clear();

        // Vertex as key (in hash format)
        std::unordered_map<Vertex, uint32_t> uniqueVertices{};
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                if (index.vertex_index >= 0) {
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

                if (index.normal_index >= 0) {
                    vertex.normal = {
                        attrib.normals[3 * index.normal_index + 0],
                        attrib.normals[3 * index.normal_index + 1],
                        attrib.normals[3 * index.normal_index + 2]
                    };
                }

                if (index.texcoord_index >= 0) {
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


    void UDModel::Builder::loadModelGltf(const std::string& filepath) {
        // .gltf file format
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret;
        if (filepath.substr(filepath.find_last_of(".") + 1) == "glb") {
            ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath); // Load .glb file
        }
        else {
            ret = loader.LoadASCIIFromFile(&model, &err, &warn, filepath); // Load .gltf file
        }

        if (!warn.empty()) {
            std::cout << "WARN: " << warn << std::endl;
        }

        if (!err.empty()) {
            std::cout << "ERR: " << err << std::endl;
        }

        if (!ret)
            std::cout << "Failed to load glTF: " << filepath << std::endl;
        else
            std::cout << "Loaded glTF: " << filepath << std::endl;

        vertices.clear();
        indices.clear();

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};
        // Load the model from the file using tinygltf
        for (const auto& mesh : model.meshes) {
            for (const auto& primitive : mesh.primitives) {
                const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

                const uint16_t* indicesData = reinterpret_cast<const uint16_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                for (size_t i = 0; i < indexAccessor.count; ++i) {
                    indices.push_back(indicesData[i]);
                }

                // load vertices 
                const tinygltf::Accessor& positionAccessor =
                    model.accessors[primitive.attributes.find("POSITION")->second];
                const tinygltf::BufferView& positionBufferView =
                    model.bufferViews[positionAccessor.bufferView];
                const tinygltf::Buffer& positionBuffer =
                    model.buffers[positionBufferView.buffer];

                const float* positionsData =
                    reinterpret_cast<const float*>(&positionBuffer.data[positionBufferView.byteOffset + positionAccessor.byteOffset]);
                for (size_t i = 0; i < positionAccessor.count; ++i) {
                    Vertex vertex{};
                    vertex.position = glm::vec3(positionsData[i * 3 + 0], positionsData[i * 3 + 1], positionsData[i * 3 + 2]);
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                // load normals
                if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                    const tinygltf::Accessor& normalAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
                    const tinygltf::BufferView& normalBufferView = model.bufferViews[normalAccessor.bufferView];
                    const tinygltf::Buffer& normalBuffer = model.buffers[normalBufferView.buffer];

                    const float* normalsData = reinterpret_cast<const float*>(&normalBuffer.data[normalBufferView.byteOffset + normalAccessor.byteOffset]);
                    for (size_t i = 0; i < normalAccessor.count; ++i) {
                        vertices[i].normal = glm::vec3(normalsData[i * 3 + 0], normalsData[i * 3 + 1], normalsData[i * 3 + 2]);
                    }
                }

                // load texture coordinates
                if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                    const tinygltf::Accessor& texcoordAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
                    const tinygltf::BufferView& texcoordBufferView = model.bufferViews[texcoordAccessor.bufferView];
                    const tinygltf::Buffer& texcoordBuffer = model.buffers[texcoordBufferView.buffer];

                    const float* texcoordsData = reinterpret_cast<const float*>(&texcoordBuffer.data[texcoordBufferView.byteOffset + texcoordAccessor.byteOffset]);
                    for (size_t i = 0; i < texcoordAccessor.count; ++i) {
                        vertices[i].uv = glm::vec2(texcoordsData[i * 2 + 0], texcoordsData[i * 2 + 1]);
                    }
                }

                // load colors
                if (primitive.material >= 0) {
                    const tinygltf::Material& material = model.materials[primitive.material];
                    if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
                        const tinygltf::Texture& texture = model.textures[material.pbrMetallicRoughness.baseColorTexture.index];
                        const tinygltf::Image& image = model.images[texture.source];
                        // Load textures using swap_chain.cpp code
                        images.push_back(image);
                    }
                }
            }
        }
    }
}
#include "model.hpp"

#include "ud_utils.hpp"

//libs
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* cpp file for the entire project
#include <tinyobjloader/tiny_obj_loader.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/tinygltf/tiny_gltf.h"
// #include "lib/tinygltf/stb_image.h"

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
    UDModel::UDModel(UDRenderer& renderer, const std::string& filepath, const std::string& texturePath) :
        renderer{ renderer }, filepath{ filepath }, texturePath{ texturePath }, device{
        renderer.getDevice()
        } {
        loadData();
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

        Builder builder{ device };
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
        else if (!texturePath.empty()) {
            createTextureImage();
            createTextureImageView();
            createTextureSampler();
        }

        dataLoaded = true;

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

        // attributeDescriptions.push_back(
        //     { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) }
        // );
        // attributeDescriptions.push_back(
        //     { 0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) }
        // );
        // attributeDescriptions.push_back(
        //     { 0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) }
        // );
        // attributeDescriptions.push_back(
        //     { 0, 3, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) }
        // );

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
            if (!indexBuffer) {
                throw std::runtime_error("Index buffer not created");
            }
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer->getBuffer(), 0, VK_INDEX_TYPE_UINT32);
        }
    }

    void UDModel::draw(VkCommandBuffer commandBuffer) {
        loadData(); // Load data on demand

        if (hasIndexBuffer) {
            if (indexCount == 0) {
                throw std::runtime_error("Index count is zero");
            }
            vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
        }
        else {
            if (vertexCount == 0) {
                throw std::runtime_error("Vertex count is zero");
            }
            vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
        }
    }


    void UDModel::createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(texturePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }
        createTextureImageFromPixels(pixels, texWidth, texHeight);
        stbi_image_free(pixels);
    }

    void UDModel::createTextureImage(const tinygltf::Image& image) {
        int texWidth = image.width;
        int texHeight = image.height;
        int texChannels = image.component;
        const unsigned char* pixels = image.image.data();
        if (!pixels) {
            throw std::runtime_error("failed to load texture image from GLTF!");
        }
        createTextureImageFromPixels(pixels, texWidth, texHeight);
    }

    /*
        Parte desse codigo esta em src/swap_chain.cpp e src/device.cpp. O ideal seria
        colocar esses metodos juntos em um arquivo chamado texture.cpp ou algo do tipo. ou aqui mesmo
    */
    // void UDModel::createTextureImage() {
    //     int texWidth, texHeight, texChannels;
    //     stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    //     VkDeviceSize imageSize = texWidth * texHeight * 4;

    //     if (!pixels) {
    //         throw std::runtime_error("failed to load texture image!");
    //     }

    //     UDBuffer stagingBuffer{
    //         device,
    //         imageSize,
    //         1,
    //         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    //         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    //     };
    //     stagingBuffer.map();
    // stagingBuffer.writeToBuffer(reinterpret_cast<void*>(const_cast<unsigned char*>(pixels)));

    //     stbi_image_free(pixels);

    //     VkImageCreateInfo imageInfo{};
    //     imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    //     imageInfo.imageType = VK_IMAGE_TYPE_2D;
    //     imageInfo.extent.width = texWidth;
    //     imageInfo.extent.height = texHeight;
    //     imageInfo.extent.depth = 1;
    //     imageInfo.mipLevels = 1;
    //     imageInfo.arrayLayers = 1;
    //     imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    //     imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    //     imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    //     imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    //     imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    //     imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    //     device.createImageWithInfo(
    //         imageInfo,
    //         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    //         textureImage,
    //         textureImageMemory
    //     );

    //     device.transitionImageLayout(
    //         textureImage,
    //         VK_FORMAT_R8G8B8A8_SRGB,
    //         VK_IMAGE_LAYOUT_UNDEFINED, // For now we don't care about the its contents
    //         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    //     );
    //     device.copyBufferToImage(stagingBuffer.getBuffer(),
    //         textureImage,
    //         static_cast<uint32_t>(texWidth),
    //         static_cast<uint32_t>(texHeight)
    //     );
    //     device.transitionImageLayout(
    //         textureImage,
    //         VK_FORMAT_R8G8B8A8_SRGB,
    //         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    //         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    //     );
    // }

    void UDModel::createTextureImageFromPixels(const unsigned char* pixels, int texWidth, int texHeight) {
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        UDBuffer stagingBuffer{
            device,
            imageSize,
            1,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        };
        stagingBuffer.map();
        // stagingBuffer.writeToBuffer(reinterpret_cast<void*>(const_cast<unsigned char*>(pixels)));
        stagingBuffer.writeToBuffer(reinterpret_cast<void*>(const_cast<unsigned char*>(pixels)));


        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = texWidth;
        imageInfo.extent.height = texHeight;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        device.createImageWithInfo(
            imageInfo,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            textureImage,
            textureImageMemory
        );

        device.transitionImageLayout(
            textureImage,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );
        device.copyBufferToImage(stagingBuffer.getBuffer(),
            textureImage,
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight)
        );
        device.transitionImageLayout(
            textureImage,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }


    void UDModel::createTextureImageView() {
        textureImageView = device.createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
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
        samplerInfo.maxAnisotropy = device.properties.limits.maxSamplerAnisotropy;

        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

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
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1] // Flip the y-axis
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
    void UDModel::updateDescriptorSets(
        std::vector<std::unique_ptr<UDBuffer>>& uboBuffers,
        std::vector<VkDescriptorSet>& descriptorSets,
        UDDescriptorSetLayout& globalSetLayout, UDDescriptorPool& globalPool) {
        for (int i = 0; i < descriptorSets.size(); i++) {
            auto bufferInfo = uboBuffers[i]->descriptorInfo();

            UDDescriptorWriter writer(globalSetLayout, globalPool);
            writer.writeBuffer(0, &bufferInfo);

            if (hasBoundTexture()) { // Check if the model has a texture
                VkDescriptorImageInfo textureImageInfo{};
                textureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                textureImageInfo.imageView = getTextureImageView();
                textureImageInfo.sampler = getTextureSampler();
                writer.writeImage(1, &textureImageInfo);
            }

            if (!writer.build(descriptorSets[i])) {
                throw std::runtime_error("Failed to build descriptor set " + std::to_string(i));
            }
        }
    }
}

#include "pipeline.hpp"

#include <fstream>
#include <stdexcept>
#include <iostream>

namespace ud {
    UDPipeline::UDPipeline(
        UDDevice &device,
        const std::string& vertFilepath,
        const std::string& fragFilepath,
        const PipelineConfigInfo& configInfo
    ) : udDevice(device) {
        createGraphicsPipeline(vertFilepath, fragFilepath, configInfo);
    }

    std::vector<char> UDPipeline::readFile(const std::string& filepath) {
        // ate flag: start reading at the end of the file to get the file size
        // binary flag: read the file as binary
        std::ifstream file(filepath, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        // size_t fileSize = (size_t) file.tellg();
        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    void UDPipeline::createGraphicsPipeline(
        const std::string& vertFilepath, 
        const std::string& fragFilepath,
        const PipelineConfigInfo& configInfo
    ) {
        auto vertCode = readFile(vertFilepath);
        auto fragCode = readFile(fragFilepath);

        std::cout << "Vertex Shader Code Size: " << vertCode.size() <<  std::endl;
        std::cout << "Fragment Shader Code Size: " << fragCode.size() <<  std::endl;
    }

    void UDPipeline::createShaderModule(const std::vector<char>& code, VkShaderModule* shaderModule) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        // reinterpret_cast is used to convert the char* to uint32_t* because the code is a char* 
        // but vulkan expects the code to be a uint32_t*
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        // Slighly different from the video tutorial
        VkShaderModule shaderModule_;
        if (vkCreateShaderModule(udDevice.device(), &createInfo, nullptr, &shaderModule_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }

        *shaderModule = shaderModule_;
    }

    PipelineConfigInfo UDPipeline::defaultPipelineConfigInfo(uint32_t width, uint32_t height) {
        PipelineConfigInfo configInfo{};

        return configInfo;
    }
}
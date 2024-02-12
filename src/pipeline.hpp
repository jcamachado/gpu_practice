#pragma once

#include "device.hpp"

// std
#include <string>
#include <vector>


namespace ud {

    /*
        It is outside the Pipeline class to make the application layer code to be 
        easily able to configure the pipeline completely, as well as to share
        the configuration between different pipelines.
    */
    struct PipelineConfigInfo {
        VkViewport viewport;
        VkRect2D scissor;
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
        VkPipelineRasterizationStateCreateInfo rasterizationInfo;
        VkPipelineMultisampleStateCreateInfo multisampleInfo;
        VkPipelineColorBlendAttachmentState colorBlendAttachment;
        VkPipelineColorBlendStateCreateInfo colorBlendInfo;
        VkPipelineDepthStencilStateCreateInfo depthStencilInfo;    
        VkPipelineLayout pipelineLayout = nullptr;
        VkRenderPass renderPass = nullptr;
        uint32_t subpass = 0;
    };


    class UDPipeline {
        public:
            UDPipeline(
                UDDevice &device, 
                const std::string& vertFilepath, 
                const std::string& fragFilepath,
                const PipelineConfigInfo& configInfo
            );
            ~UDPipeline();

            UDPipeline(const UDPipeline&) = delete;
            UDPipeline& operator=(const UDPipeline&) = delete;

            void bind(VkCommandBuffer commandBuffer);
            static PipelineConfigInfo defaultPipelineConfigInfo(uint32_t width, uint32_t height);

        private:
            // void createGraphicsPipeline();
            static std::vector<char> readFile(const std::string& filepath);

            void createGraphicsPipeline(
                const std::string& vertFilepath, 
                const std::string& fragFilepath,
                const PipelineConfigInfo& configInfo
                );
            
            void createShaderModule(const std::vector<char>& code, VkShaderModule* shaderModule);
            
            /*
                This reference is potentially memory unsafe, if the device is released before the pipeline
                is released. This is because the pipeline will still have a reference to the device,
            */
            UDDevice& udDevice; 
            VkPipeline graphicsPipeline;    // Handle to the pipeline object
            VkShaderModule vertShaderModule;    // Handle to the vertex shader module
            VkShaderModule fragShaderModule;    // Handle to the fragment shader module
    };
}
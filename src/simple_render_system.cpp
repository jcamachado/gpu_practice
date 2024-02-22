#include "simple_render_system.hpp"


#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>    

// std
#include <array>
#include <stdexcept>

namespace ud {

    struct SimplePushConstantData { // each member occupies 16 bytes even if less than 16 bytes, the remainder bytes are padding
        glm::mat4 transform{1.0f};      // 64 bytes
        glm::mat4 normalMatrix{1.0f};    //
    };

    SimpleRenderSystem::SimpleRenderSystem(UDDevice& device, VkRenderPass renderPass): udDevice(device) {
        createPipelineLayout();
        createPipeline(renderPass);
    }

    SimpleRenderSystem::~SimpleRenderSystem() { vkDestroyPipelineLayout(udDevice.device(), pipelineLayout, nullptr); }

    void SimpleRenderSystem::createPipelineLayout() {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SimplePushConstantData);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pSetLayouts = nullptr;
        pipelineLayoutInfo.pushConstantRangeCount = 1;  // Related to pushConstantRange above
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        if (vkCreatePipelineLayout(udDevice.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    void SimpleRenderSystem::createPipeline(VkRenderPass renderPass) {
        assert(pipelineLayout != nullptr && "Cannot create pipeline before pipeline layout");

        PipelineConfigInfo pipelineConfig{};
        UDPipeline::defaultPipelineConfigInfo(pipelineConfig);
        pipelineConfig.renderPass = renderPass;
        pipelineConfig.pipelineLayout = pipelineLayout;
        udPipeline = std::make_unique<UDPipeline>(
            udDevice,
            "src/shaders/simple_shader.vert.spv",
            "src/shaders/simple_shader.frag.spv",
            pipelineConfig);
    }
    
    void SimpleRenderSystem::renderGameObjects(
        FrameInfo &frameInfo,
        std::vector<UDGameObject> &gameObjects
    ) {
        udPipeline->bind(frameInfo.commandBuffer);

        auto projectionView = frameInfo.camera.getProjection() * frameInfo.camera.getView();

        for (auto& obj : gameObjects) {
            SimplePushConstantData push{};
            auto modelMatrix = obj.transform.mat4();
            push.transform = projectionView * modelMatrix; // TODO move matrix multiplication to the vertex shader
            push.normalMatrix = obj.transform.normalMatrix(); // GLM converts the mat3 to a mat4
            vkCmdPushConstants(
                frameInfo.commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(SimplePushConstantData),
                &push
            );
            obj.model->bind(frameInfo.commandBuffer);
            obj.model->draw(frameInfo.commandBuffer);
        }
    }
} 

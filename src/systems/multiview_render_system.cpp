#include "multiview_render_system.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>    

// std
#include <array>
#include <stdexcept>

namespace ud {

    struct SimplePushConstantData { // each member occupies 16 bytes even if less than 16 bytes, the remainder bytes are padding
        glm::mat4 modelMatrix{ 1.0f };      // 64 bytes
        glm::mat4 normalMatrix{ 1.0f };    //
    };

    MultiViewRenderSystem::MultiViewRenderSystem(UDDevice& device,
        VkRenderPass renderPass,
        VkDescriptorSetLayout globalSetLayout) : udDevice(device)
    {
        createPipelineLayout(globalSetLayout);
        createPipeline(renderPass);
    }

    MultiViewRenderSystem::~MultiViewRenderSystem() {
        vkDestroyPipelineLayout(udDevice.device(), pipelineLayout, nullptr);
    }

    void MultiViewRenderSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SimplePushConstantData);

        std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { globalSetLayout };

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 1;  // Related to pushConstantRange above
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        if (vkCreatePipelineLayout(udDevice.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    void MultiViewRenderSystem::createPipeline(VkRenderPass renderPass) {
        assert(pipelineLayout != nullptr && "Cannot create pipeline before pipeline layout");

        PipelineConfigInfo pipelineConfig{};
        UDPipeline::defaultPipelineConfigInfo(pipelineConfig);
        pipelineConfig.renderPass = renderPass;
        pipelineConfig.pipelineLayout = pipelineLayout;

        // Enable dynamic state for viewports and scissors
        pipelineConfig.viewportInfo.viewportCount = 2;
        pipelineConfig.viewportInfo.scissorCount = 2;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        pipelineConfig.dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        pipelineConfig.dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        pipelineConfig.dynamicStateInfo.pDynamicStates = dynamicStates.data();

        udPipeline = std::make_unique<UDPipeline>(
            udDevice,
            "build/shaders/multiview.vert.spv",
            "build/shaders/multiview.frag.spv",
            pipelineConfig);
    }

    void MultiViewRenderSystem::renderGameObjects(
        FrameInfo& frameInfo,
        const UDCamera& camera,
        const int eyeIndex
    ) {
        udPipeline->bind(frameInfo.commandBuffer);

        // Must specify the starting set
        vkCmdBindDescriptorSets(
            frameInfo.commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            0,
            1,
            &frameInfo.globalDescriptorSet,
            0,
            nullptr
        );


        // Render for eye
        if (eyeIndex == 0) {
            frameInfo.leftEyeCamera = &camera;
        }
        else {
            frameInfo.rightEyeCamera = &camera;
        }
        render(frameInfo);
    }

    void MultiViewRenderSystem::render(FrameInfo& frameInfo) {
        for (auto& kv : frameInfo.gameObjects) {
            auto& obj = kv.second;
            if (obj.model == nullptr) {
                continue;
            }
            SimplePushConstantData push{};
            push.modelMatrix = obj.transform.mat4(); // TODO move matrix multiplication to the vertex shader
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
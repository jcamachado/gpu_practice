#include "simple_render_system.hpp"


#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>    

// std
#include <array>
#include <stdexcept>

namespace ud {

    struct SimplePushConstantData {
        glm::mat4 transform{1.0f};      // 16 bytes identity matrix
        alignas(16) glm::vec3 color;    // 12 bytes
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
        VkCommandBuffer commandBuffer, 
        std::vector<UDGameObject> &gameObjects, 
        const UDCamera &camera
    ) {
        udPipeline->bind(commandBuffer);

        auto projectionView = camera.getProjection() * camera.getView();

        for (auto& obj : gameObjects) {
            obj.transform.rotation.y = glm::mod(obj.transform.rotation.y + 0.01f, glm::two_pi<float>()); // Full circle rotation
            obj.transform.rotation.x = glm::mod(obj.transform.rotation.x + 0.005f, glm::two_pi<float>()); // Full circle rotation

            SimplePushConstantData push{};
            push.color = obj.color;
            push.transform = projectionView * obj.transform.mat4(); // TODO move matrix multiplication to the vertex shader
            vkCmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(SimplePushConstantData),
                &push
            );
            obj.model->bind(commandBuffer);
            obj.model->draw(commandBuffer);
        }
    }
} 

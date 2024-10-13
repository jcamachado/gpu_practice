#include "point_light_system.hpp"


#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>    

// std
#include <array>
#include <stdexcept>

namespace ud {

    // Push constants are a way to pass data to a shader from the CPU
    // They are small and have a limited size (128 bytes)
    // They are faster than uniforms, but they are not as flexible
    struct PointLightPushConstants {
        glm::vec4 position{};
        glm::vec4 color{};
        float radius;
        int eyeIndex;
    };

    PointLightSystem::PointLightSystem(UDDevice& device,
        VkRenderPass renderPass,
        VkDescriptorSetLayout globalSetLayout) : udDevice(device)
    {
        createPipelineLayout(globalSetLayout);
        createPipeline(renderPass);
    }

    PointLightSystem::~PointLightSystem() { vkDestroyPipelineLayout(udDevice.device(), pipelineLayout, nullptr); }

    void PointLightSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PointLightPushConstants);

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

    void PointLightSystem::createPipeline(VkRenderPass renderPass) {
        assert(pipelineLayout != nullptr && "Cannot create pipeline before pipeline layout");

        PipelineConfigInfo pipelineConfig{};
        UDPipeline::defaultPipelineConfigInfo(pipelineConfig);
        pipelineConfig.attributeDescriptions.clear();
        pipelineConfig.bindingDescriptions.clear();
        pipelineConfig.renderPass = renderPass;
        pipelineConfig.pipelineLayout = pipelineLayout;
        udPipeline = std::make_unique<UDPipeline>(
            udDevice,
            "build/shaders/point_light.vert.spv",
            "build/shaders/point_light.frag.spv",
            pipelineConfig);
    }

    void PointLightSystem::update(FrameInfo& frameInfo, GlobalUBO& ubo) {
        auto rotateLight = glm::rotate(glm::mat4(1.0f), frameInfo.frameTime, glm::vec3(0.0f, -1.0f, 0.0f));

        int lightIndex = 0;
        for (auto& kv : frameInfo.gameObjects) {
            auto& gameObject = kv.second;
            if (gameObject.pointLight == nullptr) continue;

            assert(lightIndex < MAX_LIGHTS && "Point lights exceed maximum number of lights!");

            // update light position
            gameObject.transform.translation = glm::vec3(rotateLight * glm::vec4(gameObject.transform.translation, 1.0f));

            // copy light to ubo
            ubo.pointLights[lightIndex].position = glm::vec4(gameObject.transform.translation, 1.0f);
            ubo.pointLights[lightIndex].color = glm::vec4(gameObject.color, gameObject.pointLight->lightIntensity);
            lightIndex += 1;
        }
        ubo.numLights = lightIndex;
    }

    void PointLightSystem::render(FrameInfo& frameInfo) {
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

        for (auto& kv : frameInfo.gameObjects) {
            auto& gameObject = kv.second;
            if (gameObject.pointLight == nullptr) continue;

            PointLightPushConstants push{};
            push.position = glm::vec4(gameObject.transform.translation, 1.0f);
            push.color = glm::vec4(gameObject.color, gameObject.pointLight->lightIntensity);
            push.radius = gameObject.transform.scale.x;

            vkCmdPushConstants(
                frameInfo.commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(PointLightPushConstants),
                &push
            );
            // No need to draw model objects. 
            vkCmdDraw(frameInfo.commandBuffer, 6, 1, 0, 0);
        }
    }

    // void PointLightSystem::render(FrameInfo& frameInfo, const UDCamera& leftEyeCamera, const UDCamera& rightEyeCamera) {
    //     for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
    //         // Set the eye index in the push constants
    //         PointLightPushConstants push{};
    //         push.eyeIndex = eyeIndex;

    //         // Bind the pipeline and descriptor sets
    //         udPipeline->bind(frameInfo.commandBuffer);
    //         vkCmdBindDescriptorSets(
    //             frameInfo.commandBuffer,
    //             VK_PIPELINE_BIND_POINT_GRAPHICS,
    //             pipelineLayout,
    //             0,
    //             1,
    //             &frameInfo.globalDescriptorSet,
    //             0,
    //             nullptr
    //         );

    //         // Select the appropriate camera for the current eye
    //         const UDCamera& currentCamera = (eyeIndex == 0) ? leftEyeCamera : rightEyeCamera;

    //         // Render the scene for the current eye
    //         for (auto& kv : frameInfo.gameObjects) {
    //             auto& gameObject = kv.second;
    //             if (gameObject.pointLight == nullptr) continue;

    //             push.position = glm::vec4(gameObject.transform.translation, 1.0f);
    //             push.color = glm::vec4(gameObject.color, gameObject.pointLight->lightIntensity);
    //             push.radius = gameObject.transform.scale.x;

    //             vkCmdPushConstants(
    //                 frameInfo.commandBuffer,
    //                 pipelineLayout,
    //                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    //                 0,
    //                 sizeof(PointLightPushConstants),
    //                 &push
    //             );
    //             vkCmdDraw(frameInfo.commandBuffer, 6, 1, 0, 0);
    //         }
    //     }
    // }
}

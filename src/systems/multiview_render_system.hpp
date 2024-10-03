#pragma once

#include "camera.hpp"   
#include "device.hpp"
#include "frame_info.hpp"
#include "game_object.hpp"
#include "pipeline.hpp"

// std
#include <memory>
#include <vector>

namespace ud {
    /*
        ### fluxo basico (tirado de outro video) para 1 viewport ###
        (begin)
        begin frame / clear
        start recording command buffer

        load global uniforms (bloco para objects) // once per frame
            load instance uniforms
            load local uniforms
            draw

        (end)
        end recording command buffer
        submit to queue
        present
        ###

        ### fluxo basico para 2 viewports ###
        (begin)
        begin frame / clear
        start recording command buffer  (fazer esse passo 2 vezes)

        load global uniforms (bloco para objects) // once per frame
            load instance uniforms
            load local uniforms
            draw

        (end)
        end recording command buffer (fazer esse passo 2 vezes)
        submit to queue (fazer esse passo 2 vezes)
        present
        ###
    */
    class MultiViewRenderSystem {
    public:
        // struct MultiviewPass {
        //     struct FrameBufferAttachment {
        //         VkImage image{ VK_NULL_HANDLE };
        //         VkDeviceMemory memory{ VK_NULL_HANDLE };
        //         VkImageView view{ VK_NULL_HANDLE };
        //     } color, depth;
        //     VkFramebuffer frameBuffer{ VK_NULL_HANDLE };
        //     VkRenderPass renderPass{ VK_NULL_HANDLE };
        //     VkDescriptorImageInfo descriptor{ VK_NULL_HANDLE };
        //     VkSampler sampler{ VK_NULL_HANDLE };
        //     VkSemaphore semaphore{ VK_NULL_HANDLE };
        //     std::vector<VkCommandBuffer> commandBuffers{};
        //     std::vector<VkFence> waitFences{};
        // } multiviewPass;

        // struct UniformData {
        //     glm::mat4 projection[2];
        //     glm::mat4 modelview[2];
        //     glm::vec4 lightPos = glm::vec4(-2.5f, -3.5f, 0.0f, 1.0f);
        //     float distortionAlpha = 0.2f;
        // } uniformData;
        MultiViewRenderSystem(UDDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout);
        ~MultiViewRenderSystem();

        MultiViewRenderSystem(const MultiViewRenderSystem&) = delete;
        MultiViewRenderSystem& operator=(const MultiViewRenderSystem&) = delete;

        void renderGameObjects(FrameInfo& frameInfo, const UDCamera& leftEyeCamera, const UDCamera& rightEyeCamera);
        void renderForEye(FrameInfo& frameInfo, int eyeIndex);

    private:
        void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
        void createPipeline(VkRenderPass renderPass);

        UDDevice& udDevice;

        std::unique_ptr<UDPipeline> udPipeline;
        VkPipelineLayout pipelineLayout;
    };

    // Camera and view properties
    // float eyeSeparation = 0.08f;
    // const float focalLength = 0.5f;
    // const float fov = 90.0f;
    // const float zNear = 0.1f;
    // const float zFar = 256.0f;

};
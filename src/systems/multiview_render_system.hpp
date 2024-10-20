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

        void renderGameObjects(FrameInfo& frameInfo);
        void render(FrameInfo& frameInfo);

    private:
        void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
        void createPipeline(VkRenderPass renderPass);

        UDDevice& udDevice;

        std::unique_ptr<UDPipeline> udPipeline;
        VkPipelineLayout pipelineLayout;
    };
};
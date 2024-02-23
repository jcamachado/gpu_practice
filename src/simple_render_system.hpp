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
        Render System lifecyle is not tied to the render passes'.
    */
    class SimpleRenderSystem {
        public:
            SimpleRenderSystem(UDDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout);
            ~SimpleRenderSystem();

            SimpleRenderSystem(const SimpleRenderSystem&) = delete;
            SimpleRenderSystem& operator=(const SimpleRenderSystem&) = delete; 

            void renderGameObjects(
                FrameInfo &frameInfo,
                std::vector<UDGameObject> &gameObjects
            );
        private:
            void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
            void createPipeline(VkRenderPass renderPass); // Renderpass will be used specifically to create the pipeline

            UDDevice &udDevice;
            // unique_ptr is a smart pointer that manages another object through a pointer and 
            // disposes of that object when the unique_ptr goes out of scope
            std::unique_ptr<UDPipeline> udPipeline;
            VkPipelineLayout pipelineLayout;
    };
};
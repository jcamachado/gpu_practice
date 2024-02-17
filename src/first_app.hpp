#pragma once

#include "device.hpp"
#include "model.hpp" 
#include "pipeline.hpp"
#include "swap_chain.hpp"
#include "window.hpp"

// std
#include <memory>
#include <vector>

namespace ud {
    class FirstApp {
        public:
            static constexpr int WIDTH = 800;
            static constexpr int HEIGHT = 600;

            FirstApp();
            ~FirstApp();

            FirstApp(const FirstApp&) = delete;
            FirstApp& operator=(const FirstApp&) = delete;

            void run();

        private:
            void loadModels();
            void createPipelineLayout();
            void createPipeline();
            void createCommandBuffers();
            void freeCommandBuffers();
            void drawFrame();
            void recreateSwapChain();
            void recordCommandBuffer(int imageIndex);

            UDWindow udWindow{WIDTH, HEIGHT, "Vulkan"};
            UDDevice udDevice{udWindow};
            // unique_ptr is a smart pointer that manages another object through a pointer and 
            // disposes of that object when the unique_ptr goes out of scope
            std::unique_ptr<UDSwapChain> udSwapChain;
            std::unique_ptr<UDPipeline> udPipeline;
            VkPipelineLayout pipelineLayout;
            std::vector<VkCommandBuffer> commandBuffers;
            std::unique_ptr<UDModel> udModel;
    };
};
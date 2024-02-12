#pragma once

#include "device.hpp"
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
            void createPipelineLayout();
            void createPipeline();
            void createCommandBuffers();
            void drawFrame();

            UDWindow udWindow{WIDTH, HEIGHT, "Vulkan"};
            UDDevice udDevice{udWindow};
            UDSwapChain udSwapChain{udDevice, udWindow.getExtent()};
            // unique_ptr is a smart pointer that manages another object through a pointer and 
            // disposes of that object when the unique_ptr goes out of scope
            std::unique_ptr<UDPipeline> udPipeline;
            VkPipelineLayout pipelineLayout;
            std::vector<VkCommandBuffer> commandBuffers;
    };
};
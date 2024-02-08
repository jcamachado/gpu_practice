#pragma once

#include "device.hpp"
#include "pipeline.hpp"
#include "window.hpp"

namespace ud {
    class FirstApp {
        public:
            static constexpr int WIDTH = 800;
            static constexpr int HEIGHT = 600;

            void run();

        private:
            // void initWindow();
            // void initVulkan();
            // void mainLoop();
            // void cleanup();

            UDWindow udWindow{WIDTH, HEIGHT, "Vulkan"};
            UDDevice udDevice{udWindow};
            UDPipeline udPipeline
            { 
                udDevice, 
                "src/shaders/simple_shader.vert.spv", 
                "src/shaders/simple_shader.frag.spv", 
                UDPipeline::defaultPipelineConfigInfo(WIDTH, HEIGHT)
            };
    };
};
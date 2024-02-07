#pragma once

#include "pipeline.hpp"
#include "window.hpp"

namespace uffdejavu {
    class FirstApp {
        public:
            static constexpr int WIDTH = 800;
            static constexpr int HEIGHT = 600;

            void run();

        private:
            void initWindow();
            void initVulkan();
            void mainLoop();
            void cleanup();

            UDWindow window{WIDTH, HEIGHT, "Vulkan"};
            UDPipeline pipeline{"src/shaders/simple_shader.vert.spv", "src/shaders/simple_shader.frag.spv"};
    };
};
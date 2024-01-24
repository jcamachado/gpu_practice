#pragma once

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

            Window window{WIDTH, HEIGHT, "Vulkan"};
    };
};
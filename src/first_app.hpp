#pragma once

#include "device.hpp"
#include "game_object.hpp"
#include "renderer.hpp"
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
            void loadGameObjects();

            UDWindow udWindow{WIDTH, HEIGHT, "Vulkan"};
            UDDevice udDevice{udWindow};
            UDRenderer udRenderer{udWindow, udDevice};
            // unique_ptr is a smart pointer that manages another object through a pointer and 
            // disposes of that object when the unique_ptr goes out of scope
            std::vector<UDGameObject> gameObjects;
    };
};
#pragma once

#include "descriptors.hpp"
#include "device.hpp"
#include "game_object.hpp"
#include "renderer.hpp"
#include "window.hpp"

// std
#include <memory>
#include <vector>

#define NEAR_PLANE 0.1f
#define FAR_PLANE 1000.0f
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
        // Declaration order matters. Allocation happens from top to bottom, and deallocation from bottom to top
        void loadGameObjects();

        UDWindow udWindow{ WIDTH, HEIGHT, "Vulkan" };
        UDDevice udDevice{ udWindow };
        UDRenderer udRenderer{ udWindow, udDevice };
        // unique_ptr is a smart pointer that manages another object through a pointer and 
        // disposes of that object when the unique_ptr goes out of scope
        std::unique_ptr<UDDescriptorPool> globalPool{}; // Must be declared after device bc must be destroyed before the device
        UDGameObject::Map gameObjects;
    };
};
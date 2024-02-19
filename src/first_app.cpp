#include "first_app.hpp"

#include "simple_render_system.hpp"


#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>    

// std
#include <array>
#include <stdexcept>

namespace ud {
    FirstApp::FirstApp() {
        loadGameObjects();
    }

    FirstApp::~FirstApp() {}

    void FirstApp::run() {
        SimpleRenderSystem simpleRenderSystem{udDevice, udRenderer.getSwapChainRenderPass()};

        while (!udWindow.shouldClose()) {
            glfwPollEvents();
            if (auto commandBuffer = udRenderer.beginFrame()) { // If nullptr, swapchain needs to be recreated
                udRenderer.beginSwapChainRenderPass(commandBuffer);
                simpleRenderSystem.renderGameObjects(commandBuffer, gameObjects);
                udRenderer.endSwapChainRenderPass(commandBuffer);
                udRenderer.endFrame();
            }
        }

        vkDeviceWaitIdle(udDevice.device());
    }

    void FirstApp::loadGameObjects() {
        /*
            3 pairs os brackets. 
            The outermost pair is for the vector, 
            the second pair is for each Vertex struct, 
            and the innermost pair is for the glm::vec2
        */ 
        std::vector<UDModel::Vertex> vertices{
            {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
            {{0.5f,  0.5f},  {0.0f, 1.0f, 0.0f}},
            {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
        };
        auto udModel = std::make_shared<UDModel>(udDevice, vertices);

        auto triangle = UDGameObject::createGameObject();
        triangle.model = udModel;
        triangle.color = {0.1f, 0.8f, 0.1f};
        triangle.transform2d.translation.x = 0.2f;
        triangle.transform2d.scale = {2.0f, 0.5f};
        triangle.transform2d.rotation = 0.25f * glm::two_pi<float>();   // 360/4 = 90 degrees

        gameObjects.push_back(std::move(triangle));
    }
} 

#include "first_app.hpp"

#include "camera.hpp"
#include "keyboard_movement_controller.hpp"
#include "simple_render_system.hpp"


#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>    

// std
#include <array>
#include <chrono>
#include <stdexcept>

namespace ud {
    FirstApp::FirstApp() {
        loadGameObjects();
    }

    FirstApp::~FirstApp() {}

    void FirstApp::run() {  // The main loop
        SimpleRenderSystem simpleRenderSystem{udDevice, udRenderer.getSwapChainRenderPass()};
        UDCamera camera{};

        auto viewerObject = UDGameObject::createGameObject(); // stores camera current state
        KeyboardMovementController cameraController{};

        auto currentTime = std::chrono::high_resolution_clock::now();

        while (!udWindow.shouldClose()) {
            glfwPollEvents();
            auto newTime = std::chrono::high_resolution_clock::now();
            float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(newTime - currentTime).count();
            currentTime = newTime;

            cameraController.moveInPlaneXZ(udWindow.getGLFWWindow(), frameTime, viewerObject);
            camera.setViewYXZ(viewerObject.transform.translation, viewerObject.transform.rotation);

            float aspect = udRenderer.getAspectRatio();
            // Inside the loop, the orthographic projection will be kept to date with the aspect ratio
            // camera.setOrthographicProjection(-aspect, aspect, -1.0f, 1.0f, -1.0f, 1.0f); // Works only when bottom = -1 and top = 1
            camera.setPerspectiveProjection(glm::radians(50.0f), aspect, 0.1f, 10.0f);
            if (auto commandBuffer = udRenderer.beginFrame()) { // If nullptr, swapchain needs to be recreated
                udRenderer.beginSwapChainRenderPass(commandBuffer);
                simpleRenderSystem.renderGameObjects(commandBuffer, gameObjects, camera);
                udRenderer.endSwapChainRenderPass(commandBuffer);
                udRenderer.endFrame();
            }
        }

        vkDeviceWaitIdle(udDevice.device());
    }


    void FirstApp::loadGameObjects() {
        std::shared_ptr<UDModel> udModel = UDModel::createModelFromFile(udDevice, "src/models/flat_vase.obj");

        auto gameObject = UDGameObject::createGameObject();
        gameObject.model = udModel;
        gameObject.transform.translation = {-0.5f, 0.5f, 2.5f};
        gameObject.transform.scale = glm::vec3(3.0f, 1.5f, 3.0f);
        gameObjects.push_back(std::move(gameObject));

        udModel = UDModel::createModelFromFile(udDevice, "src/models/smooth_vase.obj");

        gameObject = UDGameObject::createGameObject();
        gameObject.model = udModel;
        gameObject.transform.translation = {0.5f, 0.5f, 2.5f};
        gameObject.transform.scale = glm::vec3(3.0f, 1.5f, 3.0f);
        gameObjects.push_back(std::move(gameObject));
    }
} 

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

    std::unique_ptr<UDModel> createCubeModel(UDDevice &device, glm::vec3 offset){
        UDModel::Builder modelBuilder{};

        modelBuilder.vertices = {
            // left face (white)
            {{-.5f, -.5f, -.5f}, {.9f, .9f, .9f}},
            {{-.5f, .5f, .5f},   {.9f, .9f, .9f}},
            {{-.5f, -.5f, .5f},  {.9f, .9f, .9f}},
            {{-.5f, .5f, -.5f},  {.9f, .9f, .9f}},
        
            // right face (yellow)
            {{.5f, -.5f, -.5f}, {.8f, .8f, .1f}},
            {{.5f, .5f, .5f}, {.8f, .8f, .1f}},
            {{.5f, -.5f, .5f}, {.8f, .8f, .1f}},
            {{.5f, .5f, -.5f}, {.8f, .8f, .1f}},
        
            // top face (orange, remember y axis points down)
            {{-.5f, -.5f, -.5f}, {.9f, .6f, .1f}},
            {{.5f, -.5f, .5f}, {.9f, .6f, .1f}},
            {{-.5f, -.5f, .5f}, {.9f, .6f, .1f}},
            {{.5f, -.5f, -.5f}, {.9f, .6f, .1f}},
        
            // bottom face (red)
            {{-.5f, .5f, -.5f}, {.8f, .1f, .1f}},
            {{.5f, .5f, .5f}, {.8f, .1f, .1f}},
            {{-.5f, .5f, .5f}, {.8f, .1f, .1f}},
            {{.5f, .5f, -.5f}, {.8f, .1f, .1f}},
        
            // nose face (blue)
            {{-.5f, -.5f, 0.5f}, {.1f, .1f, .8f}},
            {{.5f, .5f, 0.5f}, {.1f, .1f, .8f}},
            {{-.5f, .5f, 0.5f}, {.1f, .1f, .8f}},
            {{.5f, -.5f, 0.5f}, {.1f, .1f, .8f}},
        
            // tail face (green)
            {{-.5f, -.5f, -0.5f}, {.1f, .8f, .1f}},
            {{.5f, .5f, -0.5f}, {.1f, .8f, .1f}},
            {{-.5f, .5f, -0.5f}, {.1f, .8f, .1f}},
            {{.5f, -.5f, -0.5f}, {.1f, .8f, .1f}},
        };

        for (auto& vertex : modelBuilder.vertices) {
            vertex.position += offset;
        }

        // 4 vertices per face, because each face has a different color
        // If we had a single color, we could use 8 vertices
        modelBuilder.indices = {
            0,  1,  2,  0,  3,  1,  4,  5,  6, 
            4,  7,  5,  8,  9,  10, 8,  11, 9,
            12, 13, 14, 12, 15, 13, 16, 17, 18, 
            16, 19, 17, 20, 21, 22, 20, 23, 21
        };

        return std::make_unique<UDModel>(device, modelBuilder);
    }


    void FirstApp::loadGameObjects() {
        std::shared_ptr<UDModel> udModel = createCubeModel(udDevice, {0.0f, 0.0f, 0.0f});

        auto cube = UDGameObject::createGameObject();
        cube.model = udModel;
        cube.transform.translation = {0.0f, 0.0f, 2.5f};
        cube.transform.scale = {0.5f, 0.5f, 0.5f}; // x(-1,1) y(-1,1) z(0,1)
        gameObjects.push_back(std::move(cube));
    }
} 

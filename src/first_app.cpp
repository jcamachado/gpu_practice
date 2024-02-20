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
        std::vector<UDModel::Vertex> vertices{
            // left face (white)
            {{-0.5f, -0.5f, -0.5f}, {.9f, .9f, .9f}},
            {{-0.5f,0.5f, .5f}, {.9f, .9f, .9f}},
            {{-0.5f, -0.5f, .5f}, {.9f, .9f, .9f}},
            {{-0.5f, -0.5f, -0.5f}, {.9f, .9f, .9f}},
            {{-0.5f,0.5f, -0.5f}, {.9f, .9f, .9f}},
            {{-0.5f,0.5f, .5f}, {.9f, .9f, .9f}},
        
            // right face (yellow)
            {{.5f, -0.5f, -0.5f}, {.8f, .8f, .1f}},
            {{.5f,0.5f, .5f}, {.8f, .8f, .1f}},
            {{.5f, -0.5f, .5f}, {.8f, .8f, .1f}},
            {{.5f, -0.5f, -0.5f}, {.8f, .8f, .1f}},
            {{.5f,0.5f, -0.5f}, {.8f, .8f, .1f}},
            {{.5f,0.5f, .5f}, {.8f, .8f, .1f}},
        
            // top face (orange, remember y axis points down)
            {{-0.5f, -0.5f, -0.5f}, {.9f, .6f, .1f}},
            {{.5f, -0.5f, .5f}, {.9f, .6f, .1f}},
            {{-0.5f, -0.5f, .5f}, {.9f, .6f, .1f}},
            {{-0.5f, -0.5f, -0.5f}, {.9f, .6f, .1f}},
            {{.5f, -0.5f, -0.5f}, {.9f, .6f, .1f}},
            {{.5f, -0.5f, .5f}, {.9f, .6f, .1f}},
        
            // bottom face (red)
            {{-0.5f,0.5f, -0.5f}, {.8f, .1f, .1f}},
            {{.5f,0.5f, .5f}, {.8f, .1f, .1f}},
            {{-0.5f,0.5f, .5f}, {.8f, .1f, .1f}},
            {{-0.5f,0.5f, -0.5f}, {.8f, .1f, .1f}},
            {{.5f,0.5f, -0.5f}, {.8f, .1f, .1f}},
            {{.5f,0.5f, .5f}, {.8f, .1f, .1f}},
        
            // nose face (blue)
            {{-0.5f, -0.5f, 0.5f}, {.1f, .1f, .8f}},
            {{.5f,0.5f, 0.5f}, {.1f, .1f, .8f}},
            {{-0.5f,0.5f, 0.5f}, {.1f, .1f, .8f}},
            {{-0.5f, -0.5f, 0.5f}, {.1f, .1f, .8f}},
            {{.5f, -0.5f, 0.5f}, {.1f, .1f, .8f}},
            {{.5f,0.5f, 0.5f}, {.1f, .1f, .8f}},
        
            // tail face (green)
            {{-0.5f, -0.5f, -0.5f}, {.1f, .8f, .1f}},
            {{.5f,0.5f, -0.5f}, {.1f, .8f, .1f}},
            {{-0.5f,0.5f, -0.5f}, {.1f, .8f, .1f}},
            {{-0.5f, -0.5f, -0.5f}, {.1f, .8f, .1f}},
            {{.5f, -0.5f, -0.5f}, {.1f, .8f, .1f}},
            {{.5f,0.5f, -0.5f}, {.1f, .8f, .1f}},
        };

        for (auto& vertex : vertices) {
            vertex.position += offset;
        }

        return std::make_unique<UDModel>(device, vertices);
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

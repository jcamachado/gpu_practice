#include "first_app.hpp"

#include "buffer.hpp"
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
#include <numeric>
#include <stdexcept>

namespace ud {

    struct GlobalUBO { // Uniform Buffer Object
        glm::mat4 projectionView{1.0f};
        glm::vec3 lightDirection = glm::normalize(glm::vec3{1.0f, -3.0f, -1.0f});
    };

    FirstApp::FirstApp() {
        loadGameObjects();
    }

    FirstApp::~FirstApp() {}

    
    void FirstApp::run() {  // The main loop
        // This property is used to align the offset of the uniform buffer object
        std::vector<std::unique_ptr<UDBuffer>> uboBuffers(UDSwapChain::MAX_FRAMES_IN_FLIGHT);
        for (int i = 0; i < uboBuffers.size(); i++) {
            uboBuffers[i] = std::make_unique<UDBuffer>(
                udDevice,                           
                sizeof(GlobalUBO),                  
                1, // Only one instance per buffer
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                udDevice.properties.limits.minUniformBufferOffsetAlignment
            );
            uboBuffers[i]->map();
        }
        /*
            Instance count is the number of frames to be rendered simultaneously
            This way we can safely write to a frames ubo without worrying about 
            possible synchronization

            Double buffering. Frame 0 is being rendered while frame 1 is being prepared
            and vice versa

            Not using Host Coherent Memory because we want to selectively flush parts
            of the memory of the buffer in order to not interfere with the previous
            frame that may still be rendering
        */

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
                int frameIndex = udRenderer.getFrameIndex();
                FrameInfo frameInfo{frameIndex, frameTime, commandBuffer, camera};

                // update
                GlobalUBO ubo{};
                ubo.projectionView = camera.getProjection() * camera.getView();
                uboBuffers[frameIndex]->writeToBuffer(&ubo);
                uboBuffers[frameIndex]->flush();

                // render (draw calls)
                udRenderer.beginSwapChainRenderPass(commandBuffer);
                simpleRenderSystem.renderGameObjects(frameInfo, gameObjects);
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

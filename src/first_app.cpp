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

    struct GlobalUBO { // Uniform Buffer Object (needs alignment)
        glm::mat4 projectionView{ 1.0f };
        glm::vec4 ambientLightColor{ 1.0f, 1.0f, 1.0f, 0.02f };
        glm::vec3 lightPosition{-1.0f};
        alignas(16) glm::vec4 lightColor{1.0f}; // w is the intensity. rgb [0, 1]. r*w, g*w, b*w
    };

    FirstApp::FirstApp() {
        // pool with max 2 sets that contains at most 2 uniform buffers in total
        // Cant have more than sets
        // Cant have more descriptors than those specified in the pool
        // One set can have all the descriptors, but then the pool cannot provide more sets
        // could add more descriptor to the pool using chain call .addPoolSize(..).addPoolSize(..
        globalPool = UDDescriptorPool::Builder(udDevice)
            .setMaxSets(UDSwapChain::MAX_FRAMES_IN_FLIGHT) // 2 sets
            .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, UDSwapChain::MAX_FRAMES_IN_FLIGHT) // 2 uniform descriptors
            .build();
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
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            );
            uboBuffers[i]->map();
        }

        auto globalSetLayout = UDDescriptorSetLayout::Builder(udDevice)
            .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)
            .build();

        std::vector<VkDescriptorSet> globalDescriptorSets(UDSwapChain::MAX_FRAMES_IN_FLIGHT);
        for (int i = 0; i < globalDescriptorSets.size(); i++) {
            auto bufferInfo = uboBuffers[i]->descriptorInfo();
            UDDescriptorWriter(*globalSetLayout, *globalPool)
                .writeBuffer(0, &bufferInfo)
                .build(globalDescriptorSets[i]);
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

        SimpleRenderSystem simpleRenderSystem{ 
            udDevice, 
            udRenderer.getSwapChainRenderPass(), 
            globalSetLayout->getDescriptorSetLayout()
        };
        UDCamera camera{};

        auto viewerObject = UDGameObject::createGameObject(); // stores camera current state
        viewerObject.transform.translation.z = -2.5f;
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
            camera.setPerspectiveProjection(glm::radians(50.0f), aspect, NEAR_PLANE, FAR_PLANE);
            if (auto commandBuffer = udRenderer.beginFrame()) { // If nullptr, swapchain needs to be recreated
                int frameIndex = udRenderer.getFrameIndex();
                FrameInfo frameInfo{ frameIndex, 
                    frameTime, 
                    commandBuffer, 
                    camera , 
                    globalDescriptorSets[frameIndex],
                    gameObjects
                };

                // update
                GlobalUBO ubo{};
                ubo.projectionView = camera.getProjection() * camera.getView();
                uboBuffers[frameIndex]->writeToBuffer(&ubo);
                uboBuffers[frameIndex]->flush();

                // render (draw calls)
                udRenderer.beginSwapChainRenderPass(commandBuffer);
                simpleRenderSystem.renderGameObjects(frameInfo);
                udRenderer.endSwapChainRenderPass(commandBuffer);
                udRenderer.endFrame();
            }
        }

        vkDeviceWaitIdle(udDevice.device());
    }


    void FirstApp::loadGameObjects() {
        std::shared_ptr<UDModel> udModel = UDModel::createModelFromFile(udDevice, "src/models/flat_vase.obj");

        auto flatVase = UDGameObject::createGameObject();
        flatVase.model = udModel;
        flatVase.transform.translation = { -0.5f, 0.5f, 0.0f };
        flatVase.transform.scale = glm::vec3(3.0f, 1.5f, 3.0f);
        gameObjects.emplace(flatVase.getId(), std::move(flatVase));

        udModel = UDModel::createModelFromFile(udDevice, "src/models/smooth_vase.obj");
        auto smoothVase = UDGameObject::createGameObject();
        smoothVase.model = udModel;
        smoothVase.transform.translation = { 0.5f, 0.5f, 0.0f };
        smoothVase.transform.scale = glm::vec3(3.0f, 1.5f, 3.0f);
        gameObjects.emplace(smoothVase.getId(), std::move(smoothVase));

        udModel = UDModel::createModelFromFile(udDevice, "src/models/quad.obj");
        auto floor = UDGameObject::createGameObject();
        floor.model = udModel;
        floor.transform.translation = { 0.0f, 0.5f, 0.0f };
        floor.transform.scale = glm::vec3(3.0f, 1.0f, 3.0f);
        gameObjects.emplace(floor.getId(), std::move(floor));
    }
}

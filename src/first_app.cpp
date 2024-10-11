#include "first_app.hpp"

#include "buffer.hpp"
#include "camera.hpp"
#include "keyboard_movement_controller.hpp"
#include "systems/multiview_render_system.hpp"
#include "systems/point_light_system.hpp"
#include "systems/simple_render_system.hpp"

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
        loadObjects();
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
        PointLightSystem pointLightSystem{
            udDevice,
            udRenderer.getSwapChainRenderPass(),
            globalSetLayout->getDescriptorSetLayout()
        };
        MultiViewRenderSystem multiviewRenderSystem{
            udDevice,
            udRenderer.getSwapChainRenderPass(),
            globalSetLayout->getDescriptorSetLayout()
        };

        // Create two cameras for the left and right eye views
        UDCamera leftEyeCamera{};
        UDCamera rightEyeCamera{};

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

            // Set the view for both cameras
            leftEyeCamera.setViewYXZ(viewerObject.transform.translation, viewerObject.transform.rotation);
            rightEyeCamera.setViewYXZ(viewerObject.transform.translation, viewerObject.transform.rotation);

            float aspect = udRenderer.getAspectRatio();
            // Set the perspective projection for both cameras
            leftEyeCamera.setPerspectiveProjection(glm::radians(50.0f), aspect, NEAR_PLANE, FAR_PLANE);
            rightEyeCamera.setPerspectiveProjection(glm::radians(50.0f), aspect, NEAR_PLANE, FAR_PLANE);

            if (auto commandBuffer = udRenderer.beginFrame()) { // If nullptr, swapchain needs to be recreated
                int frameIndex = udRenderer.getFrameIndex();
                FrameInfo frameInfo{ frameIndex,
                    frameTime,
                    commandBuffer,
                    &leftEyeCamera, // Use pointer to left eye camera
                    globalDescriptorSets[frameIndex],
                    gameObjects
                };

                // update
                GlobalUBO ubo{};
                ubo.projection[0] = leftEyeCamera.getProjection();
                ubo.view[0] = leftEyeCamera.getView();
                ubo.projection[1] = rightEyeCamera.getProjection();
                ubo.view[1] = rightEyeCamera.getView();

                ubo.inverseView[0] = glm::inverse(leftEyeCamera.getView());
                ubo.inverseView[1] = glm::inverse(rightEyeCamera.getView());
                pointLightSystem.update(frameInfo, ubo);
                uboBuffers[frameIndex]->writeToBuffer(&ubo);
                uboBuffers[frameIndex]->flush();

                // render (draw calls)
                udRenderer.beginSwapChainRenderPass(commandBuffer);
                // Set viewports and scissors
                udRenderer.setViewport(commandBuffer);

                // Render using MultiviewRenderSystem
                multiviewRenderSystem.renderGameObjects(frameInfo, leftEyeCamera, rightEyeCamera);
                pointLightSystem.render(frameInfo);

                udRenderer.endSwapChainRenderPass(commandBuffer);
                udRenderer.endFrame();
            }
        }

        vkDeviceWaitIdle(udDevice.device());
    }

    void FirstApp::loadObjects() {
        loadGameObjects();
        loadParticles();
    }

    void FirstApp::loadParticles() {
        // auto particleModel = UDModel::createModelFromFile(udDevice, "models/particle.obj");
        // auto particleTexture = UDTexture::createTextureFromFile(udDevice, "textures/particle.png");
        // auto particleMaterial = UDMaterial::createMaterial();
        // particleMaterial->setDiffuseTexture(particleTexture);
        // particleMaterial->setAmbientColor({ 0.0f, 0.0f, 0.0f });
        // particleMaterial->setDiffuseColor({ 0.0f, 0.0f, 0.0f });
        // particleMaterial->setSpecularColor({ 0.0f, 0.0f, 0.0f });
        // particleMaterial->setSpecularPower(1.0f);
        // auto particle = UDGameObject::createGameObject();
        // particle.model = particleModel;
        // particle.material = particleMaterial;
        // particle.transform.translation = { 0.0f, 0.0f, 0.0f };
        // particle.transform.scale = glm::vec3(0.1f);
        // gameObjects.emplace(particle.getId(), std::move(particle));
        return;
    }

    void FirstApp::loadGameObjects() {
        // Since I wont be modifying the models, I can use a shared pointer
        std::shared_ptr<UDModel> udModel = nullptr;

        // Solids
        placeNewObject(udModel,
            udDevice,
            "models/flat_vase.obj",
            { -0.5f, 0.5f, 0.0f },
            { 3.0f, 1.5f, 3.0f });

        placeNewObject(udModel,
            udDevice,
            "models/smooth_vase.obj",
            { 0.5f, 0.5f, 0.0f },
            { 3.0f, 1.5f, 3.0f });

        placeNewObject(udModel,
            udDevice,
            "models/quad.obj",
            { 0.0f, 0.5f, 0.0f },
            { 3.0f, 1.0f, 3.0f });

        std::vector<glm::vec3> lightColors{
            {1.f, .1f, .1f},
            {.1f, .1f, 1.f},
            {.1f, 1.f, .1f},
            {1.f, 1.f, .1f},
            {.1f, 1.f, 1.f},
            {1.f, 1.f, 1.f}
        };

        // Lights
        for (int i = 0; i < lightColors.size(); i++) {
            auto pointLight = UDGameObject::makePointLight(0.2f);
            pointLight.color = lightColors[i];
            auto rotateLight = glm::rotate(
                glm::mat4(1.0f),
                (i * glm::two_pi<float>()) / lightColors.size(),
                { 0.0f, -1.0f, 0.0f }
            );
            pointLight.transform.translation = glm::vec3(rotateLight * glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f));
            gameObjects.emplace(pointLight.getId(), std::move(pointLight));
        }
    }

    void FirstApp::placeNewObject(std::shared_ptr<UDModel> udModel,
        UDDevice& udDevice,
        const std::string& objFilePath,
        glm::vec3 translation,
        glm::vec3 scale)
    {
        udModel = UDModel::createModelFromFile(udDevice, objFilePath);
        auto newObj = UDGameObject::createGameObject();
        newObj.model = udModel;
        newObj.transform.translation = translation;
        newObj.transform.scale = scale;

        gameObjects.emplace(newObj.getId(), std::move(newObj));
    }
}
#pragma once

#include "device.hpp"
#include "swap_chain.hpp"
#include "window.hpp"

// std
#include <cassert>
#include <memory>
#include <vector>

namespace ud {
    class UDRenderer {
        public:
            UDRenderer(UDWindow& udWindow, UDDevice& udDevice);
            ~UDRenderer();

            UDRenderer(const UDRenderer&) = delete;
            UDRenderer& operator=(const UDRenderer&) = delete;

            bool isFrameInProgress() const { return isFrameStarted; }

            VkCommandBuffer getCurrentCommandBuffer() const { 
                assert(isFrameStarted && "Cannot get command buffer when frame not in progress.");
                return commandBuffers[currentFrameIndex]; 
            }

            int getFrameIndex() const { 
                assert(isFrameStarted && "Cannot get frame index when frame not in progress.");
                return currentFrameIndex; 
            }

            /*
                Since drawframe will be called outside of the class, it will have to be public.
                So, we will make 2 functions to draw frames.
                The first will be called beginFrame and a second function called endFrame().
                And we will need to keep track of the current frame state.

                We will keep beginFrame and BeginSwapChainRenderPass separate, because we may want to
                be able down the line to easily integrate multiple render passes for things like
                reflections, shadows, and post-processing effects.
            */

            VkCommandBuffer beginFrame();
            void endFrame();

            VkRenderPass getSwapChainRenderPass() const { return udSwapChain->getRenderPass(); }
            void beginSwapChainRenderPass(VkCommandBuffer commandBuffer);
            void endSwapChainRenderPass(VkCommandBuffer commandBuffer);

        private:
            void createCommandBuffers();
            void freeCommandBuffers();
            void recreateSwapChain();

            UDWindow& udWindow;
            UDDevice& udDevice;
            // unique_ptr is a smart pointer that manages another object through a pointer and 
            // disposes of that object when the unique_ptr goes out of scope
            std::unique_ptr<UDSwapChain> udSwapChain;
            std::vector<VkCommandBuffer> commandBuffers;

            uint32_t currentImageIndex;
            int currentFrameIndex=0;
            bool isFrameStarted{false};
    };
};
/*
    Disclaimer: In this context, a system is anything that acts upon a subset of a gameobjects' components.
    Example: A rainbow system would act upon the color component of a gameobject.

    There will be entities that are not compatible with a system. The application is responsible
    for ensuring that the system only acts upon compatible entities.

    Renderer:
        - SwapChain
        - CommandBuffers
        - Draw

    SimpleRenderSystem
        - pipeline
        - pipelineLayout
        - SimplePushConstantData
        - render GameObjects

    Application will have 1 renderer and many Render Systems
    Examples of render systems: MaterialRenderSystem, MeshRenderSystem, LightRenderSystem, etc
*/
#pragma once

#include "buffer.hpp"
#include "device.hpp"

// vulkan headers
#include <vulkan/vulkan.h>

// tinygltf headers
#include "lib/tinygltf/tiny_gltf.h"

// std lib headers
#include <string>
#include <vector>
#include <memory>

namespace ud {
    class UDSwapChain {
    public:
        static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

        UDSwapChain(UDDevice& deviceRef, VkExtent2D windowExtent);
        UDSwapChain(UDDevice& deviceRef, VkExtent2D windowExtent, std::shared_ptr<UDSwapChain> previous);
        ~UDSwapChain();

        UDSwapChain(const UDSwapChain&) = delete;
        UDSwapChain operator=(const UDSwapChain&) = delete;

        VkFramebuffer getFrameBuffer(int index) { return swapChainFramebuffers[index]; }
        VkRenderPass getRenderPass() { return renderPass; }
        VkImageView getImageView(int index) { return swapChainImageViews[index]; }
        size_t imageCount() { return swapChainImages.size(); }
        VkFormat getSwapChainImageFormat() { return swapChainImageFormat; }
        VkExtent2D getSwapChainExtent() { return swapChainExtent; }
        uint32_t width() { return swapChainExtent.width; }
        uint32_t height() { return swapChainExtent.height; }

        float extentAspectRatio() {
            return static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
        }
        VkFormat findDepthFormat();

        VkResult acquireNextImage(uint32_t* imageIndex);
        VkResult submitCommandBuffers(const VkCommandBuffer* buffers, uint32_t* imageIndex);

        bool compareSwapFormats(const UDSwapChain& swapChain) const {
            return swapChainDepthFormat == swapChain.swapChainDepthFormat &&
                swapChainImageFormat == swapChain.swapChainImageFormat;
        }

        void loadTextureImage(const tinygltf::Image& image);
        VkDescriptorImageInfo imageDescriptorInfo();
        void transitionImageLayout(
            VkDevice device,
            VkCommandPool commandPool,
            VkQueue graphicsQueue,
            VkImage image,
            VkFormat format,
            VkImageLayout oldLayout,
            VkImageLayout newLayout
        );

    private:
        void init();
        void createSwapChain();
        void createImageViews();
        void createDepthResources();
        void createColorTexture();
        void createRenderPass();
        void createFramebuffers();
        void createSyncObjects();
        // VkImageCreateInfo createImage(
        //     uint32_t width, uint32_t height,
        //     VkFormat format,
        //     VkImageTiling tiling,
        //     VkImageUsageFlags usage,
        //     VkMemoryPropertyFlags properties,
        //     VkImage& image,
        //     VkDeviceMemory& imageMemory);

        // Helper functions
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(
            const std::vector<VkSurfaceFormatKHR>& availableFormats);
        VkPresentModeKHR chooseSwapPresentMode(
            const std::vector<VkPresentModeKHR>& availablePresentModes);
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

        VkFormat swapChainImageFormat;
        VkFormat swapChainDepthFormat;
        VkExtent2D swapChainExtent;

        std::vector<VkFramebuffer> swapChainFramebuffers;
        VkRenderPass renderPass;

        // Depth resources
        std::vector<VkImage> depthImages;
        std::vector<VkDeviceMemory> depthImageMemorys;
        std::vector<VkImageView> depthImageViews;
        // Color resources
        std::vector<VkImage> colorImages;
        std::vector<VkDeviceMemory> colorImageMemorys;
        std::vector<VkImageView> colorImageViews;
        std::vector<VkSampler> colorSamplers;

        VkImage textureImage;
        VkDeviceMemory textureImageMemory;
        VkImageView textureImageView;
        VkSampler textureSampler;


        std::vector<VkImage> swapChainImages;
        std::vector<VkImageView> swapChainImageViews;

        UDDevice& device;
        VkExtent2D windowExtent;

        VkSwapchainKHR swapChain;
        std::shared_ptr<UDSwapChain> oldSwapChain;

        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence> inFlightFences;
        std::vector<VkFence> imagesInFlight;
        size_t currentFrame = 0;
    };
}
#include "renderer.hpp"


// std
#include <array>
#include <stdexcept>

namespace ud {
    UDRenderer::UDRenderer(UDWindow& udWindow, UDDevice& udDevice) : udWindow(udWindow), udDevice(udDevice) {
        recreateSwapChain();
        createCommandBuffers();
    }

    UDRenderer::~UDRenderer() { freeCommandBuffers(); }

    void UDRenderer::recreateSwapChain() {
        auto extent = udWindow.getExtent();
        while (extent.width == 0 || extent.height == 0) {
            extent = udWindow.getExtent();
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(udDevice.device());

        if (udSwapChain == nullptr) {
            udSwapChain = std::make_unique<UDSwapChain>(udDevice, extent);
        }
        else {
            std::shared_ptr<UDSwapChain> oldSwapChain = std::move(udSwapChain);
            udSwapChain = std::make_unique<UDSwapChain>(udDevice, extent, oldSwapChain);

            if (!oldSwapChain->compareSwapFormats(*udSwapChain.get())) {
                // Ideally instead of throwing an exception, it would be better to have a callback function
                // to notify the application that a new incompatible renderpass has been created.
                throw std::runtime_error("Swap chain image (or depth) format has changed!");
            }
        }

        // We will come back here, TODO 
        // (renderpass will almost always be compatible, and pipeline may not need to be recreated)
    }

    void UDRenderer::createCommandBuffers() {
        commandBuffers.resize(UDSwapChain::MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = udDevice.getCommandPool();
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(udDevice.device(), &allocInfo, commandBuffers.data()) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void UDRenderer::freeCommandBuffers() {
        vkFreeCommandBuffers(
            udDevice.device(),
            udDevice.getCommandPool(),
            static_cast<uint32_t>(commandBuffers.size()),
            commandBuffers.data());
        commandBuffers.clear();
    }


    VkCommandBuffer UDRenderer::beginFrame() {
        assert(!isFrameStarted && "Can't call beginFrame while already in progress");
        auto result = udSwapChain->acquireNextImage(&currentImageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return nullptr;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        isFrameStarted = true;

        auto commandBuffer = getCurrentCommandBuffer();
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        return commandBuffer;
    }

    void UDRenderer::endFrame() {
        assert(isFrameStarted && "Can't call endFrame while frame is not in progress");
        auto commandBuffer = getCurrentCommandBuffer();
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        auto result = udSwapChain->submitCommandBuffers(&commandBuffer, &currentImageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || udWindow.wasWindowResized()) {
            udWindow.resetWindowResizedFlag();
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        isFrameStarted = false;
        currentFrameIndex = (currentFrameIndex + 1) % UDSwapChain::MAX_FRAMES_IN_FLIGHT;
    }

    void UDRenderer::beginSwapChainRenderPass(VkCommandBuffer commandBuffer) {
        assert(isFrameStarted && "Can't begin render pass while frame is not in progress");
        assert(
            commandBuffer == getCurrentCommandBuffer() &&
            "Can't begin render pass on command buffer from a different frame"
        );
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = udSwapChain->getRenderPass();
        renderPassInfo.framebuffer = udSwapChain->getFrameBuffer(currentImageIndex);

        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = udSwapChain->getSwapChainExtent();

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { 0.1f, 0.1f, 0.1f, 1.0f };
        clearValues[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // set the viewport
    }

    void UDRenderer::setViewport(VkCommandBuffer commandBuffer) {
        // para 1 viewport
        // VkViewport viewport{};
        // viewport.x = 0.0f;
        // viewport.y = 0.0f;
        // // viewport.width = static_cast<float>(udSwapChain->width());
        // // viewport.height = static_cast<float>(udSwapChain->height());
        // viewport.width = static_cast<float>(udSwapChain->getSwapChainExtent().width);
        // viewport.height = static_cast<float>(udSwapChain->getSwapChainExtent().height);
        // viewport.minDepth = 0.0f;
        // viewport.maxDepth = 1.0f;
        // // VkRect2D scissor{};
        // VkRect2D scissor{ {0, 0}, udSwapChain->getSwapChainExtent() };
        // vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        // vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // para 2 viewports
        std::array<VkViewport, 2> viewports{};
        viewports[0].x = 0.0f;
        viewports[0].y = 0.0f;
        viewports[0].width = static_cast<float>(udSwapChain->getSwapChainExtent().width) / 2.0f;
        viewports[0].height = static_cast<float>(udSwapChain->getSwapChainExtent().height);
        viewports[0].minDepth = 0.0f;
        viewports[0].maxDepth = 1.0f;

        viewports[1].x = static_cast<float>(udSwapChain->getSwapChainExtent().width) / 2.0f;
        viewports[1].y = 0.0f;
        viewports[1].width = static_cast<float>(udSwapChain->getSwapChainExtent().width) / 2.0f;
        viewports[1].height = static_cast<float>(udSwapChain->getSwapChainExtent().height);
        viewports[1].minDepth = 0.0f;
        viewports[1].maxDepth = 1.0f;

        vkCmdSetViewport(commandBuffer, 0, static_cast<uint32_t>(viewports.size()), viewports.data());

        std::array<VkRect2D, 2> scissors{};
        scissors[0].offset = { 0, 0 };
        scissors[0].extent.width = udSwapChain->getSwapChainExtent().width / 2;
        scissors[0].extent.height = udSwapChain->getSwapChainExtent().height;

        scissors[1].offset = { static_cast<int32_t>(udSwapChain->getSwapChainExtent().width / 2), 0 };
        scissors[1].extent.width = udSwapChain->getSwapChainExtent().width / 2;
        scissors[1].extent.height = udSwapChain->getSwapChainExtent().height;

        vkCmdSetScissor(commandBuffer, 0, static_cast<uint32_t>(scissors.size()), scissors.data());
    }

    void UDRenderer::endSwapChainRenderPass(VkCommandBuffer commandBuffer) {
        assert(isFrameStarted && "Can't end render pass while frame is not in progress");
        assert(
            commandBuffer == getCurrentCommandBuffer() &&
            "Can't end render pass on command buffer from a different frame"
        );
        vkCmdEndRenderPass(commandBuffer);
    }
}

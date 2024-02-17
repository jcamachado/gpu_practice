#include "window.hpp"

#include <stdexcept>

namespace ud {
    UDWindow::UDWindow(int width, int height, std::string name) : width(width), height(height), windowName(name) {
        initWindow();
    }

    UDWindow::~UDWindow() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void UDWindow::initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // Don't create an OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);     // Disable window resizing

        window = glfwCreateWindow(width, height, windowName.c_str(), nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizedCallback);
    }

    void UDWindow::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface) {
        if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface!");
        }
    }

    void UDWindow::framebufferResizedCallback(GLFWwindow* window, int width, int height) {
        auto udWindow = reinterpret_cast<UDWindow*>(glfwGetWindowUserPointer(window));
        udWindow->framebufferResized = true;
        udWindow->width = width;
        udWindow->height = height;
    }
};

#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
namespace ud {
    class UDWindow {
        public:
            UDWindow(int width, int height, std::string name);
            ~UDWindow();

            /*
                Delete copy constructor 
                (Copy constructor is a constructor which creates an object by initializing it 
                with an object of the same class, which has been created previously
                But it is bad for resources management
                )
            */
            UDWindow(const UDWindow&) = delete; 
            UDWindow& operator=(const UDWindow&) = delete; // Delete copy assignment operator, so that we can't copy the window
            
            bool shouldClose() { return glfwWindowShouldClose(window); }
            VkExtent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}; }
            bool wasWindowResized() { return framebufferResized; }
            void resetWindowResizedFlag() { framebufferResized = false; }

            // KHR is the extension for Vulkan
            void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);


        private:
            static void framebufferResizedCallback(GLFWwindow* window, int width, int height);
            void initWindow();

            int width;
            int height;
            bool framebufferResized = false;

            std::string windowName;

            GLFWwindow* window;
    };
}
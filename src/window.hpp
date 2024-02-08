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

            // KHR is the extension for Vulkan
            void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);


        private:
            void initWindow();

            const int width;
            const int height;

            std::string windowName;

            GLFWwindow* window;
    };
}
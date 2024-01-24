#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
namespace uffdejavu {
    class Window {
        public:
            Window(int width, int height, std::string name);
            ~Window();

            /*
                Delete copy constructor 
                (Copy constructor is a constructor which creates an object by initializing it 
                with an object of the same class, which has been created previously
                But it is bad for resources management
                )
            */
            Window(const Window&) = delete; 
            Window& operator=(const Window&) = delete; // Delete copy assignment operator, so that we can't copy the window
            
            bool shouldClose() { return glfwWindowShouldClose(window); }


        private:
            void initWindow();

            const int width;
            const int height;

            std::string windowName;

            GLFWwindow* window;
    };
}
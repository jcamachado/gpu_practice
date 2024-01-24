#include "window.hpp"

namespace uffdejavu {
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
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);     // Disable window resizing

        window = glfwCreateWindow(width, height, windowName.c_str(), nullptr, nullptr);
    }
};

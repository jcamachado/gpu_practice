#include "first_app.hpp"

namespace ud {
    void FirstApp::run() {
        while (!udWindow.shouldClose()) {
            glfwPollEvents();
        }
    }
}
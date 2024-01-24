#include "first_app.hpp"

namespace uffdejavu {
    void FirstApp::run() {
        while (!window.shouldClose()) {
            glfwPollEvents();
        }
    }
}
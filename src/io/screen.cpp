#include "screen.h"

#include "keyboard.h"
#include "mouse.h"
// #include "joystick.h"

unsigned int Screen::SCR_WIDTH = 800;
unsigned int Screen::SCR_HEIGHT = 600;

void Screen::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);

    SCR_WIDTH = width;
    SCR_HEIGHT = height;
}

Screen::Screen()
    : window(nullptr) {}

bool Screen::init(){
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Particle System", NULL, NULL);
    if (window == NULL) {
        return false;
    }
    glfwMakeContextCurrent(window);
    return true;
}

void Screen::setParameters() {
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

    glfwSetFramebufferSizeCallback(window, Screen::framebufferSizeCallback);

    glfwSetKeyCallback(window, Keyboard::keyCallback); // Set callback function for keyboard input
    
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Disable cursor, to look like a fps camera
    glfwSetCursorPosCallback(window, Mouse::cursorPositionCallback); // Set callback function for mouse movement
    glfwSetMouseButtonCallback(window, Mouse::mouseButtonCallback); // Set callback function for mouse buttons
    glfwSetScrollCallback(window, Mouse::mouseScrollCallback); // Set callback function for mouse scroll

    glEnable(GL_DEPTH_TEST); // Enable depth testing
}

void Screen::update() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Window background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color buffer
}

void Screen::newFrame() {
    glfwSwapBuffers(window); // Swap buffers
    glfwPollEvents(); // Check for events
}

bool Screen::shouldClose() {
    return glfwWindowShouldClose(window);
}

void Screen::setShouldClose(bool value) {
    glfwSetWindowShouldClose(window, value);
}
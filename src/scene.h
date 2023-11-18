#ifndef SCENE_H
#define SCENE_H
// scene has everything that screen have and more.

#include "../glad/glad.h"
#include <GLFW/glfw3.h>

#include <vector>

#include <glm/glm.hpp>

#include "graphics/light.h"
#include "graphics/shader.h"


#include "io/camera.h"
#include "io/keyboard.h"
#include "io/mouse.h"

#include "algorithms/states.hpp"

class Scene {
    /*
        Callbacks
    */
    static void frameBufferSizeCallback(GLFWwindow* window, int width, int height);
    
    /*
        Constructors
    */
    Scene() {}
    Scene(int glfwVersionMajor, 
        int glfwVersionMinor, 
        const char* title, 
        unsigned int scrWidth, 
        unsigned int scrHeight
    );

    /* 
        Initialization
    */
    bool init();

    /*
        Main loop methods
    */
    // process input
    void processInput();
    
    // update screen before each frame
    void update();

    // update screen before after each frame
    void newFrame();

    // set uniform shader variables (lighting, etc)
    void render(Shader shader, bool applyLighting = true);

    /*
        Cleanup method
    */
    void cleanup();

    /* 
        Accessors
    */


    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
}

#endif
#ifndef SCENE_H
#define SCENE_H
// scene has everything that screen have and more.

#include "../lib/glad/glad.h"
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
    public:
        /*
            Callbacks
        */
        static void frameBufferSizeCallback(GLFWwindow* window, int width, int height);
        
        /*
            Constructors
        */
        Scene();
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
        // Process input
        void processInput(float dt);
        
        // Update screen before each frame
        void update();

        // Update screen before after each frame
        void newFrame();

        // Set uniform shader variables (lighting, etc)
        void render(Shader shader, bool applyLighting = true);

        /*
            Cleanup method
        */
        void cleanup();

        /* 
            Accessors
        */
        bool shouldClose();
        Camera* getActiveCamera(); 

        /*
            Modifiers
        */
        void setShouldClose(bool shouldClose);
        void setWindowColor(float r, float g, float b, float a);

        /*
            Lights
        */
        // list of point lights
        std::vector<PointLight*> pointLights;
        unsigned int activePointLights;
        // list of spot lights
        std::vector<SpotLight*> spotLights;
        unsigned int activeSpotLights;
        // directional light
        DirLight* dirLight;
        bool dirLightActive;

        /*
            Camera
        */
        std::vector<Camera*> cameras;
        unsigned int activeCamera;
        glm::mat4 view;
        glm::mat4 projection;
        glm::vec3 cameraPos;

    protected:
        // Window object
        GLFWwindow* window;

        // Window values
        const char* title;
        // static because frameBufferSizeCallback is static
        static unsigned int scrWidth;
        static unsigned int scrHeight; 

        float bgColor[4]; // background color

        // GLFW info
        int glfwVersionMajor;
        int glfwVersionMinor;
};

#endif
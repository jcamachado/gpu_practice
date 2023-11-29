#ifndef SCENE_H
#define SCENE_H
// scene has everything that screen have and more.

#include <glad/glad.h>
// #include "../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <vector>

#include <glm/glm.hpp>

#include "graphics/model.h"
#include "graphics/light.h"
#include "graphics/shader.h"

#include "graphics/models/box.hpp"

#include "io/camera.h"
#include "io/keyboard.h"
#include "io/mouse.h"
// #include "io/joystick.h"

#include "algorithms/states.hpp"
#include "algorithms/octree.h"
#include "algorithms/trie.hpp"

/*
    Forward declaration
*/

namespace Octree {
    class node;
}

class Model; 

class Scene {
    public:
        trie::Trie<Model*> models;
        trie::Trie<RigidBody*> instances;
        std::vector<RigidBody*> instancesToDelete;
        Octree::node* octree;                           // Root for the scene

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
            - init() calls gl functions for window and io initialization and init octree
            - prepare() calls the octree->insert() all objects, since the octree is not built yet
        */
        bool init();    
        void prepare(Box &box);                                                 

        /*
            Main loop methods
        */
        void processInput(float dt);                                    // Process input
        void update();                                                  // Update screen before each frame
        void newFrame(Box &box);                                        // Update screen before after each frame
        void renderShader(Shader shader, bool applyLighting = true);    // Set uniform shader variables (lighting, etc)
        void renderInstances(                                           // Render all instances of a model
            std::string modelId, 
            Shader shader, 
            float dt
        );                           

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
            Models/Instances methods and variables
        */
        std::string currentId;                                          // Has to be 8 chars long, from aaaaaaaa to zzzzzzzz
        
        std::string generateId();
        RigidBody* generateInstance(
            std::string modelId, 
            glm::vec3 size, 
            float mass, 
            glm::vec3 pos
        );
        void initInstances();                                           // Will call model.initInstances()
        void loadModels();                                              // Will call model.init()
        void registerModel(Model* model);
        void removeInstance(std::string instanceId);
        void markForDeletion(std::string instanceId);                   // Mark instances for death
        void clearDeadInstances();                                      // Delete instances marked for death

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
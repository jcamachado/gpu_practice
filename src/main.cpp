/*
    CODE
    ENGINE (Scene) we are creating this
    OPENGL
    GPU
*/
#include "../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <iostream>
#include <stack>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "scene.h"

#include "algorithms/states.hpp"

#include "graphics/cubemap.h"
#include "graphics/framememory.hpp"
#include "graphics/light.h"
#include "graphics/model.h"
#include "graphics/shader.h"
#include "graphics/texture.h"

#include "graphics/models/box.hpp" 
#include "graphics/models/cube.hpp"
#include "graphics/models/gun.hpp"
#include "graphics/models/lamp.hpp"
#include "graphics/models/plane.hpp"
#include "graphics/models/sphere.hpp"

#include "io/camera.h"
#include "io/keyboard.h"
#include "io/mouse.h"
#include "io/joystick.h"

#include "physics/environment.h"

void processInput(double dt);
Scene scene;

Camera cam;

double dt = 0.0f;       // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

unsigned int nSpheres = 10;
unsigned int nLamps = 4;

Sphere sphere(nSpheres);

int main(){
    scene = Scene(3, 3, "Particle System", 800, 600); // Create scene
    
    if (!scene.init()){ // Initialize scene
        std::cout << "Could not open window" << std::endl;
        glfwTerminate();
        return -1;
    }

    scene.cameras.push_back(&cam);
    scene.activeCamera = 0; // It is an index 

    /*
         Shaders
    */
    Shader boxShader("assets/shaders/instanced/box.vs", "assets/shaders/instanced/box.fs");
    Shader bufferShader("assets/shaders/buffer.vs", "assets/shaders/buffer.fs");
    Shader lampShader("assets/shaders/instanced/instanced.vs", "assets/shaders/lamp.fs");
    Shader dirLightShader("assets/shaders/shadows/directionalshadow.vs", "assets/shaders/shadows/shadow.fs");
    Shader outlineShader("assets/shaders/outline.vs", "assets/shaders/outline.fs");
    Shader shader("assets/shaders/instanced/instanced.vs", "assets/shaders/object.fs");
    Shader textShader("assets/shaders/text.vs", "assets/shaders/text.fs");



    /*
        Models
    */
    // Lamp lamp(nLamps);
    // scene.registerModel(&lamp);
    scene.registerModel(&sphere);

    Cube cube(10);
    scene.registerModel(&cube);

    Box box;
    box.init();                 // Box is not instanced

    scene.defaultFBO.bind(); // rebind default framebuffer
    DirLight dirLight(
        glm::vec3(-0.2f, -1.0f, -0.3f), 
        glm::vec4(0.1f, 0.1f, 0.1f, 1.0f), 
        glm::vec4(0.4f, 0.4f, 0.4f, 1.0f),
        glm::vec4(0.5f, 0.5f, 0.5f, 1.0f), 
        BoundingRegion(glm::vec3(-20.0f, -20.0f, 0.5f), glm::vec3(20.0f, 20.0f, 20.0f))
    );

    // Setup plane to display texture
    Plane map;
    map.init(dirLight.shadowFBO.textures[0]);
    scene.registerModel(&map);
    scene.loadModels();         // Load all model data

    /*
        Lights
    */
    
    scene.dirLight = &dirLight;

    // glm::vec3 pointLightPositions[] = {
    //     glm::vec3(0.7f,  0.2f,  2.0f),
    //     glm::vec3(2.3f, -3.3f, -4.0f),
    //     glm::vec3(-4.0f,  2.0f, -12.0f),
    //     glm::vec3(0.0f,  0.0f, -3.0f)
    // };
    
    // glm::vec4 ambient(0.05f, 0.05f, 0.05f, 1.0f);
    // glm::vec4 diffuse(0.8f, 0.8f, 0.8f, 1.0f);
    // glm::vec4 specular(1.0f);
    // float k0 = 1.0f;
    // float k1 = 0.09f;
    // float k2 = 0.032f;

    // PointLight pointLights[nLamps];

    // for (unsigned int i = 0; i < nLamps; i++) {
    //     pointLights[i] = {
    //         pointLightPositions[i],
    //         k0, k1, k2,
    //         ambient, diffuse, specular
    //     };
    //     scene.generateInstance(lamp.id, glm::vec3(0.25f), 0.25f, pointLightPositions[i]);
    //     scene.pointLights.push_back(&pointLights[i]);
    //     States::activate(&scene.activePointLights, i);
    // }
    // SpotLight spotLight = {
    //     cam.cameraPos,
    //     cam.cameraFront,
    //     glm::cos(glm::radians(12.5f)),
    //     glm::cos(glm::radians(20.0f)),
    //     1.0f,
    //     0.07f,
    //     0.032f,
    //     glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
    //     glm::vec4(1.0f),
    //     glm::vec4(1.0f)
    // };
    // scene.spotLights.push_back(&spotLight);
    // scene.activeSpotLights = 1;                         // 0b00000001
    
    scene.generateInstance(cube.id, glm::vec3(20.0f, 0.1f, 20.0f), 100.0f, glm::vec3(0.0f, -3.0f, 0.0f));
    glm::vec3 cubePositions[] = {
        { 1.0f, 3.0f, -5.0f },
        { -7.25f, 2.1f, 1.5f },
        { -15.0f, 2.55f, 9.0f },
        { 4.0f, -3.5f, 5.0f },
        { 2.8f, 1.9f, -6.2f },
        { 3.5f, 6.3f, -1.0f },
        { -3.4f, 10.9f, -5.5f },
        { 10.0f, -2.0f, 13.2f },
        { 2.1f, 7.9f, -8.3f },
    };
    for (unsigned int i = 0; i < 9; i++) {
        scene.generateInstance(cube.id, glm::vec3(0.5f), 1.0f, cubePositions[i]);
    }

    // instantiate texture plane
    scene.generateInstance(map.id, glm::vec3(2.0f, 2.0f, 0.0f), 0.0f, glm::vec3(0.0f)); 
    scene.initInstances();                              // Instantiate instances
    scene.prepare(box);                                 // Builds octree  
    scene.variableLog["time"] = (double)0.0;

    while (!scene.shouldClose()){                       // Check if window should close
        double currentTime = glfwGetTime();
        dt = currentTime - lastFrame;
        lastFrame = currentTime;

        scene.variableLog["time"] += dt;
        scene.variableLog["fps"] = 1.0f/dt;
        
        scene.update();                                 // Update screen values
        processInput(dt);                               // Process input

        // Activate the directional light's FBO
        // Everything rendered after this will be rendered to this FBO


        for (int i = 0; i < sphere.currentNumInstances; i++){
            if (glm::length(cam.cameraPos - sphere.instances[i]->pos) > 250.0f){
                scene.markForDeletion(sphere.instances[i]->instanceId);
            }
        }

        // scene.renderShader(shader);  
        dirLight.shadowFBO.activate();
        scene.renderDirLightShader(dirLightShader);    // Render scene from light's perspective     
        if (sphere.currentNumInstances > 0){            // Render launch objects
            scene.renderInstances(sphere.id, dirLightShader, dt);
        }
        scene.renderInstances(cube.id, dirLightShader, dt);     // Render cubes
        // scene.renderShader(boxShader, false);           // Render boxes
        // box.render(boxShader);                          // Box is not instanced

        /*
            Render texture
        */
        // rebind default framebuffer
        scene.defaultFBO.activate();

        // Render quad
        scene.renderInstances(map.id, bufferShader, dt);

        // Send new frame to window
        scene.newFrame(box);
        scene.clearDeadInstances();             // Delete instances after updating octree
    }
    scene.cleanup();
    return 0;
}

void launchItem(float dt){
    RigidBody* rb = scene.generateInstance(sphere.id, glm::vec3(0.05f), 1.0f, cam.cameraPos-glm::vec3(0.0f, 0.0f, 0.0f));
    // RigidBody* rb = scene.generateInstance(sphere.id, glm::vec3(1.0f), 1.0f, cam.cameraPos-glm::vec3(-15.0f, 10.0f, 10.0f));
    if (rb){
        rb->transferEnergy(100.0f, cam.cameraFront);
        rb->applyAcceleration(Environment::gravity);
    }
}

void processInput(double dt){ // Function for processing input
    scene.processInput(dt); // Process input for scene

    // if(Keyboard::key(GLFW_KEY_ESCAPE) || mainJ.buttonState(GLFW_JOYSTICK_BTN_RIGHT)){ // Check if escape key is pressed
    if(Keyboard::key(GLFW_KEY_ESCAPE)){
        scene.setShouldClose(true); // Set window to close
    }

    // Update flash light
    if (States::isIndexActive(&scene.activeSpotLights, 0)){
        scene.spotLights[0]->position = scene.getActiveCamera()->cameraPos;
        scene.spotLights[0]->direction = scene.getActiveCamera()->cameraFront;
    }

    if(Keyboard::keyWentUp(GLFW_KEY_L)){
        States::toggleIndex(&scene.activeSpotLights, 0);
    }

    if (Keyboard::keyWentDown(GLFW_KEY_F)){
        launchItem(dt);
    }
    if (Keyboard::keyWentDown(GLFW_KEY_T)){
        for (int i = 0; i < sphere.currentNumInstances; i++){
            if (!sphere.instances[i]->freeze()){
                sphere.instances[i]->unfreeze();
            }
        }
    }
    //reset octree
    if (Keyboard::keyWentDown(GLFW_KEY_R)){
        scene.octree = new Octree::node(BoundingRegion(glm::vec3(-16.0f), glm::vec3(16.0f)));
    }
    for (int i=0; i<4; i++){
        if (Keyboard::keyWentDown(GLFW_KEY_1 + i)){
            States::toggleIndex(&scene.activePointLights, i);
        }
    }
} 

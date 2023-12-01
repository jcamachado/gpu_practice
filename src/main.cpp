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
#include "graphics/light.h"
#include "graphics/model.h"
#include "graphics/shader.h"
#include "graphics/texture.h"

#include "graphics/models/box.hpp" 
#include "graphics/models/cube.hpp"
#include "graphics/models/gun.hpp"
#include "graphics/models/lamp.hpp"
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

Sphere sphere(10);

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
    Shader boxShader("assets/instanced/box.vs", "assets/instanced/box.fs");
    Shader lampShader("assets/instanced/instanced.vs", "assets/lamp.fs");
    Shader shader("assets/instanced/instanced.vs", "assets/object.fs");
    Shader skyboxShader("assets/skybox/skybox.vs", "assets/skybox/sky.fs");
    Shader textShader("assets/text.vs", "assets/text.fs");
    skyboxShader.activate();
    skyboxShader.set3Float("min", 0.047f, 0.016f, 0.239f);
    skyboxShader.set3Float("max", 0.945f, 1.000f, 0.682f);


    /*
        Skybox
    */
    Cubemap skybox;
    skybox.init();
    // skybox.loadTextures("assets/skybox");

    /*
        Models
    */
    Lamp lamp(4);
    scene.registerModel(&lamp);
    scene.registerModel(&sphere);

    Cube cube(1);
    scene.registerModel(&cube);

    Box box;
    box.init();                 // Box is not instanced

    scene.loadModels();         // Load all model data

    /*
        Lights
    */
    DirLight dirLight{
        glm::vec3(-0.2f, -1.0f, -0.3f), 
        glm::vec4(0.1f, 0.1f, 0.1f, 1.0f), 
        glm::vec4(0.4f, 0.4f, 0.4f, 1.0f),
        glm::vec4(0.5f, 0.5f, 0.5f, 1.0f)
    };
    scene.dirLight = &dirLight;

    glm::vec3 pointLightPositions[] = {
        glm::vec3(0.7f,  0.2f,  2.0f),
        glm::vec3(2.3f, -3.3f, -4.0f),
        glm::vec3(-4.0f,  2.0f, -12.0f),
        glm::vec3(0.0f,  0.0f, -3.0f)
    };
    
    glm::vec4 ambient(0.05f, 0.05f, 0.05f, 1.0f);
    glm::vec4 diffuse(0.8f, 0.8f, 0.8f, 1.0f);
    glm::vec4 specular(1.0f);
    float k0 = 1.0f;
    float k1 = 0.09f;
    float k2 = 0.032f;

    PointLight pointLights[4];

    for (unsigned int i = 0; i < 4; i++) {
        pointLights[i] = {
            pointLightPositions[i],
            k0, k1, k2,
            ambient, diffuse, specular
        };
        scene.generateInstance(lamp.id, glm::vec3(0.25f), 0.25f, pointLightPositions[i]);
        scene.pointLights.push_back(&pointLights[i]);
        States::activate(&scene.activePointLights, i);
    }
    SpotLight spotLight = {
        cam.cameraPos,
        cam.cameraFront,
        glm::cos(glm::radians(12.5f)),
        glm::cos(glm::radians(20.0f)),
        1.0f,
        0.07f,
        0.032f,
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(1.0f),
        glm::vec4(1.0f)
    };
    scene.spotLights.push_back(&spotLight);
    scene.activeSpotLights = 1;                         // 0b00000001
    
    scene.generateInstance(cube.id, glm::vec3(20.0f, 0.1f, 20.0f), 100.0f, glm::vec3(0.0f, -3.0f, 0.0f));

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

        skyboxShader.activate();
        skyboxShader.setFloat("time", scene.variableLog["time"].val<float>());
        skybox.render(skyboxShader, &scene); //Render skybox

        scene.renderText(
            "comic", 
            textShader, 
            "Hello World!!", 
            50.0f, 
            50.0f, 
            glm::vec2(1.0f), 
            glm::vec3(0.5f, 0.6f, 1.0f)
        );
        scene.renderText(
            "comic", 
            textShader, 
            "Time: " + scene.variableLog["time"].dump(), 
            50.0f, 
            550.0f, 
            glm::vec2(1.0f), 
            glm::vec3(0.5f, 0.6f, 1.0f)
        );
        scene.renderText(
            "comic", 
            textShader, 
            "FPS: " + scene.variableLog["fps"].dump(), 
            50.0f, 
            550.0f - 40.0f, 
            glm::vec2(1.0f), 
            glm::vec3(0.5f, 0.6f, 1.0f)
        );

        for (int i = 0; i < sphere.currentNumInstances; i++){
            if (glm::length(cam.cameraPos - sphere.instances[i]->pos) > 250.0f){
                scene.markForDeletion(sphere.instances[i]->instanceId);
            }
        }
        scene.renderShader(shader);                     
        if (sphere.currentNumInstances > 0){            // Render launch objects
            scene.renderInstances(sphere.id, shader, dt);
        }
        scene.renderInstances(cube.id, shader, dt);     // Render cube

        scene.renderShader(lampShader, false);                  // Render lamps
        scene.renderInstances(lamp.id, lampShader, dt);

        scene.renderShader(boxShader, false);           // Render boxes
        box.render(boxShader);                          // Box is not instanced
        

        // Send new frame to window
        scene.newFrame(box);
        scene.clearDeadInstances();             // Delete instances after updating octree
    }
    skybox.cleanup();
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

    // Update flash light
    if (States::isIndexActive(&scene.activeSpotLights, 0)){
        scene.spotLights[0]->position = scene.getActiveCamera()->cameraPos;
        scene.spotLights[0]->direction = scene.getActiveCamera()->cameraFront;
    }

    // if(Keyboard::key(GLFW_KEY_ESCAPE) || mainJ.buttonState(GLFW_JOYSTICK_BTN_RIGHT)){ // Check if escape key is pressed
    if(Keyboard::key(GLFW_KEY_ESCAPE)){
        scene.setShouldClose(true); // Set window to close
    }
    if(Keyboard::keyWentUp(GLFW_KEY_L)){
        States::toggleIndex(&scene.activeSpotLights, 0);
    }
    if (Keyboard::keyWentDown(GLFW_KEY_F)){
        launchItem(dt);
    }
    for (int i=0; i<4; i++){
        if (Keyboard::keyWentDown(GLFW_KEY_1 + i)){
            States::toggleIndex(&scene.activePointLights, i);
        }
    }
} 

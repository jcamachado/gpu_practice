/*
    CODE
    ENGINE (Scene) we are creating this
    OPENGL
    GPU
*/

#include <vector>
#include <stack>

#include "graphics/shader.h" //Including majority of the OpenGL headers
#include "graphics/texture.h" // Also includes other OpenGL headers and stb_image.h 

#include "graphics/models/box.hpp" 
#include "graphics/models/cube.hpp" // Also includes other OpenGL headers
#include "graphics/models/lamp.hpp"
#include "graphics/models/gun.hpp"
#include "graphics/models/sphere.hpp"
#include "graphics/light.h"
#include "graphics/model.h"

#include "physics/environment.h"

#include "io/joystick.h"

#include "algorithms/states.hpp"

#include "scene.h"
// TODO put all mutually used headers in a single header file

void processInput(double dt); // Function for processing input
Scene scene;

Camera cam;

double dt = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

SphereArray launchObjects; 

int main(){
    scene = Scene(3, 3, "Particle System", 800, 600); // Create scene
    
    if (!scene.init()){ // Initialize scene
        std::cout << "Could not open window" << std::endl;
        glfwTerminate();
        return -1;
    }

    scene.cameras.push_back(&cam);
    scene.activeCamera = 0; // It is an index 

    // Shaders
    Shader shader("assets/object.vs", "assets/object.fs");
    Shader lampShader("assets/instanced/instanced.vs", "assets/lamp.fs");
    Shader launchShader("assets/instanced/instanced.vs", "assets/object.fs");
    Shader boxShader("assets/instanced/box.vs", "assets/instanced/box.fs");

    // Models
    launchObjects.init();

    Box box;
    box.init();

    Model m(BoundTypes::AABB, glm::vec3(0.0f), glm::vec3(0.05f));
    m.loadModel("assets/models/lotr_troll/scene.gltf");
    // Gun m;
    // m.init();

    // Lights
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

    LampArray lamps;
    lamps.init();

    for (unsigned int i = 0; i < 4; i++) {
        pointLights[i] = {
            pointLightPositions[i],
            k0, k1, k2,
            ambient, diffuse, specular
        };
        lamps.lightInstances.push_back(pointLights[i]);
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
    States::activate(&scene.activeSpotLights, 1); // 0b00000001

    while (!scene.shouldClose()){ // Check if window should close
        box.positions.clear();
        box.sizes.clear();

        double currentTime = glfwGetTime();
        dt = currentTime - lastFrame;
        lastFrame = currentTime;
        
        // Process input
        processInput(dt); 

        // Update screen values
        scene.update(); 
        
        scene.render(shader);
        m.render(shader, dt, &box);
        // m.render(shader, dt, &box, &scene); // For any model extending Model class

        std::stack<int> removeObjects;
        for (int i = 0; i<launchObjects.instances.size(); i++){
            if(glm::length(scene.getActiveCamera()->cameraPos - launchObjects.instances[i].pos) > 50.0f){
                removeObjects.push(i);
                continue;
            }
        }
        for (int i = 0; i<removeObjects.size(); i++){
            launchObjects.instances.erase(launchObjects.instances.begin() + removeObjects.top());
            removeObjects.pop();
        }

        if (launchObjects.instances.size() > 0){
            scene.render(launchShader); 
            launchObjects.render(launchShader, dt, &box);
        }

        // Lamps
        scene.render(lampShader, false);
        lamps.render(lampShader, dt, &box);

        // Render boxes
        if (box.positions.size() > 0){
            //if instances exist
            scene.render(boxShader, false);
            box.render(boxShader);
        }

        // Send new frame to window
        scene.newFrame();
    }

    lamps.cleanup();
    box.cleanup();
    launchObjects.cleanup();
    m.cleanup();

    scene.cleanup();
    return 0;
}

void launchItem(float dt){
    RigidBody rb(1.0f, scene.getActiveCamera()->cameraPos);
    rb.transferEnergy(100.0f, scene.getActiveCamera()->cameraFront);
    rb.applyAcceleration(Environment::gravity);
    launchObjects.instances.push_back(rb);
}

void processInput(double dt){ // Function for processing input
    scene.processInput(dt); // Process input for scene

    // Update flash light
    if (States::isActive(&scene.activeSpotLights, 0)){
        scene.spotLights[0]->position = scene.getActiveCamera()->cameraPos;
        scene.spotLights[0]->direction = scene.getActiveCamera()->cameraFront;
    }

    // if(Keyboard::key(GLFW_KEY_ESCAPE) || mainJ.buttonState(GLFW_JOYSTICK_BTN_RIGHT)){ // Check if escape key is pressed
    if(Keyboard::key(GLFW_KEY_ESCAPE)){
        scene.setShouldClose(true); // Set window to close
    }
    if(Keyboard::keyWentUp(GLFW_KEY_L)){
        States::toggle(&scene.activeSpotLights, 0);
    }
    if (Keyboard::keyWentDown(GLFW_KEY_F)){
        launchItem(dt);
    }
    for (int i=0; i<4; i++){
        if (Keyboard::keyWentDown(GLFW_KEY_1 + i)){
            States::toggle(&scene.activePointLights, i);
        }

    }
} 

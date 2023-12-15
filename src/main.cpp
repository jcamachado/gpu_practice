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

#include "io/camera.h"
#include "io/keyboard.h"
#include "io/mouse.h"
#include "io/joystick.h"

#include "physics/environment.h"


#include "graphics/memory/framememory.hpp"
#include "graphics/memory/uniformmemory.hpp"

#include "graphics/objects/model.h"

#include "graphics/models/box.hpp" 
#include "graphics/models/brickwall.hpp"
#include "graphics/models/cube.hpp"
#include "graphics/models/gun.hpp"
#include "graphics/models/lamp.hpp"
#include "graphics/models/plane.hpp"
#include "graphics/models/sphere.hpp"

#include "graphics/rendering/cubemap.h"
#include "graphics/rendering/light.h"
#include "graphics/rendering/shader.h"
#include "graphics/rendering/texture.h"
#include "graphics/rendering/text.h"

#include "physics/collisionmesh.h"

Scene scene;


Camera cam;

double dt = 0.0f;       // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

unsigned int nSpheres = 20;
unsigned int nLamps = 1;
std::string Shader::defaultDirectory = "assets/shaders";

BrickWall wall;
Cube cube(1);
Lamp lamp(nLamps);
Sphere sphere(nSpheres);
Plane map;

void processInput(double dt);
void renderScene(Shader shader);
void setSceneLights(Scene *scene);

int main(){
    scene = Scene(3, 3, "Particle System", 1200, 720); // Create scene
    
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
    Shader::loadIntoDefault("defaultHeader.gh");
    // Shader bufferShader("buffer.vs", "buffer.fs");
    // Shader outlineShader("outline.vs", "outline.fs");
    // Shader textShader("text.vs", "text.fs");
    
    Shader boxShader(false, "instanced/box.vs", "instanced/box.fs");
    Shader shader(true, "instanced/instanced.vs", "object.fs");
    Shader pointShadowShader(
        false, 
        "shadows/pointShadow.vs", 
        "shadows/pointSpotShadow.fs", 
        "shadows/pointShadow.gs"
    );
    Shader dirShadowShader(
        false, 
        "shadows/dirSpotShadow.vs", 
        "shadows/dirShadow.fs"
    );
    Shader spotShadowShader(
        false, 
        "shadows/dirSpotShadow.vs", 
        "shadows/pointSpotShadow.fs"
    );

    Shader::clearDefault();
    TextRenderer font(32);
    if (!scene.registerFont(&font, "comic", "assets/fonts/Comic_Sans_MS.ttf")) {
        std::cout << "Could not load font" << std::endl;
    }

    /*
        Models
    */
    scene.registerModel(&cube);
    scene.registerModel(&lamp);
    scene.registerModel(&sphere);
    // scene.registerModel(&wall);

    Box box;
    box.init();                 // Box is not instanced

    // Setup plane to display texture
    
    
    
    scene.loadModels();         // Load all model data

    /*
        Lights
    */
    
    setSceneLights(&scene);
    map.init({scene.dirLight->shadowFBO.textures[0]});
    scene.registerModel(&map);

    glm::vec3 pointLightPositions[] = {
        glm::vec3(1.0f,  1.0f,  0.0f),
        glm::vec3(0.0,  15.0f,  0.0f),
        glm::vec3(2.3f, -3.3f, -4.0f),
        glm::vec3(-4.0f,  2.0f, -12.0f),
        glm::vec3(0.0f,  0.0f, -3.0f)
    };
    
    glm::vec4 ambient(0.05f, 0.05f, 0.05f, 1.0f);
    glm::vec4 diffuse(0.8f, 0.8f, 0.8f, 1.0f);
    glm::vec4 specular(1.0f);
    float k0 = 1.0f;
    float k1 = 0.0014f;
    float k2 = 0.000007f;

    PointLight pointLights[nLamps];

    for (unsigned int i = 0; i < nLamps; i++) {
        pointLights[i] = PointLight(
            pointLightPositions[i],
            k0, k1, k2,
            ambient, diffuse, specular,
            0.5f, 50.0f
        );
        scene.generateInstance(lamp.id, glm::vec3(0.25f), 0.25f, pointLightPositions[i]);
        scene.pointLights.push_back(&pointLights[i]);
        States::activateIndex(&scene.activePointLights, i);
    }

    // Spot Light
    SpotLight spotLight(                                            // Perpendicular to direction
        cam.cameraPos, cam.cameraFront, cam.cameraUp,
        // glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
        glm::cos(glm::radians(12.5f)), glm::cos(glm::radians(20.0f)),
        1.0f, 0.0014f, 0.000007f,
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f), glm::vec4(1.0f),
        0.1f, 100.0f
    );
    scene.spotLights.push_back(&spotLight);
    // scene.activeSpotLights = 1;                         // 0b00000001
    
    // scene.generateInstance(cube.id, glm::vec3(20.0f, 0.1f, 20.0f), 100.0f, glm::vec3(0.0f, -3.0f, 0.0f));
    glm::vec3 cubePositions[] = {
        { 0.0f, -3.0f, 0.0f },  //base floor
        { -7.25f, 2.1f, 1.5f },
        { -15.0f, 2.55f, 9.0f },
        { 4.0f, -3.5f, 5.0f },
        { 2.8f, 1.9f, -6.2f },
        { 3.5f, 6.3f, -1.0f },
        { -3.4f, 10.9f, -5.5f },
        { 0.0f, 11.0f, 0.2f },
        { 0.0f, 5.0f, 0.0f },
    };
    for (unsigned int i = 0; i < 9; i++) {
        scene.generateInstance(
            cube.id,
            glm::vec3(10.0f, 10.0f, 1.0f), 
            1.0f, cubePositions[i],
            glm::vec3(glm::radians(90.0f), 0.0f, 0.0f)
        );
    }
    // Instantiate brickwall
    // scene.generateInstance(
    //     wall.id, 
    //     glm::vec3(1.0f, 1.0f, 1.0f), 
    //     1.0f, 
    //     glm::vec3(0.0f, -2.0f, -2.0f), 
    //     glm::vec3(glm::radians(90.0f), 0.0f, 0.0f)
    // );

    // instantiate texture plane
    // scene.generateInstance(map.id, glm::vec3(2.0f, 2.0f, 0.0f), 0.0f, glm::vec3(0.0f)); 
    scene.initInstances();                              // Instantiate instances
    scene.prepare(box, { shader });                                 // Builds octree  
    scene.variableLog["time"] = (double)0.0;

    scene.defaultFBO.bind(); // rebind default framebuffer
    try {
        while (!scene.shouldClose()){                       // Check if window should close
            double currentTime = glfwGetTime();
            dt = currentTime - lastFrame;
            lastFrame = currentTime;

            scene.variableLog["time"] += dt;
            scene.variableLog["fps"] = 1.0f/dt;
            scene.update();                                 // Update screen values
            processInput(dt);                               // Process input

            for (int i = 0; i < sphere.currentNInstances; i++){
                if (glm::length(cam.cameraPos - sphere.instances[i]->pos) > 100.0f)
                {
                    scene.markForDeletion(sphere.instances[i]->instanceId);
                }
            }
            // Activate the directional light's FBO
            // Everything rendered after this will be rendered to this FBO

            // // Render scene to dirlight FBO
            // scene.dirLight->shadowFBO.activate();
            // scene.renderDirLightShader(dirShadowShader);    // Render scene from light's perspective     
            // renderScene(dirShadowShader);


            // // Render scene to point light FBO
            // for (unsigned int i = 0, len = scene.pointLights.size(); i < len; i++){
            //     if (States::isIndexActive(&scene.activePointLights, i)){
            //         scene.pointLights[i]->shadowFBO.activate();
            //         /*
            //             Render scene from light's perspective  
            //             1 - Goes through model transformation to put in world coordinates (.vs)  
            //             2 - then goes to geometry shader and its transformed 6 times to get each face
            //             and then each one of those 6 triangles is then emmited as a vertex and then passes
            //             the fragment shader to be colored customly to the depth buffer
            //         */
            //         scene.renderPointLightShader(pointShadowShader, i);       
            //         renderScene(pointShadowShader);
            //     }
            // }

            // // Render scene to spot light FBO
            // for (unsigned int i = 0, len = scene.spotLights.size(); i < len; i++){
            //     if (States::isIndexActive(&scene.activeSpotLights, i)){
            //         scene.spotLights[i]->shadowFBO.activate();
            //         scene.renderSpotLightShader(spotShadowShader, i);    // Render scene from light's perspective     
            //         renderScene(spotShadowShader);
            //     }
            // }

            // Render scene normally
            scene.defaultFBO.activate();
            scene.renderShader(shader);               // Render scene normally
            renderScene(shader);

            scene.renderShader(boxShader, false);           // Render boxes
            box.render(boxShader);                          // Box is not instanced

            // Send new frame to window
            scene.newFrame(box);
            scene.clearDeadInstances();             // Delete instances after updating octree
        }
    } catch (const std::exception& e) {
        std::cout << "BROKE IN MAIN" << e.what() << std::endl;
        throw e;
    }
    scene.cleanup();
    return 0;
}

void renderScene(Shader shader){                // assumes shader is prepared accordingly
    if (sphere.currentNInstances > 0) {            // Render launch objects
            scene.renderInstances(sphere.id, shader, dt);
    }
    scene.renderInstances(cube.id, shader, dt);     // Render cubes
    scene.renderInstances(lamp.id, shader, dt);     // Render lamps
    // scene.renderInstances(wall.id, shader, dt);     // Render wall
}

void setSceneDirLights(Scene *scene) {
    DirLight dirLight(
        glm::vec3(-0.2f, -0.9f, -0.2f), 
        glm::vec4(0.1f, 0.1f, 0.1f, 1.0f), 
        glm::vec4(0.6f, 0.6f, 0.6f, 1.0f),
        glm::vec4(0.7f, 0.7f, 0.7f, 1.0f), 
        BoundingRegion(glm::vec3(-20.0f, -20.0f, 0.5f), glm::vec3(20.0f, 20.0f, 50.0f))
    );

    scene->dirLight = &dirLight;
}

void setSceneLights(Scene *scene){
    setSceneDirLights(scene);
    // setScenePointLights();
    // setSceneSpotLights();
}

void launchItem(float dt){
    RigidBody* rb = scene.generateInstance(sphere.id, glm::vec3(0.05f), 1.0f, cam.cameraPos);

    if (rb != nullptr){
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

    // Update flash light attached to the camera
    if (States::isIndexActive(&scene.activeSpotLights, 0)){
        scene.spotLights[0]->position = scene.getActiveCamera()->cameraPos;
        scene.spotLights[0]->direction = scene.getActiveCamera()->cameraFront;
        scene.spotLights[0]->up = scene.getActiveCamera()->cameraUp;
        scene.spotLights[0]->updateMatrices();
    }

    if(Keyboard::keyWentUp(GLFW_KEY_L)){
        States::toggleIndex(&scene.activeSpotLights, 0);
    }

    if (Keyboard::keyWentDown(GLFW_KEY_F)){

        launchItem(dt);
    }
    if (Keyboard::keyWentDown(GLFW_KEY_T)){
        for (int i = 0; i < sphere.currentNInstances; i++){
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

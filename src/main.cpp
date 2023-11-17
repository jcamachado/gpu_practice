// TODO put all mutually used headers in a single header file
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

#include "io/joystick.h"
#include "io/keyboard.h"
#include "io/mouse.h"
#include "io/camera.h"
#include "io/screen.h"

#include "physics/environment.h"

void processInput(double dt); // Function for processing input

Screen screen;

Joystick mainJ(0);
Camera Camera::defaultCamera(glm::vec3(0.0f, 0.0f, 0.0f));

double dt = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

bool flashLightOn = false;

Box box;
SphereArray launchObjects; 

int main(){
    int success;
    char infoLog[512];

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // Set major version to 3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // Set minor version to 3

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Set OpenGL profile to core

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Set OpenGL forward compatibility to true    
#endif

    if(!screen.init()){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){ // Check if glad was loaded
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    screen.setParameters();
    //render troll

    // Shaders
    Shader shader("assets/object.vs", "assets/object.fs");
    Shader lampShader("assets/instanced/instanced.vs", "assets/lamp.fs");
    Shader launchShader("assets/instanced/instanced.vs", "assets/object.fs");
    Shader boxShader("assets/instanced/box.vs", "assets/instanced/box.fs");

    // Models
    launchObjects.init();
    box.init();

    // Lights
    DirLight dirLight{
        glm::vec3(-0.2f, -1.0f, -0.3f), 
        glm::vec4(0.1f, 0.1f, 0.1f, 1.0f), 
        glm::vec4(0.4f, 0.4f, 0.4f, 1.0f),
        glm::vec4(0.5f, 0.5f, 0.5f, 1.0f)
    };

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

    Model troll(glm::vec3(5.0f), glm::vec3(0.05f));
    troll.loadModel("assets/models/lotr_troll/scene.gltf");



    LampArray lamps;
    lamps.init();

    for (unsigned int i = 0; i < 4; i++) {
        lamps.lightInstances.push_back({
            pointLightPositions[i],
            k0, k1, k2,
            ambient, diffuse, specular
        });
    }
    SpotLight s = {
        Camera::defaultCamera.cameraPos,
        Camera::defaultCamera.cameraFront,
        1.0f,
        0.07f,
        0.032f,
        glm::cos(glm::radians(12.5f)),
        glm::cos(glm::radians(20.0f)),
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(1.0f),
        glm::vec4(1.0f)
    };

    // joystick recognition
    // mainJ.update();
    // if (mainJ.isPresent()){ 
    //     std::cout << "Joystick connected" << std::endl;
    // }

    while (!screen.shouldClose()){ // Check if window should close
        double currentTime = glfwGetTime();
        dt = currentTime - lastFrame;
        lastFrame = currentTime;
        
        // Process input
        processInput(dt); 

        //update screen values
        screen.update(); 

        //draw shapes
        shader.activate(); //apply shader
        launchShader.activate(); //apply shader

        shader.activate(); //apply shader
        shader.set3Float("viewPos", Camera::defaultCamera.cameraPos);
        launchShader.activate(); //apply shader
        launchShader.set3Float("viewPos", Camera::defaultCamera.cameraPos);

        shader.activate(); //apply shader
        dirLight.render(shader);
        launchShader.activate(); //apply shader
        dirLight.render(launchShader);


        for(unsigned int i = 0; i < 4; i++){
            shader.activate();
            lamps.lightInstances[i].render(shader, i);
            launchShader.activate();
            lamps.lightInstances[i].render(launchShader, i);
        }
        
        shader.activate(); //apply shader
        shader.setInt("nPointLights", 4);
        launchShader.activate();
        launchShader.setInt("nPointLights", 4);
        
        if (flashLightOn){
            s.position = Camera::defaultCamera.cameraPos;
            s.direction = Camera::defaultCamera.cameraFront;
            shader.activate(); //apply shader
            s.render(shader, 0);
            shader.setInt("nSpotLights", 1);
            launchShader.activate();
            s.render(launchShader, 0);
            launchShader.setInt("nSpotLights", 1);
        }
        else {
            shader.activate(); //apply shader
            shader.setInt("nSpotLights", 0);
            launchShader.activate();
            launchShader.setInt("nSpotLights", 0);
        }

        //create transformation for screen
        glm::mat4 view = glm::mat4(1.0f);        
        glm::mat4 projection = glm::mat4(1.0f);

        view = Camera::defaultCamera.getViewMatrix();

        // the parameters are the field of view, the aspect ratio, the near clipping plane and the far clipping plane
        projection = glm::perspective(
            glm::radians(Camera::defaultCamera.getZoom()), 
            (float)Screen::SCR_WIDTH/(float)Screen::SCR_HEIGHT, 0.1f, 100.0f
        ); // Create perspective projection

        shader.activate(); //apply shader
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        troll.render(shader , dt);

        std::stack<int> removeObjects;
        for (int i = 0; i<launchObjects.instances.size(); i++){
            if(glm::length(Camera::defaultCamera.cameraPos - launchObjects.instances[i].pos) > 50.0f){
                removeObjects.push(i);
                continue;
            }
        }
        for (int i = 0; i<removeObjects.size(); i++){
            launchObjects.instances.erase(launchObjects.instances.begin() + removeObjects.top());
            removeObjects.pop();
        }


        if (launchObjects.instances.size() > 0){
            launchShader.activate();
            launchShader.setMat4("view", view);
            launchShader.setMat4("projection", projection);
            launchObjects.render(launchShader, dt);
        }

        lampShader.activate(); 
        lampShader.setMat4("view", view);
        lampShader.setMat4("projection", projection);
        lamps.render(lampShader, dt);

        //render boxes
        if (box.offsets.size() > 0){
            //if instances exist
            boxShader.activate();
            boxShader.setMat4("view", view);
            boxShader.setMat4("projection", projection);
            box.render(boxShader);
        }

        // Swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        screen.newFrame(); 
    }

    lamps.cleanup();
    box.cleanup();
    launchObjects.cleanup();
    troll.cleanup();

    glfwTerminate(); // Terminate glfw

    return 0;
}

void launchItem(float dt){
    RigidBody rb(1.0f, Camera::defaultCamera.cameraPos);
    rb.transferEnergy(100.0f, Camera::defaultCamera.cameraFront);
    // rb.applyImpulse( Camera::defaultCamera.cameraFront, 10000.0f, dt );
    rb.applyAcceleration(Environment::gravity);
    launchObjects.instances.push_back(rb);
}

void processInput(double dt){ // Function for processing input
    if(Keyboard::key(GLFW_KEY_ESCAPE) || mainJ.buttonState(GLFW_JOYSTICK_BTN_RIGHT)){ // Check if escape key is pressed
        screen.setShouldClose(true); // Set window to close
    }
    if(Keyboard::keyWentUp(GLFW_KEY_L)){
        flashLightOn = !flashLightOn;
    }

    //move camera
    if (Keyboard::key(GLFW_KEY_W)){
        Camera::defaultCamera.updateCameraPosition(CameraDirection::FORWARD, dt);
    }
    if (Keyboard::key(GLFW_KEY_S)){
        Camera::defaultCamera.updateCameraPosition(CameraDirection::BACKWARD, dt);
    }
    if (Keyboard::key(GLFW_KEY_D)){
        Camera::defaultCamera.updateCameraPosition(CameraDirection::RIGHT, dt);
    }
    if (Keyboard::key(GLFW_KEY_A)){
        Camera::defaultCamera.updateCameraPosition(CameraDirection::LEFT, dt);
    }
    if (Keyboard::key(GLFW_KEY_SPACE)){
        Camera::defaultCamera.updateCameraPosition(CameraDirection::UP, dt);
    }
    if (Keyboard::key(GLFW_KEY_LEFT_SHIFT)){
        Camera::defaultCamera.updateCameraPosition(CameraDirection::DOWN, dt);
    }

    if (Keyboard::keyWentDown(GLFW_KEY_F)){
        launchItem(dt);
    }
    if (Keyboard::keyWentDown(GLFW_KEY_I)){
        box.offsets.push_back(glm::vec3(box.offsets.size() * 1.0f));
        box.sizes.push_back(glm::vec3(box.sizes.size()*0.5f));
    }

    /*
        provavelmente deprecated
        if using Joystick
    */
   /*
   float lx = mainJ.axesState(GLFW_JOYSTICK_AXES_LEFT_STICK_X);
   float ly = mainJ.axesState(GLFW_JOYSTICK_AXES_LEFT_STICK_Y);

    if(std::abs(lx)>0.05f){// This value is the denser threshold(O quanto pro lado esta o stick)
        x += lx/5.0f;
    }
    if(std::abs(ly)>0.05f){ 
        y += ly/5.0f;
    }

    //triggers starts at -1 -1 and goes to 1 1
    float rt = mainJ.axesState(GLFW_JOYSTICK_AXES_RIGHT_TRIGGER) / 2.0f + 0.5f;
    float lt = -mainJ.axesState(GLFW_JOYSTICK_AXES_LEFT_TRIGGER) / 2.0f + 0.5f;
    if (rt>0.05f){
        transform = glm::scale(transform, glm::vec3(1+ rt/10.0f, 1+ rt/10.0f, 0.0f));
    }

    if (lt>0.05f){
        transform = glm::scale(transform, glm::vec3(lt/10.0f, lt/10.0f, 0.0f));
    }
    */

	// mainJ.update();

} 

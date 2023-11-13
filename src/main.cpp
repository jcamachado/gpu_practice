// TODO put all mutually used headers in a single header file
#include "graphics/shader.h" //Including majority of the OpenGL headers
#include "graphics/texture.h" // Also includes other OpenGL headers and stb_image.h 

#include "graphics/models/cube.hpp" // Also includes other OpenGL headers
#include "graphics/models/lamp.hpp"
#include "graphics/light.h"
#include "graphics/model.h"

#include "io/joystick.h"
#include "io/keyboard.h"
#include "io/mouse.h"
#include "io/camera.h"
#include "io/screen.h"

void processInput(double dt); // Function for processing input
float mixValue = 0.5f;


glm::mat4 transform = glm::mat4(1.0f); // Create identity matrix (no transformation)
Joystick mainJ(0);
unsigned int SCR_WIDTH = 800, SCR_HEIGHT = 600;

Screen screen;
Camera cameras[2] = {
    Camera(glm::vec3(0.0f, 0.0f, 3.0f)),
    Camera(glm::vec3(10.0f, 10.0f, 10.0f))
};
int activeCamera = 0;

float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

bool flashLightOn = true;

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

    Shader shader("assets/object.vs", "assets/object.fs");
    Shader lampShader("assets/object.vs", "assets/lamp.fs");

    Model m(glm::vec3(0.0f, 0.0f, -5.0f), glm::vec3(0.05f), false);
    m.loadModel("assets/models/lotr_troll/scene.gltf");

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
    Lamp lamps[4];
    for (unsigned int i = 0; i < 4; i++) {
        lamps[i] = Lamp(glm::vec3(1.0f),
            glm::vec4(0.05f, 0.05f, 0.05f, 1.0f), glm::vec4(0.8f, 0.8f, 0.8f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
            1.0f, 0.07f, 0.032f,
            pointLightPositions[i], glm::vec3(0.25f));
        lamps[i].init();
    }

    SpotLight s = {
        cameras[activeCamera].cameraPos,
        cameras[activeCamera].cameraFront,
        1.0f,
        0.07f,
        0.032f,
        glm::cos(glm::radians(12.5f)),
        glm::cos(glm::radians(20.0f)),
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)
    };
        
    mainJ.update();
    if (mainJ.isPresent()){ 
        std::cout << "Joystick connected" << std::endl;
    }

    while (!screen.shouldClose()){ // Check if window should close
        double currentTime = glfwGetTime();
        deltaTime = currentTime - lastFrame;
        lastFrame = currentTime;
        processInput(deltaTime); // Process input

        //render
        screen.update(); 

        shader.activate(); //apply shader
        shader.set3Float("viewPos", cameras[activeCamera].cameraPos);

        dirLight.direction = glm::vec3(
            glm::rotate(glm::mat4(1.0f),
            glm::radians(1.0f), 
            glm::vec3(0.5, 1.0, 0.0)) * glm::vec4(dirLight.direction, 1.0f));
        dirLight.render(shader);
        
        for (unsigned int i = 0; i < 4; i++) {
            lamps[i].pointLight.render(shader, i); //lamp now third party its light position and data
        }
        shader.setInt("nPointLights", 4);

        if (flashLightOn){
            s.position = cameras[activeCamera].cameraPos;
            s.direction = cameras[activeCamera].cameraFront;
            s.render(shader, 0);
            shader.setInt("nSpotLights", 1);
        }
        else {
            shader.setInt("nSpotLights", 0);
        }

        //create transformation for screen
        glm::mat4 view = glm::mat4(1.0f);        
        glm::mat4 projection = glm::mat4(1.0f);

        view = cameras[activeCamera].getViewMatrix();

        // the parameters are the field of view, the aspect ratio, the near clipping plane and the far clipping plane
        projection = glm::perspective(glm::radians(cameras[activeCamera].getZoom()), (float)SCR_WIDTH/(float)SCR_HEIGHT, 0.1f, 100.0f); // Create perspective projection
    

        shader.activate(); //apply shader
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);

        m.render(shader);
        lampShader.activate(); 
        lampShader.setMat4("view", view);
        lampShader.setMat4("projection", projection);

        for (unsigned int i = 0; i < 4; i++) {
            lamps[i].render(lampShader);
        }
        // Swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        screen.newFrame(); 
    }

    m.cleanup();
    for (unsigned int i = 0; i < 4; i++) {
        lamps[i].cleanup();
    }
    glfwTerminate(); // Terminate glfw

    return 0;
}


void processInput(double dt){ // Function for processing input
    if(Keyboard::key(GLFW_KEY_ESCAPE) || mainJ.buttonState(GLFW_JOYSTICK_BTN_RIGHT)){ // Check if escape key is pressed
        screen.setShouldClose(true); // Set window to close
    }
    if(Keyboard::keyWentUp(GLFW_KEY_L)){
        flashLightOn = !flashLightOn;
    }

    // change mixValue
    if(Keyboard::keyWentUp(GLFW_KEY_UP)){
        mixValue += 0.05f; // Increase mix value
        if(mixValue > 1.0f){ // Check if mix value is greater than 1
            mixValue = 1.0f; // Set mix value to 1
        }
    }
    if(Keyboard::keyWentDown(GLFW_KEY_DOWN)){
        mixValue -= 0.05f; // Decrease mix value
        if(mixValue < 0.0f){ // Check if mix value is less than 0
            mixValue = 0.0f; // Set mix value to 0
        }
    }
    if(Keyboard::keyWentDown(GLFW_KEY_TAB)){
        activeCamera = activeCamera == 0 ? 1 : 0;
    }

    //move camera
    if (Keyboard::key(GLFW_KEY_W)){
        cameras[activeCamera].updateCameraPosition(CameraDirection::FORWARD, dt);
    }
    if (Keyboard::key(GLFW_KEY_S)){
        cameras[activeCamera].updateCameraPosition(CameraDirection::BACKWARD, dt);
    }
    if (Keyboard::key(GLFW_KEY_D)){
        cameras[activeCamera].updateCameraPosition(CameraDirection::RIGHT, dt);
    }
    if (Keyboard::key(GLFW_KEY_A)){
        cameras[activeCamera].updateCameraPosition(CameraDirection::LEFT, dt);
    }
    if (Keyboard::key(GLFW_KEY_SPACE)){
        cameras[activeCamera].updateCameraPosition(CameraDirection::UP, dt);
    }
    if (Keyboard::key(GLFW_KEY_LEFT_SHIFT)){
        cameras[activeCamera].updateCameraPosition(CameraDirection::DOWN, dt);
    }

    double dx = Mouse::getDX(), dy = Mouse::getDY();
    if (dx != 0 || dy != 0){
        cameras[activeCamera].updateCameraDirection(dx, dy);
    }

    double scrollDy = Mouse::getScrollDY();
    if (scrollDy != 0){
        cameras[activeCamera].updateCameraZoom(scrollDy);
    }

    mainJ.update();

    /*
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

    /*
        if using mouse
    */
	// if (Mouse::button(GLFW_MOUSE_BUTTON_LEFT)) {
	// 	double _x = Mouse::getMouseX();
	// 	double _y = Mouse::getMouseY();
	// 	std::cout << x << ' ' << y << std::endl;
    //     x = -_x/SCR_WIDTH;
    //     y = _y/SCR_HEIGHT;
	// }

	mainJ.update();

} 

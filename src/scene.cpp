#include "scene.h"

unsigned int Scene::scrWidth = 0;
unsigned int Scene::scrHeight = 0;

/*
    Callbacks
*/
void Scene::frameBufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    Scene::scrWidth = width;
    Scene::scrHeight = height;
}

/*
    Constructors
*/

Scene::Scene() {}

Scene::Scene(int glfwVersionMajor, 
    int glfwVersionMinor, 
    const char* title, 
    unsigned int scrWidth, 
    unsigned int scrHeight
) : glfwVersionMajor(glfwVersionMajor), 
    glfwVersionMinor(glfwVersionMinor),
    title(title),
    activeCamera(-1),
    activePointLights(0),
    activeSpotLights(0) {

    Scene::scrWidth = scrWidth;
    Scene::scrHeight = scrHeight;

    setWindowColor(0.1f, 0.15f, 0.15f, 1.0f);
}

/* 
    Initialization
*/
bool Scene::init() {
    glfwInit();

    // set glfw version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, glfwVersionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, glfwVersionMinor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Initialize window
    window = glfwCreateWindow(scrWidth, scrHeight, title, NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        return false;
    }
    glfwMakeContextCurrent(window);

    // Set glad
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return false;
    }

    // Setup screen
    glViewport(0, 0, scrWidth, scrHeight);

    /*
        Callbacks
    */
    // Frame size
    glfwSetFramebufferSizeCallback(window, frameBufferSizeCallback);

    // Key pressed
    glfwSetKeyCallback(window, Keyboard::keyCallback);

    // Mouse movement
    glfwSetCursorPosCallback(window, Mouse::cursorPositionCallback);

    // Mouse scroll
    glfwSetScrollCallback(window, Mouse::mouseScrollCallback);

    // Mouse button pressed
    glfwSetMouseButtonCallback(window, Mouse::mouseButtonCallback);


    // Joystick recognition (I moved it here out of my own mind, it was in the main)
    // Joystick mainJ(0);
    // mainJ.update();
    // if (mainJ.isPresent()){ 
    //     std::cout << "Joystick connected" << std::endl;
    // }

    
    /*
        Set rendering parameters
    */
    glEnable(GL_DEPTH_TEST); // Doesn't show vertices not visible to camera (back of objects)
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    return true;
}

/*
    Main loop methods
*/
// Process input
void Scene::processInput(float dt){
    if(activeCamera != -1 && activeCamera < cameras.size()){
        // Active camera exists

        // Set camera direction
        // if (!(Mouse::getDX() == 0 && Mouse::getDY() == 0)){ // to save resources
        cameras[activeCamera]->updateCameraDirection(Mouse::getDX(), Mouse::getDY());
        // }

        // Set camera zoom
        // if (Mouse::getScrollDY() != 0){                 // to save resources
        cameras[activeCamera]->updateCameraZoom(Mouse::getScrollDY());
        // }

        // Set camera position
        if (Keyboard::key(GLFW_KEY_W)){
            cameras[activeCamera]->updateCameraPosition(CameraDirection::FORWARD, dt);
        }
        if (Keyboard::key(GLFW_KEY_S)){
            cameras[activeCamera]->updateCameraPosition(CameraDirection::BACKWARD, dt);
        }
        if (Keyboard::key(GLFW_KEY_D)){
            cameras[activeCamera]->updateCameraPosition(CameraDirection::RIGHT, dt);
        }
        if (Keyboard::key(GLFW_KEY_A)){
            cameras[activeCamera]->updateCameraPosition(CameraDirection::LEFT, dt);
        }
        if (Keyboard::key(GLFW_KEY_SPACE)){
            cameras[activeCamera]->updateCameraPosition(CameraDirection::UP, dt);
        }
        if (Keyboard::key(GLFW_KEY_LEFT_SHIFT)){
            cameras[activeCamera]->updateCameraPosition(CameraDirection::DOWN, dt);
        }
        
        // Set matrices
        view = cameras[activeCamera]->getViewMatrix();
        projection = glm::perspective(
            glm::radians(cameras[activeCamera]->getZoom()), // FOV
            (float)scrWidth / (float)scrHeight,             // Aspect ratio
            0.1f, 100.0f                                    // Near and far clipping planes
        );

        // Set position at end
        cameraPos = cameras[activeCamera]->cameraPos;

        /*
            if using Joystick (probably deprecated, but the logic is here)
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
        mainJ.update();
        */ 
    }
}

// Update screen before each frame
void Scene::update(){
    // apply shaders for lighting and textures
    glClearColor(bgColor[0], bgColor[1], bgColor[2], bgColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

// Update screen before after each frame
void Scene::newFrame(){
    // Send new frame to window
    glfwSwapBuffers(window);
    glfwPollEvents();
}

// Set uniform shader variables (lighting, etc)
void Scene::render(Shader shader, bool applyLighting){
    // Activate shader
    shader.activate();

    // Set camera values
    shader.setMat4("view", view);
    shader.setMat4("projection", projection);
    shader.set3Float("viewPos", cameraPos);

    // Lighting
    if (applyLighting){
        // Point lights
        unsigned int nLights = pointLights.size();
        unsigned int nActiveLights = 0;
        for (unsigned int i = 0; i < nLights; i++){
            if (States::isActive(&activePointLights, i)){
                // i'th light is active
                pointLights[i]->render(shader, nActiveLights);
                nActiveLights++;
            }
        }
        shader.setInt("nPointLights", nActiveLights);
        
        // Spot lights
        nLights = spotLights.size();
        nActiveLights = 0;
        for (unsigned int i = 0; i < nLights; i++){
            if (States::isActive(&activeSpotLights, i)){
                // i'th spot light is active
                spotLights[i]->render(shader, nActiveLights);
                nActiveLights++;
            }
        }
        shader.setInt("nSpotLights", nActiveLights);

        // Directional light (only one)
        dirLight->render(shader);
    }
}

/*
    Cleanup method
*/
void Scene::cleanup(){
    glfwTerminate();
}

/* 
    Accessors
*/
bool Scene::shouldClose(){
    return glfwWindowShouldClose(window);
}

Camera* Scene::getActiveCamera(){
    return (activeCamera >= 0 && activeCamera < cameras.size()) ? cameras[activeCamera] : nullptr;
}

/*
    Modifiers
*/
void Scene::setShouldClose(bool shouldClose){
    glfwSetWindowShouldClose(window, shouldClose);
}

void Scene::setWindowColor(float r, float g, float b, float a){
    bgColor[0] = r;
    bgColor[1] = g;
    bgColor[2] = b;
    bgColor[3] = a;
}

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

Scene::Scene() : currentId("aaaaaaaa") {}

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
    activeSpotLights(0),
    currentId("aaaaaaaa") {

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
        double dx = Mouse::getDX(), dy = Mouse::getDY();
        if (dx != 0 || dy != 0){
            cameras[activeCamera]->updateCameraDirection(dx, dy);
        }

        // Set camera zoom
        double scrollDY = Mouse::getScrollDY();
        if (scrollDY != 0) {
            cameras[activeCamera]->updateCameraZoom(scrollDY);
        }

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

void Scene::renderShader(Shader shader, bool applyLighting){
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

void Scene::renderInstances(std::string modelId, Shader shader, float dt){
        models[modelId]->render(shader, dt, this);
}

/*
    Cleanup method
*/
void Scene::cleanup(){
    // Cleanup models
    models.traverse([](Model* model) -> void {      // Lambda function, return is type(->) void
        model->cleanup();
    });

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

/*
    Models/Instances methods
*/
std::string Scene::generateId(){
    for (int i = currentId.length()-1; i >= 0; i--){
        if((int)currentId[i] != (int) 'z') {
            currentId[i] = (char)(((int)currentId[i]) + 1);
            break;
        }
        else{
            currentId[i] = 'a';
        }
    }
    return currentId;
}

std::string Scene::generateInstance(std::string modelId, glm::vec3 size, float mass, glm::vec3 pos){
    unsigned int idx = models[modelId]->generateInstance(size, mass, pos);
    if (idx != -1) {
        // Instance was created successfully
        std::string id = generateId();
        models[modelId]->instances[idx].instanceId = id;
        instances.insert(id, modelId);
        return id;
    }
    return "";
}


void Scene::initInstances(){
    models.traverse([](Model* model) -> void {                  // Iteration over models in Trie structure 
        model->initInstances();
    });
}

void Scene::loadModels(){
    models.traverse([](Model* model) -> void {
        model->init();
    });
}

void Scene::registerModel(Model* model){
    models.insert(model->id, model);
}

void Scene::removeInstance(std::string instanceId){
    /*
        Remove all locations
        -Scene::instances
        -Model::instances
    */
    std::string targetModel = instances[instanceId];
    models[targetModel]->removeInstance(instanceId);
    instances.erase(instanceId);
}


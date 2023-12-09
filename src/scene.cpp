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
    defaultFBO = FramebufferObject(scrWidth, scrHeight, 
    GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT
    );

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
    /*
        Depth testing
        - GL_DEPTH_TEST: Doesn't show vertices not visible to camera (back of objects)
        Blending fortext tures
        - GL_BLEND: Allows transparency between objects (text texture)
        - glBlendFunc(): Sets blending function (how to blend)
        - glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED): Disable cursor like in FPS games
    */
    glEnable(GL_DEPTH_TEST);        

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    /*
        Stencil Testing
        -glStencilOp: has 3 parameters: fail, zfail, zpass. fail that represents 3 cases.
        1- fail means that both stencil and depth tests failed. 
        2- zfail means that stencil test passed but depth test failed. 
        3- zpass means that both tests passed.

        GL_Keep keep fragmets if stencil or depth fails. And replace if both pass.
    */
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);



    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    /*
        Init octree
    */
    octree = new Octree::node(BoundingRegion(glm::vec3(-16.0f), glm::vec3(16.0f)));

    /*
        Init FreeType library
    */
    if (FT_Init_FreeType(&ft)){
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
        return false;
    }

    /*
        Insert font
    */
    fonts.insert("comic", TextRenderer(32));
    if (!fonts["comic"].loadFont(ft, "assets/fonts/Comic_Sans_MS.ttf")){
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
        return false;
    }

    FT_Done_FreeType(ft);

    // Setup lighting values
    variableLog["useBlinn"] = true;
    variableLog["useGamma"] = true;

    return true;
}

/*
    Prepare for mainloop (after object generation, etc and before main while loop)
*/
void Scene::prepare(Box &box){
    // octree->build();
    octree->update(box);        // Calls octree->build() if it hasn't been built yet
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
            0.1f, 250.0f                                    // Near and far clipping planes
        );
        textProjection = glm::ortho(0.0f, (float)scrWidth, 0.0f, (float)scrHeight);

        // Set position at end
        cameraPos = cameras[activeCamera]->cameraPos;

        // Update blinn parameter if necessary
        if (Keyboard::keyWentUp(GLFW_KEY_B)){
            variableLog["useBlinn"] = !variableLog["useBlinn"].val<bool>();
        }

        // Toggle gamma correction parameter if necessary
        if (Keyboard::keyWentUp(GLFW_KEY_G)){
            variableLog["useGamma"] = !variableLog["useGamma"].val<bool>();
        }

        // Update outline parameter if necessary
        if (Keyboard::keyWentUp(GLFW_KEY_O)){
            variableLog["displayOutline"] = !variableLog["displayOutline"].val<bool>();
        }

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
    // Clear occupied bits
    defaultFBO.clear();
}

// Update screen before after each frame
void Scene::newFrame(Box &box){
    box.positions.clear();
    box.sizes.clear();

    // Process pending.
    octree->processPending();       // "Process new objects"
    octree->update(box);            // "Are there any destroyed objects?"

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
        unsigned int textureIdx = 31;
        // Directional light (only one)
        dirLight->render(shader, textureIdx--);     // set as last texture to guarantee that wont override other textures(solution could be better)

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
            if (States::isIndexActive(&activeSpotLights, i)){
                // i'th spot light is active
                spotLights[i]->render(shader, nActiveLights);
                nActiveLights++;
            }
        }
        shader.setInt("nSpotLights", nActiveLights);

        
        shader.setBool("useBlinn", variableLog["useBlinn"].val<bool>());
        shader.setBool("useGamma", variableLog["useGamma"].val<bool>());
    }
}

void Scene::renderDirLightShader(Shader shader){
    shader.activate();

    // Set camera values
    shader.setMat4("lightSpaceMatrix", dirLight->lightSpaceMatrix);
}

void Scene::renderInstances(std::string modelId, Shader shader, float dt){
    shader.activate();
    models[modelId]->render(shader, dt, this);
}

void Scene::renderText(
    std::string font,
    Shader shader,
    std::string text,
    float x, 
    float y, 
    glm::vec2 scale, 
    glm::vec3 color
){
    shader.activate();
    shader.setMat4("projection", textProjection);

    fonts[font].render(shader, text, x, y, scale, color);
}

/*
    Cleanup method
*/
void Scene::cleanup(){
    // Cleanup models
    models.traverse([](Model* model) -> void {      // Lambda function, return is type(->) void
        model->cleanup();
    });

    // Cleanup Tries
    models.cleanup();
    instances.cleanup();

    // Cleanup fonts
    fonts.traverse([](TextRenderer tr) -> void {
        tr.cleanup();
    });
    
    // Destroy octree
    octree->destroy();

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

RigidBody* Scene::generateInstance(std::string modelId, glm::vec3 size, float mass, glm::vec3 pos){
    /*
        octree->addToPending(rb, models);     Add all bounding regions from the models to the pending queue
        and since processPending calls update, prepare() doesnt need to call update.
    */
    RigidBody* rb = models[modelId]->generateInstance(size, mass, pos);
    if (rb) {
        // Instance was created successfully
        std::string id = generateId();
        rb->instanceId = id;
        instances.insert(id, rb);
        octree->addToPending(rb, models);               // Add all bounding regions from the models to the pending queue
        return rb;
    }
    return nullptr;
}


void Scene::initInstances(){
    models.traverse([](Model* model) -> void {          // Iteration over models in Trie structure 
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
    std::string targetModel = instances[instanceId]->modelId;
    models[targetModel]->removeInstance(instanceId);
    instances[instanceId] = nullptr;
    instances.erase(instanceId);                        // erasee() doesnt know the type of <T>Trie, so, deletes the nullptr
}

void Scene::markForDeletion(std::string instanceId){
    States::activate(&instances[instanceId]->state, INSTANCE_DEAD);
    instancesToDelete.push_back(instances[instanceId]);
}

void Scene::clearDeadInstances(){
    for (RigidBody *rb : instancesToDelete){
        removeInstance(rb->instanceId);
    }
    instancesToDelete.clear();
}
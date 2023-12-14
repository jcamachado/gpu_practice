#include "scene.h"

/*
    These macros are the same as in defaultHeader.gh
*/
#define MAX_POINT_LIGHTS 10
#define MAX_SPOT_LIGHTS 2

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
Scene::Scene() : currentId("aaaaaaaa"), lightUBO(0) {}

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
    currentId("aaaaaaaa"), 
    lightUBO(0) {

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
        Init model/instance trees
    */
    models = avl_createEmptyRoot(strkeycmp);
    instances = avl_createEmptyRoot(strkeycmp);

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
    fonts = avl_createEmptyRoot(strkeycmp);

    variableLog["skipNormalMapping"] = false;

    return true;
}

bool Scene::registerFont(TextRenderer* tr, std::string name, std::string path) {
    if (tr->loadFont(ft, path)) {
        fonts = avl_insert(fonts, (void*)name.c_str(), tr);
        return true;
    }
    else {
        return false;
    }
}

/*
    Prepare for mainloop (after object generation, etc and before main while loop)
*/
void Scene::prepare(Box &box, std::vector<Shader> shaders){
    // Close FT library
    FT_Done_FreeType(ft);
    // Process current instances
    octree->update(box);        // Calls octree->build() if it hasn't been built yet

    // Set lighting UBO, mapped with Lights in defaultHeader.gh
    lightUBO = UBO::UBO(0, {
        UBO::newStruct({// Directional light
            UBO::Type::VEC3,

            UBO::Type::VEC4,
            UBO::Type::VEC4,
            UBO::Type::VEC4,

            UBO::Type::SCALAR,

            UBO::newColMat(4, 4)
        }),

        UBO::Type::SCALAR,  // nPointLights
        UBO::newArray(MAX_POINT_LIGHTS, UBO::newStruct({ // Point lights
            UBO::Type::VEC3,

            UBO::Type::VEC4,
            UBO::Type::VEC4,
            UBO::Type::VEC4,

            UBO::Type::SCALAR,
            UBO::Type::SCALAR,
            UBO::Type::SCALAR,

            UBO::Type::SCALAR
        })),

        UBO::Type::SCALAR,  // nSpotLights
        UBO::newArray(MAX_SPOT_LIGHTS, UBO::newStruct({ // Spot lights
            UBO::Type::VEC3,
            UBO::Type::VEC3,

            UBO::Type::SCALAR,
            UBO::Type::SCALAR,

            UBO::Type::VEC4,
            UBO::Type::VEC4,
            UBO::Type::VEC4,

            UBO::Type::SCALAR,
            UBO::Type::SCALAR,
            UBO::Type::SCALAR,

            UBO::Type::SCALAR,
            UBO::Type::SCALAR,

            UBO::newColMat(4, 4)
        }))

    });

    // Attach the UBO to specified shaders
    for (Shader shader : shaders){
        lightUBO.attachToShader(shader, "Lights");
    }

    // Setup memory for UBO
    lightUBO.generate();
    lightUBO.bind();
    lightUBO.initNullData(GL_STATIC_DRAW);
    lightUBO.bindRange();

    // Write initial values
    lightUBO.startWrite();

    // Write directional light
    lightUBO.writeElement<glm::vec3>(&dirLight->direction);
    lightUBO.writeElement<glm::vec4>(&dirLight->ambient);
    lightUBO.writeElement<glm::vec4>(&dirLight->diffuse);
    lightUBO.writeElement<glm::vec4>(&dirLight->specular);
    lightUBO.writeElement<float>(&dirLight->br.max.z);  // Far plane
    lightUBO.writeArrayContainer<glm::mat4, glm::vec4>(&dirLight->lightSpaceMatrix, 4);

    // Write point lights   
    nPointLights = std::min<unsigned int>(pointLights.size(), MAX_POINT_LIGHTS);
    lightUBO.writeElement<unsigned int>(&nPointLights);
    unsigned int i = 0;
    for (; i < nPointLights; i++){
        lightUBO.writeElement<glm::vec3>(&pointLights[i]->position);
        lightUBO.writeElement<glm::vec4>(&pointLights[i]->ambient);
        lightUBO.writeElement<glm::vec4>(&pointLights[i]->diffuse);
        lightUBO.writeElement<glm::vec4>(&pointLights[i]->specular);
        lightUBO.writeElement<float>(&pointLights[i]->k0);
        lightUBO.writeElement<float>(&pointLights[i]->k1);
        lightUBO.writeElement<float>(&pointLights[i]->k2);
        lightUBO.writeElement<float>(&pointLights[i]->farPlane);
    }
    lightUBO.advanceArray(MAX_POINT_LIGHTS - i);    // Advance to finish array

    // Write spot lights
    nSpotLights = std::min<unsigned int>(spotLights.size(), MAX_SPOT_LIGHTS);
    lightUBO.writeElement<unsigned int>(&nSpotLights);
    for (int i = 0; i < nSpotLights; i++) {
        lightUBO.writeElement<glm::vec3>(&spotLights[i]->position);
        lightUBO.writeElement<glm::vec3>(&spotLights[i]->direction);
        lightUBO.writeElement<float>(&spotLights[i]->cutOff);
        lightUBO.writeElement<float>(&spotLights[i]->outerCutOff);
        lightUBO.writeElement<glm::vec4>(&spotLights[i]->ambient);
        lightUBO.writeElement<glm::vec4>(&spotLights[i]->diffuse);
        lightUBO.writeElement<glm::vec4>(&spotLights[i]->specular);
        lightUBO.writeElement<float>(&spotLights[i]->k0);
        lightUBO.writeElement<float>(&spotLights[i]->k1);
        lightUBO.writeElement<float>(&spotLights[i]->k2);
        lightUBO.writeElement<float>(&spotLights[i]->nearPlane);
        lightUBO.writeElement<float>(&spotLights[i]->farPlane);
        lightUBO.writeArrayContainer<glm::mat4, glm::vec4>(&spotLights[i]->lightSpaceMatrix, 4);
    }
    lightUBO.clear();
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

        if (Keyboard::key(GLFW_KEY_N)){
            variableLog["skipNormalMapping"] = !variableLog["skipNormalMapping"].val<bool>();
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
        // set as last texture to guarantee that wont override other textures(solution could be better)
        dirLight->render(shader, textureIdx--);     

        // Point lights
        unsigned int nLights = pointLights.size();
        unsigned int nActiveLights = 0;
        for (unsigned int i = 0; i < nLights; i++){
            if (States::isIndexActive(&activePointLights, i)){
                // i'th light is active
                pointLights[i]->render(shader, nActiveLights, textureIdx--);
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
                spotLights[i]->render(shader, nActiveLights, textureIdx--);
                nActiveLights++;
            }
        }
        shader.setInt("nSpotLights", nActiveLights);

        
        shader.setBool("skipNormalMapping", variableLog["skipNormalMapping"].val<bool>());
    }
}

void Scene::renderDirLightShader(Shader shader){
    shader.activate();

    // Set camera values
    shader.setMat4("lightSpaceMatrix", dirLight->lightSpaceMatrix);
}

void Scene::renderSpotLightShader(Shader shader, unsigned int idx){
    shader.activate();

    // Set light space matrix
    shader.setMat4("lightSpaceMatrix", spotLights[idx]->lightSpaceMatrix);

    // light position
    shader.set3Float("lightPos", spotLights[idx]->position);

    // far plane
    shader.setFloat("farPlane", spotLights[idx]->farPlane);


}

void Scene::renderPointLightShader(Shader shader, unsigned int idx){
    shader.activate();

    // Set light space matrices
    for (unsigned int i = 0; i < 6; i++){
        // idx is the index of the point light and i is the index of the matrix within that light
        shader.setMat4("lightSpaceMatrices[" + std::to_string(i) + "]", pointLights[idx]->lightSpaceMatrices[i]);
    }

    // light position
    shader.set3Float("lightPos", pointLights[idx]->position);

    // far plane
    shader.setFloat("farPlane", pointLights[idx]->farPlane);
}

void Scene::renderInstances(std::string modelId, Shader shader, float dt){
    void* val = avl_get(models, (void*)modelId.c_str());
    
    if (val) {
        // render each mesh in specified model
        shader.activate();
        ((Model*)val)->render(shader, dt, this);
    }
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
    void* val = avl_get(fonts, (void*)font.c_str());
    if (val) {
        shader.activate();
        shader.setMat4("projection", textProjection);
        ((TextRenderer*)val)->render(shader, text, x, y, scale, color);
    }
}

/*
    Cleanup method
*/
void Scene::cleanup(){
    // clean up instances
    avl_free(instances);

    // Cleanup models
    avl_postorderTraverse(models, [](avl* node) -> void {
        ((Model*)node->val)->cleanup();
    });
    avl_free(models);
    avl_free(fonts);
    
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

RigidBody* Scene::generateInstance(
    std::string modelId, 
    glm::vec3 size, 
    float mass, 
    glm::vec3 pos,
    glm::vec3 eRotation
){
    /*
        octree->addToPending(rb, models);     Add all bounding regions from the models to the pending queue
        and since processPending calls update, prepare() doesnt need to call update.
    */
    void* val = avl_get(models, (void*)modelId.c_str());
    if (val) {
        Model* model = (Model*)val;
        RigidBody* rb = model->generateInstance(size, mass, pos, eRotation);
        if (rb) {
            // successfully generated, set new and unique id for instance
            std::string id = generateId();
            rb->instanceId = id;
            // insert into trie
            instances = avl_insert(instances, (void*)id.c_str(), rb);
            // insert into pending queue
            octree->addToPending(rb, model);
            return rb;
        }
    }
    return nullptr;
}


void Scene::initInstances(){
    avl_inorderTraverse(models, [](avl* node) -> void {
        ((Model*)node->val)->initInstances();
    });
}

void Scene::loadModels(){
    avl_inorderTraverse(models, [](avl* node) -> void {
        ((Model*)node->val)->init();
    });
}

void Scene::registerModel(Model* model){
    models = avl_insert(models, (void*)model->id.c_str(), model);
}

void Scene::removeInstance(std::string instanceId){
    /*
        Remove all locations
        -Scene::instances
        -Model::instances
    */
    RigidBody* instance = (RigidBody*)avl_get(instances, (void*)instanceId.c_str());
    
    std::string targetModel = instance->modelId;
    Model* model = (Model*)avl_get(models, (void*)targetModel.c_str());
    
    // delete instance from model
    model->removeInstance(instanceId);

    // remove from tree
    instances = avl_remove(instances, (void*)instanceId.c_str());
}

void Scene::markForDeletion(std::string instanceId){
    RigidBody* instance = (RigidBody*)avl_get(instances, (void*)instanceId.c_str());
    States::activate(&instance->state, INSTANCE_DEAD);  // Activate kill switch
    instancesToDelete.push_back(instance);
}

void Scene::clearDeadInstances(){
    for (RigidBody *rb : instancesToDelete){
        removeInstance(rb->instanceId);
    }
    instancesToDelete.clear();
}
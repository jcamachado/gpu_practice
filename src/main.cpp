#include "shader.h" //Including majority of the OpenGL headers
#include <GLFW/glfw3.h>
#include "../lib/stb/stb_image.h" // include glad to get all the required OpenGL headers

#include "io/joystick.h"
#include "io/keyboard.h"
#include "io/mouse.h"
#include "io/camera.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height); // Callback function for window resizing
void processInput(GLFWwindow *window, double dt); // Function for processing input
float mixValue = 0.5f;


glm::mat4 transform = glm::mat4(1.0f); // Create identity matrix (no transformation)
Joystick mainJ(0);
unsigned int SCR_WIDTH = 800, SCR_HEIGHT = 600;

Camera cameras[2] = {
    Camera(glm::vec3(0.0f, 0.0f, 3.0f)),
    Camera(glm::vec3(10.0f, 10.0f, 10.0f))
};
int activeCamera = 0;

float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame
float x, y, z;



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



    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL); // Create window
    glfwMakeContextCurrent( window ); // Set window as current context


    if (window == NULL){ // Check if window was created
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent( window ); // Set window as current context
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){ // Check if glad was loaded
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }


    glViewport(0,0, SCR_WIDTH, SCR_HEIGHT); // Set viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // Set callback function for window resizing

    glfwSetKeyCallback(window, Keyboard::keyCallback); // Set callback function for keyboard input
    
    glfwSetCursorPosCallback(window, Mouse::cursorPositionCallback); // Set callback function for mouse movement
    glfwSetMouseButtonCallback(window, Mouse::mouseButtonCallback); // Set callback function for mouse buttons
    glfwSetScrollCallback(window, Mouse::mouseScrollCallback); // Set callback function for mouse scroll

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Disable cursor, to look like a fps camera



    Shader shader("assets/vertex_core.glsl", "assets/fragment_core.glsl");
    Shader shader2("assets/vertex_core.glsl", "assets/fragment_core2.glsl");
    glEnable(GL_DEPTH_TEST); // Enable depth testing

    //each face of the cube have to have a texture
    float vertices[] = {
        //positions          //texture coords
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };
    

    // VAO is the vertex array object, it stores the vertex attribute calls
    // VBO is the vertex buffer object, it stores the vertex data

    // VAO and VBO
    // VAO, VBOare bound by the shader program

    unsigned int VAO, VBO; // Create VAO and VBO
    glGenBuffers(1, &VBO); // Generate VBO
    glGenVertexArrays(1, &VAO); // Generate VAO

    // Bind VAO and VBO
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // Set vertex data

    // Set vertex attributes pointers
    
    //positions, 0 is the first attribute, 6 is the stride, ja que vertices agora tem 6 floats por linha (vertex + color)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0); // Set vertex attribute pointer
    glEnableVertexAttribArray(0); // Enable vertex attribute pointer, index 0

    //texture coordinates, 8 floats per vertex, offset 6 floats 2 3d vectors to go through
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3*sizeof(float))); // Set vertex attribute pointer
    glEnableVertexAttribArray(1); // Enable vertex attribute pointer, index 2

    //Textures
    unsigned int texture1, texture2;
    glGenTextures(1, &texture1); // Generate texture
    glBindTexture(GL_TEXTURE_2D, texture1); // Bind texture to unit 0

    // Set texture wrapping and filtering options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); //x -> s, y -> t, z -> r
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Set texture filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // Load image, create texture and generate mipmaps
    int width, height, nChannels;
    stbi_set_flip_vertically_on_load(true); // Tell stb_image.h to flip loaded texture's on the y-axis.
    unsigned char *data = stbi_load("assets/image1.jpg", &width, &height, &nChannels, 0);
    if (data){
        //Texture1 is the currently one bound,
        //so it automatically knows and sets this data to the currently bound texture
        //by default, texture unit 0 is selected
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data); 
        glGenerateMipmap(GL_TEXTURE_2D); // Generate mipmaps
    }else{
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data); // Free image memory

    glGenTextures(1, &texture2); // Generate texture
    glBindTexture(GL_TEXTURE_2D, texture2); //Bind texture to unit 0 removing previous bond

    data = stbi_load("assets/image2.png", &width, &height, &nChannels, 0);
    if (data){
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data); 
        glGenerateMipmap(GL_TEXTURE_2D); // Generate mipmaps
    }else{
        std::cout << "Failed to load texture" << std::endl;
    }
    shader.activate(); //apply shader
    shader.setInt("texture1", 0); // Set texture uniform
    shader.setInt("texture2", 1); // Set texture uniform


    mainJ.update();
    if (mainJ.isPresent()){ 
        std::cout << "Joystick connected" << std::endl;
    }else{
        std::cout << "Joystick not connected" << std::endl;
    }

    x = 0.0f;
    y = 0.0f;
    z = 3.0f;
    while (!glfwWindowShouldClose(window)){ // Check if window should close
        double currentTime = glfwGetTime();
        deltaTime = currentTime - lastFrame;
        lastFrame = currentTime;
        processInput(window, deltaTime); // Process input

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Window background color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color buffer

        glActiveTexture(GL_TEXTURE0); // Activate texture
        glBindTexture(GL_TEXTURE_2D, texture1); // Bind texture to unit 0

        glActiveTexture(GL_TEXTURE1); // Activate texture
        glBindTexture(GL_TEXTURE_2D, texture2);  // Bind texture to unit 1 removing previous bond from unit 0 to texture2

        //draw shapes
        glBindVertexArray(VAO); // Bind VAO

        //create transformation for screen
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::mat4(1.0f);        
        glm::mat4 projection = glm::mat4(1.0f);

        model = glm::rotate(model, (float)glfwGetTime() * glm::radians(-55.0f), glm::vec3(0.5f)); // Rotate model
        // view = glm::translate(view, glm::vec3(-x, -y, -z)); // Translate view
        view = cameras[activeCamera].getViewMatrix();

        // the parameters are the field of view, the aspect ratio, the near clipping plane and the far clipping plane
        projection = glm::perspective(glm::radians(cameras[activeCamera].zoom), (float)SCR_WIDTH/(float)SCR_HEIGHT, 0.1f, 100.0f); // Create perspective projection

        shader.activate(); //apply shader
        shader.setMat4("model", model);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);

        shader.setFloat("mixValue", mixValue); // Set mix value uniform

        model = glm::rotate(model, (float)glfwGetTime()/100.0f, glm::vec3(0.5f, 1.0f, 0.0f)); // Rotate model
        
        glDrawArrays(GL_TRIANGLES, 0, 36); 

        glBindVertexArray(0); // Bind VAO

        glfwSwapBuffers(window); // Swap buffers
        glfwPollEvents(); // Check for events
    }

    glfwTerminate(); // Terminate glfw
    glDeleteVertexArrays(1, &VAO); // Delete VAO
    glDeleteBuffers(1, &VBO); // Delete VBO
    // glDeleteBuffers(1, &EBO); // Delete EBO

    glfwTerminate(); // Terminate glfw

    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height){ // Callback function for window resizing
    glViewport(0,0,width,height);
    SCR_WIDTH = width;
    SCR_HEIGHT = height;
}

void processInput(GLFWwindow *window, double dt){ // Function for processing input
    if(Keyboard::key(GLFW_KEY_ESCAPE) || mainJ.buttonState(GLFW_JOYSTICK_BTN_RIGHT)){ // Check if escape key is pressed
        glfwSetWindowShouldClose(window, true); // Set window to close
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

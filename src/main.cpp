// #include <iostream>
// #include "../lib/glad/glad.h" // GLAD is for loading OpenGL functions
// #include <GLFW/glfw3.h>
// #include <glm/glm.hpp> // GLM is for matrix math
// #include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtc/type_ptr.hpp>

// #include <fstream>
// #include <sstream>
// #include <streambuf>
// #include <string>

// #include "shader.h"

#include "shader.h" //Including majority of the OpenGL headers
#include <GLFW/glfw3.h>
#include "../lib/stb/stb_image.h" // include glad to get all the required OpenGL headers

#include "io/Joystick.h"
#include "io/Keyboard.h"
#include "io/Mouse.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height); // Callback function for window resizing
void processInput(GLFWwindow *window); // Function for processing input
float mixValue = 0.5f;


glm::mat4 transform = glm::mat4(1.0f); // Create identity matrix (no transformation)
Joystick mainJ(0);

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



    GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL); // Create window
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


    glViewport(0,0,800,600); // Set viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // Set callback function for window resizing

    glfwSetKeyCallback(window, Keyboard::keyCallback); // Set callback function for keyboard input
    glfwSetCursorPosCallback(window, Mouse::cursorPositionCallback); // Set callback function for mouse movement
    glfwSetMouseButtonCallback(window, Mouse::mouseButtonCallback); // Set callback function for mouse buttons
    glfwSetScrollCallback(window, Mouse::mouseScrollCallback); // Set callback function for mouse scroll


    Shader shader("assets/vertex_core.glsl", "assets/fragment_core.glsl");
    Shader shader2("assets/vertex_core.glsl", "assets/fragment_core2.glsl");

    // float vertices[] = { 
    //     //position              colors // Color interpolates through range!
    //     -0.25f, -0.5f, 0.0f,    1.0f, 1.0f, 0.5f,
    //      0.15f,  0.0f, 0.0f,    0.5f, 1.0f, 0.75f,
    //      0.0f,   0.5f, 0.0f,    0.6f, 1.0f, 0.2f,
    //      0.5f,  -0.4f, 0.0f,    1.0f, 0.2f, 1.0f,

    //     // //second triangle
    //     // 0.5f, -0.5f, 0.0f,
    //     // 0.25f, 0.5f, 0.0f,
    //     // 0.1f, -0.5f, 0.0f,
    // };

    //VBO will receive the texture coordinates
    //colors Interpolates through range!
    float vertices[] = { 
        //position              colors              texture coords (2D)
        -0.5f, -0.5f, 0.0f,     1.0f, 1.0f, 0.5f,   0.0f, 0.0f, //bottom left
        -0.5f,  0.5f, 0.0f,     0.5f, 1.0f, 0.75f,  0.0f, 1.0f, //top left
         0.5f, -0.5f, 0.0f,     0.6f, 1.0f, 0.2f,   1.0f, 0.0f, //bottom right
         0.5f,  0.5f, 0.0f,     1.0f, 0.2f, 1.0f,   1.0f, 1.0f, //top right

        // //second triangle
        // 0.5f, -0.5f, 0.0f,
        // 0.25f, 0.5f, 0.0f,
        // 0.1f, -0.5f, 0.0f,
    };


    unsigned int indices[] = { // Create indices
        0,1,2, // first triangle
        3,1,2// second triangle
    };
    
    
    // VAO and VBO
    // VAO, VBO and EBO are bound by the shader program

    unsigned int VAO, VBO, EBO; // Create VAO and VBO
    glGenBuffers(1, &VBO); // Generate VBO
    glGenVertexArrays(1, &VAO); // Generate VAO
    glGenBuffers(1, &EBO); // Generate EBO (Element Buffer Object)

    // Bind VAO and VBO
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // Set vertex data

    // Bind EBO (put index array in EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW); // Set element data

    // Set vertex attributes pointers
    
    //positions, 0 is the first attribute, 6 is the stride, ja que vertices agora tem 6 floats por linha (vertex + color)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0); // Set vertex attribute pointer
    glEnableVertexAttribArray(0); // Enable vertex attribute pointer, index 0

    //colors 1 is the second attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); // Set vertex attribute pointer
    glEnableVertexAttribArray(1); // Enable vertex attribute pointer, index 1

    //texture coordinates, 8 floats per vertex, offset 6 floats 2 3d vectors to go through
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6*sizeof(float))); // Set vertex attribute pointer
    glEnableVertexAttribArray(2); // Enable vertex attribute pointer, index 2

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


    glm::mat4 trans = glm::mat4(1.0f); // Create identity matrix (no transformation)
    trans = rotate (trans, glm::radians(45.0f), glm::vec3(0.0f,0.0f,1.0f));
    shader.activate();
    shader.setMat4("transform", trans);

    // glm::mat4 trans2 = glm::mat4(1.0f); // Create identity matrix (no transformation)
    // // trans2 = scale (trans2, glm::vec3(1.5f));
    // // trans2 = rotate (trans2, glm::radians(15.0f), glm::vec3(0.0f,0.0f,1.0f));
    // shader2.activate();
    // shader2.setMat4("transform", trans);

    mainJ.update();
    if (mainJ.isPresent()){ 
        std::cout << "Joystick connected" << std::endl;
    }else{
        std::cout << "Joystick not connected" << std::endl;
    }

    while (!glfwWindowShouldClose(window)){ // Check if window should close
        processInput(window); // Process input

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Window background color
        glClear(GL_COLOR_BUFFER_BIT); // Clear color buffer

        glActiveTexture(GL_TEXTURE0); // Activate texture
        glBindTexture(GL_TEXTURE_2D, texture1); // Bind texture to unit 0

        glActiveTexture(GL_TEXTURE1); // Activate texture
        glBindTexture(GL_TEXTURE_2D, texture2);  // Bind texture to unit 1 removing previous bond from unit 0 to texture2

        trans = rotate(trans, glm::radians((float)glfwGetTime()/100.0f), glm::vec3(0.0f,0.0f,1.0f));
        shader.activate(); //The subsequent changes are applied to the matrix just before the draw call
        shader.setMat4("transform", trans);

        //draw shapes
        glBindVertexArray(VAO); // Bind VAO
        shader.activate(); //apply shader
        shader.setFloat("mixValue", mixValue); // Set mix value uniform
        shader.setMat4("transform", transform);
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); // Draw triangle

        // trans2 = rotate(trans2, glm::radians((float)glfwGetTime()/-100.0f), glm::vec3(0.0f,0.0f,1.0f));
        // shader2.activate(); 
        // shader2.setMat4("transform", trans2);

        //draw second triangle, moving offset
        // shader2.activate();
        // glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (void*)((3*sizeof(unsigned int)))); // Draw triangle

        glBindVertexArray(0); // Bind VAO

        glfwSwapBuffers(window); // Swap buffers
        glfwPollEvents(); // Check for events
    }

    glfwTerminate(); // Terminate glfw
    glDeleteVertexArrays(1, &VAO); // Delete VAO
    glDeleteBuffers(1, &VBO); // Delete VBO
    glDeleteBuffers(1, &EBO); // Delete EBO

    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height){ // Callback function for window resizing
    glViewport(0,0,width,height);
}

void processInput   (GLFWwindow *window){ // Function for processing input
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

    if(Keyboard::key(GLFW_KEY_W)){
        transform = glm::translate(transform, glm::vec3(0.0f, 0.1f, 0.0f)); // Move up
    }
    if(Keyboard::key(GLFW_KEY_S)){
        transform = glm::translate(transform, glm::vec3(0.0f, -0.1f, 0.0f)); // Move down
    }
    if(Keyboard::key(GLFW_KEY_A)){
        transform = glm::translate(transform, glm::vec3(-0.1f, 0.0f, 0.0f)); // Move left
    }
    if(Keyboard::key(GLFW_KEY_D)){
        transform = glm::translate(transform, glm::vec3(0.1f, 0.0f, 0.0f)); // Move right
    }

    mainJ.update();

    /*
        if using Joystick
    */
   /*
   float lx = mainJ.axesState(GLFW_JOYSTICK_AXES_LEFT_STICK_X);
   float ly = mainJ.axesState(GLFW_JOYSTICK_AXES_LEFT_STICK_Y);

   if(std::abs(lx)>0.05f){// This value is the denser threshold(O quanto pro lado esta o stick)
         transform = glm::translate(transform, glm::vec3(lx/10.0f, 0.0f, 0.0f)); // Move up
   }
    if(std::abs(ly)>0.05f){ 
            transform = glm::translate(transform, glm::vec3(0.0f, ly/10.0f, 0.0f)); // Move up
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
}

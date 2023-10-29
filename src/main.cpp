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

#include "shader.h"
#include <GLFW/glfw3.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height); // Callback function for window resizing
void processInput(GLFWwindow *window); // Function for processing input

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

    Shader shader("assets/vertex_core.glsl", "assets/fragment_core.glsl");
    Shader shader2("assets/vertex_core.glsl", "assets/fragment_core2.glsl");

    float vertices[] = { 
        //position              colors // Color interpolates through range!
        -0.25f, -0.5f, 0.0f,    1.0f, 1.0f, 0.5f,
         0.15f,  0.0f, 0.0f,    0.5f, 1.0f, 0.75f,
         0.0f,   0.5f, 0.0f,    0.6f, 1.0f, 0.2f,
         0.5f,  -0.4f, 0.0f,    1.0f, 0.2f, 1.0f,

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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); // Set vertex attribute pointer
    glEnableVertexAttribArray(0); // Enable vertex attribute pointer, index 0

    //colors 1 is the second attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3*sizeof(float))); // Set vertex attribute pointer
    glEnableVertexAttribArray(1); // Enable vertex attribute pointer, index 1


    glm::mat4 trans = glm::mat4(1.0f); // Create identity matrix (no transformation)
    trans = rotate (trans, glm::radians(45.0f), glm::vec3(0.0f,0.0f,1.0f));
    shader.activate();
    shader.setMat4("transform", trans);

    glm::mat4 trans2 = glm::mat4(1.0f); // Create identity matrix (no transformation)
    trans2 = scale (trans2, glm::vec3(1.5f));
    trans2 = rotate (trans2, glm::radians(15.0f), glm::vec3(0.0f,0.0f,1.0f));
    shader2.activate();
    shader2.setMat4("transform", trans);

    while (!glfwWindowShouldClose(window)){ // Check if window should close
        processInput(window); // Process input

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Window background color
        glClear(GL_COLOR_BUFFER_BIT); // Clear color buffer

        trans = rotate(trans, glm::radians((float)glfwGetTime()/100.0f), glm::vec3(0.0f,0.0f,1.0f));
        shader.activate(); //The subsequent changes are applied to the matrix just before the draw call
        shader.setMat4("transform", trans);

        //draw shapes
        glBindVertexArray(VAO); // Bind VAO
        shader.activate(); //apply shader
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0); // Draw triangle

        trans2 = rotate(trans2, glm::radians((float)glfwGetTime()/-100.0f), glm::vec3(0.0f,0.0f,1.0f));
        shader2.activate(); 
        shader2.setMat4("transform", trans2);

        //draw second triangle, moving offset
        shader2.activate();
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (void*)((3*sizeof(unsigned int)))); // Draw triangle

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
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){ // Check if escape key is pressed
        glfwSetWindowShouldClose(window, true); // Set window to close
    }
}

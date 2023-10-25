#include <iostream>
#include "../lib/glad/glad.h" // GLAD is for loading OpenGL functions
#include <GLFW/glfw3.h>
#include <glm/glm.hpp> // GLM is for matrix math
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void framebuffer_size_callback(GLFWwindow* window, int width, int height); // Callback function for window resizing
void processInput(GLFWwindow *window); // Function for processing input


int main(){

    // //glm test
    // glm::vec4 vec(1.0f, 0.0f, 0.0f, 1.0f); // Create vector
    // glm::mat4 trans = glm::mat4(1.0f);
    
    
    // trans = glm::translate(trans, glm::vec3(1.0f, 1.0f, 0.0f));
    // trans = glm::rotate(trans, glm::radians(90.0f), glm::vec3(0.0f,0.0f,1.0f);
    // trans = glm::scale(trans, glm::vec3(0.5f, 0.5f, 0.5f));
    // vec = trans * vec;
    // std::cout << vec.x << ' ' << vec.y << ' ' << vec.z << std::endl;


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

    while (!glfwWindowShouldClose(window)){ // Check if window should close
        processInput(window); // Process input

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Window background color
        glClear(GL_COLOR_BUFFER_BIT); // Clear color buffer

        glfwSwapBuffers(window); // Swap buffers
        glfwPollEvents(); // Check for events
    }

    glfwTerminate(); // Terminate glfw

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
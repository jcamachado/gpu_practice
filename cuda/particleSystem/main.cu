// #include <GL/glew.h>
// #include <GLFW/glfw3.h>
// #include <iostream>
// #include <vector>
// #include <ctime>
// #include <cstdlib>
// #include <glm/glm.hpp>
// #include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtc/type_ptr.hpp>
// #include "jaxeUtils/jx_utils.h"

// using namespace std;

// vector<particle> particles;

// void draw_particles() {
//     glLoadIdentity();                 // Reset the model-view matrix
//     glPointSize(5.0f);
//     glBegin(GL_POINTS);
//     for (int i = 0; i < particles.size(); i++) {
//         glVertex3f(particles[i].position.x, particles[i].position.y, particles[i].position.z);
//     }
//     glEnd();
// }

// // void update_particles(float dt) {
// //     // Ideally, summing update would receive a sum of forces
// //     for (int i = 0; i < particles.size(); i++) {
// //         particles[i].applyForce(weightForce(particles[i].mass));
// //         particles[i].update(dt);
// //     }
// // }
// __global__ void update_particles_kernel(particle* particles, float dt, int num_particles) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < num_particles) {
//         particles[i].applyForce(weightForce(particles[i].mass));
//         particles[i].update(dt);
//     }
// }
// void update_particles(float dt) {
//     particle* d_particles;
//     cudaMalloc(&d_particles, particles.size() * sizeof(particle));
//     cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(particle), cudaMemcpyHostToDevice);

//     int block_size = 256;
//     int num_blocks = (particles.size() + block_size - 1) / block_size;
//     update_particles_kernel<<<num_blocks, block_size>>>(d_particles, dt, particles.size());

//     cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(particle), cudaMemcpyDeviceToHost);
//     cudaFree(d_particles);
// }

// void drawCube(float scale=1.0f, bool isWireframe=true) {
//     glLoadIdentity();                 // Reset the model-view matrix
//     if (isWireframe){
//         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);  // this tells it to only render lines
//     }
//     glTranslatef(0.0f, -1.0f, -10.0f);  // Move further into the screen
//     glScalef(scale,scale,scale); // keep proportion
    
//     glBegin(GL_QUADS);                // Begin drawing the color cube with 6 quads
//         // Top face (y = 1.0f)
//         // Define vertices in counter-clockwise (CCW) order with normal pointing out
//         //   glColor3f(0.0f, 1.0f, 0.0f);     // Green
//         glVertex3f( 1.0f, 1.0f, -1.0f);
//         glVertex3f(-1.0f, 1.0f, -1.0f);
//         glVertex3f(-1.0f, 1.0f,  1.0f);
//         glVertex3f( 1.0f, 1.0f,  1.0f);

//         // Bottom face (y = -1.0f)
//         //   glColor3f(1.0f, 0.5f, 0.0f);     // Orange
//         glVertex3f( 1.0f, -1.0f,  1.0f);
//         glVertex3f(-1.0f, -1.0f,  1.0f);
//         glVertex3f(-1.0f, -1.0f, -1.0f);
//         glVertex3f( 1.0f, -1.0f, -1.0f);

//         // Front face  (z = 1.0f)
//         //   glColor3f(1.0f, 0.0f, 0.0f);     // Red
//         glVertex3f( 1.0f,  1.0f, 1.0f);
//         glVertex3f(-1.0f,  1.0f, 1.0f);
//         glVertex3f(-1.0f, -1.0f, 1.0f);
//         glVertex3f( 1.0f, -1.0f, 1.0f);

//         // Back face (z = -1.0f)
//         //  glColor3f(1.0f, 1.0f, 0.0f);     // Yellow
//         glVertex3f( 1.0f, -1.0f, -1.0f);
//         glVertex3f(-1.0f, -1.0f, -1.0f);
//         glVertex3f(-1.0f,  1.0f, -1.0f);
//         glVertex3f( 1.0f,  1.0f, -1.0f);

//         // Left face (x = -1.0f)
//         //   glColor3f(0.0f, 0.0f, 1.0f);     // Blue
//         glVertex3f(-1.0f,  1.0f,  1.0f);
//         glVertex3f(-1.0f,  1.0f, -1.0f);
//         glVertex3f(-1.0f, -1.0f, -1.0f);
//         glVertex3f(-1.0f, -1.0f,  1.0f);

//         // Right face (x = 1.0f)
//         //   glColor3f(1.0f, 0.0f, 1.0f);     // Magenta
//         glVertex3f(1.0f,  1.0f, -1.0f);
//         glVertex3f(1.0f,  1.0f,  1.0f);
//         glVertex3f(1.0f, -1.0f,  1.0f);
//         glVertex3f(1.0f, -1.0f, -1.0f);
//     glEnd();  // End of drawing color-cube
// }

// int main() {
//     // Initialize GLFW
//     if (!glfwInit()) {
//         std::cerr << "Failed to initialize GLFW" << std::endl;
//         return -1;
//     }

//     // Create a windowed mode window and its OpenGL context
//     GLFWwindow* window = glfwCreateWindow(1920, 1080, "Particle System", NULL, NULL);

//     if (!window) {
//         std::cerr << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return -1;
//     }
//     // Make the window's context current (must be done before initializing GLEW)
//     glfwMakeContextCurrent(window);

//     // Initialize GLEW
//     if (glewInit() != GLEW_OK) {
//         std::cerr << "Failed to initialize GLEW" << std::endl;
//         glfwTerminate();
//         return -1;
//     }

//     // Set up projection matrix
//     glMatrixMode(GL_PROJECTION);
//     glLoadIdentity();
//     gluPerspective(45.0f, 640.0f / 480.0f, 0.1f, 100.0f);

//     // Set up view matrix
//     glMatrixMode(GL_MODELVIEW);
//     glLoadIdentity();
//     gluLookAt(0.0f, 0.0f, 5.0f,  // Camera position
//               0.0f, 0.0f, 0.0f,  // Look at point
//               0.0f, 1.0f, 0.0f); // Up vector
    
    

//     // Initialize particles
//     particles = generateParticles();

//     // Loop until the user closes the window
//     while (!glfwWindowShouldClose(window)) {
//         // Clear the screen
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//         // Draw cube
//         drawCube(2.0f);

//         // Update particles
//         float dt = 0.01f;
//         update_particles(dt);

//         // Draw particles
//         draw_particles();

//         // Swap buffers
//         glfwSwapBuffers(window);

//         // Poll for and process events
//         glfwPollEvents();
//     }

//     // Clean up
//     glfwTerminate();

//     return 0;
// }


// #include <GL/glew.h>
#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "jaxeUtils/jx_utils.h"

using namespace std;

vector<particle> particles;

void draw_particles() {
    glLoadIdentity();                 // Reset the model-view matrix
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < particles.size(); i++) {
        glVertex3f(particles[i].position.x, particles[i].position.y, particles[i].position.z);
    }
    glEnd();
}

// void update_particles(float dt) {
//     // Ideally, summing update would receive a sum of forces
//     for (int i = 0; i < particles.size(); i++) {
//         particles[i].applyForce(weightForce(particles[i].mass));
//         particles[i].update(dt);
//     }
// }
__global__ void update_particles_kernel(particle* particles, float dt, int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        particles[i].applyForce(weightForce(particles[i].mass));
        particles[i].update(dt);
    }
}
void update_particles(float dt) {
    particle* d_particles;
    cudaMalloc(&d_particles, particles.size() * sizeof(particle));
    cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(particle), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (particles.size() + block_size - 1) / block_size;
    update_particles_kernel<<<num_blocks, block_size>>>(d_particles, dt, particles.size());

    cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);
}

void drawCube(float scale=1.0f, bool isWireframe=true) {
    glLoadIdentity();                 // Reset the model-view matrix
    if (isWireframe){
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);  // this tells it to only render lines
    }
    glTranslatef(0.0f, -1.0f, -10.0f);  // Move further into the screen
    glScalef(scale,scale,scale); // keep proportion
    
    glBegin(GL_QUADS);                // Begin drawing the color cube with 6 quads
        // Top face (y = 1.0f)
        // Define vertices in counter-clockwise (CCW) order with normal pointing out
        //   glColor3f(0.0f, 1.0f, 0.0f);     // Green
        glVertex3f( 1.0f, 1.0f, -1.0f);
        glVertex3f(-1.0f, 1.0f, -1.0f);
        glVertex3f(-1.0f, 1.0f,  1.0f);
        glVertex3f( 1.0f, 1.0f,  1.0f);

        // Bottom face (y = -1.0f)
        //   glColor3f(1.0f, 0.5f, 0.0f);     // Orange
        glVertex3f( 1.0f, -1.0f,  1.0f);
        glVertex3f(-1.0f, -1.0f,  1.0f);
        glVertex3f(-1.0f, -1.0f, -1.0f);
        glVertex3f( 1.0f, -1.0f, -1.0f);

        // Front face  (z = 1.0f)
        //   glColor3f(1.0f, 0.0f, 0.0f);     // Red
        glVertex3f( 1.0f,  1.0f, 1.0f);
        glVertex3f(-1.0f,  1.0f, 1.0f);
        glVertex3f(-1.0f, -1.0f, 1.0f);
        glVertex3f( 1.0f, -1.0f, 1.0f);

        // Back face (z = -1.0f)
        //  glColor3f(1.0f, 1.0f, 0.0f);     // Yellow
        glVertex3f( 1.0f, -1.0f, -1.0f);
        glVertex3f(-1.0f, -1.0f, -1.0f);
        glVertex3f(-1.0f,  1.0f, -1.0f);
        glVertex3f( 1.0f,  1.0f, -1.0f);

        // Left face (x = -1.0f)
        //   glColor3f(0.0f, 0.0f, 1.0f);     // Blue
        glVertex3f(-1.0f,  1.0f,  1.0f);
        glVertex3f(-1.0f,  1.0f, -1.0f);
        glVertex3f(-1.0f, -1.0f, -1.0f);
        glVertex3f(-1.0f, -1.0f,  1.0f);

        // Right face (x = 1.0f)
        //   glColor3f(1.0f, 0.0f, 1.0f);     // Magenta
        glVertex3f(1.0f,  1.0f, -1.0f);
        glVertex3f(1.0f,  1.0f,  1.0f);
        glVertex3f(1.0f, -1.0f,  1.0f);
        glVertex3f(1.0f, -1.0f, -1.0f);
    glEnd();  // End of drawing color-cube
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Particle System", NULL, NULL);

    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    // Make the window's context current (must be done before initializing GLEW)
    glfwMakeContextCurrent(window);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Set up projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, 640.0f / 480.0f, 0.1f, 100.0f);

    // Set up view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0f, 0.0f, 5.0f,  // Camera position
              0.0f, 0.0f, 0.0f,  // Look at point
              0.0f, 1.0f, 0.0f); // Up vector
    
    

    // Initialize particles
    particles = generateParticles();

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw cube
        drawCube(2.0f);

        // Update particles
        float dt = 0.01f;
        update_particles(dt);

        // Draw particles
        draw_particles();

        // Swap buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Clean up
    glfwTerminate();

    return 0;
}


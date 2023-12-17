#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "jaxeUtils/jx_physics.h"
#include "jaxeUtils/jx_particle.h"
#include "io/keyboard.cpp"



using namespace std;



void drawCube(float scale, bool isWireframe);
void draw_particles();
void update_particles(float dt);
void processInput(double dt, GLFWwindow* window); // Function for processing input
double dt = 0.0f;       // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame
int num_blocks;
int block_size;

Particle *particles;
int currentAmount = 1;

// void update_particles(float dt) {
//     // Ideally, summing update would receive a sum of forces
//     for (int i = 0; i < particles.size(); i++) {
//         particles[i].applyForce(weightForce(particles[i].mass));
//         particles[i].update(dt);
//     }
// }
__global__ void update_particles_kernel(Particle *particles, float dt, int Nparticles) {
    glm::vec3 gravity = glm::vec3(0.0f, -9.8f, 0.0f);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nparticles) {
        updateVelocity(particles[i].vel, gravity,  dt);
        updatePosition(particles[i].pos, particles[i].vel, dt);
    }
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
    glfwSetKeyCallback(window, Keyboard::keyCallback);

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
    particles = (Particle*)malloc(sizeof(Particle) * currentAmount);
    generateParticles(particles, currentAmount, 30);
    int block_size = 256;
    int num_blocks = (sizeof(Particle) + block_size - 1) / block_size;

    dt = 0.0f;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        double currentTime = glfwGetTime();
        dt = currentTime - lastFrame;
        lastFrame = currentTime;
        // processInput(dt, window);
        std::cout << dt << std::endl;
        // Draw cube
        drawCube(2.0f, true);

        // Update particles
        update_particles(dt);

        // Swap buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Clean up
    glfwTerminate();

    return 0;
}

void update_particles(float dt) {   // gpu version
    Particle *d_particles;
    cudaMalloc(&d_particles, sizeof(Particle) * currentAmount);
    cudaMemcpy(d_particles, (void*)(sizeof(Particle) * currentAmount), sizeof(Particle) * currentAmount, cudaMemcpyHostToDevice);
    
    update_particles_kernel<<<num_blocks, block_size>>>(d_particles, dt, sizeof(Particle) * currentAmount);

    cudaMemcpy(particles, d_particles, sizeof(Particle) * currentAmount, cudaMemcpyDeviceToHost);
    cudaFree(d_particles);
    
    // Draw particles
    draw_particles();
}

void processInput(double dt, GLFWwindow* window){ // Function for processing input
    // if(Keyboard::key(GLFW_KEY_ESCAPE) || mainJ.buttonState(GLFW_JOYSTICK_BTN_RIGHT)){ // Check if escape key is pressed
    if(Keyboard::key(GLFW_KEY_ESCAPE)){
        glfwSetWindowShouldClose(window, true);
    }

    // if (Keyboard::keyWentDown(GLFW_KEY_F)){
    //     launchItem(dt);
    // }
    if (Keyboard::keyWentDown(GLFW_KEY_F)){
        generateParticles(particles, currentAmount, 30);
        
    }

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

void draw_particles() {
    glLoadIdentity();                 // Reset the model-view matrix
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < currentAmount; i++) {
        std::cout << particles[i].pos.x << " " << particles[i].pos.y << " " << particles[i].pos.z << std::endl;
        glVertex3f(particles[i].pos.x, particles[i].pos.y, particles[i].pos.z);
    }
    glEnd();
}
// #include "../../lib/glad/glad.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "jaxeUtils/jx_particle.h"
#include "jaxeUtils/jx_physics.h"
#include "jaxeUtils/jx_math.h"

#define NUM_PARTICLES 100000
#define GRAV_ACC -9.8f
// Posx Posy Posz Velx Vely Velz Accx Accy Accz
float nParts[9*NUM_PARTICLES] = {0.0f};
float *d_nParts;

using namespace std;
void drawCube(float scale);
void draw_particles();


__global__ void update_particles_kernel(float *part, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index * 9;
    int stride = blockDim.x * gridDim.x;
    for (; i < NUM_PARTICLES; i += stride){
        part[i+7] =  -9.8f;           // Apply gravity
        // Update velocity            
        part[i+3] += part[i+6] * dt;
        part[i+4] += part[i+7] * dt;
        part[i+5] += part[i+8] * dt;

        part[i+0] += part[i+3] * dt;
        part[i+1] += part[i+4] * dt;
        part[i+2] += part[i+5] * dt;
    }
}

void update_particles_host(float *part, float dt) {
    for (int i = 0; i < NUM_PARTICLES*9; i+=9) {
        part[i+7] =  -9.8f;;           // Apply gravity
        // Update velocity          
        
        part[i+3] += part[i+6] * dt;
        part[i+4] += part[i+7] * dt;
        part[i+5] += part[i+8] * dt;
        // Update position
        part[i+0] += part[i+3] * dt;
        part[i+1] += part[i+4] * dt;
        part[i+2] += part[i+5] * dt;
    }
}

void generatePartsRaw(){
    cudaMalloc((void**)&d_nParts, 9*NUM_PARTICLES*sizeof(float));
    for (int i=0; i<NUM_PARTICLES*9; i+=9){
        float rightNum = (float)randomInt(-100, 100)/100.0f;
        rightNum = rightNum ? rightNum : 0.1f;
        float velX = (float)randomInt(-1, 1)+rightNum;
        float velY = (float)randomInt(0, 30)+rightNum;
        float velZ = (float)randomInt(-2, 2)+rightNum;
        // pos
        nParts[i + 0] = velX+1.0f;
        nParts[i + 1] = velY + 50.0f;
        nParts[i + 2] = velZ+1.0f;    
        // vel
        nParts[i + 3] = velX;
        nParts[i + 4] = velY + 100.0f;
        nParts[i + 5] = velZ;
        // acc + 
        nParts[i + 6] = 0.0f;
        nParts[i + 7] = 0.0f;
        nParts[i + 8] = 0.0f;
    }
}


void update_particles(float dt,int nBlocks=0,  int nThreads=0) {
    if (nBlocks == 0 || nThreads == 0){
        update_particles_host(nParts, dt);
        return;
    }
    cudaMemcpy(d_nParts, nParts, 9*NUM_PARTICLES*sizeof(float), cudaMemcpyHostToDevice);
    update_particles_kernel<<<nBlocks, nThreads>>>(d_nParts, dt);
    cudaMemcpy(nParts, d_nParts, 9*NUM_PARTICLES*sizeof(float), cudaMemcpyDeviceToHost);
}


int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Particle System", NULL, NULL);

    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    // Make the window's context current (must be done before initializing GLEW)
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        return -1;
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, 640.0f / 480.0f, 0.1f, 100.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0f, 0.0f, 5.0f,  // Camera position
              0.0f, 0.0f, 0.0f,  // Look at point
              0.0f, 1.0f, 0.0f); // Up vector
    generatePartsRaw();
    int nThreads = 36;
    int nBlocks = 40;
        
    double dt = 0.0f;
    double lastFrame = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        double currentTime = glfwGetTime();
        dt = currentTime - lastFrame;
        lastFrame = currentTime;
        std::cout << "FPS: " << 1.0f/dt << std::endl;

        update_particles(dt);
        // update_particles(dt, nBlocks, nThreads);
        draw_particles();
        drawCube(2.0f);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    cudaFree(d_nParts);
    glfwTerminate();
    return 0;
}


void draw_particles() {
    glLoadIdentity();                 // Reset the model-view matrix
    glPointSize(5.0f);

    glBegin(GL_POINTS);
    // set color to white
    for (int i = 0; i < NUM_PARTICLES*9; i+=9) {
        if (nParts[i + 1] < -300.0f) {
            nParts[i + 1] = 30.0f;
            nParts[i + 4] = 0.0f;
        }
        glVertex3f(nParts[i + 0], nParts[i + 1], nParts[i + 2]);
    }
    glEnd();
}

void drawCube(float scale) {
    glLoadIdentity();                 // Reset the model-view matrix
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);  // this tells it to only render lines
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

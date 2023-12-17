//Abstracao da particula

// Std. Includes
#include <vector>

//cuda
#include <cuda_runtime.h>

// GL Includes

#include <GL/glut.h>
#include "glm/glm.hpp"
#include <vector>
#include "jx_math.h"

using namespace std;
// talvez separar generator de painter


typedef struct Particle {
    glm::vec3 pos;
    glm::vec3 vel;
    glm::vec4 color;
    float size;
} Particle;


void generateParticles(Particle *particles, int &currentAmount, unsigned int amount=1) {
    currentAmount += amount;
            
    for (int i = 0; i < currentAmount; i++) {
        int rx = randomInt(-10, 10);
        int ry = -randomInt(5, 10);
        int rz = randomInt(-10, 10);
        int randomNum = randomInt(0, 10);
        int randomDen = randomInt(1, 10);
        float randomFactor = (float)randomNum / (float)randomDen;
        glm::vec3 randomVec = glm::vec3(rx, ry, rz);
        glm::vec3 pos(0.0f, 0.0f, 0.0f);
        glm::vec3 vel(rx, ry, rz);
        glm::vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
        float size = 0.1f;
        Particle p = {pos, vel, color, size};
        particles[i] = p;
    }
}


// class particle {
// public:
//     glm::vec3 position;
//     glm::vec3 velocity;
//     glm::vec4 color;
//     float size;
//     int lifeTime;

//     glm::vec3 acceleration; //resulting from forces/mass
    
//     float mass;

//     // particle(glm::vec3 pos, glm::vec3 vel, glm::vec3 acc, glm::vec4 colr , float s) {
//     particle(glm::vec3 pos, glm::vec3 vel, glm::vec4 colr , float siz, float mas, int life=100) {
//         position = pos;
//         velocity = vel;
//         // acceleration = acc;
//         color = colr;
//         size = siz;
//         mass = mas;
//         lifeTime = life;
//     }


//     void update(float dt) {
//         // pensando como a funcao de euler, 
//         // newPosition = oldPosition + dt * (Forcas/massa)
//         // onde forcas/massa = derivEval
//         velocity += acceleration * dt;
//         position += velocity * dt;

//         // lifeTime--;
//         acceleration *=0; //reset acceleration, its not cumulative
//     }
// };
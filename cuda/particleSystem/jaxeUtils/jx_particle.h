//Abstracao da particula

// Std. Includes
#include <vector>

//cuda
#include <cuda_runtime.h>

// GL Includes
#include <glm/glm.hpp>
class particle {
public:
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec4 color;
    float size;
    int lifeTime;

    glm::vec3 acceleration; //resulting from forces/mass
    
    float mass;

    // particle(glm::vec3 pos, glm::vec3 vel, glm::vec3 acc, glm::vec4 colr , float s) {
    __host__ __device__  particle(glm::vec3 pos, glm::vec3 vel, glm::vec4 colr , float siz, float mas, int life=100) {
        position = pos;
        velocity = vel;
        // acceleration = acc;
        color = colr;
        size = siz;
        mass = mas;
        lifeTime = life;
    }
    __device__ void applyForce(glm::vec3 force) {
        acceleration += force;
    }

    // void update(float dt) {
    //     velocity += acceleration * dt;
    //     position += velocity * dt;
    // }

    __host__ __device__ void update(float dt) {
        // pensando como a funcao de euler, 
        // newPosition = oldPosition + dt * (Forcas/massa)
        // onde forcas/massa = derivEval
        velocity += acceleration * dt;
        position += velocity * dt;

        // lifeTime--;
        acceleration *=0; //reset acceleration, its not cumulative
    }


    
};
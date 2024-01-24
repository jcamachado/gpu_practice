// //Abstracao da particula

// // Std. Includes

// //cuda
// #include <cuda_runtime.h>

// // GL Includes
// #include <GL/glut.h>
// #include <glm/glm.hpp>

// #include <vector>
// #include "jx_math.h"

// #define NPARTICLES 1000000

// using namespace std;

// typedef struct particleRaw{
//     float posX, posY, posZ;
//     float velX, velY, velZ;
//     float accX, accY, accZ;
//     float size;
// } float3;


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
//     __host__ __device__  particle(glm::vec3 pos, glm::vec3 vel, glm::vec4 colr , float siz, float mas, int life=100) {
//         position = pos;
//         velocity = vel;
//         // acceleration = acc;
//         color = colr;
//         size = siz;
//         mass = mas;
//         lifeTime = life;
//     }
//     __device__ void applyForce(glm::vec3 force) {
//         acceleration += force;
//     }

//     // void update(float dt) {
//     //     velocity += acceleration * dt;
//     //     position += velocity * dt;
//     // }
//     void applyForceHost(glm::vec3 force) {
//         acceleration += force;
//     }

//     // void update(float dt) {
//     //     velocity += acceleration * dt;
//     //     position += velocity * dt;
//     // }

//     __device__ void update(float dt) {
//         // pensando como a funcao de euler, 
//         // newPosition = oldPosition + dt * (Forcas/massa)
//         // onde forcas/massa = derivEval
        
//         velocity += (acceleration * dt);
//         position += velocity * dt;

//         // lifeTime--;
//         acceleration *=0; //reset acceleration, its not cumulative
//     }
//     void updateHost(float dt) {
//         // pensando como a funcao de euler, 
//         // newPosition = oldPosition + dt * (Forcas/massa)
//         // onde forcas/massa = derivEval
        
//         velocity += (acceleration * dt);
//         position += velocity * dt;

//         // lifeTime--;
//         acceleration *=0; //reset acceleration, its not cumulative
//     }
// };

// // talvez separar generator de painter

// // void generateParticlesRaw(int amount=NPARTICLES) {
// //     vector<particleRaw> particles;

// //     srand(time(NULL));
// //     for (int i = 0; i < amount; i++) {
// //         glm::vec3 pos(randomInt(-10, 10),  2 * randomInt(0, 100) - 2, randomInt(-45, -40));
// //         glm::vec3 vel(0.0f, 0.0f, 0.0f);
// //         glm::vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
// //         float size = 0.1f;
// //         float mass = 1.0f;
// //         particle p(pos, vel, color, size, mass);
// //         particles.push_back(p);
// //     }
// //     return particles;
// // }

// // vector<particle> generateParticles(int amount=NPARTICLES) {
//     // vector<particle> particles;

//     // srand(time(NULL));
//     // for (int i = 0; i < amount; i++) {
//     //     glm::vec3 pos(randomInt(-10, 10),  2 * randomInt(0, 100) - 2, randomInt(-45, -40));
//     //     glm::vec3 vel(0.0f, 0.0f, 0.0f);
//     //     glm::vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
//     //     float size = 0.1f;
//     //     float mass = 1.0f;
//     //     particle p(pos, vel, color, size, mass);
//     //     particles.push_back(p);
//     // }
//     // return particles;
//     // generateParticlesRaw(amount);
// // }

// inclide std
#include <glm/glm.hpp>
#include <cuda_runtime.h>

// #define GRAVITY glm::vec3(0.0f, -9.8f, 0.0f)

// Weigth of a particle
// Weight = mass * gravity
// glm::vec3 weightForce(float mass){
//     return GRAVITY * mass;
// }
// __device__ glm::vec3 weightForce(float mass){
//         return GRAVITY * mass;
// }
// __device__ glm::vec3 addForce(float &x, float &y, float &z, float magnitude){
//         x += magnitude; 
//         y += magnitude;
//         z += magnitude;
// }

// __device__ void updateVelocity(
//         float &velocityX, float &velocityY, float &velocityZ, 
//         float &accelX, float &accelY, float &accelZ, 
//         float dt) 
// {
//         velocityX += accelX * dt;
//         velocityY += accelY * dt;
//         velocityZ += accelZ * dt;
// }
__device__ void updateVelocity(glm::vec3 &velocity, glm::vec3 &accel, float dt)
{
    velocity.x += accel.x * dt;
    velocity.y += accel.y * dt;
    velocity.z += accel.z * dt;
}


// __device__ void updatePosition(
//         float &posX, float &posY, float &posZ, 
//         float &velocityX, float &velocityY, float &velocityZ, 
//         float dt) 
// {
//         posX += velocityX * dt;
//         posY += velocityY * dt;
//         posZ += velocityZ * dt;
     
// }  
__device__ void updatePosition(glm::vec3 &position, glm::vec3 &velocity, float dt)
{
        position.x += velocity.x * dt;
        position.y += velocity.y * dt;
        position.z += velocity.z * dt;
}

// Drag of a particle
// Drag = -velocity * drag
// glm::vec3 returnDrag(glm::vec3 velocity, float drag){
//     return -(velocity * drag);
// }

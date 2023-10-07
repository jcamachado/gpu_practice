// inclide std
#include <glm/glm.hpp>

#define GRAVITY glm::vec3(0.0f, -9.8f, 0.0f)

// Weigth of a particle
// Weight = mass * gravity
glm::vec3 weightForce(float mass){
    return GRAVITY * mass;
}

// Drag of a particle
// Drag = -velocity * drag
// glm::vec3 returnDrag(glm::vec3 velocity, float drag){
//     return -(velocity * drag);
// }

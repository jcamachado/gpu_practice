
// Std. Includes
#include <vector>

// GL Includes
#include <glm/glm.hpp>
class particle {
public:
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec4 color;
    float size;

    glm::vec3 acceleration; //resulting from forces/mass
    float mass;


    // particle(glm::vec3 pos, glm::vec3 vel, glm::vec3 acc, glm::vec4 colr , float s) {
    particle(glm::vec3 pos, glm::vec3 vel, glm::vec4 colr , float siz, float mas) {
        position = pos;
        velocity = vel;
        // acceleration = acc;
        color = colr;
        size = siz;
        mass = mas;
    }

    void update(float dt, glm::vec3 force) {
        acceleration = force/mass;
        velocity += acceleration * dt;
        position += velocity * dt;
    }
};

// Std. Includes
#include <vector>

// GL Includes
#include <glm/glm.hpp>
class particle {
public:
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 acceleration;
    glm::vec4 color;
    float size;


    particle(glm::vec3 pos, glm::vec3 vel, glm::vec3 acc, glm::vec4 colr , float s) {
        position = pos;
        velocity = vel;
        acceleration = acc;
        color = colr;
        size = s;
    }

    void update(float dt) {
        velocity += acceleration * dt;
        position += velocity * dt;
    }
};
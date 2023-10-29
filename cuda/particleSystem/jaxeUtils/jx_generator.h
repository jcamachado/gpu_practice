#include <GL/glut.h>
#include "glm/glm.hpp"
#include <vector>
#include "jx_particle.h"
#include "jx_math.h"
#include "jx_geometry.h"

#define NPARTICLES 100000

using namespace std;
// talvez separar generator de painter

vector<particle> generateParticles(int amount=NPARTICLES) {
    vector<particle> particles;

    srand(time(NULL));
    for (int i = 0; i < amount; i++) {
        glm::vec3 pos(randomInt(-10, 10),  2 * randomInt(0, 100) - 2, randomInt(-45, -40));
        glm::vec3 vel(0.0f, 0.0f, 0.0f);
        glm::vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
        float size = 0.1f;
        float mass = 1.0f;
        particle p(pos, vel, color, size, mass);
        particles.push_back(p);
    }
    return particles;
}


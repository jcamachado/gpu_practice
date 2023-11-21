#ifndef RIGIDBODY_H
#define RIGIDBODY_H

#include <glm/glm.hpp>

#include <string>

#define INSTANCE_DEAD       (unsigned char)0b00000001
#define INSTANCE_MOVED      (unsigned char)0b00000010

class RigidBody {
    public:
        /*
            State
            -_state_ will be read by the octree to know if it should update the position of the instance
        */
        unsigned char state;

        float mass;
        glm::vec3 pos, velocity, acceleration, size;

        std::string modelId;
        std::string instanceId;

        bool operator==(RigidBody rb);
        bool operator==(std::string id);

        RigidBody();

        RigidBody(std::string modelId, 
            glm::vec3 size = glm::vec3(1.0f), 
            float mass= 1.0f,
            glm::vec3 pos = glm::vec3(0.0f)
        );

        void update(float dt);
        /*
            Movement methods        
            -Apply force -> change in acceleration
            -Apply impulse -> change in velocity J = delta p = F * dt = Favg * dt -> v = v0 + a * dt
            -Energy transfer -> momentum E = 1/2 * m * v^2
            -Same as applyForce but it doesn't divide by mass (optimization?)

        */
        void applyForce(glm::vec3 force);
        void applyForce(glm::vec3 direction, float magnitude);

        void applyAcceleration(glm::vec3 a);
        void applyAcceleration(glm::vec3 direction, float magnitude);
        
        void applyImpulse(glm::vec3 force, float dt);
        void applyImpulse(glm::vec3 direction, float magnitude, float dt);

        void transferEnergy(float joules, glm::vec3 direction);
};


#endif
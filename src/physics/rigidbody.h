#ifndef RIGIDBODY_H
#define RIGIDBODY_H

#include <glm/glm.hpp>

#include <string>

class RigidBody {
    public:
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
        //apply force -> change in acceleration
        //apply impulse -> change in velocity J = delta p = F * dt = Favg * dt -> v = v0 + a * dt
        //Energy transfer -> momentum E = 1/2 * m * v^2

        void applyForce(glm::vec3 force);
        void applyForce(glm::vec3 direction, float magnitude);

        //same as applyForce but it doesn't divide by mass (optimization?)
        void applyAcceleration(glm::vec3 a);
        void applyAcceleration(glm::vec3 direction, float magnitude);
        
        void applyImpulse(glm::vec3 force, float dt);
        void applyImpulse(glm::vec3 direction, float magnitude, float dt);

        void transferEnergy(float joules, glm::vec3 direction);

};


#endif
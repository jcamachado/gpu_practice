#ifndef RIGIDBODY_H
#define RIGIDBODY_H

#include <glm/glm.hpp>

#include <string>

#define INSTANCE_DEAD       (unsigned char)0b00000001
#define INSTANCE_MOVED      (unsigned char)0b00000010

#define COLLISION_THRESHOLD 0.05f   // in seconds

class RigidBody {
    public:
        unsigned char state;                    // Combination of above switches in octree

        // Physics
        float mass;                             // kg
        glm::vec3 size;                         // Dimensions of the object

        //Linear
        glm::vec3 pos, velocity, acceleration;  // (m, m/s, m/s^2  
        
        // Rotation in Euler angles
        glm::vec3 eRotation; // (rad, rad/s, rad/s^2)

        // Model matrix
        glm::mat4 model;
        glm::mat3 normalModel;   // From tangent spaces

        /*
            Freeze values. Custom made by me.
            Used to freeze motion of model objects
        */
        glm::vec3 storedVelocity, storedAcceleration;
        
        /*
            Ids. For fast access to the model and instance
        */
        std::string modelId;
        std::string instanceId;

        /*
            Data from previous collision
        */
        float lastCollision;            // Time of last collision
        std::string lastCollisionId;

        bool operator==(RigidBody rb);
        bool operator==(std::string id);

        /*
            Constructors
        */
        RigidBody(std::string modelId = "", 
            glm::vec3 size = glm::vec3(1.0f), 
            float mass= 1.0f,
            glm::vec3 pos = glm::vec3(0.0f),
            glm::vec3 eRotation = glm::vec3(0.0f)
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
        bool freeze();
        bool unfreeze();

        /*
            Collisions
        */
        void handleCollision(RigidBody* inst, glm::vec3 normal);
};


#endif
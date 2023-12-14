#include "rigidbody.h"

#include <glm/gtc/matrix_transform.hpp>
// Quaternion: Number system that extends complex numbers to 3 dimensions
// a+b*i+c*j+d*k where i^2 = j^2 = k^2 = ijk = -1
#include <glm/gtc/quaternion.hpp>   
#include <glm/gtx/quaternion.hpp>

bool RigidBody::operator==(RigidBody rb){
    return instanceId == rb.instanceId;
}

bool RigidBody::operator==(std::string id){
    return instanceId == id;
}

RigidBody::RigidBody(
    std::string modelId, 
    glm::vec3 size,
    float mass,
    glm::vec3 pos,
    glm::vec3 eRotation
) : modelId(modelId),
    size(size),
    mass(mass), 
    pos(pos), 
    eRotation(eRotation),
    velocity(0.0f), 
    acceleration(0.0f),
    state(0) 
{
    // if object is not dynamic, we have to set baseline for transformation matrix
    update(0.0f);           
}

void RigidBody::update(float dt){
    pos += velocity * dt + 0.5f * acceleration * (dt * dt);
    velocity += acceleration * dt;

    // Calculate rotation matrix. We use 4d matrix to apply rotation
    glm::mat4 rotationMat = glm::toMat4(glm::quat(eRotation));

    /*
        Model matrix
        Multiplication order matters! - I: identity matrix
        modelMatrix(M) = translation(T) * rotation(R) * scale(S) = M = T * R * S
    */ 
    model = glm::translate(glm::mat4(1.0f), pos);   // M = I * T
    model = model * rotationMat;                    // M = M * R = T * R
    model = glm::scale(model, size);                // M = M * S = T * R * S

    /*
        Tangent space
        - Direction doesn't change with translation, only rotation and scale
        - Translation takes place in the 4th column of the matrix
        So, only gets the upper left 3x3 matrix from model matrix
    */
    normalModel = glm::mat3(glm::transpose(glm::inverse(model)));
}

void RigidBody::applyForce(glm::vec3 force){
    acceleration += force / mass;
}

void RigidBody::applyForce(glm::vec3 direction, float magnitude){
    applyForce(direction * magnitude);
}

void RigidBody::applyAcceleration(glm::vec3 a){
    acceleration += a;
}

void RigidBody::applyAcceleration(glm::vec3 direction, float magnitude){
    applyAcceleration(direction * magnitude);
}

void RigidBody::applyImpulse(glm::vec3 force, float dt){
    velocity += force / mass * dt;
}

void RigidBody::applyImpulse(glm::vec3 direction, float magnitude, float dt){
    applyImpulse(direction * magnitude, dt);
}

void RigidBody::transferEnergy(float joules, glm::vec3 direction){
    if(joules == 0){
        return;
    
    }
    //comes from formula KE = 1/2 * m * v^2
    glm::vec3 deltaV = ((float)sqrt(2 * abs(joules) / mass)) * direction;
    velocity += joules > 0 ? deltaV : -deltaV;

    // float velocityMagnitude = glm::length(velocity);
    // float newVelocityMagnitude = sqrt(2 * joules / mass);
    // velocity = glm::normalize(velocity) * newVelocityMagnitude;
}

bool RigidBody::freeze(){//TODO: create a frozen state
    if (velocity == glm::vec3(0.0f) && acceleration == glm::vec3(0.0f)){
        return false;
    }
    storedVelocity = velocity;
    storedAcceleration = acceleration;

    velocity = glm::vec3(0.0f);
    acceleration = glm::vec3(0.0f);
    return true;
}

bool RigidBody::unfreeze(){
    if (velocity != glm::vec3(0.0f) && acceleration != glm::vec3(0.0f)){
        return false;
    }
    velocity = storedVelocity;
    acceleration = storedAcceleration;
    return true;
}
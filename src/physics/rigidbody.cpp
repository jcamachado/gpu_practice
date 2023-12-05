#include "rigidbody.h"

bool RigidBody::operator==(RigidBody rb){
    return instanceId == rb.instanceId;
}

bool RigidBody::operator==(std::string id){
    return instanceId == id;
}

RigidBody::RigidBody() {}

RigidBody::RigidBody(
    std::string modelId, 
    glm::vec3 size,
    float mass,
    glm::vec3 pos
) : modelId(modelId),
        size(size),
        mass(mass), 
        pos(pos), 
        velocity(0.0f), 
        acceleration(0.0f),
        state(0) {}


void RigidBody::update(float dt){
    pos += velocity * dt + 0.5f * acceleration * (dt * dt);
    velocity += acceleration * dt;
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
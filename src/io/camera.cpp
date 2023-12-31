#include "camera.h"
#include <iostream>

Camera::Camera(glm::vec3 position)
        : cameraPos(position),
        worldUp(glm::vec3(0.0f, 1.0f, 0.0f)),
        yaw(0.0f),
        pitch(0.0f),
        speed(5.5f),
        sensitivity(1.0f),
        zoom(45.0f),
        cameraFront(glm::vec3(1.0f, 0.0f, 0.0f))
{
    updateCameraVectors();
}

void Camera::updateCameraDirection(double dx, double dy) {
    // yaw += dx * sensitivity;
    yaw += dx;
    // pitch += dy * sensitivity;
    pitch += dy;
    if (pitch > 89.0f) {
        pitch = 89.0f;
    }
    if (pitch < -89.0f) {
        pitch = -89.0f;
    }
    updateCameraVectors();
}

void Camera::updateCameraPosition(CameraDirection direction, double dt) {
    float velocity = (float)dt * speed;

    switch(direction) {
        case CameraDirection::FORWARD:
            cameraPos += cameraFront * velocity;
            break;
        case CameraDirection::BACKWARD:
            cameraPos -= cameraFront * velocity;
            break;
        case CameraDirection::RIGHT:
            cameraPos += cameraRight * velocity;
            break;
        case CameraDirection::LEFT:
            cameraPos -= cameraRight * velocity;
            break;
        case CameraDirection::UP:
            cameraPos += cameraUp * velocity;
            break;
        case CameraDirection::DOWN:
            cameraPos -= cameraUp * velocity;
            break;
        default:
            break;
    
    }

}

void Camera::updateCameraZoom(double dy) {
    if (zoom >= 1.0f && zoom <= 45.0f) {
        zoom -= dy;
    }
    else if (zoom < 1.0f) {
        zoom = 1.0f;
    }
    else  { // zoom > 45.0f
        zoom = 45.0f;
    }
}

glm::mat4 Camera::getViewMatrix() {
    glm::mat4 rest = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    
    return rest;
}


float Camera::getZoom() {
    return zoom;
}

void Camera::updateCameraVectors() {
    glm::vec3 direction;
    //making yaw and pitch in radians will make the math cheaper
    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch)); 
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(direction);
    
    cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
    cameraUp = glm::normalize(glm::cross(cameraRight, cameraFront));
}

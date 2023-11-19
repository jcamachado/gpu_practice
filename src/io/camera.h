#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//the axis here will be (i guess) with: the y being the up axis, x the right axis, and z being the forward axis
enum class CameraDirection {
    NONE = 0,
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Camera {
public:
    //camera position
    glm::vec3 cameraPos;
    
    //camera directional vectors
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    glm::vec3 cameraRight;
    glm::vec3 worldUp;
    
    //rotation
    float yaw;
    float pitch;
    
    //movement
    float speed;
    float sensitivity;
    float zoom;
    Camera(glm::vec3 position = glm::vec3(0.0f));

    void updateCameraDirection(double dx, double dy);
    void updateCameraPosition(CameraDirection direction, double dt);
    void updateCameraZoom(double dy); //offset y

    glm::mat4 getViewMatrix();
    float getZoom();

private:
    void updateCameraVectors();
};

#endif
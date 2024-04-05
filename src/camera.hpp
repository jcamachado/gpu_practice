#pragma once

//libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

/*
    Model transform (model matrix) makes the vertices of local object space into vertices in world space
    Projection transform makes the vertices in world space into vertices in clip space (Vulkan Canonical View Volume)
    Viewport transform makes the vertices in clip space into vertices in screen space (mapping to the screen pixels)
    Camera space is the space where the camera is at the origin and looking down the positive z axis 
    In camera space, the objects are moved into the view frustum and the camera is moved to the origin
    1- Translate the camera to the origin
    2- Rotate the camera to point in the +z direction    
    Mcam = Rotate X Translate 
*/
namespace ud {

    class UDCamera {
        public:
            UDCamera() = default;
            ~UDCamera() = default;

            UDCamera(const UDCamera&) = delete;
            UDCamera& operator=(const UDCamera&) = delete;

            void setOrthographicProjection(float left, float right, float top, float bottom, float near, float far);
            // fovy is the field of view in the y direction, vertical fov
            void setPerspectiveProjection(float fovy, float aspect, float near, float far);

            // position is the eye
            void setViewDirection(
                const glm::vec3 position, 
                const glm::vec3 direction, 
                const glm::vec3 up = glm::vec3(0.0f, -1.0f, 0.0f)
            );
            void setViewTarget(
                const glm::vec3 position, 
                const glm::vec3 target, 
                const glm::vec3 up = glm::vec3(0.0f, -1.0f, 0.0f)
            );

            void setViewYXZ(const glm::vec3 position, const glm::vec3 rotation); // euler angles, tait-bryan angles

            const glm::mat4& getProjection() const { return projectionMat; }
            const glm::mat4& getView() const { return viewMatrix; }
            const glm::mat4& getInverseView() const { return inverseViewMatrix; }
        
        private:
            glm::mat4 projectionMat = glm::mat4(1.0f);
            glm::mat4 viewMatrix{1.0f}; // view matrix used to transform the world space to camera space
            glm::mat4 inverseViewMatrix{1.0f}; // inverse of the view matrix used to transform the camera space to world space
    };
}
#pragma once

//libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>


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
            const glm::mat4& getProjection() const { return projectionMat; }
        
        private:
            glm::mat4 projectionMat = glm::mat4(1.0f);
    };
}
#include "camera.hpp"

//libs
#include <cassert>
#include <limits>

namespace ud {

    void UDCamera::setOrthographicProjection(
        float left, float right, 
        float top, float bottom, 
        float near, float far
    ) {
        projectionMat = glm::mat4{1.0f};
        projectionMat[0][0] = 2.f / (right - left);
        projectionMat[1][1] = 2.f / (bottom - top);
        projectionMat[2][2] = 1.f / (far - near);
        projectionMat[3][0] = -(right + left) / (right - left);
        projectionMat[3][1] = -(bottom + top) / (bottom - top);
        projectionMat[3][2] = -near / (far - near);
    }
 
    void UDCamera::setPerspectiveProjection(float fovy, float aspect, float near, float far) {
        assert(glm::abs(aspect - std::numeric_limits<float>::epsilon()) > 0.0f);
        const float tanHalfFovy = tan(fovy / 2.f);
        projectionMat = glm::mat4{0.0f};
        projectionMat[0][0] = 1.f / (aspect * tanHalfFovy);
        projectionMat[1][1] = 1.f / (tanHalfFovy);
        projectionMat[2][2] = far / (far - near);
        projectionMat[2][3] = 1.f;
        projectionMat[3][2] = -(far * near) / (far - near);
    }
} // namespace ud

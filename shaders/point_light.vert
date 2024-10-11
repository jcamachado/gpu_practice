#version 450

const vec2 OFFSETS[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, 1.0)
);

layout (location = 0) out vec2 fragOffset;

struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};

// layout(set = 0, binding = 0) uniform GlobalUbo { 
//     mat4 projection;
//     mat4 view;
//     mat4 inverseView;
//     vec4 ambientLightColor; // w is intensity
//     // Specialization Constants is a method to define constants that are known at compile time. ath the time of pipeline creation.
//     // This value of 10 that is hardcoded here can be replaced by a specialization constant. And it have to match the value in the C++ code.
//     PointLight pointLights[10]; 
//     int numLights;
// } ubo;

layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 projection[2]; // Projection matrices for left and right eyes
    mat4 view[2];       // View matrices for left and right eyes
    mat4 inverseView[2]; // Inverse view matrices for left and right eyes
    vec4 ambientLightColor; // w is intensity
    PointLight pointLights[10]; 
    int numLights;
} ubo;

// layout(push_constant) uniform Push {
//     vec4 position;
//     vec4 color;
//     float radius;
// } push;

layout(push_constant) uniform Push {
    vec4 position;
    vec4 color;
    float radius;
    int eyeIndex; // 0 for left eye, 1 for right eye
} push;

/*
    An alternative way to calculate the position is to use camera space instead of world space.
    This way we can avoid the multiplication with the view matrix. Try it later TODO.
    Transform light position to camera space and then apply offset in camera space.
*/
void main() {
    fragOffset = OFFSETS[gl_VertexIndex];
    vec3 cameraRightWorld = vec3(ubo.view[push.eyeIndex][0][0], ubo.view[push.eyeIndex][1][0], ubo.view[push.eyeIndex][2][0]);
    vec3 cameraUpWorld = vec3(ubo.view[push.eyeIndex][0][1], ubo.view[push.eyeIndex][1][1], ubo.view[push.eyeIndex][2][1]);

    vec3 positionWorld = push.position.xyz + 
        push.radius * fragOffset.x * cameraRightWorld +
        push.radius * fragOffset.y * cameraUpWorld;
    gl_Position = ubo.projection[push.eyeIndex] * ubo.view[push.eyeIndex] * vec4(positionWorld, 1.0);
}
// void main() {
//     fragOffset = OFFSETS[gl_VertexIndex];
//     vec3 cameraRightWorld = vec3(ubo.view[0][0][0], ubo.view[0][1][0], ubo.view[0][2][0]); // old solution
//     vec3 cameraUpWorld = vec3(ubo.view[0][0][1], ubo.view[0][1][1], ubo.view[0][2][1]);

//     vec3 positionWorld = push.position.xyz + 
//         push.radius * fragOffset.x * cameraRightWorld +
//         push.radius * fragOffset.y * cameraUpWorld;
//     gl_Position = ubo.projection[0] * ubo.view[0] * vec4(positionWorld, 1.0);

//     // Alternative way to calculate the position
//     // vec4 lightInCameraSpace = ubo.view * vec4(ubo.lightPosition, 1.0);
//     // vec4 positionCameraSpace = lightInCameraSpace + LIGHT_RADIUS * vec4(fragOffset, 0.0, 0.0);
//     // gl_Position = ubo.projection * vec4(positionCameraSpace);
// }
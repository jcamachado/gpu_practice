#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv; // texCoord

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragPosWorld;
layout(location = 2) out vec3 fragNormalWorld;
layout(location = 3) out flat int eyeIndex; // Pass the eye index as a flat variable

struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};

// if ubo is too big, we can use push constants to pass the view-projection matrices
// I can pass projection and view matrices as push constants
// also multiply the model matrix with the normal matrix to get the normal in world space
// also pass multiplied projection and view matrices to the fragment shader
layout(set = 0, binding = 0) uniform GlobalUBO { 
    mat4 projection[2]; 
    mat4 view[2];
    mat4 inverseView[2];
    vec4 ambientLightColor; // w is intensity
} ubo;

layout(set = 0, binding = 1) uniform PointLightsUBO {
    PointLight pointLights[10];
    int numLights;
} pointLightsUbo;

layout(push_constant) uniform Push {    // Limit is 128 bytes to make it compatible with all hardware.
    mat4 modelMatrix;
    mat4 normalMatrix;   // is 3x3 but we pass it as a 4x4 to be aligned to 16 bytes.                              
} push;

void main() {
    // Transform the vertex position
    vec4 positionWorld = push.modelMatrix * vec4(position, 1.0);
    gl_Position = positionWorld; // Pass the world position to the geometry shader

    // Pass the normal and position to the geometry shader
    fragNormalWorld = normalize(mat3(push.normalMatrix) * normal);
    fragPosWorld = positionWorld.xyz;
    fragColor = color;
}
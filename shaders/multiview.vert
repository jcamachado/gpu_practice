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

layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 viewProj[2]; // Array to hold view-projection matrices for both eyes
    mat4 inverseView[2];
    vec4 ambientLightColor; // w is intensity
    PointLight pointLights[10]; 
    int numLights;
} ubo;

layout(push_constant) uniform Push {    // Limit is 128 bytes to make it compatible with all hardware.
    mat4 modelMatrix;
    mat4 normalMatrix;   // is 3x3 but we pass it as a 4x4 to be aligned to 16 bytes.                              
} push;

void main() {
    // Determine which eye we are rendering for
    int eyeIndex = gl_InstanceIndex % 2;

    // Transform the vertex position
    vec4 positionWorld = push.modelMatrix * vec4(position, 1.0);
    gl_Position = ubo.viewProj[eyeIndex] * positionWorld;

    // Pass the normal and position to the fragment shader
    fragNormalWorld = normalize(mat3(push.normalMatrix) * normal);
    fragPosWorld = positionWorld.xyz;
    fragColor = color;
}
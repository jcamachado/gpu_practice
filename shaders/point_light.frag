#version 450

layout(location = 0) in vec2 fragOffset;
layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};

layout(set = 0, binding = 0) uniform GlobalUBO { 
    mat4 projection[2]; // Projection matrices for left and right eyes
    mat4 view[2];       // View matrices for left and right eyes
    mat4 inverseView[2]; // Inverse view matrices for left and right eyes
    vec4 ambientLightColor; // w is intensity
} ubo;

layout(set = 0, binding = 1) uniform PointLightsUBO {
    PointLight pointLights[10];
    int numLights;
} pointLightsUbo;

layout(push_constant) uniform PointLightPushConstants {
    vec4 position;
    vec4 color;
    float radius;
} push;

void main() {
    // float distance = sqrt(dot(fragOffset, fragOffset));
    // if (distance >= 1.0) {
    //     discard;
    // }
    outColor = vec4(push.color.xyz, 1.0);
}
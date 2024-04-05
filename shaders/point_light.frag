#version 450

layout(location = 0) in vec2 fragOffset;
layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};

layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 projection;
    mat4 view;
    mat4 inverseView;
    vec4 ambientLightColor; // w is intensity
    // Specialization Constants is a method to define constants that are known at compile time. ath the time of pipeline creation.
    // This value of 10 that is hardcoded here can be replaced by a specialization constant. And it have to match the value in the C++ code.
    PointLight pointLights[10]; 
    int numLights;
} ubo;

layout(push_constant) uniform Push {
    vec4 position;
    vec4 color;
    float radius;
} push;

void main() {
    float distance = sqrt(dot(fragOffset, fragOffset));
    if (distance >= 1.0) {
        discard;
    }
    outColor = vec4(push.color.xyz, 1.0);
}
#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

layout(location = 0) in vec2 fragOffset[];

layout(location = 0) out vec2 gs_out_fragOffset;

struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};

layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 projection[2]; // Projection matrices for left and right eyes
    mat4 view[2];       // View matrices for left and right eyes
    mat4 inverseView[2]; // Inverse view matrices for left and right eyes
    vec4 ambientLightColor; // w is intensity
    PointLight pointLights[10]; 
    int numLights;
} ubo;

layout(push_constant) uniform Push {
    vec4 position;
    vec4 color;
    float radius;
    int eyeIndex; // 0 for left eye, 1 for right eye
} push;

void main() {
}
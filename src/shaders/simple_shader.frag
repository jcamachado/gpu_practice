#version 450

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
    mat2 transform;
    vec2 offset;  // 8 bytes + 8 bytes padding
    vec3 color;
} push;

void main() {
    outColor = vec4(push.color, 1.0);
}
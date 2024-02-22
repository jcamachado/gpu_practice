#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {    // Limit is 128 bytes to make it compatible with all hardware.
    mat4 transform;     // projection * view * model     64 bytes
    mat4 normalMatrix;   // 64 bytes                                
} push;

void main() {
    outColor = vec4(fragColor, 1.0);
}
#version 450
/*
    Input and output layouts are not associated. Even if both are location = 0, they are not related inside the shader.
    But they must be in same location and type on the receiving end.
*/
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    fragColor = color;
}
#version 330 core
//These are from buffer
//layout(qualifier1​, qualifier2​ = value, ...) variable definition
layout (location = 0) in vec3 aPos; //the layout is the location of the vertex attribute in the VBO
layout (location = 1) in vec3 aOffset;
layout (location = 2) in vec3 aSize;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec3 pos = vec3(aPos.x * aSize.x, aPos.y * aSize.y, aPos.z * aSize.z);

    gl_Position = projection * view * model * vec4(pos + aOffset, 1.0); //Order Matters!
}
#version 330 core

//These are from buffer
//layout(qualifier1​, qualifier2​ = value, ...) variable definition
layout (location = 0) in vec3 aPos; //layout variable
layout (location = 1) in vec3 aColor;

out vec3 ourColor; //This is an attribute

uniform mat4 transform;


void main() {
    gl_Position = transform * vec4(aPos, 1.0); //Order Matters!
    ourColor = aColor; //Attribute receives value from buffer
}
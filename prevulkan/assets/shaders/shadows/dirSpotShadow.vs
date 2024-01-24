#version 330 core
layout (location = 0) in vec3 aPos; //the layout is the location of the vertex attribute in the VBO
/*
    In mesh class, 0 is position, 1 is normal, 2 texture coordinate and 3 is tangent, and they are all light specific, 
    they only affect the color, not the position of the vertex, so we can ignore them in the vertex shader
*/
layout (location = 4) in vec3 aOffset;
layout (location = 5) in vec3 aSize;

uniform mat4 lightSpaceMatrix;

out vec4 FragPos;

// What differs from directional light and spot light is the projection matrix.
// So we can reuse this code.
void main() {
    FragPos = vec4(aSize * aPos + aOffset, 1.0);
    // Projection * view * model * pos,  synonimous to gl_Position in object.vs but calculating lightSpaceMatrix on the CPU
    gl_Position = lightSpaceMatrix * FragPos; 
} 
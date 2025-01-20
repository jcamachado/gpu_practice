#version 330 core

//These are from buffer
//layout(qualifier1​, qualifier2​ = value, ...) variable definition
layout (location = 0) in vec3 aPos; //the layout is the location of the vertex attribute in the VBO
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0)); // Its position in the world
    Normal = mat3(transpose(inverse(model))) * aNormal; //Normal in world space

    gl_Position = projection * view * vec4(FragPos, 1.0); //Order Matters!
    TexCoord = aTexCoord;
}
#version 330 core
layout (location = 0) in vec3 aPos; //the layout is the location of the vertex attribute in the VBO
layout (location = 2) in vec2 aTexCoord;
layout (location = 4) in vec3 aSize;

out vec2 TexCoord;

void main() {
    TexCoord = aTexCoord;
    gl_Position = vec4(aSize * aPos, 1.0);
} 
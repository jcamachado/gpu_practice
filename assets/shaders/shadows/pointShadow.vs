#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 3) in vec3 aOffset;
layout (location = 4) in vec3 aSize;

// We wont make matrix transformations in the vertex shader because only geometry shader can emit multiple vertices of vertex shader 
void main()
{   
    // World space coordinates of the vertex, geometry shader will translate it into projection and view of each specified face
    gl_Position = vec4(aSize * aPos + aOffset, 1.0);    
}
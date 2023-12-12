#version 330 core
// 1 is normal, 2 is texture, 3 is tangent
layout (location = 0) in vec3 aPos;     // mesh position vbo
layout (location = 4) in vec3 aOffset;  // posVBO
layout (location = 5) in vec3 aSize;    // sizeVBO

// We wont make matrix transformations in the vertex shader because only geometry shader can emit multiple vertices of vertex shader 
void main()
{   
    // World space coordinates of the vertex, geometry shader will translate it into projection and view of each specified face
    gl_Position = vec4(aSize * aPos + aOffset, 1.0);    
}
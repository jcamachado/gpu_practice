#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;  // 6 faces = 6 triangles = 18 vertices

uniform mat4 lightSpaceMatrices[6];

out vec4 FragPos;

void main()
{
    for (int face = 0; face < 6; face++) {
        // gl_Layer = face; built-in variable that specifies to which face we render on cubemap.
        gl_Layer = face;

        // render triangle
        for (int i = 0; i < 3; i++) {
            FragPos = gl_in[i].gl_Position;
            gl_Position = lightSpaceMatrices[face] * FragPos;   // Transformation from world space to light space
            EmitVertex();   // Emit each vertex separately and then after this inner loop were going to end the primitive
        }
        EndPrimitive();
    }
}
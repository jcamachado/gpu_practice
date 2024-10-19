#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 gs_in_fragColor[];
layout(location = 1) in vec3 gs_in_fragPosWorld[];
layout(location = 2) in vec3 gs_in_fragNormalWorld[];
layout(location = 3) flat in int gs_in_eyeIndex[];

layout(location = 0) out vec3 fs_out_fragColor;
layout(location = 1) out vec3 fs_out_fragPosWorld;
layout(location = 2) out vec3 fs_out_fragNormalWorld;
layout(location = 3) flat out int fs_out_eyeIndex;

void main() {
    for (int i = 0; i < 3; ++i) {
        gl_Position = gl_in[i].gl_Position;
        fs_out_fragColor = gs_in_fragColor[i];
        fs_out_fragPosWorld = gs_in_fragPosWorld[i];
        fs_out_fragNormalWorld = gs_in_fragNormalWorld[i];
        fs_out_eyeIndex = gs_in_eyeIndex[i];
        EmitVertex();
    }
    EndPrimitive();
}
#version 450
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 gs_in_position[];
layout(location = 1) in vec3 gs_in_normal[];
layout(location = 2) in vec3 gs_in_color[];

layout(location = 0) out vec3 fs_out_normal;
layout(location = 1) out vec3 fs_out_pos;
layout(location = 2) out vec3 fs_out_color;
void main() {
}
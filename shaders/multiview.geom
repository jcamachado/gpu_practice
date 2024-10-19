#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 gs_in_fragColor[];
layout(location = 1) in vec3 gs_in_fragPosWorld[];
layout(location = 2) in vec3 gs_in_fragNormalWorld[];

layout(location = 0) out vec3 fs_out_fragColor;
layout(location = 1) out vec3 fs_out_fragPosWorld;
layout(location = 2) out vec3 fs_out_fragNormalWorld;
layout(location = 3) flat out int fs_out_eyeIndex;

struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};

layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 projection[2]; 
    mat4 view[2];
    mat4 inverseView[2];
    vec4 ambientLightColor;
    PointLight pointLights[10]; 
    int numLights;
} ubo;

void main() {
    // Emit vertices for the first viewport (left eye)
    for (int i = 0; i < 3; ++i) {
        gl_Position = ubo.projection[0] * ubo.view[0] * gl_in[i].gl_Position;
        gl_ViewportIndex = 0;
        fs_out_eyeIndex = 0;

        fs_out_fragColor = gs_in_fragColor[i];
        fs_out_fragPosWorld = gs_in_fragPosWorld[i];
        fs_out_fragNormalWorld = gs_in_fragNormalWorld[i];
        EmitVertex();
    }
    EndPrimitive();

    // Emit vertices for the second viewport (right eye)
    for (int i = 0; i < 3; ++i) {
        gl_Position = ubo.projection[1] * ubo.view[1] * gl_in[i].gl_Position;
        gl_ViewportIndex = 1;
        fs_out_eyeIndex = 1;
        
        fs_out_fragColor = gs_in_fragColor[i];
        fs_out_fragPosWorld = gs_in_fragPosWorld[i];
        fs_out_fragNormalWorld = gs_in_fragNormalWorld[i];
        EmitVertex();
    }
    EndPrimitive();
}
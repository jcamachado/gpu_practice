#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

layout(location = 0) in vec3 gs_in_fragColor[];
layout(location = 1) in vec3 gs_in_fragPosWorld[];
layout(location = 2) in vec3 gs_in_fragNormalWorld[];

layout(location = 0) out vec3 fs_out_fragColor;
layout(location = 1) out vec3 fs_out_fragPosWorld;
layout(location = 2) out vec3 fs_out_fragNormalWorld;
layout(location = 3) flat out int fs_out_eyeIndex;
layout(location = 4) out vec2 gsFragOffset;

struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};

layout(set = 0, binding = 0) uniform GlobalUBO { 
    mat4 projection[2]; 
    mat4 view[2];
    mat4 inverseView[2];
    vec4 ambientLightColor; // w is intensity
} ubo;

layout(set = 0, binding = 1) uniform PointLightsUBO {
    PointLight pointLights[10];
    int numLights;
} pointLightsUbo;

void main() {
    for (int viewport = 0; viewport < 2; ++viewport) {
        for (int i = 0; i < 3; ++i) {
            vec4 ndcPos = ubo.projection[viewport] * ubo.view[viewport] * gl_in[i].gl_Position;
            
            gl_Position = ndcPos;

            gl_ViewportIndex = viewport; // necessary for multi-view rendering
            fs_out_eyeIndex = viewport;

            fs_out_fragColor = gs_in_fragColor[i];
            fs_out_fragPosWorld = gs_in_fragPosWorld[i];
            fs_out_fragNormalWorld = gs_in_fragNormalWorld[i];
            
            // gsFragOffset = gl_Position.xy; // Calculate and pass gsFragOffset based on NDC
            EmitVertex();
        }
        EndPrimitive();
    }
}
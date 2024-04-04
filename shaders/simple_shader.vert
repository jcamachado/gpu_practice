#version 450

//Input and output layouts are not related inside the shader, so layout 0 in != 0 out.
//But they must be in same location and type on the receiving end.
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv; // texCoord

// Per-vertex color, not a color for whole object.
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragPosWorld;
layout(location = 2) out vec3 fragNormalWorld;
struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};


// Descriptor set. set and binding must match the C++ code of descriptor set layout
// This will be the same to all shaders. If needed, we will create a new descriptor set to be shader specific.
layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 projection;
    mat4 view;
    vec4 ambientLightColor; // w is intensity
    // Specialization Constants is a method to define constants that are known at compile time. ath the time of pipeline creation.
    // This value of 10 that is hardcoded here can be replaced by a specialization constant. And it have to match the value in the C++ code.
    PointLight pointLights[10]; 
    int numLights;
} ubo;

/*
    The order of the members in the push constant block must match the order of the push constant block 
    in the C++ code. (SimplePushConstantData)
    Only one push constant block is allowed per shader entry point.

    Members must be aligned to 4 bytes and the entire struct must be aligned to 16 bytes. In any case, look docs. 
    Ex: Offset will occupy 16 bytes even if it only uses 8 bytes, color will be 16 bytes. 
    So the entire struct will be 32 bytes. A padding of 8 bytes will be added to offset make the entire struct 32 bytes.
    {x, y, -, -, r, g, b, padding} = 32 bytes
    To avoid the padding, we can use alignas(16) for the members. (mannually aligning)
*/
layout(push_constant) uniform Push {    // Limit is 128 bytes to make it compatible with all hardware.
    mat4 modelMatrix;
    mat4 normalMatrix;   // is 3x3 but we pass it as a 4x4 to be aligned to 16 bytes.                              
} push;
// When dealing with lighting, make sure that the vectors are unit vectors. For this, we can use normalize() function.
// (n)ormalize (v) = v / length(v) => n(v) = v/||v||

// Remember that the light direction is in world space and vertex position is in world space. 
// We need to transform the vertex position to world space to calculate the light direction.
// 1.0 in vec4 position is homogeneous coordinate, if it was 0.0, it would be a vector.
void main() {
    vec4 positionWorld = push.modelMatrix * vec4(position, 1.0);
    gl_Position = ubo.projection * ubo.view * positionWorld;

     // normalMatrix is 4x4 but we only need 3x3
    fragNormalWorld = normalize(mat3(push.normalMatrix) * normal);
    fragPosWorld = positionWorld.xyz;
    fragColor = color;
}
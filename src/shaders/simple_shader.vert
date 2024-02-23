#version 450
/*
    Input and output layouts are not associated. Even if both are location = 0, they are not related inside the shader.
    But they must be in same location and type on the receiving end.
*/
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv; // texCoord

// Descriptor set. set and binding must match the C++ code of descriptor set layout
// Available only in vertex shader stage for now.
layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 projectionViewMatrix;
    vec3 directionToLight;
} ubo;

// Per-vertex color, not a color for whole object.
layout(location = 0) out vec3 fragColor;

/*
    The order of the members in the push constant block is important.
    And must match the order of the push constant block in the C++ code. (SimplePushConstantData)

    Only one push constant block is allowed per shader entry point.

    Members must be aligned to 4 bytes and the entire struct must be aligned to 16 bytes
    The details are on the documentation. 
    offset will be 16 bytes, color will be 16 bytes. So the entire struct will be 32 bytes.
    even though the size of offset is 8 bytes, a padding of 8 bytes will be added to make the entire struct 32 bytes.
    {x, y, -, -, r, g, b, padding} = 32 bytes

    To avoid the padding, we can use alignas(16) for the members.

*/
layout(push_constant) uniform Push {    // Limit is 128 bytes to make it compatible with all hardware.
    mat4 modelMatrix;
    mat4 normalMatrix;   // is 3x3 but we pass it as a 4x4 to be aligned to 16 bytes.                              
} push;

// When dealing with lighting, make sure that the vectors are unit vectors. For this, we can use normalize() function.
// (n)ormalize (v) = v / length(v) => n(v) = v/||v||
const float AMBIENT = 0.02;

void main() {
    // 1.0 is homogeneous coordinate, if it was 0.0, it would be a vector.
    gl_Position = ubo.projectionViewMatrix * push.modelMatrix * vec4(position, 1.0); 

     // normalMatrix is 4x4 but we only need 3x3
    vec3 normalWorldSpace = normalize(mat3(push.normalMatrix) * normal);

    float lightIntensity = AMBIENT + max(dot(normalWorldSpace, ubo.directionToLight), 0);
    fragColor = lightIntensity * color;

}
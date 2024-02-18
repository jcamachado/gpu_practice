#version 450
/*
    Input and output layouts are not associated. Even if both are location = 0, they are not related inside the shader.
    But they must be in same location and type on the receiving end.
*/
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

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
layout(push_constant) uniform Push {
    mat2 transform;
    vec2 offset;
    vec3 color;
} push;

void main() {
    gl_Position = vec4(push.transform * position + push.offset, 0.0, 1.0);
}
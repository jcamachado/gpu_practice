#version 450

const vec2 OFFSETS[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, 1.0)
);

layout (location = 0) out vec2 fragOffset;

layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 projection;
    mat4 view;
    vec4 ambientLightColor; // w is intensity
    vec3 lightPosition;
    vec4 lightColor;
} ubo;

const float LIGHT_RADIUS = 0.1; // Make it push constant

/*
    An alternative way to calculate the position is to use camera space instead of world space.
    This way we can avoid the multiplication with the view matrix. Try it later TODO.
    Transform light position to camera space and then apply offset in camera space.
*/
void main() {
    fragOffset = OFFSETS[gl_VertexIndex];
    // vec3 cameraRightWorld = { ubo.view[0][0], ubo.view[1][0], ubo.view[2][0] }; // old solution
    // vec3 cameraUpWorld = { ubo.view[0][1], ubo.view[1][1], ubo.view[2][1] };

    // vec3 positionWorld = ubo.lightPosition.xyz + 
    //     LIGHT_RADIUS * fragOffset.x * cameraRightWorld +
    //     LIGHT_RADIUS * fragOffset.y * cameraUpWorld;
    
    // gl_Position = ubo.projection * ubo.view * vec4(positionWorld, 1.0);3
    // Alternative way to calculate the position
    vec4 lightInCameraSpace = ubo.view * vec4(ubo.lightPosition, 1.0);
    vec4 positionCameraSpace = lightInCameraSpace + LIGHT_RADIUS * vec4(fragOffset, 0.0, 0.0);
    gl_Position = ubo.projection * vec4(positionCameraSpace);
}
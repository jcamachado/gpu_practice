#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragPosWorld;
layout(location = 2) in vec3 fragNormalWorld;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position; // ignore w
    vec4 color; // w is intensity
};

layout(set = 0, binding = 0) uniform GlobalUbo { 
    mat4 projection;
    mat4 view;
    vec4 ambientLightColor; // w is intensity
    // Specialization Constants is a method to define constants that are known at compile time. ath the time of pipeline creation.
    // This value of 10 that is hardcoded here can be replaced by a specialization constant. And it have to match the value in the C++ code.
    PointLight pointLights[10]; 
    int numLights;
} ubo;

layout(push_constant) uniform Push {    // Limit is 128 bytes to make it compatible with all hardware.
    mat4 modelMatrix;
    mat4 normalMatrix;   // is 3x3 but we pass it as a 4x4 to be aligned to 16 bytes.                              
} push;

void main() {
    vec3 diffuseLight = ubo.ambientLightColor.xyz * ubo.ambientLightColor.w;
    vec3 surfaceNormal = normalize(fragNormalWorld);

    for (int i=0; i<ubo.numLights; i++) {
        PointLight light = ubo.pointLights[i];
        vec3 directionToLight = light.position.xyz- fragPosWorld;
        // dot product of a vector with itself is an efficient way to calculate the length of the vector.
        float attenuation = 1.0 / dot(directionToLight, directionToLight); // 1 / length^2
        // Cosine of the angle of incidence of the light ray on the surface.
        float cosAngIncidence = max(dot(surfaceNormal, normalize(directionToLight)), 0);
        vec3 intencity = light.color.xyz * light.color.w * attenuation;

        diffuseLight += intencity * cosAngIncidence;
    }

    outColor = vec4(diffuseLight* fragColor, 1.0);
}
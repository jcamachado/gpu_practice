#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragPosWorld;
layout(location = 2) in vec3 fragNormalWorld;
layout(location = 3) flat in int eyeIndex; // Receive the eye index as a flat variable

layout(location = 0) out vec4 outColor;

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

layout(push_constant) uniform Push {    // Limit is 128 bytes to make it compatible with all hardware.
    mat4 modelMatrix;
    mat4 normalMatrix;   // is 3x3 but we pass it as a 4x4 to be aligned to 16 bytes.                              
} push;

void main() {
    vec3 diffuseLight = ubo.ambientLightColor.xyz * ubo.ambientLightColor.w;
    vec3 specularLight = vec3(0.0);
    vec3 surfaceNormal = normalize(fragNormalWorld);

    vec3 cameraPosWorld = ubo.inverseView[gl_ViewportIndex][3].xyz;
    // Calculated for half angle vector.
    vec3 viewDirection = normalize(cameraPosWorld - fragPosWorld); // Direction from the fragment to the camera.

    for (int i = 0; i < ubo.numLights; i++) {
        PointLight light = ubo.pointLights[i];
        vec3 directionToLight = light.position.xyz - fragPosWorld;
        // dot product of a vector with itself is an efficient way to calculate the length of the vector.
        float attenuation = 1.0 / dot(directionToLight, directionToLight); // 1 / length^2
        directionToLight = normalize(directionToLight); // After attenuation.
        // Cosine of the angle of incidence of the light ray on the surface.
        float cosAngIncidence = max(dot(surfaceNormal, directionToLight), 0);
        vec3 intensity = light.color.xyz * light.color.w * attenuation;

        diffuseLight += intensity * cosAngIncidence;

        // Specular lighting
        vec3 halfAngle = normalize(directionToLight + viewDirection);
        float blinnTerm = dot(halfAngle, surfaceNormal);
        blinnTerm = clamp(blinnTerm, 0, 1);
        blinnTerm = pow(blinnTerm, 64.0); // higher values -> sharper highlights
        specularLight += intensity * blinnTerm;
    }

    outColor = vec4(diffuseLight * fragColor + specularLight * fragColor, 1.0);
}
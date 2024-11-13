#version 450

// layout(location = 0) in vec3 fragColor;
// layout(location = 1) in vec3 fragPosWorld;
// layout(location = 2) in vec3 fragNormalWorld;
// layout(location = 3) flat in int eyeIndex; // Receive the eye index as a flat variable

layout(location = 0) in vec3 fs_out_fragColor;
layout(location = 1) in vec3 fs_out_fragPosWorld;
layout(location = 2) in vec3 fs_out_fragNormalWorld;
layout(location = 3) flat in int fs_out_eyeIndex; // Receive the eye index as a flat variable
// layout(location = 4) in vec2 gsFragOffset; // Receive gsFragOffset from geometry shader

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

layout(set = 0, binding = 1) uniform sampler2D textureSampler; // Add a sampler2D uniform

// Function to sample neighboring pixels for blurring
vec3 sampleNeighbors(vec2 uv, float blurRadius) {
    vec3 color = vec3(0.0);
    int samples = 0;
    ivec2 texSize = textureSize(textureSampler, 0); // Get the texture size
    for (float x = -blurRadius; x <= blurRadius; x++) {
        for (float y = -blurRadius; y <= blurRadius; y++) {
            vec2 offset = vec2(x, y) / vec2(texSize);
            color += texture(textureSampler, uv + offset).rgb;
            samples++;
        }
    }
    return color / float(samples);
}

void main() {
    // Transform the fragment position to clip space
    vec4 fragPosClip = ubo.projection[fs_out_eyeIndex] * ubo.view[fs_out_eyeIndex] * vec4(fs_out_fragPosWorld, 1.0);
    
    // Transform the fragment position to normalized device coordinates (NDC)
    vec3 fragPosNDC = fragPosClip.xyz / fragPosClip.w;
    
    
    // Calculate the angle from the center in NDC space
    float angle = atan(fragPosNDC.y, fragPosNDC.x);
    
    // Define the field of view (FOV) in radians
    float fov = radians(60.0); // 60 degrees FOV
    
    // Define the maximum radius based on the angle
    float maxRadius;
    if (angle < radians(30.0) || angle > radians(-30.0)) {
        maxRadius = 1.0; // Central vision
    } else if (angle < radians(60.0) || angle > radians(-60.0)) {
        maxRadius = 0.8; // Peripheral vision
    } else {
        maxRadius = 0.6; // Far peripheral vision
    }

    // Calculate the distance from the center in NDC space
    float distance = length(fragPosNDC.xy);

    // Discard the fragment if it is outside the radius
    if (distance > maxRadius) {
        discard;
    }

    // Calculate blur radius based on distance
    float blurRadius = smoothstep(0.0, maxRadius, distance) * 5.0; // Adjust the multiplier for more/less blur
    // Remap fragPosNDC.xy to UV coordinates in [0,1]
    vec2 uv = fragPosNDC.xy * 0.5 + 0.5;
    // Sample neighboring pixels for blurring
    vec3 blurredColor = sampleNeighbors(uv, blurRadius);
    
    vec3 diffuseLight = ubo.ambientLightColor.xyz * ubo.ambientLightColor.w;
    vec3 specularLight = vec3(0.0);
    vec3 surfaceNormal = normalize(fs_out_fragNormalWorld);

    vec3 cameraPosWorld = ubo.inverseView[fs_out_eyeIndex][3].xyz;
    // Calculated for half angle vector.
    vec3 viewDirection = normalize(cameraPosWorld - fs_out_fragPosWorld); // Direction from the fragment to the camera.

    for (int i = 0; i < ubo.numLights; i++) {
        PointLight light = ubo.pointLights[i];
        vec3 directionToLight = light.position.xyz - fs_out_fragPosWorld;
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
    vec3 finalColor = mix(fs_out_fragColor, blurredColor, smoothstep(0.0, maxRadius, distance));

    // outColor = vec4(diffuseLight * fs_out_fragColor + specularLight * fs_out_fragColor, 1.0);
    outColor = vec4(diffuseLight * finalColor + specularLight * finalColor, 1.0);

}
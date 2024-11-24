#version 450

/*
    Received values from the vertex or geometry shader:

    -fs_out_eyeIndex: Eye index as a flat variable (0 for left eye, 1 for right eye)
*/

layout(location = 0) in vec3 fs_out_fragColor;
layout(location = 1) in vec3 fs_out_fragPosWorld;
layout(location = 2) in vec3 fs_out_fragNormalWorld;
layout(location = 3) in vec2 fs_out_fragTexCoord;
layout(location = 4) flat in int fs_out_eyeIndex; 
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
    // clamp uv to [0, 1] to avoid sampling outside the texture
    uv = clamp(uv, 0.0, 1.0);
    int numSamples = int(blurRadius * 1);
    for (float x = -numSamples; x <= numSamples; x++) {
        for (float y = -numSamples; y <= numSamples; y++) {
            vec2 offset = vec2(x, y) / vec2(texSize);
            color += texture(textureSampler, uv + offset).rgb;
            samples++;
        }
    }
    return color / float(samples);
}

// 5 regions of vision: central, paracentral, near peripheral, mid peripheral, far peripheral
void drawFovealCircles(
    float distanceDegrees,
    float lineThickness,
    float centralVision, 
    float paracentralVision, 
    float nearPeripheralVision, 
    float midPeripheralVision, 
    float farPeripheralVision) {
        // todo
}
        

void main() {
    // Transform the fragment position to clip space
    vec4 fragPosClip = ubo.projection[fs_out_eyeIndex] * ubo.view[fs_out_eyeIndex] * vec4(fs_out_fragPosWorld, 1.0);
    // Transform the fragment position to normalized device coordinates (NDC)
    vec3 fragPosNDC = fragPosClip.xyz / fragPosClip.w;
    
    float distance = length(fragPosNDC.xy);  // Distance from the center in NDC space

    // Convert the distance to degrees based on the horizontal field of view
    // Use the vertical FOV to convert the distance to degrees
    float aspect = 1280.0 / 720.0; // Example aspect ratio
    float fovx = radians(110.0); // Assuming a 110-degree horizontal field of view
    float fovy = 2.0 * atan(tan(fovx / 2.0) / aspect); // Calculate vertical FOV from horizontal FOV
    float distanceDegrees = degrees(atan(distance * tan(fovy / 2.0)));
    
    // Define the degrees for different vision regions from the center
    // Since it is related to the distance from the center, we have to divide by 2.0
    float centralVision = 5.0/2.0; // Central vision in degrees
    float paracentralVision = 8.0/2.0; // Paracentral vision in degrees
    float nearPeripheralVision = 30.0/2.0; // Near peripheral vision in degrees
    float midPeripheralVision = 60.0/2.0; // Mid peripheral vision in degrees
    float farPeripheralVision = 100.0/2.0; // Far peripheral vision in degrees

    float lineThickness = 0.05; // Thickness of the line in degrees

    /* 
        Draw where the central and peripheral vision meet 
        - Draw a red line at the boundary between central and peripheral vision
    */

    // Check if the fragment is at the boundary between central and paracentral vision
    if (distanceDegrees > centralVision - lineThickness && distanceDegrees < centralVision + lineThickness) {
        outColor = vec4(0.0, 1.0, 0.0, 1.0); // Green color
        return;
    }

    // Check if the fragment is at the boundary between paracentral and near peripheral vision
    if (distanceDegrees > paracentralVision - lineThickness && distanceDegrees < paracentralVision + lineThickness) {
        outColor = vec4(0.0, 1.0, 0.0, 1.0); // Green color
        return;
    }

    // Check if the fragment is at the boundary between near peripheral and mid peripheral vision
    if (distanceDegrees > nearPeripheralVision - lineThickness && distanceDegrees < nearPeripheralVision + lineThickness) {
        outColor = vec4(0.0, 1.0, 0.0, 1.0); // Green color
        return;
    }

    // Check if the fragment is at the boundary between mid peripheral and far peripheral vision
    if (distanceDegrees > midPeripheralVision - lineThickness && distanceDegrees < midPeripheralVision + lineThickness) {
        outColor = vec4(0.0, 1.0, 0.0, 1.0); // Green color
        return; 
    }

    if (distanceDegrees > farPeripheralVision - lineThickness && distanceDegrees < farPeripheralVision + lineThickness) {
        outColor = vec4(0.0, 1.0, 0.0, 1.0); // Green color
        return;
    }

    // Discard the fragment if it is outside the far peripheral vision
    if (distanceDegrees > farPeripheralVision) {
        discard;
    }

    // Removing pixels from one eye
    // Discard the fragment if it is between near peripheral and mid peripheral vision
    if (
        (distanceDegrees > nearPeripheralVision && distanceDegrees < midPeripheralVision)
        && (fs_out_eyeIndex == 1)
        ) 
    {
        discard;
    }

    /* 
        Blur effect 
        - Calculate blur radius based on distance
        - Remap fragPosNDC.xy to UV coordinates in [0,1]
        - Sample neighboring pixels for blurring
    */

    float blurRadius = 0.0;
    if (distanceDegrees > centralVision && distanceDegrees <= paracentralVision) {
        blurRadius = 1.0; // Blur radius for paracentral vision
    } else if (distanceDegrees > paracentralVision && distanceDegrees <= nearPeripheralVision) {
        blurRadius = 2.0; // Blur radius for near peripheral vision
    } else if (distanceDegrees > nearPeripheralVision && distanceDegrees <= midPeripheralVision) {
        blurRadius = 4.0; // Blur radius for mid peripheral vision
    } else if (distanceDegrees > midPeripheralVision && distanceDegrees <= farPeripheralVision) {
        blurRadius = 8.0; // Blur radius for far peripheral vision
    }

    // if (distanceDegrees > centralVision && distanceDegrees < farPeripheralVision) {
        // blurRadius = smoothstep(centralVision, farPeripheralVision, distanceDegrees) * 5.0; // Adjust the multiplier for more/less blur
    // }

    vec2 uv = fragPosNDC.xy; // Remap fragPosNDC.xy to UV coordinates in [0,1]
    vec3 blurredColor = sampleNeighbors(uv, blurRadius);

    /*
        Lighting calculations
        - Calculate ambient light
        - Calculate diffuse light
        - Calculate specular light
        - Calculate final color based on other effects (e.g., blur)
    */
    
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
    // Sample the texture color
    vec3 textureColor = texture(textureSampler, fs_out_fragTexCoord).rgb;

    // if no texture color, use the fragment color
    if (textureColor == vec3(0.0)) {
        textureColor = fs_out_fragColor;
    }
    // Apply lighting to the texture color
    vec3 litColor = textureColor * (diffuseLight + specularLight);

    // Apply foveal blurred color change
    vec3 fovealColor = mix(litColor, blurredColor, smoothstep(centralVision, farPeripheralVision, distanceDegrees));

    outColor = vec4(fovealColor, 1.0);

}


/*
    TODO 1: Implement a blur effect based on the distance from the center of the screen using
    degrees based on foveal vision. The blur effect should be more intense as the distance from 
    the center increases.

    central vision: 0-5 degrees
    paracentral vision: 5-8 degrees
    near peripheral vision: 8-30 degrees
    mid peripheral vision: 30-60 degrees
    far peripheral vision: 60-100 degrees


    // Calculate the angle from the center in NDC space in radians
    // float angle = atan(fragPosNDC.y, fragPosNDC.x);
    
    // float centralToPeriVision = radians(30.0); // Angle in radians from the center to the peripheral vision
    // float centralToFarPeriVision = radians(60.0); // Angle in radians from the center to the far peripheral vision

    // float maxRadius;
    // if (angle < radians(centralToPeriVision) || angle > radians(-centralToPeriVision)) {
    //     maxRadius = centerMaxRadius; // Central vision
    // } else if (angle < radians(centralToFarPeriVision) || angle > radians(-centralToFarPeriVision)) {
    //     maxRadius = periMaxRadius; // Peripheral vision
    // } else {
    //     maxRadius = farPeriMaxRadius; // Far peripheral vision
    // }
*/


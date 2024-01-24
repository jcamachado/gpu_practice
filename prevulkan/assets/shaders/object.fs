/*
    The values of uniform arrays cannot be dynamically changed, so we need to set a maximum number of lights
*/
#define DEFAULT_GAMA 2.2        // 2.2 is the usual value for gamma

struct Material {
    vec4 diffuse;               //texture, 1 per mesh
    vec4 specular;              //texture, 1 per mesh
    float shininess;
};

uniform sampler2D diffuse0;
uniform sampler2D specular0;
uniform sampler2D normal0;

uniform sampler2D dirLightBuffer;
uniform samplerCube pointLightBuffers[MAX_POINT_LIGHTS];
uniform sampler2D spotLightBuffers[MAX_SPOT_LIGHTS];

out vec4 FragColor;

uniform Material material;

in VS_OUT {
    vec3 FragPos;
    vec2 TexCoord;

    TangentLights tanLights;
} fs_in;
uniform bool noNormalMap;
uniform bool noTexture;
uniform bool skipNormalMapping;

vec4 calcDirLight(vec3 norm, vec3 viewVec, vec3 viewDir, vec4 diffMap, vec4 specMap);
vec4 calcPointLight(int idx, vec3 norm, vec3 viewVec, vec3 viewDir, vec4 diffMap, vec4 specMap);
vec4 calcSpotLight(int idx, vec3 norm, vec3 viewVec, vec3 viewDir, vec4 diffMap, vec4 specMap);

/*
    gridSamplingDisk
    - We need to sample the cubemap in a grid pattern, so we need to calculate the offsets
    These values are kinda arbitrary. They need to be a good representation of all directions.
*/
#define NUM_SAMPLES 20
vec3 sampleOffsetDirections[NUM_SAMPLES] = vec3[]
(
   vec3(1, 1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, 1,  1), 
   vec3(1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
   vec3(1, 1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1, 1,  0),
   vec3(1, 0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1, 0, -1),
   vec3(0, 1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0, 0, 0)
);

// TODO - make Blinn always true
void main(){
    vec3 norm = normalize(fs_in.tanLights.Normal);

    if (!skipNormalMapping && !noNormalMap){
        norm = normalize(texture(normal0, fs_in.TexCoord).rgb * 2.0 - 1.0); // map from [0, 1] to [-1, 1]
    }
    vec3 viewVec = fs_in.tanLights.ViewPos - fs_in.tanLights.FragPos; // Will be used for soft shadow
    vec3 viewDir = normalize(viewVec);

    vec4 diffMap;
    vec4 specMap;

    if (noTexture){
        diffMap = material.diffuse;
        specMap = material.specular;
    }
    else{
        diffMap = texture(diffuse0, fs_in.TexCoord);
        specMap = texture(specular0, fs_in.TexCoord);
    }

    vec4 result;

    // Directional light
    // result = calcDirLight(norm, viewVec, viewDir, diffMap, specMap);

    // Point lights
    for(int i = 0; i < nPointLights; i++){
        result += calcPointLight(i, norm, viewVec, viewDir, diffMap, specMap);
    }

    // Spot lights
    for(int i = 0; i < nSpotLights; i++){
        // result += calcSpotLight(i, viewVec, norm, viewDir, diffMap, specMap);
    }

    // Gamma correction
    result.rgb = pow(result.rgb, vec3(1.0/DEFAULT_GAMA));

    /*
        Depth testing
        -Z is related to depth (distance from camera to fragment)
        -Since we are using perspective, we need to transform the depth value to linear depth
        by taking the inverse of the projection matrix (only for perspective) using the formula:
            linearDepth = (2.0 * near * far) / (z * (far - near) - (far + near));
        -Then we need to transform the linear depth to [0, 1] range using the formula:
    */
    float near = 0.1;
    float far = 100.0;
    float z = gl_FragCoord.z * 2.0 - 1.0;   // Transform to Normalized Device Coordinates (NDC) [0, 1] -> [-1, 1]
    float linearDepth = (2.0 * near * far) / (z * (far - near) - (far + near)); 
    float factor = (near + linearDepth) / (near - far); // Transform back to [0, 1] range

    result.rgb *= 1 - factor;   // Darken the fragment the further away it is

    FragColor = result;
}

/*
    About shadows
      Shadows are calculated based on the light in the world space
*/

float calcDirLightShadow(vec3 norm, vec3 viewVec, vec3 lightDir){
    /*
        - fragPosLightSpace:  fs_in.FragPos is only affected by the model, gl_pos is affected by changes in perspective. 
        It wouldnt make sense passing projection coordinates because we dont want to render light from the camera's point of view.
        So its easier to handle where the fragment is in the world.

        - We need to calculate a Bias factor to avoid Shadow Acne, where the shadow is rendered in a striped pattern

        - We dont need to linearize the depth because we are using an orthographic projection, so the depth is already linear

        - In this scenario, outerCutoff cannot be greater than 45 degrees because of cubemap limitations
    */
    vec4 fragPosLightSpace = dirLight.lightSpaceMatrix * vec4(fs_in.FragPos, 1.0);

    // Perspective divide (Transforming coordinates NDC, normalized device coordinates)
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w; //[depth relative to light] => [-1, 1]

    // NDC to depth range (renders everything inside bounding region in lightSpaceMatrix from light.cpp [i guess??])
    projCoords = projCoords * 0.5 + 0.5; //[-1, 1] => [0, 1]

    // If too far from light, do not return shadow
    if (projCoords.z > 1.0){
        return 0.0;
    }

    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z; // in normalized coordinates, z is the depth

    // Calculate bias (based on depth map resolution and slope)
    float maxBias = 0.05;
    float minBias = 0.005;
    float bias = max(maxBias * (1.0 - dot(norm, lightDir)), minBias);

    /*
     PCF (Percentage Closer Filtering)
    - Texel is a pixel in a texture
    We need to go from -1 to 1 and do the averages
    - pcfDepth is the depth of the neighbouring texel 
    We create the offset by -1 to 1 and multiply by the texelSize, converting the offset in pixels to normalized coordinates
    */

    float shadowSum = 0.0;
    vec2 texelSize = 1.0 / textureSize(dirLightBuffer, 0); // 0 because its a 2D texture
    float viewDist = length(viewVec);
    float diskRadius = (1.0 + (viewDist / dirLight.farPlane)) / 30.0; 
    for (int y = -1; y <= 1; y++){
        for (int x = -1; x <= 1; x++){
            float pcfDepth = texture(dirLightBuffer, 
                projCoords.xy + vec2(x, y) * texelSize * diskRadius
            ).r;
            // If depth is greater (further), return 1
            shadowSum += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }


    return shadowSum / 9.0; // 9 because we are doing 3x3 PCF
}

vec4 calcDirLight(vec3 norm, vec3 viewVec, vec3 viewDir, vec4 diffMap, vec4 specMap) {
    /*
        Ambient -> constant
    */
    vec4 ambient = dirLight.ambient * diffMap;
    
    /*
        Diffuse
        diff value: When the angle between pointLight vector and  normal are more than 90 degrees, dot product is 0
    */
    vec3 lightDir = normalize(-fs_in.tanLights.dirLightDirection);
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 diffuse = dirLight.diffuse * (diff * diffMap);

    /*
        Specular
        - If diff <= 0, object is behind the light
    */
    vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
    if (diff > 0) {
        float dotProd = 0.0;
        vec3 halfwayDir = normalize(lightDir + viewDir);
        dotProd = dot(norm, halfwayDir);
        float spec = pow(max(dotProd, 0.0), material.shininess*128);
        specular = dirLight.specular * (spec * specMap);
    }

    // float shadow = calcDirLightShadow(norm, viewVec, lightDir);    // Only affects diffuse and specular
    float shadow = 0.0;
    return vec4(ambient + (1.0 - shadow) * (diffuse + specular));
}

float calcPointLightShadow(int idx, vec3 norm, vec3 viewVec, vec3 lightDir){
    // Get vector from the light to the fragment (similar to pointShadow.fs)
    vec3 lightToFrag = fs_in.FragPos - pointLights[idx].position;

    // // Get depth from cubemap
    // float closestDepth = texture(pointLightBuffers[idx], lightToFrag).r;

    // // [0,1] => original depth value
    // closestDepth *= pointLights[idx].farPlane;

    // Get current depth
    float currentDepth = length(lightToFrag);

    // Calculate bias
    float minBias = 0.005;
    float maxBias = 0.05;
    float bias = max(maxBias * (1.0 - dot(norm, lightDir)), minBias);

    // PCF
    float shadow = 0.0;
    float viewDist = length(viewVec);
    /*
        30.0 is arbitrary. 
        The bigger the value, the more samples we take and closer to the real shadow we get
        because the disk radius will be closer to 0 (or 1).
        The further out the fragment is, the bigger the disk radius will be and the 
        offset will be further away from the point.
    */
    float diskRadius = (1.0 + (viewDist / pointLights[idx].farPlane)) / 30.0; 
    for (int i = 0; i < NUM_SAMPLES; i++){
        float pcfDepth = texture(pointLightBuffers[idx], 
            lightToFrag + sampleOffsetDirections[i] * diskRadius).r;
        pcfDepth *= pointLights[idx].farPlane;

        shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
    }
    shadow /= float(NUM_SAMPLES);

    return shadow;  // We can add PCF here

}

vec4 calcPointLight(int idx, vec3 norm, vec3 viewVec, vec3 viewDir, vec4 diffMap, vec4 specMap){
    // Ambient -> constant
    vec4 ambient = pointLights[idx].ambient * diffMap;

    vec3 lightDir = normalize(fs_in.tanLights.pointLightPositions[idx] - fs_in.tanLights.FragPos);
    // When pointLight is > perpendicular to the normal, dot product is 0
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 diffuse = pointLights[idx].diffuse * (diff * diffMap);

    /*
        Specular 
        - It is more about the position of the camera in relation to the object, 
        as of how the pointLight gets more concentrated as you get closer to a reflected ray
        - If diff <= 0, object is behind the light
    */
    vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
    if (diff > 0) {
        float dotProd = 0.0;
        // Calculate using Blinn-Phong model
        vec3 halfwayDir = normalize(lightDir + viewDir);
        dotProd = dot(norm, halfwayDir);

        float spec = pow(max(dotProd, 0.0), material.shininess*128);
        specular = pointLights[idx].specular * (spec * specMap);
    }

    float dist = length(pointLights[idx].position - fs_in.FragPos);
    float attenuation = 1.0/(pointLights[idx].k0 + pointLights[idx].k1*dist + pointLights[idx].k2*(dist*dist));

    // float shadow = calcPointLightShadow(idx, norm, viewVec, lightDir);
    float shadow = 0.0;

    return vec4(ambient + (1.0 - shadow) * (diffuse + specular)) * attenuation;
}


float calcSpotLightShadow(int idx, vec3 norm, vec3 viewVec, vec3 lightDir){
    vec4 fragPosLightSpace = spotLights[idx].lightSpaceMatrix * vec4(fs_in.FragPos, 1.0);

    // Perspective divide (Transforming coordinates NDC)
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w; //[depth relative to light] => [-1, 1]

    // NDC to depth range (renders everything inside bounding region in lightSpaceMatrix from light.cpp [i guess??])
    projCoords = projCoords * 0.5 + 0.5; //[-1, 1] => [0, 1]

    // If too far from light, do not return shadow
    if (projCoords.z > 1.0){
        return 0.0;
    }

    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z; // in normalized coordinates, z is the depth

    // Calculate bias (based on depth map resolution and slope)
    float maxBias = 0.05;
    float minBias = 0.005;
    float bias = max(maxBias * (1.0 - dot(norm, lightDir)), minBias);

    /*
        PCF (Percentage Closer Filtering)
    */

    float shadowSum = 0.0;
    vec2 texelSize = 1.0 / textureSize(spotLightBuffers[idx], 0); // 0 because its a 2D texture
    float viewDist = length(viewVec);
    float diskRadius = (1.0 + (viewDist / spotLights[idx].farPlane)) / 30.0; 
    for (int y = -1; y <= 1; y++){
        for (int x = -1; x <= 1; x++){
            float pcfDepth = texture(spotLightBuffers[idx], 
                projCoords.xy + vec2(x, y) * texelSize * diskRadius
            ).r;
            pcfDepth *= spotLights[idx].farPlane;   // [0,1] => original depth value
            // If depth is greater (further), return 1
            shadowSum += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }

    // Return average. 9 because we are doing 3x3 PCF
    return shadowSum / 9.0;

}

vec4 calcSpotLight(int idx, vec3 norm, vec3 viewVec, vec3 viewDir, vec4 diffMap, vec4 specMap){
    vec3 lightDir = normalize(                                                  //same as pointLight
        fs_in.tanLights.spotLightPositions[idx] - fs_in.tanLights.FragPos
    );
     // Angle between lightDir and spotLight direction, cossine
    float theta = dot(lightDir, normalize(-fs_in.tanLights.spotLightDirections[idx]));

    //cossine and theta have inverse relationship
    vec4 ambient = spotLights[idx].ambient * diffMap;
    if(theta > spotLights[idx].outerCutOff){ //using cosines and not degrees
        //if in cutoff, render light

        float diff = max(dot(norm, lightDir), 0.0); //when pointLight > perpendicular or more to the normal, dot product is 0
        vec4 diffuse = spotLights[idx].diffuse * (diff * diffMap);
        /*
            Specular
            - If diff <= 0, object is behind the light
        */
        
        vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
        if (diff > 0) {
            float dotProd = 0.0;
            vec3 halfwayDir = normalize(lightDir + viewDir);
            dotProd = dot(norm, halfwayDir);
            float spec = pow(max(dotProd, 0.0), material.shininess*128);
            specular = spotLights[idx].specular * (spec * specMap);
        }

        float intensity = (theta - spotLights[idx].outerCutOff)/(spotLights[idx].cutOff - spotLights[idx].outerCutOff);
        intensity = clamp(intensity, 0.0, 1.0);
        diffuse *= intensity;
        specular *= intensity;

        float dist = length(spotLights[idx].position - fs_in.FragPos);
        float attenuation = 1.0/(spotLights[idx].k0 + spotLights[idx].k1*dist + spotLights[idx].k2*(dist*dist));

        // float shadow = calcSpotLightShadow(idx, norm, viewVec, lightDir);
        float shadow = 0.0;


        ambient *= attenuation;
        diffuse *= attenuation;
        specular *= attenuation;

        return vec4(ambient + (1.0 - shadow) * (diffuse + specular));
    }
    else{
        return ambient;
    }
}
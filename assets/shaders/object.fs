#version 330 core

/*
    The values of uniform arrays cannot be dynamically changed, so we need to set a maximum number of lights
*/
#define MAX_POINT_LIGHTS 20     
#define MAX_SPOT_LIGHTS 5       

#define DEFAULT_GAMA 2.2        // 2.2 is the usual value for gamma

struct Material {
    vec4 diffuse;               //texture, 1 per mesh
    vec4 specular;              //texture, 1 per mesh
    float shininess;
};

uniform sampler2D diffuse0;
uniform sampler2D specular0;


struct PointLight {
    vec3 position;

    float k0;
    float k1;
    float k2;

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

struct DirLight {
    vec3 direction;

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;

    sampler2D depthBuffer;    // set from light.cpp
    mat4 lightSpaceMatrix;
};

struct SpotLight {
    vec3 position;
    vec3 direction;

    float cutOff;
    float outerCutOff;

    float k0;
    float k1;
    float k2;

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform int nPointLights;
uniform DirLight dirLight;
uniform SpotLight spotLights[MAX_SPOT_LIGHTS];
uniform int nSpotLights;

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform Material material;

uniform int noTexture;
uniform vec3 viewPos;
uniform bool useBlinn;
uniform bool useGamma;


float calcDirLightShadow(vec3 norm, vec3 lightDir);
vec4 calcPointLight(int idx, vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap);
vec4 calcDirLight(vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap);
vec4 calcSpotLight(int idx, vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap);


void main(){
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec4 diffMap;
    vec4 specMap;

    if (noTexture == 1){
        diffMap = material.diffuse;
        specMap = material.specular;
    }
    else{
        diffMap = texture(diffuse0, TexCoord);
        specMap = texture(specular0, TexCoord);
    }

    vec4 result;

    // Directional light
    result = calcDirLight(norm, viewDir, diffMap, specMap);

    // Point lights
    for(int i = 0; i < nPointLights; i++){
        result += calcPointLight(i, norm, viewDir, diffMap, specMap);
    }

    // Spot lights
    for(int i = 0; i < nSpotLights; i++){
        result += calcSpotLight(i, norm, viewDir, diffMap, specMap);
    }

    // Gamma correction
    if (useGamma){
        result.rgb = pow(result.rgb, vec3(1.0/DEFAULT_GAMA));
    }

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

// TODO - make Blinn always true
vec4 calcPointLight(int idx, vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap){
    // Ambient -> constant
    vec4 ambient = pointLights[idx].ambient * diffMap;

    vec3 lightDir = normalize(pointLights[idx].position - FragPos);
    // When pointLight is > perpendicular to the normal, dot product is 0
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 diffuse = pointLights[idx].diffuse * (diff * diffMap);

    /*
        Specular 
        - It is more about the position of the camera in relation to the object, 
        as of how the pointLight gets more concentrated as you get closer to a reflected ray
        - If diff <= 0, object is behind the light
    */
    /*
        Specular
    */
    vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
    if (diff > 0) {
        float dotProd = 0.0;
        if (useBlinn){  //Calculate using Blinn-Phong model
            vec3 halfwayDir = normalize(lightDir + viewDir);
            dotProd = dot(norm, halfwayDir);
        }
        else{           //Calculate using Phong model
            vec3 reflectDir = reflect(-lightDir, norm);
            dotProd = dot(viewDir, reflectDir);
        }
        float spec = pow(max(dotProd, 0.0), material.shininess*128);
        specular = dirLight.specular * (spec * specMap);
    }

    float dist = length(pointLights[idx].position - FragPos);
    float attenuation = 1.0/(pointLights[idx].k0 + pointLights[idx].k1*dist + pointLights[idx].k2*(dist*dist));

    return vec4(ambient + diffuse + specular) * attenuation;
}

float calcDirLightShadow(vec3 norm, vec3 lightDir){
    /*
        - fragPosLightSpace:  FragPos is only affected by the model, gl_pos is affected by changes in perspective. 
        It wouldnt make sense passing projection coordinates because we dont want to render light from the camera's point of view.
        So its easier to handle where the fragment is in the world.

        - We need to calculate a Bias factor to avoid Shadow Acne, where the shadow is rendered in a striped pattern
    */
    vec4 fragPosLightSpace = dirLight.lightSpaceMatrix * vec4(FragPos, 1.0);

    // Perspective divide (Transforming coordinates NDC, normalized device coordinates)
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w; //[depth relative to light] => [-1, 1]

    // NDC to depth range (renders everything inside bounding region in lightSpaceMatrix from light.cpp [i guess??])
    projCoords = projCoords * 0.5 + 0.5; //[-1, 1] => [0, 1]

    // Get closest depth in depthBuffer
    float closestDepth = texture(dirLight.depthBuffer, projCoords.xy).r; //r because its a depth texture

    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z; // in normalized coordinates, z is the depth

    // Calculate bias (based on depth map resolution and slope)
    float maxBias = 0.05;
    float minBias = 0.005;
    float bias = max(maxBias * (1.0 - dot(norm, lightDir)), minBias);

    // If depth is greater (further), return 1
    return currentDepth - bias > closestDepth ? 1.0 : 0.0;
}


vec4 calcDirLight(vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap) {
    /*
        Ambient -> constant
    */
    vec4 ambient = dirLight.ambient * diffMap;
    
    /*
        Diffuse
        diff value: When the angle between pointLight vector and  normal are more than 90 degrees, dot product is 0
    */
    vec3 lightDir = normalize(-dirLight.direction);
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 diffuse = dirLight.diffuse * (diff * diffMap);

    /*
        Specular
        - If diff <= 0, object is behind the light
    */
    vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
    if (diff > 0) {
        float dotProd = 0.0;
        if (useBlinn){  //Calculate using Blinn-Phong model
            vec3 halfwayDir = normalize(lightDir + viewDir);
            dotProd = dot(norm, halfwayDir);
        }
        else{           //Calculate using Phong model
            vec3 reflectDir = reflect(-lightDir, norm);
            dotProd = dot(viewDir, reflectDir);
        }
        float spec = pow(max(dotProd, 0.0), material.shininess*128);
        specular = dirLight.specular * (spec * specMap);
    }

    float shadow = calcDirLightShadow(norm, lightDir);    // Only affects diffuse and specular
    
    return vec4(ambient + (1.0 - shadow) * (diffuse + specular));
}

vec4 calcSpotLight(int idx, vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap){
    vec3 lightDir = normalize(spotLights[idx].position - FragPos); //same as pointLight

     // Angle between lightDir and spotLight direction, cossine
    float theta = dot(lightDir, normalize(-spotLights[idx].direction));

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
            if (useBlinn){  //Calculate using Blinn-Phong model
                vec3 halfwayDir = normalize(lightDir + viewDir);
                dotProd = dot(norm, halfwayDir);
            }
            else{           //Calculate using Phong model
                vec3 reflectDir = reflect(-lightDir, norm);
                dotProd = dot(viewDir, reflectDir);
            }
            float spec = pow(max(dotProd, 0.0), material.shininess*128);
            specular = dirLight.specular * (spec * specMap);
        }

        float intensity = (theta - spotLights[idx].outerCutOff)/(spotLights[idx].cutOff - spotLights[idx].outerCutOff);
        intensity = clamp(intensity, 0.0, 1.0);
        diffuse *= intensity;
        specular *= intensity;

        float dist = length(spotLights[idx].position - FragPos);
        float attenuation = 1.0/(spotLights[idx].k0 + spotLights[idx].k1*dist + spotLights[idx].k2*(dist*dist));

        return vec4(ambient + diffuse + specular) * attenuation;
    }
    else{
        return ambient;
    }
}
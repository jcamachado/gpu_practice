#version 330 core
// define constant por max point lights
#define MAX_POINT_LIGHTS 20
// define constant por max spot lights
#define MAX_SPOT_LIGHTS 5

struct Material {
    vec4 diffuse; //texture, 1 per mesh
    vec4 specular;  //texture, 1 per mesh
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

// uniform PointLight pointLight;
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform int nPointLights;
uniform DirLight dirLight;
// uniform SpotLight spotLight;
uniform SpotLight spotLights[MAX_SPOT_LIGHTS];
uniform int nSpotLights;



out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform Material material;

uniform vec3 viewPos;
uniform int noTexture;

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
        diffMap = vec4(texture(diffuse0, TexCoord)); //Diffuse map
        specMap = vec4(texture(specular0, TexCoord)); //Specular map
    }

    vec4 result;

    // directional light
    result = calcDirLight(norm, viewDir, diffMap, specMap);

    // point lights
    for(int i = 0; i < nPointLights; i++){
        if(i < pointLights.length()){
            result += calcPointLight(i, norm, viewDir, diffMap, specMap);
        }
    }

    // spot lights
    for(int i = 0; i < nSpotLights; i++){
        if(i < nSpotLights){
            result += calcSpotLight(i, norm, viewDir, diffMap, specMap);
        }
    }

    FragColor = result;
}

vec4 calcPointLight(int idx, vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap){
    // ambient -> constant
    vec4 ambient = pointLights[idx].ambient * diffMap;

    vec3 lightDir = normalize(pointLights[idx].position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0); //when pointLight is perpendicular+ to the normal, dot product is 0
    vec4 diffuse = pointLights[idx].diffuse * (diff * diffMap);

    // specular (is more about the position of the camera in relation to the object, 
    //as how the pointLight gets more concentrated as you get closer to a reflected ray)
    vec3 reflectDir = reflect(-lightDir, norm); //the bounce of the pointLight direction
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128); //128 is arbitrary
    vec4 specular = pointLights[idx].specular * (spec * specMap);

    float dist = length(pointLights[idx].position - FragPos);
    float attenuation = 1.0/(pointLights[idx].k0 + pointLights[idx].k1*dist + pointLights[idx].k2*(dist*dist));

    return vec4(ambient + diffuse + specular) * attenuation;
}

vec4 calcDirLight(vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap) {
    // ambient -> constant
    vec4 ambient = dirLight.ambient * diffMap;
    
    vec3 lightDir = normalize(-dirLight.direction);
    float diff = max(dot(norm, lightDir), 0.0); //when pointLight > perpendicular or more to the normal, dot product is 0
    vec4 diffuse = dirLight.diffuse * (diff * diffMap);

    vec3 reflectDir = reflect(-lightDir, norm); //use lightDir to not call normalize again
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128);
    vec4 specular = dirLight.specular * (spec * specMap);

    return vec4(ambient + diffuse + specular);
}

vec4 calcSpotLight(int idx, vec3 norm, vec3 viewDir, vec4 diffMap, vec4 specMap){
    vec3 lightDir = normalize(spotLights[idx].position - FragPos); //same as pointLight

     //angle between lightDir and spotLight direction, cossine
    float theta = dot(lightDir, normalize(-spotLights[idx].direction));

    //cossine and theta have inverse relationship
    vec4 ambient = spotLights[idx].ambient * diffMap;
    if(theta > spotLights[idx].outerCutOff){ //using cosines and not degrees
        //if in cutoff, render light

        float diff = max(dot(norm, lightDir), 0.0); //when pointLight > perpendicular or more to the normal, dot product is 0
        vec4 diffuse = spotLights[idx].diffuse * (diff * diffMap);

        vec3 reflectDir = reflect(-lightDir, norm); //use lightDir to not call normalize again
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128);
        vec4 specular = spotLights[idx].specular * (spec * specMap);

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
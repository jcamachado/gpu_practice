#version 330 core
// define constant por max point lights
#define MAX_POINT_LIGHTS 20
// define constant por max spot lights
#define MAX_SPOT_LIGHTS 5

struct Material {
    vec3 ambient;
    // vec3 diffuse;
    sampler2D diffuse; //texture
    // vec3 specular;
    sampler2D specular;
    float shininess;
};

struct PointLight {
    vec3 position;

    float k0;
    float k1;
    float k2;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct DirLight {
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;

    float k0;
    float k1;
    float k2;

    float cutOff;
    float outerCutOff;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
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


vec3 calcPointLight(int idx, vec3 norm, vec3 viewDir, vec3 diffMap, vec3 specMap);
vec3 calcDirLight(vec3 norm, vec3 viewDir, vec3 diffMap, vec3 specMap);
vec3 calcSpotLight(int idx, vec3 norm, vec3 viewDir, vec3 diffMap, vec3 specMap);


void main(){
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 diffMap = vec3(texture(material.diffuse, TexCoord)); //Diffuse map
    vec3 specMap = vec3(texture(material.specular, TexCoord)); //Specular map

    vec3 result;

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

    FragColor = vec4(result, 1.0);
}

vec3 calcPointLight(int idx, vec3 norm, vec3 viewDir, vec3 diffMap, vec3 specMap){
    // ambient -> constant
    vec3 ambient = pointLights[idx].ambient * diffMap;

    vec3 lightDir = normalize(pointLights[idx].position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0); //when pointLight is perpendicular+ to the normal, dot product is 0
    vec3 diffuse = pointLights[idx].diffuse * (diff * diffMap);

    // specular (is more about the position of the camera in relation to the object, 
    //as how the pointLight gets more concentrated as you get closer to a reflected ray)
    vec3 reflectDir = reflect(-lightDir, norm); //the bounce of the pointLight direction
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128); //128 is arbitrary
    vec3 specular = pointLights[idx].specular * (spec * specMap);

    float dist = length(pointLights[idx].position - FragPos);
    float attenuation = 1.0/(pointLights[idx].k0 + pointLights[idx].k1*dist + pointLights[idx].k2*(dist*dist));

    return vec3(ambient + diffuse + specular) * attenuation;
}

vec3 calcDirLight(vec3 norm, vec3 viewDir, vec3 diffMap, vec3 specMap) {
    // ambient -> constant
    vec3 ambient = dirLight.ambient * diffMap;

    
    vec3 lightDir = normalize(-dirLight.direction);
    float diff = max(dot(norm, lightDir), 0.0); //when pointLight > perpendicular or more to the normal, dot product is 0
    vec3 diffuse = dirLight.diffuse * (diff * diffMap);

    vec3 reflectDir = reflect(-lightDir, norm); //use lightDir to not call normalize again
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128);
    vec3 specular = dirLight.specular * (spec * specMap);

    return vec3(ambient + diffuse + specular);
}

vec3 calcSpotLight(int idx, vec3 norm, vec3 viewDir, vec3 diffMap, vec3 specMap){
    vec3 lightDir = normalize(spotLights[idx].position - FragPos); //same as pointLight

     //angle between lightDir and spotLight direction, cossine
    float theta = dot(lightDir, normalize(-spotLights[idx].direction));

    //cossine and theta have inverse relationship
    vec3 ambient = spotLights[idx].ambient * diffMap;
    if(theta > spotLights[idx].outerCutOff){ //using cosines and not degrees
        //if in cutoff, render light

        float diff = max(dot(norm, lightDir), 0.0); //when pointLight > perpendicular or more to the normal, dot product is 0
        vec3 diffuse = spotLights[idx].diffuse * (diff * diffMap);

        vec3 reflectDir = reflect(-lightDir, norm); //use lightDir to not call normalize again
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128);
        vec3 specular = spotLights[idx].specular * (spec * specMap);

        float intensity = (theta - spotLights[idx].outerCutOff)/(spotLights[idx].cutOff - spotLights[idx].outerCutOff);
        
        intensity = clamp(intensity, 0.0, 1.0);

        diffuse *= intensity;
        specular *= intensity;

        float dist = length(spotLights[idx].position - FragPos);
        float attenuation = 1.0/(spotLights[idx].k0 + spotLights[idx].k1*dist + spotLights[idx].k2*(dist*dist));

        return vec3(ambient + diffuse + specular) * attenuation;
    }
    else{
        return ambient;
    }

}
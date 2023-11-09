#version 330 core

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

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform PointLight pointLight;

out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform Material material;

uniform vec3 viewPos;
// uniform sampler2D texture1;
// uniform sampler2D texture2;
// uniform float mixValue;

void calcPointLight(vec3 norm, vec3 viewDir, vec3 diffMap, vec3 specMap);

void main(){
    // FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), mixValue);

    // ambient - constant
    // vec3 ambient = pointLight.ambient * material.ambient;
    vec3 ambient = pointLight.ambient * vec3(texture(material.diffuse, TexCoord));

    // diffuse - is more about the position of the object in relation to the pointLight
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(pointLight.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0); //when pointLight is perpendicular+ to the normal, dot product is 0
    vec3 diffuse = pointLight.diffuse * (diff * vec3(texture(material.diffuse, TexCoord)));

    // specular (is more about the position of the camera in relation to the object, 
    //as how the pointLight gets more concentrated as you get closer to a reflected ray)
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm); //the bounce of the pointLight direction
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128); //128 is arbitrary
    vec3 specular = pointLight.specular * (spec * vec3(texture(material.specular, TexCoord)));

    FragColor = vec4(vec3(ambient + diffuse + specular), 1.0);
}

void calcPointLight(vec3 norm, vec3 viewDir, vec3 diffMap, vec3 specMap){
    // FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), mixValue);

    // ambient - constant
    // vec3 ambient = pointLight.ambient * material.ambient;
    vec3 ambient = pointLight.ambient * vec3(texture(material.diffuse, TexCoord));

    // diffuse - is more about the position of the object in relation to the pointLight
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(pointLight.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0); //when pointLight is perpendicular+ to the normal, dot product is 0
    vec3 diffuse = pointLight.diffuse * (diff * vec3(texture(material.diffuse, TexCoord)));

    // specular (is more about the position of the camera in relation to the object, 
    //as how the pointLight gets more concentrated as you get closer to a reflected ray)
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm); //the bounce of the pointLight direction
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess*128); //128 is arbitrary
    vec3 specular = pointLight.specular * (spec * vec3(texture(material.specular, TexCoord)));

    FragColor = vec4(vec3(ambient + diffuse + specular), 1.0);
}
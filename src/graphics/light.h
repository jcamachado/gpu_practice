#ifndef LIGHT_H
#define LIGHT_H


#include <glm/glm.hpp>
#include "shader.h"

struct PointLight {
    glm::vec3 position;

    // attenuation constants
    float k0; // constant
    float k1; // linear
    float k2; // quadratic

    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;


    void render(Shader shader, int idx);


};

struct DirLight {
    glm::vec3 direction;

    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;


    void render(Shader shader);
};

struct SpotLight {
    glm::vec3 position;
    glm::vec3 direction;

    float cutOff; // inner line cone where the light is stronger
    float outerCutOff; //from inner cut to outer cut off, the light weakens and limits the light cone

    // attenuation constants
    float k0; // constant
    float k1; // linear
    float k2; // quadratic

    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;

    void render(Shader shader, int idx);
};

#endif
#ifndef LIGHT_H
#define LIGHT_H


#include <glm/glm.hpp>
#include "shader.h"

struct PointLight {
    glm::vec3 position;

    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;


    void render(Shader shader);


};

#endif
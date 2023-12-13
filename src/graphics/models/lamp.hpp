#ifndef LAMP_HPP
#define LAMP_HPP

#include <glm/glm.hpp>

#include "cube.hpp"
#include "../rendering/material.h"
#include "../rendering/light.h"
#include "../rendering/shader.h"

class Lamp : public Cube {
    public:
        glm::vec3 lightColor;

        Lamp(unsigned int maxNInstances, glm::vec3 lightColor = glm::vec3(1.0f)) 
            : Cube(maxNInstances, Material::white_rubber) {
            id = "lamp";
            this->lightColor = lightColor;
        }
};

#endif
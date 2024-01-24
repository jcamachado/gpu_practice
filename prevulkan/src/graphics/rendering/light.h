#ifndef LIGHT_H
#define LIGHT_H


#include <glm/glm.hpp>

#include "shader.h"
#include "../memory/framememory.hpp"
#include "../../algorithms/bounds.h"

struct DirLight {
    glm::vec3 direction;

    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;

    // Bounding region for shadow, defines the shadow format
    BoundingRegion br;

    // Transformation to a light space (projection * view)
    glm::mat4 lightSpaceMatrix;

    // FBO for shadows
    FramebufferObject shadowFBO;    // This is working basically as a texture

    // Constructors
    DirLight();     // Default constructor
    DirLight(glm::vec3 direction,
             glm::vec4 ambient,
             glm::vec4 diffuse,
             glm::vec4 specular,
             BoundingRegion br
    );

    // Render light into shader
    void render(Shader shader, unsigned int textureIdx);

    // Update light space matrix
    void updateMatrices();

};


struct PointLight {
    glm::vec3 position;

    // attenuation constants
    float k0; // constant
    float k1; // linear
    float k2; // quadratic

    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;

    // Bounds
    float nearPlane;
    float farPlane;

    // List of view matrices for each face of the cube
    glm::mat4 lightSpaceMatrices[6];

    // FBO for shadows
    FramebufferObject shadowFBO;

    // Constructors
    PointLight();       // Default constructor
    PointLight(glm::vec3 position,
               float k0, float k1, float k2,
               glm::vec4 ambient, glm::vec4 diffuse, glm::vec4 specular,
               float nearPlane, float farPlane
    );

    void render(Shader shader, int idx, unsigned int textureIdx);

    // Update light space matrix
    void updateMatrices();

    // List of directions (static, since they are shared by all instances. Point lights are radially symmetric)
    static glm::vec3 directions[6];

    // List of up vectors
    static glm::vec3 ups[6];
};

struct SpotLight {
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 up;

    float cutOff; // inner line cone where the light is stronger
    float outerCutOff; //from inner cut to outer cut off, the light weakens and limits the light cone

    // attenuation constants
    float k0; // constant
    float k1; // linear
    float k2; // quadratic

    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;

    // Bounds for the shadow
    float nearPlane;
    float farPlane;

    // Light space transformation
    glm::mat4 lightSpaceMatrix;

    // FBO for shadows
    FramebufferObject shadowFBO;

    // Constructors
    SpotLight();        // Default constructor
    SpotLight(glm::vec3 position, glm::vec3 direction, glm::vec3 up,
              float cutOff, float outerCutOff,
              float k0, float k1, float k2,
              glm::vec4 ambient, glm::vec4 diffuse, glm::vec4 specular,
              float nearPlane, float farPlane
    );

    /*
        Render light into shader
        - textureIdx: index of the texture that will be assigned for a depth map (texture index that we are rendering to)
    */
    void render(Shader shader, int idx, unsigned int textureIdx);

    // Update light space matrix
    void updateMatrices();
};

#endif
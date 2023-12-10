
#include "light.h"

void PointLight::render(Shader shader, int idx){
    std::string name = "pointLights[" + std::to_string(idx) + "]";

    shader.set3Float(name + ".position", position);

    shader.setFloat(name + ".k0", k0);
    shader.setFloat(name + ".k1", k1);
    shader.setFloat(name + ".k2", k2);

    shader.set4Float(name + ".ambient", ambient);
    shader.set4Float(name + ".diffuse", diffuse);
    shader.set4Float(name + ".specular", specular);
}

DirLight::DirLight(glm::vec3 direction,
                   glm::vec4 ambient,
                   glm::vec4 diffuse,
                   glm::vec4 specular,
                   BoundingRegion br
) : direction(direction), ambient(ambient), diffuse(diffuse), specular(specular), br(br), shadowFBO(2048, 2048, GL_DEPTH_BUFFER_BIT) {
    // Generate FBO
    shadowFBO.generate();
    shadowFBO.bind();
    shadowFBO.disableColorBuffer();
    shadowFBO.allocateAndAttachTexture(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT, GL_FLOAT);

    updateMatrices();   // this methodWill setup the inital space matrix for the light
}

void DirLight::render(Shader shader, unsigned int textureIdx){
    std::string name = "dirLight";

    shader.set3Float(name + ".direction", direction);

    shader.set4Float(name + ".ambient", ambient);
    shader.set4Float(name + ".diffuse", diffuse);
    shader.set4Float(name + ".specular", specular);

    // Set depth texture
    glActiveTexture(GL_TEXTURE0 + textureIdx);
    shadowFBO.textures[0].bind();
    shader.setInt(name + ".depthBuffer", textureIdx);

    // Set light space matrix
    shader.setMat4(name + ".lightSpaceMatrix", lightSpaceMatrix);
}

void DirLight::updateMatrices(){
    // Setup light space matrix
    glm::mat4 projection = glm::ortho(br.min.x, br.max.x, br.min.y, br.max.y, br.min.z, br.max.z); // Everything in dirlight is parallel, therefore ortho
    // d = -kp (direction of light vec = -constant * object position vec)
    // K can vary, here we will use 2.0f (arbitrary)
    // Do jeito que ele falou eh a conta de cima. Mas , no caso k = 1/2.0f, pois d = -k * p, e embaixo esta p = -d/k
    glm::vec3 pos = -2.0f * direction;
    // eye is the position of the Sun (directional light source)
    glm::mat4 lightView = glm::lookAt(pos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    lightSpaceMatrix = projection * lightView;  // same as we did in object.fs, but calculating on the CPU instead. (May change later)
}

SpotLight::SpotLight(glm::vec3 position, glm::vec3 direction, glm::vec3 up,
                     float cutOff, float outerCutOff,
                     float k0, float k1, float k2,
                     glm::vec4 ambient, glm::vec4 diffuse, glm::vec4 specular,
                     float nearPlane, float farPlane) 
    : position(position), direction(direction), up(up), 
    cutOff(cutOff), outerCutOff(outerCutOff),
    k0(k0), k1(k1), k2(k2), 
    ambient(ambient), diffuse(diffuse), specular(specular),
    nearPlane(nearPlane), farPlane(farPlane), 
    shadowFBO(2048, 2048, GL_DEPTH_BUFFER_BIT) 
{
    shadowFBO.generate();
    shadowFBO.bind();
    shadowFBO.disableColorBuffer();
    shadowFBO.allocateAndAttachTexture(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT, GL_FLOAT);

    updateMatrices();
    
}

void SpotLight::render(Shader shader, int idx, unsigned int textureIdx){
    std::string name = "spotLights[" + std::to_string(idx) + "]";

    shader.set3Float(name + ".position", position);
    shader.set3Float(name + ".direction", direction);

    shader.setFloat(name + ".cutOff", cutOff);
    shader.setFloat(name + ".outerCutOff", outerCutOff);

    shader.setFloat(name + ".k0", k0);
    shader.setFloat(name + ".k1", k1);
    shader.setFloat(name + ".k2", k2);

    shader.set4Float(name + ".ambient", ambient);
    shader.set4Float(name + ".diffuse", diffuse);
    shader.set4Float(name + ".specular", specular);

    // Set depth texture
    glActiveTexture(GL_TEXTURE0 + textureIdx);
    shadowFBO.textures[0].bind();
    shader.setInt(name + ".depthBuffer", textureIdx);

    // Set light space matrix
    shader.setMat4(name + ".lightSpaceMatrix", lightSpaceMatrix);
}

void SpotLight::updateMatrices(){
    // Similar to the camera, projection matrix and view matrix
    // But getting the degrees of the cone with arccos
    glm::mat4 projection = glm::perspective(glm::acos(outerCutOff) * 2.0f,      // fov
        (float)shadowFBO.height / (float)shadowFBO.width,                       // aspect ratio
        nearPlane, farPlane
    );
    
    glm::mat4 lightView = glm::lookAt(position, position + direction, up);

    lightSpaceMatrix = projection * lightView;
}
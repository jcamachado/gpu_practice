
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
) : direction(direction), ambient(ambient), diffuse(diffuse), specular(specular), br(br), shadowFBO(1024, 1024, GL_DEPTH_BUFFER_BIT) {
    // Generate FBO
    shadowFBO.generate();
    shadowFBO.bind();
    shadowFBO.disableDrawColorBuffer();
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

void SpotLight::render(Shader shader, int idx){
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
}
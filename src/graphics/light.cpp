#include "light.h"

DirLight::DirLight() {}

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

    shader.setFloat(name + ".farPlane", br.max.z);

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

// List of directions //CUBEMAP_DEFAULT_NFACES
glm::vec3 PointLight::directions[6] = {
    { 1.0f,  0.0f,  0.0f},      // Right  x+
    {-1.0f,  0.0f,  0.0f},      // Left   x-
    { 0.0f,  1.0f,  0.0f},      // Top    y+
    { 0.0f, -1.0f,  0.0f},      // Bottom y-
    { 0.0f,  0.0f,  1.0f},      // Front  z+
    { 0.0f,  0.0f, -1.0f}       // Back   z-
};

// List of up vectors
glm::vec3 PointLight::ups[6] = {
    {0.0f, -1.0f,  0.0f},       // Right  x+
    {0.0f, -1.0f,  0.0f},       // Left   x-
    {0.0f,  0.0f,  1.0f},       // Top    y+
    {0.0f,  0.0f, -1.0f},       // Bottom y-
    {0.0f, -1.0f,  0.0f},       // Front  z+
    {0.0f, -1.0f,  0.0f}        // Back   z-
};

PointLight::PointLight() {}

PointLight::PointLight(glm::vec3 position,
                       float k0, float k1, float k2,
                       glm::vec4 ambient, glm::vec4 diffuse, glm::vec4 specular,
                       float nearPlane, float farPlane
) : position(position), 
    k0(k0), k1(k1), k2(k2), 
    ambient(ambient), diffuse(diffuse), specular(specular), 
    nearPlane(nearPlane), farPlane(farPlane), 
    shadowFBO(2048, 2048, GL_DEPTH_BUFFER_BIT)
{
    shadowFBO.generate();
    shadowFBO.bind();
    shadowFBO.disableColorBuffer();
    shadowFBO.allocateAndAttachCubemap(GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT, GL_FLOAT);

    updateMatrices();
}

void PointLight::render(Shader shader, int idx, unsigned int textureIdx){
    std::string name = "pointLights[" + std::to_string(idx) + "]";

    shader.set3Float(name + ".position", position);

    shader.setFloat(name + ".k0", k0);
    shader.setFloat(name + ".k1", k1);
    shader.setFloat(name + ".k2", k2);

    shader.set4Float(name + ".ambient", ambient);
    shader.set4Float(name + ".diffuse", diffuse);
    shader.set4Float(name + ".specular", specular);

    // Set near and far planes
    shader.setFloat(name + ".nearPlane", nearPlane);
    shader.setFloat(name + ".farPlane", farPlane);

    // Set depth texture
    glActiveTexture(GL_TEXTURE0 + textureIdx);  // OpenGL doesnt distinguish between 2d and cubemap textures
    shadowFBO.cubemap.bind();
    shader.setInt(name + ".depthBuffer", textureIdx);
}

// Update light space matrices
void PointLight::updateMatrices() {
    // 90 degrees because it is a cube
    glm::mat4 projection = glm::perspective(glm::radians(90.0f),        // FOV 
        (float)shadowFBO.height / (float)shadowFBO.width,               // Aspect ratio
        nearPlane, farPlane                                             // Near and far bounds       
    );

    for (unsigned int i = 0; i < 6; i++) {
        // The position of the camera is the position of the light
        glm::mat4 lightView = glm::lookAt(
            position,
            position + PointLight::directions[i],
            PointLight::ups[i]
        );
    }        
}

SpotLight::SpotLight() {}

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
#include "../objects/model.h"

#include "../../io/camera.h"
#include "../../io/keyboard.h"

#include "../../scene.h"

class Gun : public Model {
    public:
        // CONST_INSTANCES eh relativo a vbo e pois as mudancas de gun vao ocorrer na propria classe model
        Gun(unsigned int maxNInstances)      
            : Model("m4a1", maxNInstances, CONST_INSTANCES | NO_TEX) {} 

        void init(){
            loadModel("assets/models/m4a1/scene.gltf");
        }

        
        // void render(Shader shader, float dt, Scene *scene, glm::mat4 model){
            /*
                WE ARE DOING STATIC ROTATIONS NOW
                SYMBOL ROTATIONS ALONG THE EULER AXES
                WE WILL ADD CUSTOM ROTATIONS LATER
            */
        //     // Set position
        //     // Multiply offset by unit vector in 2 directions
        //     rb.pos = scene->getActiveCamera()->cameraPos + 
        //     glm::vec3((scene->getActiveCamera()->cameraFront * 0.5f)) - 
        //     glm::vec3((scene->getActiveCamera()->cameraUp * 0.205f));
        //     model = glm::translate(model, rb.pos);
            
        //     // Rotate around cameraRight using dot product
        //     float theta = acos(glm::dot(scene->getActiveCamera()->worldUp, scene->getActiveCamera()->cameraFront)/ 
        //         glm::length(scene->getActiveCamera()->cameraUp) / glm::length(scene->getActiveCamera()->cameraFront));
        //     // Offset by pi/2 because angle between cameraUp and gunFront
        //     model = glm::rotate(model, atanf(1)*2 - theta, scene->getActiveCamera()->cameraRight);

        //     //rotate around cameraup using dot product
        //     glm::vec2 front2d = glm::vec2(scene->getActiveCamera()->cameraFront.x, scene->getActiveCamera()->cameraFront.z);
        //     theta = acos(glm::dot(glm::vec2(1.0f, 0.0f), front2d)/ glm::length(front2d));
        //     model = glm::rotate(model, scene->getActiveCamera()->cameraFront.z < 0 ? theta : -theta, scene->getActiveCamera()->worldUp);


        //     //scale 
        //     model = glm::scale(model, rb.size);

        //     shader.setMat4("model", model);

        //     Model::render(shader, dt, scene);

        // }
};
#include "../model.h"
#include "../../io/camera.h"
#include "../../io/keyboard.h"


class Gun : public Model {
    public:
        Gun()
            : Model(glm::vec3(0.0f), glm::vec3(0.05f), true) {}
        
        void render(Shader shader, bool setModel = false ){
            glm::mat4 model = glm::mat4(1.0f);
            
            glm::vec3 down = glm::cross(Camera::defaultCamera.cameraFront, Camera::defaultCamera.cameraRight);
            //set position
            pos = Camera::defaultCamera.cameraPos + 
            glm::vec3((Camera::defaultCamera.cameraFront * 2.0f)) - 
            glm::vec3((Camera::defaultCamera.cameraUp * 0.90f));
            model = glm::translate(model, pos);
            
            //rotate around cameraRight using dot product
            float theta = acos(glm::dot(Camera::defaultCamera.worldUp, Camera::defaultCamera.cameraFront)/ 
                glm::length(Camera::defaultCamera.cameraUp) / glm::length(Camera::defaultCamera.cameraFront));
            //offset by pi/2 because angle between cameraUp and gunFront
            model = glm::rotate(model, atanf(1)*2 - theta, Camera::defaultCamera.cameraRight);

            //rotate around cameraup using dot product
            glm::vec2 front2d = glm::vec2(Camera::defaultCamera.cameraFront.x, Camera::defaultCamera.cameraFront.z);
            theta = acos(glm::dot(glm::vec2(1.0f, 0.0f), front2d)/ glm::length(front2d));
            model = glm::rotate(model, Camera::defaultCamera.cameraFront.z < 0 ? theta : -theta, Camera::defaultCamera.worldUp);

            


            if(Keyboard::key(GLFW_KEY_SPACE)){
                std::cout << Camera::defaultCamera.cameraPos.x << 
                " " << Camera::defaultCamera.cameraPos.y <<
                " " << Camera::defaultCamera.cameraPos.z <<
                " " << pos.x <<
                " " << pos.y <<
                " " << pos.z << std::endl;

            }

            //scale 
            model = glm::scale(model, size);

            shader.setMat4("model", model);

            Model::render(shader, false);

        }


};

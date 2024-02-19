#pragma once

#include "model.hpp"

#include <memory>

namespace ud {
    /*
        GLM Enphasitizes on the row major order, not column major order (cada elemento eh uma coluna)

        In rotation, the multiplication matrix is: (Na duvida, procurar mais em estudo de algebra linear)
        {{cos(theta), -sin(theta)}, {{0,0} {0,1}}   i (column 0) = {1,0} (x axis)
         {sin(theta), cos(theta)}}   {1,0} {1,1}    j (column 1)= {0,1} (y axis)

        Y axis is inverted in the screen, so the rotation is inverted too. So instead of
        oriented in the counter-clockwise, it is oriented in the clockwise direction.

         SOH CAH TOA
            sin(theta) = opposite/hypotenuse
            cos(theta) = adjacent/hypotenuse
            tan(theta) = opposite/adjacent
    */
    struct RigidBody2dComponent {
        glm::vec2 velocity;
        float mass{1.0f};
    };
    struct Transform2dComponent {
        glm::vec2 translation{};                                    // {Position offset}
        glm::vec2 scale{1.f, 1.f};                                  // {Scale}
        float rotation;                                      // {Rotation in radians}
        glm::mat2 mat2() {                  
            float sin = std::sin(rotation);
            float cos = std::cos(rotation);
            glm::mat2 rotationMat{{cos, sin}, {-sin, cos}};   // col1 = {cos, sin}, col2 = {*sin, cos}

            glm::mat2 scaleMat{{scale.x, 0.0f}, {0.0f, scale.y}};   // col1 = {scale.x, 0.0f}, col2 = {0.0f, scale.y}
            return rotationMat * scaleMat; 
        }
    };

    class UDGameObject {
        public:
            using id_t = unsigned int; // id_t is an alias for unsigned int

            static UDGameObject createGameObject() {
                static id_t currentId = 0;
                return UDGameObject(currentId++);
            }

            UDGameObject(const UDGameObject&) = delete; // Copy constructor
            UDGameObject& operator=(const UDGameObject&) = delete;  // Copy assignment operator
            UDGameObject(UDGameObject&&) = default; // Move constructor
            UDGameObject& operator=(UDGameObject&&) = default; // Move assignment operator

            id_t getId() const { return id; }

            std::shared_ptr<UDModel> model; // Varios gameobjects vao compartilhar a mesma instancia de modelo
            glm::vec3 color;

            Transform2dComponent transform2d;
            RigidBody2dComponent rigidBody2d;

        private:
            UDGameObject(id_t objId) : id(objId) {}
            id_t id;
    };
};
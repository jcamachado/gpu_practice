#pragma once

#include "model.hpp"

// libs
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective

// std
#include <memory>
#include <unordered_map>

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
    struct TransformComponent {
        glm::vec3 translation{};            // {Position offset}
        glm::vec3 scale{1.0f, 1.0f, 1.0f};  // {Scale}
        glm::vec3 rotation{};               // {Rotation, radians} euler angles, tait-bryan angles or quaternions
        
        /*
            mat4 -> 3 spacial dimensions, 1 for the homogeneous coordinates
            Translation matrix: 3x3 identity matrix with another column for the translation 
            Rotation convention uses tait-bryan angle with axis order Y(1), X(2), Z(3)
            https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

            The transform matrix = translation * Ry * Rx * Rz * scale

            Intrinsic vs Extrinsic rotation:
            - Intrinsic: Rotate the object in its local coordinate system
                Multiplication order: Ry * Rx * Rz
            - Extrinsic: Rotate the object in the world coordinate system
                Multiplication order: Rz * Rx * Ry
        */
        glm::mat4 mat4();
        glm::mat3 normalMatrix();
    };

    class UDGameObject {
        public:
            using id_t = unsigned int;
            using Map = std::unordered_map<id_t, UDGameObject>;

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

            TransformComponent transform{};

        private:
            UDGameObject(id_t objId) : id(objId) {}
            id_t id;
    };
};
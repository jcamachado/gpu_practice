#ifndef PLANE_HPP
#define PLANE_HPP

#include "../objects/model.h"

class Plane: public Model{
    public:
        Plane(): Model("plane", BoundTypes::AABB, 1, CONST_INSTANCES){}

        void init(std::vector<Texture> textures){
            int nVertices = 4;
            float quadVertices[] ={
                // Position             Normal             TexCoord
                 0.5f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f,   1.0f, 1.0f, // top right
                -0.5f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f,   0.0f, 1.0f, // top left
                -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
                 0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,   1.0f, 0.0f  // bottom right
            };

            float collisionVertices[] = {
                // Position
                 0.5f,  0.5f, 0.0f,                                 // top right
                -0.5f,  0.5f, 0.0f,                                 // top left
                -0.5f, -0.5f, 0.0f,                                 // bottom left
                 0.5f, -0.5f, 0.0f                                  // bottom right
            };

            unsigned int indices[] = {
                0, 1, 3, // first triangle
                1, 2, 3  // second triangle
            };

            BoundingRegion br(glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.0f));
            
            /*
                We can only use the position from quadVertices for process mesh, so 
                we will pass in the collision vertices separately.
            */ 
            Mesh ret = processMesh(
                br,
                nVertices, quadVertices,
                6, indices,
                true, 
                nVertices, collisionVertices,
                2, indices                              // Face is the group of indices 
            );

            ret.setupTextures(textures);

            addMesh(&ret);
        }
};

#endif
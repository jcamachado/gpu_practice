
#ifndef MESH_H
#define MESH_H

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <vector>
#include <glm/glm.hpp>

#include "../memory/vertexmemory.hpp"

#include "../models/box.hpp"

#include "../rendering/material.h"
#include "../rendering/shader.h"
#include "../rendering/texture.h"

#include "../../algorithms/bounds.h"

#include "../../physics/collisionmesh.h"

/*
    All base coordinates are here

    We want to ensure tangent and bitangent vectors are perpendic.ular. 
    Their orthogonality will allow more accurate calculations.
    We will leave the bitangent calculation to the vertex shader.

*/
struct Vertex{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;

    static std::vector<struct Vertex> genList(float* vertices, int nVertices);

    // Calculate tangent vectors for each vertex
    static void calcTanVectors(
        std::vector<struct Vertex>& list, 
        std::vector<unsigned int>& indices
    );
};

typedef struct Vertex Vertex;

class Mesh {
    public:
        BoundingRegion br;                          // Bounding region for mesh
        CollisionMesh* collision;                   // Pointer to attached collision mesh
        std::vector<Vertex> vertices;               // List of vertices
        std::vector<unsigned int> indices;          // List of indices
        ArrayObject VAO;                            // VAO pointing to all data for the mesh
        
        std::vector<Texture> textures;              // List of textures
        aiColor4D diffuse;                          // Material diffuse value
        aiColor4D specular;                         // Material specular value

        /*
            Constructors
        */
        Mesh();                                                     //1 Default
        Mesh(BoundingRegion br);                                    //2 Init with bounding region
        Mesh(BoundingRegion br, std::vector<Texture> textures);     //3 Init as textured object
        Mesh(BoundingRegion br, aiColor4D diff, aiColor4D spec);    //4 Init as material object
        Mesh(BoundingRegion br, Material m);                        //5 Initialize with a material (analog to 4)  

        void setupTextures(std::vector<Texture> textures);  
        void setupColors(aiColor4D diff, aiColor4D spec);            // Setup material color   
        void setupMaterial(Material material);                              // Setup material structure

        void loadData(
            std::vector<Vertex> vertices, 
            std::vector<unsigned int> indices, 
            bool pad = false
        );
        void loadCollisionMesh(
            unsigned int nPoints, 
            float* coordinates, 
            unsigned int nFaces, 
            unsigned int* indices
        );

        void render(Shader shader, unsigned int numInstances);

        void cleanup();

    private:
        unsigned int VBO, EBO;
        bool noTextures;

        void setup();
};


#endif
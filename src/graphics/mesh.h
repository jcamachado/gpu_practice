
#ifndef MESH_H
#define MESH_H

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <vector>
#include <glm/glm.hpp>

#include "shader.h"
#include "texture.h"
#include "glmemory.hpp"

#include "models/box.hpp"

#include "../algorithms/bounds.h"

struct Vertex{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static std::vector<struct Vertex> genList(float* vertices, int nVertices);
};

typedef struct Vertex Vertex;

class Mesh {
    public:
        BoundingRegion br;
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        ArrayObject VAO;
        
        std::vector<Texture> textures;
        aiColor4D diffuse;
        aiColor4D specular;
        Mesh();
        Mesh(BoundingRegion br, std::vector<Texture> textures = {});
        Mesh(BoundingRegion br, aiColor4D diff, aiColor4D spec);

        void loadData(std::vector<Vertex> vertices, std::vector<unsigned int> indices);

        void render(Shader shader, unsigned int numInstances);

        void cleanup();

    private:
        unsigned int VBO, EBO;
        bool noTextures;

        void setup();
};


#endif
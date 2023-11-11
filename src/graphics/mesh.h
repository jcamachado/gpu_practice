
#ifndef MESH_H
#define MESH_H

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <vector>
#include <glm/glm.hpp>

#include "shader.h"
#include "texture.h"

struct Vertex{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static std::vector<struct Vertex> genList(float* vertices, int nVertices);
};

typedef struct Vertex Vertex;

class Mesh {
    public:
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        unsigned int VAO;
        
        std::vector<Texture> textures;
        aiColor4D diffuse;
        aiColor4D specular;

        Mesh();
        Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures = {});
        Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, aiColor4D diffuse, aiColor4D specular);

        void render(Shader shader);

        void cleanup();

    private:
        unsigned int VBO, EBO;
        bool noTextures;

        void setup();
};


#endif
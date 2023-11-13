#ifndef MODEL_H
#define MODEL_H

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

#include "mesh.h"


class Model {
    public:
        glm::vec3 pos;
        glm::vec3 size;
        

        Model() {}
        Model(glm::vec3 pos = glm::vec3(0.0f), glm::vec3 size  = glm::vec3(1.0f), bool noTextures = false);
        void loadModel(std::string path);
        void render(Shader shader, bool setModel = true);
        void cleanup();
    protected:
        bool noTextures;

        std::vector<Mesh> meshes;
        std::string directory;

        // Stores the textures loaded so far, optimization so textures arent loaded more than once.
        std::vector<Texture> textures_loaded;
        //ai here stands for assimp 
        void processNode(aiNode *node, const aiScene *scene);
        Mesh processMesh(aiMesh *mesh, const aiScene *scene);
        std::vector<Texture> loadTextures(aiMaterial *mat, aiTextureType type);
};

#endif
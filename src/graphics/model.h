#ifndef MODEL_H
#define MODEL_H

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

#include "mesh.h"


class Model {
    public:
        std::vector<Mesh> meshes;
        std::string directory;

        Model();
        void init();
        void render(Shader shader);
        void cleanup();
};

#endif
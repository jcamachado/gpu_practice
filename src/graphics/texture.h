#ifndef TEXTURE
#define TEXTURE

#include <glad/glad.h>
// #include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <assimp/scene.h>

#include "../../lib/stb/stb_image.h" // include glad to get all the required OpenGL headers

class Texture {
    public:
        Texture();
        Texture(std::string dir, std::string path, aiTextureType type);
        // Texture(const char* path, const char* name, bool defaultParams = true);

        void generate();
        void load(bool flip = true);

        void bind();

        unsigned int id;
        aiTextureType type;
        std::string dir;
        std::string path;

};

#endif
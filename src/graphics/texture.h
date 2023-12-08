#ifndef TEXTURE
#define TEXTURE

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <assimp/scene.h>

#include "../../lib/stb/stb_image.h" // include glad to get all the required OpenGL headers

class Texture {
    public:
        Texture();
        Texture(std::string name);
        Texture(std::string dir, std::string path, aiTextureType type);

        void generate();
        void load(bool flip = true);

        void bind();

        /*
            Texture object values
        */
        unsigned int id;
        aiTextureType type;
        std::string name;
        std::string dir;
        std::string path;

};

#endif
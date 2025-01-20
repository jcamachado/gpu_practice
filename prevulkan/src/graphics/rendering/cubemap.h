#ifndef CUBEMAP_H
#define CUBEMAP_H

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <stb/stb_image.h>

#include <string>
#include <vector>

#include "../memory/vertexmemory.hpp"
#include "../rendering/shader.h"

class Scene;

/*
    Skybox is a type of cubemap that is rendered at infinity.
*/
class Cubemap{
    public:
        // Texture object
        unsigned int id;
        
        Cubemap();

        void generate();

        void bind();

        /*
            The order matters. 
            Positive X, Negative X, Positive Y, Negative Y, Positive Z, Negative Z
        */
        void loadTextures(
            std::string directory,
            std::string right   = "right.png",
            std::string left    = "left.png",
            std::string top     = "top.png",
            std::string bottom  = "bottom.png",
            std::string front   = "front.png",
            std::string back    = "back.png"
        );

        // So we can work with the cubemap as a texture, for example shader uniforms like shadow maps
        void allocate(GLenum format, GLuint width, GLuint height, GLenum type);

        void init();

        void render(Shader shader, Scene *scene);

        void cleanup();

    private:
        std::string directory;
        std::vector<std::string> faces;
        bool hasTextures;

        // Cube
        ArrayObject VAO;
};

#endif
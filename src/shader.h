#ifndef SHADER_H
#define SHADER_H

#include "../lib/glad/glad.h" // include glad to get all the required OpenGL headers

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

class Shader{
    public:
        unsigned int id;
        Shader(const char* vertexShaderPath, const char* fragmentShaderPath);
        void activate();

        // utility functions
        std::string loadShaderSrc(const char* filepath);
        GLuint compileShader(const char* source, GLenum type);

        // uniform functions
        void setMat4(const std::string& name, glm::mat4 value);

};
#endif


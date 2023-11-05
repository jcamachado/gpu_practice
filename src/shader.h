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
        void setBool(const std::string& name, bool value);
        void setInt(const std::string& name, int value);
        void setFloat(const std::string& name, float value);
        void set4Float(const std::string& name, float v1, float v2, float v3, float v4);
        void setMat4(const std::string& name, glm::mat4 value);

};
#endif


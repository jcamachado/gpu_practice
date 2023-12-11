#ifndef SHADER_H
#define SHADER_H

#include "../../lib/glad/glad.h" // include glad to get all the required OpenGL headers

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <assimp/scene.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

class Shader{
    public:
        unsigned int id;
        Shader();
        // Initialize with paths to vertex, fragment and geometry shaders (optional)
        Shader(
            bool includeDefaultHeader,
            const char* vertexShaderPath, 
            const char* fragmentShaderPath, 
            const char* geometryShaderPath = nullptr
        );

        /*
            Process functions
        */
        // Generates using vertex, fragment and geometry shaders (optional)
        void generate(
            bool includeDefaultHeader,
            const char* vertexShaderPath, 
            const char* fragmentShaderPath, 
            const char* geometryShaderPath = nullptr);
        
        void activate();

        // utility functions
        GLuint compileShader(bool includeDefaultHeader, const char* source, GLenum type);

        // uniform functions
        void setBool(const std::string& name, bool value);
        void setInt(const std::string& name, int value);
        void setFloat(const std::string& name, float value);
        void set3Float(const std::string& name, glm::vec3 v);
        void set3Float(const std::string& name, float v1, float v2, float v3);
        void set4Float(const std::string& name, float v1, float v2, float v3, float v4);
        void set4Float(const std::string& name, aiColor4D color);
        void set4Float(const std::string& name, glm::vec4 v);
        void setMat4(const std::string& name, glm::mat4 v);

        /*
            static
        */
        // Default directory
        static std::string defaultDirectory;

        // Stream containing the default header source
        static std::stringstream defaultHeaders;

        // Load into default header
        static void loadIntoDefault(const char* filepath);

        // Clear default header (after shader compilation)
        static void clearDefault();

        // Load shader source code (string) from file
        static std::string loadShaderSrc(bool includeDefaultHeader, const char* filepath);
};
#endif


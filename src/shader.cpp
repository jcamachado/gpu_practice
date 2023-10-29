#include "shader.h"
//.h eh feito principalmente pros imports e definicoes de funcoes
//o .cpp eh feito principalmente pra implementar as funcoes definidas no .h

Shader::Shader(const char* vertexShaderPath, const char* fragmentShaderPath){
    int success;
    char infoLog[512];

    GLuint vertexShader = compileShader(vertexShaderPath, GL_VERTEX_SHADER);
    GLuint fragShader = compileShader(fragmentShaderPath, GL_FRAGMENT_SHADER);

    id = glCreateProgram();
    glAttachShader(id, vertexShader);
    glAttachShader(id, fragShader);
    glLinkProgram(id);

    glGetProgramiv(id, GL_LINK_STATUS, &success); // Check if shader program linked successfully
    if(!success){ // Check if shader program linked successfully
        glGetProgramInfoLog(id, 512, NULL, infoLog); // Get error message
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl; // Print error message
    }
    
    glDeleteShader(vertexShader); // Delete vertex shader
    glDeleteShader(fragShader); // Delete fragment shader
}

void Shader::activate(){
    glUseProgram(id);
}


std::string Shader::loadShaderSrc(const char* filename){ // Function for loading shader source code
    std::ifstream file;
    std::stringstream buf;

    std::string ret = "";
    file.open(filename); // Open file
    if (file.is_open()){ // Check if file is open
        buf << file.rdbuf(); // Read file buffer into stringstream
        ret = buf.str(); // Set return string to string from stringstream
    }
    else{
        std::cout << "Unable to open file " << filename << std::endl; // Print error message
    }

    file.close(); // Close file

    return ret; // Return string from stringstream
}

GLuint Shader::compileShader (const char* filepath, GLenum type){ // Function for compiling shader
    int success;
    char infoLog[512];

    GLuint ret = glCreateShader(type); // Create shader and set the type
    std::string shaderSrc = loadShaderSrc(filepath); // Load shader source code
    const GLchar* shader = shaderSrc.c_str(); // Get pointer to shader source code
    glShaderSource(ret, 1, &shader, NULL); // Set shader source code
    glCompileShader(ret); // Compile shader

    glGetShaderiv(ret, GL_COMPILE_STATUS, &success); // Check if fragment shader compiled successfully
    if(!success){ // Check if fragment shader compiled successfully
        glGetShaderInfoLog(ret, 512, NULL, infoLog); // Get error message
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl; // Print error message
    }

    return ret; // Return compiled shader
}

void Shader::setMat4(const std::string& name, glm::mat4 value){
    //name eh o nome da variavel no shader, ex: "transform"
    //ou seja, value sera a matrix de transformacao aplicada ao shader atraves da variavel "transform"
    glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
}
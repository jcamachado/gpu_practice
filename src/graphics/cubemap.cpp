#include "cubemap.h"
#include "../scene.h"       // we call scene methods here

Cubemap::Cubemap(): hasTextures(false) {}

void Cubemap::generate() {
    glGenTextures(1, &id);
}

void Cubemap::bind() {
    glBindTexture(GL_TEXTURE_CUBE_MAP, id);
}

void Cubemap::loadTextures(
    std::string _directory,
    std::string right,
    std::string left,
    std::string top,
    std::string bottom,
    std::string front,
    std::string back
){
    directory = _directory;
    hasTextures = true;
    faces = { right, left, top, bottom, front, back };
    
    int width, height, nChannels;

    for (unsigned int i = 0; i < 6; i++){
        unsigned char* data = stbi_load((directory + "/" + faces[i]).c_str(),
            &width, &height, &nChannels, 0);

        GLenum colorMode = GL_RED;
        switch (nChannels) {
            case 3:
                colorMode = GL_RGB;
                break;
            case 4:
                colorMode = GL_RGBA;
                break;
        }
        
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, colorMode, width, height, 0, colorMode, GL_UNSIGNED_BYTE, data);
        }
        else {
            std::cout << "Failed to load texture at " << faces[i] << std::endl;
        }

        stbi_image_free(data);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

// We have to allocate 6 texture slots for the cubemap but we wont allocate any data to them
// Just like we did on the texture allocate method
void Cubemap::allocate(GLenum format, GLuint width, GLuint height, GLenum type){
    hasTextures = true;

    for (unsigned int i = 0; i < 6; i++){
        /*
            0 is Positive X, 1 is Negative X, 
            2 is Positive Y, 3 is Negative Y, 
            4 is Positive Z, 5 is Negative Z
        */        
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
            0, format, width, height, 0, format, type, NULL);
    }

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);	
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); 
}

void Cubemap::init(){
    // Setup vertices
    float skyboxVertices[] = {
		-1.0f,  1.0f, -1.0f,        // Back face
		-1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,        // Left face
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		 1.0f, -1.0f, -1.0f,        // Right face
		 1.0f, -1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,        // Front face
		-1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		-1.0f,  1.0f, -1.0f,        // Top face
		 1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,        // Bottom face
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f
	};

    // Generate/Setup VAO
    VAO.generate();
    VAO.bind();

    // Setup VBO
    VAO["VBO"] = BufferObject(GL_ARRAY_BUFFER);
    VAO["VBO"].generate();
    VAO["VBO"].bind();
    VAO["VBO"].setData<float>(36 * 3, skyboxVertices, GL_STATIC_DRAW);

    // Set attribute pointers
    VAO["VBO"].setAttrPointer<GLfloat>(0, 3, GL_FLOAT, 3, 0);
    VAO["VBO"].clear();

    ArrayObject::clear();
}

void Cubemap::render(Shader shader, Scene *scene){
    /*
        Thinking about the Skybox rendering.
        - By default, glDepthMash is true.
        But in this case, when we want to render the skybox, we want to disable it, 
        so we the skybox is always rendered behind everything else.

        - We dont want to translate it. We want the rotation and projection of the camera, but not the view.
        So we dont see the outside of the cube, so the cube is always around the camera
        Note: This translation restriction should be in a separate class, like a Skybox class.
        But for now, we are using the Skybox and CubeMap classes interchangeably.
    */    
    glDepthMask(GL_FALSE);

    // Remove translation from view matrix. Remove 4th column and row (3x3) and turn it back into a 4x4 matrix
    glm::mat4 view = glm::mat4(glm::mat3(scene->getActiveCamera()->getViewMatrix()));
    shader.setMat4("view", view);
    shader.setMat4("projection", scene->projection);

    if (hasTextures){
        bind();
    }
    
    VAO.bind();
    VAO.draw(GL_TRIANGLES, 0, 36);
    ArrayObject::clear();

    glDepthMask(GL_TRUE);
}

void Cubemap::cleanup(){
    VAO.cleanup();
}
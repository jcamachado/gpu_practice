#include "mesh.h"

#include <iostream>

std::vector<Vertex> Vertex::genList(float* vertices, int nVertices){
    // 5 floats per vertex, 3 for position, 2 for texture coordinates
    std::vector<Vertex> ret(nVertices);
    
    int stride = sizeof(Vertex) / sizeof(float);
    
    for(int i=0; i<nVertices; i++){
        ret[i].pos = glm::vec3(
            vertices[i*stride+0], 
            vertices[i*stride+1], 
            vertices[i*stride+2]
        );

        ret[i].normal = glm::vec3(
            vertices[i*stride+3], 
            vertices[i*stride+4], 
            vertices[i*stride+5]
        );

        ret[i].texCoord = glm::vec2(
            vertices[i*stride+6], 
            vertices[i*stride+7]
        );
    }

    return ret;
}

// Default constructor
Mesh::Mesh() {}
 
// Initialize as textured object
Mesh::Mesh(BoundingRegion br, std::vector<Texture> textures)
    : br(br), textures(textures), noTextures(false) {}
 
// Initialize as material object
Mesh::Mesh(BoundingRegion br, aiColor4D diff, aiColor4D spec)
    : br(br), diffuse(diff), specular(spec), noTextures(true) {}

// Load vertex and index data
void Mesh::loadData(std::vector<Vertex> _vertices, std::vector<unsigned int> _indices) {
    this->vertices = _vertices;
    this->indices = _indices;
 
    // Bind VAO
    VAO.generate();
    VAO.bind();
 
    // Generate/Set EBO
    VAO["EBO"] = BufferObject(GL_ELEMENT_ARRAY_BUFFER);
    VAO["EBO"].generate();
    VAO["EBO"].bind();
    VAO["EBO"].setData<GLuint>(this->indices.size(), &this->indices[0], GL_STATIC_DRAW);
 
    // Load data into vertex buffers
    VAO["VBO"] = BufferObject(GL_ARRAY_BUFFER);
    VAO["VBO"].generate();
    VAO["VBO"].bind();
    VAO["VBO"].setData<Vertex>(this->vertices.size(), &this->vertices[0], GL_STATIC_DRAW);
 
    // Set the vertex attribute pointers
    VAO["VBO"].bind();
    VAO["VBO"].setAttrPointer<GLfloat>(0, 3, GL_FLOAT, 8, 0);   // Vertex Positions
    VAO["VBO"].setAttrPointer<GLfloat>(1, 3, GL_FLOAT, 8, 3);   // Normal ray
    VAO["VBO"].setAttrPointer<GLfloat>(2, 3, GL_FLOAT, 8, 6);   // Vertex texture coords
    
    VAO["VBO"].clear();
 
    ArrayObject::clear();
}

void Mesh::render(Shader shader, unsigned int numInstances){
    if(noTextures){
        //materials
        shader.set4Float("material.diffuse", diffuse);
        shader.set4Float("material.specular", specular);
        shader.setInt("noTexture", 1);
    }
    else{
        // textures
        unsigned int diffuseIdx = 0;
        unsigned int specularIdx = 0;

        for(unsigned int i = 0; i < textures.size(); i++){
            /*
                Activate proper texture unit before binding
                Retrieve texture info (the N in diffuse_textureN)
            */
            glActiveTexture(GL_TEXTURE0 + i);
            std::string name;
            switch(textures[i].type){
                case aiTextureType_DIFFUSE:
                    name = "diffuse" + std::to_string(diffuseIdx++);
                    break;
                case aiTextureType_SPECULAR:
                    name = "specular" + std::to_string(specularIdx++);
                    break;
            }
            // set shader value
            shader.setInt(name, i);
            textures[i].bind();
        }
    }

    VAO.bind();                                                                 // Bind VAO
    VAO.draw(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0, numInstances);   // Draw
    ArrayObject::clear();
    glActiveTexture(GL_TEXTURE0);                                               // Reset texture unit
}


void Mesh::setup(){
    /*
        VAO and VBO
        VAO, VBO are bound by the shader program
    */
    // bind VAO
    VAO.generate();
    VAO.bind();

    // generate/set EBO
    // We gave it a name EBO
    // The reason why we can call = is because when we overwrite the operator, we're returning a reference to the object
    // so this is a modifiable value
    VAO["EBO"] = BufferObject(GL_ELEMENT_ARRAY_BUFFER);
    VAO["EBO"].generate();
    VAO["EBO"].bind();
    VAO["EBO"].setData<GLuint>(indices.size(), &indices[0], GL_STATIC_DRAW);

    // generate/set VBO
    VAO["VBO"] = BufferObject(GL_ARRAY_BUFFER);
    VAO["VBO"].generate();
    VAO["VBO"].bind();
    VAO["VBO"].setData<Vertex>(vertices.size(), &vertices[0], GL_STATIC_DRAW);

    // set vertex attribute pointers
    // vertex positions, Vertex is 32 bytes because it has 8 floats
    VAO["VBO"].setAttrPointer<GLfloat>(0, 3, GL_FLOAT, 8, 0);
    // normal ray
    VAO["VBO"].setAttrPointer<GLfloat>(1, 3, GL_FLOAT, 8, 3);
    // texture coordinates
    VAO["VBO"].setAttrPointer<GLfloat>(2, 2, GL_FLOAT, 8, 6);

    VAO["VBO"].clear();

    ArrayObject::clear();
}

void Mesh::cleanup(){
    VAO.cleanup();
}
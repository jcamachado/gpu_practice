#include "mesh.h"

std::vector<Vertex> Vertex::genList(float* vertices, int nVertices){
    // 5 floats per vertex, 3 for position, 2 for texture coordinates
    std::vector<Vertex> ret(nVertices);
    
    int stride = sizeof(Vertex) / sizeof(float);
    
    for(int i=0; i<nVertices; i++){
        ret[i].pos = glm::vec3(
            vertices[i*stride], 
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

Mesh::Mesh(){}

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures)
    :
    vertices(vertices),
    indices(indices),
    textures(textures)
    {
    setup();
}



void Mesh::render(Shader shader){
    // Bind textures
    for(unsigned int i=0; i<textures.size(); i++){
        shader.setInt(textures[i].name, textures[i].id); // Set texture unit
        glActiveTexture(GL_TEXTURE0 + i); // Activate texture unit
        textures[i].bind();
    }

    // Bind VAO
    glBindVertexArray(VAO);

    // Draw
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE0); // Reset texture unit
}

void Mesh::cleanup(){
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1,&EBO);
}

void Mesh::setup(){
    // VAO and VBO
    // VAO, VBO are bound by the shader program

    glGenVertexArrays(1, &VAO); // Generate VAO
    glGenBuffers(1, &VBO); // Generate VBO
    glGenBuffers(1, &EBO); // Generate EBO

    // Bind VAO and VBO
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW); // Set vertex data

    // Bind EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    // Set vertex data by passing indices
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW); 

    // Set vertex attributes pointers
    
    // vertex.pos, 3 floats per vertex, offset of pos in Vertex struct
    glEnableVertexAttribArray(0); // Enable vertex attribute pointer, index 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos)); // Set vertex attribute pointer

    // vertex.normal 3 floats per vertex, offset 3 floats from vertex.
    glEnableVertexAttribArray(1); // Enable vertex attribute pointer, index 2
    // Set vertex attribute pointer
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*) offsetof(Vertex, normal));


    //vertex.texCoord, 2 floats per vertex, offset 6 floats from vertex.pos
    glEnableVertexAttribArray(2); // Enable vertex attribute pointer, index 1
    // Set vertex attribute pointer
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*) offsetof(Vertex, texCoord)); 

}
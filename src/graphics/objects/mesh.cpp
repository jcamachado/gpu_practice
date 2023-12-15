#include "mesh.h"

#include <iostream>

std::vector<Vertex> Vertex::genList(float* vertices, int nVertices){
    /*  
    8 floats per vertex in world space
        3 for position, 3 for normal, 2 for texture coordinates
        Not counting tangent vector, therefore 8 floats (11 - 3 of tangent)
    */
   std::vector<Vertex> ret(nVertices);
    
    int stride = 8;

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


/*
    Motivation: A lot of vertices are reused, so we want to average the tangent vectors so
    that we can save memory and have more accurate calculations
    Ex:
    Two values a, b and average m
    m = (a + b) / 2
    add new value c and get new average m'
    m' = (a + b + c) / 3 = k*m + l, where k is the weight of the previous average and l is the weight of the new value
    2 = existing contributions
    2 + 1 = 3 = new number of values (contributions)
    Proof:
    m' = (2/3)*(m) + c/3 =(2/3)*((a+b)/2) + c/3 = (a + b + c) / 3
    
*/
void averageVectors(glm::vec3& baseVec, glm::vec3& addition, unsigned char existingContributions){
    // if no existing contributions, the average of addition of the contribution = contribution
    if (!existingContributions){
        baseVec = addition;
    }
    else { 
        /*
            Otherwise, if there are contributions, calculate the value of the average
        */
        float f = 1.0f / (float)(existingContributions + 1);

        baseVec *= (float)existingContributions * f;

        baseVec += addition * f;
    }

}

void Vertex::calcTanVectors(
    std::vector<Vertex>& list, 
    std::vector<unsigned int>& indices
){
        unsigned char* counts = (unsigned char*)malloc(list.size() * sizeof(unsigned char));
        try{
        // if (indices.size() % 3 != 0){
        //     std::cout << "Error: Indices size is not divisible by 3. Not a face?" << std::endl;
        //     return;
        // }
        for (unsigned int i = 0, len = list.size(); i < len; i++){
            counts[i] = 0;
        }

        // Iterate through indices and calculate vectors for each face
        for (unsigned int i = 0, len = indices.size(); i < len; i += 3){
            // 3 vertices per face
            Vertex v1 = list[indices[i + 0]];
            Vertex v2 = list[indices[i + 1]];
            Vertex v3 = list[indices[i + 2]];

            // Calculate edges
            glm::vec3 edge1 = v2.pos - v1.pos;
            glm::vec3 edge2 = v3.pos - v1.pos;
            
            // Calculates delta UVs
            glm::vec2 deltaUV1 = v2.texCoord - v1.texCoord;
            glm::vec2 deltaUV2 = v3.texCoord - v1.texCoord;

            // Use inverse of UV matrix to determine tangent (f=1/det)
            float det = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;

            // inverse of determinant = 0
            if (det == 0.0f){
                std::cout << "Error: Determinant is 0." << std::endl; 
                std::cout << "Cannot calculate tangent vector" << std::endl; 
                std::cout << "Parallel" << std::endl; 
                return;
            }

            float f = 1.0f / det;

            // Tangent components
            glm::vec3 tangent = {
                f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x),
                f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y),
                f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z)
            };

            // Average in the new tangent vector
            averageVectors(list[indices[i + 0]].tangent, tangent, counts[indices[i + 0]]++);
            averageVectors(list[indices[i + 1]].tangent, tangent, counts[indices[i + 1]]++);
            averageVectors(list[indices[i + 2]].tangent, tangent, counts[indices[i + 2]]++);
        }
        free(counts);
    }
    catch(std::exception e){
        std::cout << "Error: " << e.what() << std::endl;
        free(counts);
        throw e;
    }
}

/*
    Constructors
*/
Mesh::Mesh()                                                        //1 Default
    : collision(NULL) {}                                            

Mesh::Mesh(BoundingRegion br)                                       //2 Init with bounding region
    : br(br), collision(NULL){}

Mesh::Mesh(BoundingRegion br, std::vector<Texture> textures)        //3 Init as textured object
    : Mesh(br) {
    setupTextures(textures);
}
 
Mesh::Mesh(BoundingRegion br, aiColor4D diff, aiColor4D spec)       //4 Initialize as material object
    : Mesh(br) {
    setupColors(diff, spec);
}

Mesh::Mesh(BoundingRegion br, Material m)                           //5 Initialize with a material (analog to 4)  
    : Mesh(br) {
    setupMaterial(m);
}

// Load vertex and index data
void Mesh::loadData(
    std::vector<Vertex> _vertices, 
    std::vector<unsigned int> _indices, 
    bool pad
) {
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


    unsigned int size = this->vertices.size();
    if (pad && size){
        size++;
    }

    VAO["VBO"].setData<Vertex>(size, &this->vertices[0], GL_STATIC_DRAW);
 
    /*
        Set the vertex attribute pointers
        Stride = 11;    3 for position, 3 for normal, 2 for texture coordinates, 3 for tangent
    */
    VAO["VBO"].bind();
    VAO["VBO"].setAttrPointer<GLfloat>(0, 3, GL_FLOAT, 11, 0);   // Vertex Positions
    VAO["VBO"].setAttrPointer<GLfloat>(1, 3, GL_FLOAT, 11, 3);   // Normal ray
    VAO["VBO"].setAttrPointer<GLfloat>(2, 3, GL_FLOAT, 11, 6);   // Vertex texture coords
    VAO["VBO"].setAttrPointer<GLfloat>(3, 3, GL_FLOAT, 11, 8);   // Tangent vector
    
    VAO["VBO"].clear();
 
    ArrayObject::clear();
}

void Mesh::loadCollisionMesh(
    unsigned int nPoints, 
    float* coordinates, 
    unsigned int nFaces, 
    unsigned int* indices
){    
    this->collision = new CollisionMesh(nPoints, coordinates, nFaces, indices);
    this->br = this->collision->br;
}

void Mesh::setupTextures(std::vector<Texture> textures){
    this->noTextures = false;
    this->textures.insert(this->textures.end(), textures.begin(), textures.end());
}

void Mesh::setupColors(aiColor4D diff, aiColor4D spec){
    this->noTextures = true;
    this->diffuse = diff;
    this->specular = spec;
}

void Mesh::setupMaterial(Material material){
    this->noTextures = true;
    this->diffuse = {material.diffuse.r, material.diffuse.g, material.diffuse.b, 1.0f};
    this->specular = {material.specular.r, material.specular.g, material.specular.b, 1.0f};
}

void Mesh::render(Shader shader, unsigned int numInstances){
    shader.setBool("noNormalMap", true);

    if(noTextures){
        //materials
        shader.set4Float("material.diffuse", diffuse);
        shader.set4Float("material.specular", specular);
        shader.setBool("noTexture", true);
    }
    else{
        // textures
        unsigned int diffuseIdx = 0;
        unsigned int normalIdx = 0;
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
                case aiTextureType_NORMALS:
                    name = "normal" + std::to_string(normalIdx++);
                    shader.setBool("noNormalMap", false);
                    break;
                case aiTextureType_SPECULAR:
                    name = "specular" + std::to_string(specularIdx++);
                    break;
                case aiTextureType_NONE:
                    name = textures[i].name;
                    break;
            }
            // set shader value
            shader.setInt(name, i);
            textures[i].bind();
        }
        shader.setBool("noTexture", false);
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

    for (Texture t: textures) {
        t.cleanup();
    }
}
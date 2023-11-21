#include "model.h"

#include "../physics/environment.h"

#include <iostream>


Model::Model(std::string id, BoundTypes boundType, unsigned int maxNumInstances, unsigned int flags)
    :id (id), 
    boundType(boundType), 
    switches(flags),
    currentNumInstances(0), 
    maxNumInstances(maxNumInstances) {}

unsigned int Model::generateInstance(glm::vec3 size, float mass, glm::vec3 pos){
    if (currentNumInstances >= maxNumInstances){
        return -1;                                                  // All slots filled 
    }

    instances.push_back(RigidBody(id, size, mass, pos));
    return currentNumInstances++;
}

void Model::initInstances() {
    glm::vec3* posData = nullptr;
    glm::vec3* sizeData = nullptr;
    GLenum usage = GL_DYNAMIC_DRAW;

    std::vector<glm::vec3> positions, sizes;

    if (States::isActive(&switches, CONST_INSTANCES)){
        // Set data pointers
        for(unsigned int i = 0; i < currentNumInstances; i++){
            positions.push_back(instances[i].pos);
            sizes.push_back(instances[i].size);
        }

        if (positions.size() > 0) {
            posData = &positions[0];
            sizeData = &sizes[0];
        }

        usage = GL_STATIC_DRAW;                                 // CONST_INSTANCES kind of a synonym for static instances
    }

    /*
        This is for whole objects
        - Generate positions VBO
        - Generate sizes VBO
    */
    posVBO = BufferObject(GL_ARRAY_BUFFER);
    posVBO.generate();
    posVBO.bind();
    posVBO.setData<glm::vec3>(UPPER_BOUND, posData, GL_DYNAMIC_DRAW);
    
    sizeVBO = BufferObject(GL_ARRAY_BUFFER);
    sizeVBO.generate();
    sizeVBO.bind();
    sizeVBO.setData<glm::vec3>(UPPER_BOUND, sizeData, GL_DYNAMIC_DRAW);

    /*
        Set attribute pointers for each mesh
        - For positions
        - For sizes
    */
    for(unsigned int i=0, size = meshes.size(); i<size; i++){
        meshes[i].VAO.bind();

        // Set vertex attribute pointers
        posVBO.bind();
        /*
            .setAttrPointer<template>();
            1st param: 0, 1 and 2 are used for normal mesh (i believe he refers to mesh.cpp)
            2nd param and 3rd are related, so 3 GL_FLOATs
            4th param: stride is 1 glm::vec3 (related to template)
            6th param: reset every 1 instance
        */
        posVBO.setAttrPointer<glm::vec3>(3, 3, GL_FLOAT, 1, 0, 1); 

        sizeVBO.bind();
        sizeVBO.setAttrPointer<glm::vec3>(4, 3, GL_FLOAT, 1, 0, 1);

        ArrayObject::clear();
    }
}

void Model::removeInstance(unsigned int idx){
    instances.erase(instances.begin() + idx);
    currentNumInstances--;
}

void Model::removeInstance(std::string instanceId){
    unsigned int idx = getIdx(instanceId);
    if (idx != -1){
        instances.erase(instances.begin() + idx);
        currentNumInstances--;
    }
}

unsigned int Model::getIdx(std::string id){
    for (int i = 0; i < currentNumInstances; i++){
        if (instances[i] == id){                                // Uses RB operator==
            return i;
        }
    }
    return -1;
}

void Model::init() {}

void Model::render(Shader shader, float dt, Scene *scene, bool setModel){
    /*
        We won't transform here, that data will be passed into the VBO and 
        all calculations will be done in the shaders.
        Just like we are doing for the model and the instances.
        We will be this to all models.
    */
    if (setModel){
        shader.setMat4("model", glm::mat4(1.0f));
    }

    if (!States::isActive(&switches, CONST_INSTANCES)){
        // Update VBO data
        // std::vector<glm::vec3> positions(currentNumInstances), sizes(currentNumInstances);
        std::vector<glm::vec3> positions, sizes;
        bool doUpdate = States::isActive(&switches, DYNAMIC);

        for (int i = 0; i < currentNumInstances; i++){
            if (doUpdate){
                instances[i].update(dt);                        // Update RigidBody
            }
            // positions[i] = instances[i].pos;
            // sizes[i] = instances[i].size;
            // positions[i] = instances[i]->pos;
            // sizes[i] = instances[i]->size;
            positions.push_back(instances[i].pos);
            sizes.push_back(instances[i].size);
        }

        posVBO.bind();
        posVBO.updateData<glm::vec3>(0, currentNumInstances, &positions[0]);
        sizeVBO.bind();
        sizeVBO.updateData<glm::vec3>(0, currentNumInstances, &sizes[0]);
    }

    shader.setFloat("material.shininess", 0.5f);

    for (unsigned int i = 0, noMeshes = meshes.size();
        i < noMeshes;
        i++) {
        meshes[i].render(shader, currentNumInstances);
    }
}

void Model::cleanup(){
    for (unsigned int i = 0; i < instances.size(); i++) {
        meshes[i].cleanup();
    }

    posVBO.cleanup();
    sizeVBO.cleanup();
}

void Model::loadModel(std::string path){
    Assimp::Importer import;
    const aiScene *scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
    // check for errors
    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cout << 
        "ERROR::ASSIMP::Could not load model at" << 
        path << 
        import.GetErrorString() << 
        std::endl;
        return;
    }
    directory = path.substr(0, path.find_last_of('/'));
    processNode(scene->mRootNode, scene);
}

void Model::processNode(aiNode *node, const aiScene *scene){
    // process all the node's meshes (if any)
    for(unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]]; 
        meshes.push_back(processMesh(mesh, scene));			
    }
    // then do the same for each of its children
    for(unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene){
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;

    BoundingRegion br(boundType);
    /*
        ~0: bit complement of int zero cast into float, which is the max float
        this is for every value in each axis.

        - min point = max float
        - max point = -min
    */ 
    glm::vec3 min((float)(~0));         
    glm::vec3 max(-(float)(~0));
    // process vertices
    for(unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        // position
        vertex.pos = glm::vec3(
            mesh->mVertices[i].x, 
            mesh->mVertices[i].y, 
            mesh->mVertices[i].z
        );

        for(int j=0; j<3; j++){
            // if smaller than min
            if(vertex.pos[j] < min[j]){
                min[j] = vertex.pos[j];
            }
            // if larger than max
            if(vertex.pos[j] > max[j]){
                max[j] = vertex.pos[j];
            }
        }

        //  normal vectors 
        vertex.normal = glm::vec3(
            mesh->mNormals[i].x, 
            mesh->mNormals[i].y, 
            mesh->mNormals[i].z
        );

        // texture coordinates
        // mTextireCoords stores up to 8 different texture coordinates per vertex. 
        // We only care about the first set of texture coordinates (if it does exist).
        if(mesh->mTextureCoords[0]) { // does the mesh contain texture coordinates?
            vertex.texCoord = glm::vec2(
                mesh->mTextureCoords[0][i].x, 
                mesh->mTextureCoords[0][i].y
            );

        } else {
            vertex.texCoord = glm::vec2(0.0f);

        }
        vertices.push_back(vertex);
    }

    // process min/max for bounding region BR
    if (boundType == BoundTypes::AABB){
        // min and max are already calculated
        br.min = min;
        br.max = max;
    }else{
        // calculate center and radius
        br.center = BoundingRegion(min, max).calculateCenter();
        float maxRadiusSquare = 0.0f;
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            float radiusSquared = 0.0f; //distance for this vertex
            for (int j = 0; j < 3; j++) {
                radiusSquared += (vertices[i].pos[j] - br.center[j]) * (vertices[i].pos[j] - br.center[j]);
            }
            if (radiusSquared > maxRadiusSquare) {
                // if this distance is larger than the current max, set it as the new max
                // if a^2 > b^2, then |a| > |b
                maxRadiusSquare = radiusSquared;
            }
        }
        // calling here sqrt is more efficient than calling it everytime
        br.radius = sqrt(maxRadiusSquare);
    }


    // process indices
    for(unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        // retrieve all indices of the face and store them in the indices vector
        for(unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }  

    Mesh ret;

    // process material
    if (mesh->mMaterialIndex >= 0) {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
 
        if (States::isActive<unsigned int>(&switches, NO_TEX)) {
            // 1. diffuse colors
            aiColor4D diff(1.0f);
            aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diff);
            // 2. specular colors
            aiColor4D spec(1.0f);
            aiGetMaterialColor(material, AI_MATKEY_COLOR_SPECULAR, &spec);
 
            ret = Mesh(br, diff, spec);
        }
        else {
            // 1. diffuse maps
            std::vector<Texture> diffuseMaps = loadTextures(material, aiTextureType_DIFFUSE);
            textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
            // 2. specular maps
            std::vector<Texture> specularMaps = loadTextures(material, aiTextureType_SPECULAR);
            textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
 
            ret = Mesh(br, textures);
        }
    }
 
    ret.loadData(vertices, indices);
    return ret;
}

std::vector<Texture> Model::loadTextures(aiMaterial *mat, aiTextureType type){
    std::vector<Texture> textures;

    for(unsigned int i = 0; i < mat->GetTextureCount(type); i++) {
        aiString str;
        mat->GetTexture(type, i, &str);
        
        // prevent duplicate loading
        // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
        bool skip = false;

        for(unsigned int j = 0; j < textures_loaded.size(); j++) {
            if(std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0) {
                textures.push_back(textures_loaded[j]);
                skip = true; 
                break;
            }
        }
        if(!skip) {   // if texture hasn't been loaded already, load it
            Texture texture = Texture(directory, str.C_Str(), type);
            texture.load(false);
            textures.push_back(texture);
            textures_loaded.push_back(texture);  // add to loaded textures
        }
    }
    return textures;
}
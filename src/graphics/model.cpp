#include "model.h"

#include "../physics/environment.h"

#include <iostream>


Model::Model(std::string id, BoundTypes boundType, unsigned int maxNInstances, unsigned int flags)
    :id (id), 
    boundType(boundType), 
    switches(flags),
    currentNInstances(0), 
    maxNInstances(maxNInstances) {}

RigidBody* Model::generateInstance(glm::vec3 size, float mass, glm::vec3 pos){
    if (currentNInstances >= maxNInstances){
        return nullptr;                                                  // All slots filled 
    }

    instances.push_back(new RigidBody(id, size, mass, pos));
    return instances[currentNInstances++];
}

void Model::initInstances() {
    glm::vec3* posData = nullptr;
    glm::vec3* sizeData = nullptr;
    GLenum usage = GL_DYNAMIC_DRAW;

    std::vector<glm::vec3> positions, sizes;

    if (States::isActive(&switches, CONST_INSTANCES)){
        // Set data pointers
        for(unsigned int i = 0; i < currentNInstances; i++){
            positions.push_back(instances[i]->pos);
            sizes.push_back(instances[i]->size);
        }

        if (positions.size() > 0) {
            posData = &positions[0];
            sizeData = &sizes[0];
        }
    /*
        CONST_INSTANCES kind of a synonym for static instances
    */
        usage = GL_STATIC_DRAW;                                 
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
            1st param: index of attribute
                In mesh, the indices 0, 1, 2, 3  are used for position, normal, texCoord and tangent
                So: 4 is used for instance vbo positions
                And 5 is used for instance vbo sizes


            2nd param and 3rd are related, so 3 GL_FLOATs
            4th param: stride is 1 glm::vec3 (related to template)
            6th param: reset every 1 instance
        */
        posVBO.setAttrPointer<glm::vec3>(4, 3, GL_FLOAT, 1, 0, 1); 

        sizeVBO.bind();
        sizeVBO.setAttrPointer<glm::vec3>(5, 3, GL_FLOAT, 1, 0, 1);

        ArrayObject::clear();
    }
}

void Model::removeInstance(unsigned int idx){
    instances.erase(instances.begin() + idx);
    currentNInstances--;
}

void Model::removeInstance(std::string instanceId){
    unsigned int idx = getIdx(instanceId);
    if (idx != -1){
        instances.erase(instances.begin() + idx);
        currentNInstances--;
    }
}

unsigned int Model::getIdx(std::string id){
    for (int i = 0; i < currentNInstances; i++){
        if (instances[i]->instanceId == id){                                // Uses RB operator==
            return i;
        }
    }
    return -1;
}

void Model::init() {}
    /*
        We won't transform here, that data will be passed into the VBO and 
        all calculations will be done in the shaders.
        Just like we are doing for the model and the instances.
        We will be this to all models.
    */
void Model::render(Shader shader, float dt, Scene *scene, glm::mat4 model){
    // Set model matrix
    shader.setMat4("model", model);
    // Avoid doing this per phase(Phase or face?), its the same per model 
    shader.setMat3("normalModel", glm::mat3(glm::transpose(glm::inverse(model))));
    
    
    if (!States::isActive(&switches, CONST_INSTANCES)){
        /*
            Dynamic instances - Update VBO data
        */
        std::vector<glm::vec3> positions, sizes;
        bool doUpdate = States::isActive(&switches, DYNAMIC);

        for (int i = 0; i < currentNInstances; i++){
            if (doUpdate){
                instances[i]->update(dt);               // Update RigidBody
                States::activate(&instances[i]->state, INSTANCE_MOVED);
            }else{
                States::deactivate(&instances[i]->state, INSTANCE_MOVED);
            }
            positions.push_back(instances[i]->pos);
            sizes.push_back(instances[i]->size);
        }
        
        posVBO.bind();
        posVBO.updateData<glm::vec3>(0, currentNInstances, &positions[0]);
        sizeVBO.bind();
        sizeVBO.updateData<glm::vec3>(0, currentNInstances, &sizes[0]);
    }

    shader.setFloat("material.shininess", 0.5f);

    
    for (unsigned int i = 0, noMeshes = meshes.size();
        i < noMeshes;
        i++) {
        meshes[i].render(shader, currentNInstances);
    }
}

void Model::cleanup(){
    for (unsigned int i = 0; i < instances.size(); i++) {
        meshes[i].cleanup();
    }

    posVBO.cleanup();
    sizeVBO.cleanup();
}

/*
    Our models attrb (position, normal, texCoord, tangent) are set and calculated in mesh.cpp
    But for the imported ones, assimp will calculate these attributes.
    So we need to get those attributes from assimp and set them in our model.
*/
void Model::loadModel(std::string path){
    Assimp::Importer import;
    /*
        - aiProcess_Triangulate - Makes faces into triangles
        - aiProcess_FlipUVs - Flips the texture coordinates on the y-axis where necessary during processing. 
            Some file formats (such as jpg) store texture coordinates from top to bottom while OpenGL
            (and many others) store them from bottom to top.
        - aiProcess_CalcTangentSpace - Calculates the tangents and bitangents for the imported meshes.
            Assimp stores these tangent vectors in each vertex
    */
    const aiScene *scene = import.ReadFile(
        path, 
        aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
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
    /*
        Process all the node's meshes (Generate all the meshes),
        then do the same for each of its children
    */
    for(unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]]; 
        Mesh newMesh = processMesh(mesh, scene);
        meshes.push_back(newMesh);
        boundingRegions.push_back(newMesh.br);
    }
    for(unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

/*
    aiMesh: Is an Assimp struct that contains all the data about a mesh
*/
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

        // Texture coordinates
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
        // Tangent vector
        vertex.tangent = glm::vec3(
            mesh->mTangents[i].x, 
            mesh->mTangents[i].y, 
            mesh->mTangents[i].z
        );

        vertices.push_back(vertex);
    }

    // process min/max for bounding region BR
    if (boundType == BoundTypes::AABB){
        // min and max are already calculated
        br.min = min;
        br.ogMin = min;
        br.max = max;
        br.ogMax = max;
    }else{
        // calculate center and radius
        br.center = BoundingRegion(min, max).calculateCenter();
        br.ogCenter = br.center;
        float maxRadiusSquare = 0.0f;
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            float radiusSquared = 0.0f;                 // Distance for this vertex
            for (int j = 0; j < 3; j++) {
                radiusSquared += (vertices[i].pos[j] - br.center[j]) * (vertices[i].pos[j] - br.center[j]);
            }
            if (radiusSquared > maxRadiusSquare) {      // If this distance is larger than the current max, set it as the new max
                maxRadiusSquare = radiusSquared;        // If a^2 > b^2, then |a| > |b, saves sqrt calls 
                
            }
        }
        // calling here sqrt is more efficient than calling it everytime
        br.radius = sqrt(maxRadiusSquare);
        br.ogRadius = br.radius;
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
            // Use textures
            // 1. diffuse maps
            std::vector<Texture> diffuseMaps = loadTextures(material, aiTextureType_DIFFUSE);
            textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
            // 2. specular maps
            std::vector<Texture> specularMaps = loadTextures(material, aiTextureType_SPECULAR);
            textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
            // 3. normal maps
            // if file is .obj. Use aiTextureType_HEIGHT instead of aiTextureType_NORMALS
            std::vector<Texture> normalMaps = loadTextures(material, aiTextureType_NORMALS);
            textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
            // 4. 
 
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
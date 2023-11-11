#include "model.h"

Model::Model(glm::vec3 pos, glm::vec3 size, bool noTextures)
    : pos(pos), size(size), noTextures(noTextures)  {}

void Model::init(){}

void Model::render(Shader shader){
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, pos);
    model = glm::scale(model, size);
    shader.setMat4("model", model);

    shader.setFloat("material.shininess", 0.5f);

    for(Mesh mesh : meshes){
        mesh.render(shader);
    }
}

void Model::cleanup(){
    for(Mesh mesh : meshes){
        mesh.cleanup();
    }
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

    // process vertices
    for(unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        // position
        vertex.pos = glm::vec3(
            mesh->mVertices[i].x, 
            mesh->mVertices[i].y, 
            mesh->mVertices[i].z
        );

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

    // process indices
    for(unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        // retrieve all indices of the face and store them in the indices vector
        for(unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }  
    // process material
    if(mesh->mMaterialIndex >= 0) {
        aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];

        if(noTextures){
            // diffuse color
            aiColor4D diff(1.0f);
            aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diff);

            // specular color
            aiColor4D spec(1.0f);
            aiGetMaterialColor(material, AI_MATKEY_COLOR_SPECULAR, &spec);

            return Mesh(vertices, indices, diff, spec);

        //diffuse maps
        std::vector<Texture> diffuseMaps = loadTextures(material, aiTextureType_DIFFUSE);
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
        
        //specular maps
        std::vector<Texture> specularMaps = loadTextures(material, aiTextureType_SPECULAR);
        textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
        }
        return Mesh(vertices, indices, textures);
    }
}

std::vector<Texture> Model::loadTextures(aiMaterial *mat, aiTextureType type){
    std::vector<Texture> textures;
    for(unsigned int i = 0; i < mat->GetTextureCount(type); i++) {
        aiString str;
        mat->GetTexture(type, i, &str);
        std::cout << "Loading texture at " << str.C_Str() << std::endl;
        
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
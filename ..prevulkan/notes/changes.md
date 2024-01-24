(This file has some code that were removed and may proof useful in the future)

/*  (comment on removed code On setting custom fbo in mainloop, because after this we will activate directional lights FBO)
            Render scene to the custom framebuffer
            Depth values will be output to the texture that we attach and 
            the color values will be aoutputted to the render buffer object
        */


(code removed for setting skybox and text on screen)
    Shader skyboxShader("assets/shaders/skybox/skybox.vs", "assets/shaders/skybox/sky.fs");
    // skyboxShader.activate();
    // skyboxShader.set3Float("min", 0.047f, 0.016f, 0.239f);
    // skyboxShader.set3Float("max", 0.945f, 1.000f, 0.682f);

    /*
        Skybox
    */
    Cubemap skybox;
    skybox.init();
    // skybox.loadTextures("assets/skybox");    // Load cubemap texture (image)
    ...
        (in the main loop)
        // skyboxShader.activate();
        // skyboxShader.setFloat("time", scene.variableLog["time"].val<float>());
        // skybox.render(skyboxShader, &scene); //Render skybox

        // scene.renderText(
        //     "comic", 
        //     textShader, 
        //     "Hello World!!", 
        //     50.0f, 
        //     50.0f, 
        //     glm::vec2(1.0f), 
        //     glm::vec3(0.5f, 0.6f, 1.0f)
        // );
        // scene.renderText(
        //     "comic", 
        //     textShader, 
        //     "Time: " + scene.variableLog["time"].dump(), 
        //     50.0f, 
        //     550.0f, 
        //     glm::vec2(1.0f), 
        //     glm::vec3(0.5f, 0.6f, 1.0f)
        // );
        // scene.renderText(
        //     "comic", 
        //     textShader, 
        //     "FPS: " + scene.variableLog["fps"].dump(), 
        //     50.0f, 
        //     550.0f - 40.0f, 
        //     glm::vec2(1.0f), 
        //     glm::vec3(0.5f, 0.6f, 1.0f)
        // );        
    (after mainloop)
    // skybox.cleanup();


(outline code removed from the main loop)

        if (scene.variableLog["displayOutline"].val<bool>()){
            /*
                glStencilMask tells opengl what to bitwise AND the stencil buffer with.
            */
            glStencilMask(0x00);                        // Disable stencil buffer writing for sphersd
            // scene.renderShader(outlineShader, false);    // Render outline
            // scene.renderInstances(sphere.id, outlineShader, dt);
        } 
        ...

        ...

        if (scene.variableLog["displayOutline"].val<bool>()){
            // Always write to stencil buffer with cubes
            glStencilFunc(GL_ALWAYS, 1, 0xFF);              // Set any stencil to 1
            glStencilMask(0xFF);                            // Always write to stencil buffer
            scene.renderInstances(cube.id, shader, dt);     // Render cubes

            glStencilFunc(GL_NOTEQUAL, 1, 0xFF); // render fragments if different than what is stored
            glStencilMask(0x00); // disable writing 
            glDisable(GL_DEPTH_TEST); // disable depth test so outlines are displayed behind objects

            scene.renderShader(outlineShader, false); // Render outline
            scene.renderInstances(cube.id, outlineShader, dt);

            // Reset valus
            glStencilFunc(GL_ALWAYS, 1, 0xFF);  // Every fragment written to stencil buffer
            glStencilMask(0xFF);                // Write always
            glEnable(GL_DEPTH_TEST);            // Re-enable depth test
        }
        else{
            // render cubes normally
            scene.renderInstances(cube.id, shader, dt);     // Render cubes
        }

(octree was not rendered [using box] anymore after new outline method)

(render lamps)
        scene.renderShader(lampShader, false);                  // Render lamps
        scene.renderInstances(lamp.id, lampShader, dt);



        // Update blinn parameter if necessary
        if (Keyboard::keyWentUp(GLFW_KEY_B)){
            variableLog["useBlinn"] = !variableLog["useBlinn"].val<bool>();
        }

        // Toggle gamma correction parameter if necessary
        if (Keyboard::keyWentUp(GLFW_KEY_G)){
            variableLog["useGamma"] = !variableLog["useGamma"].val<bool>();
        }

        // Update outline parameter if necessary
        if (Keyboard::keyWentUp(GLFW_KEY_O)){
            variableLog["displayOutline"] = !variableLog["displayOutline"].val<bool>();
        }

















In instanced.vs video 51, removed aOffset and aSize, therefore removing posVBO and sizeVBO
removed from model.h Model

/*
    VBOs for positions and sizes
    Are these only for instances? Since its for optimization and 
    are called offset in shader, maybe they are like 
*/
BufferObject posVBO;                // Instance position?
BufferObject sizeVBO;               // Instance size?


//These are from buffer
//layout(qualifier1​, qualifier2​ = value, ...) variable definition
//the layout is the location of the vertex attribute in the VBO
layout (location = 0) in vec3 aPos;         // vertex position
layout (location = 1) in vec3 aNormal;      // vertex normal
layout (location = 2) in vec2 aTexCoord;    // vertex texture coordinates
layout (location = 3) in vec3 aTangent;     // vertex tangent
layout (location = 4) in mat4 aModel;       // model matrix, replaces offSet and size (4x4 occupies 4 memory slots)
layout (location = 8) in mat3 normalModel;  // values in tangent space

// layout (location = 4) in vec3 aOffset;      // vertex position in instanced array (posVBO)
// layout (location = 5) in vec3 aSize;        // vertex size in instanced array      (posVBO)

// We will keep some data in world space but we will also pass in tangent data
out VS_OUT {     
    vec3 FragPos;   // Available in world space coordinates and in tangent space coordinates
    vec2 TexCoord;

    TangentLights tanLights;
} vs_out;

uniform mat4 view;
uniform mat4 projection;

uniform vec3 viewPos;
uniform bool noNormalMap;

void main() {
    // Get position in world space
    vec3 pos = aPos * aSize + aOffset;

    // Apply model transformation
    vs_out.FragPos = vec3(model * vec4(pos, 1.0)); //**- model is an uniform

    // Set texture coordinates
    vs_out.TexCoord = aTexCoord;

    // Determine normal vector in tangent space
    if (noNormalMap) {
		// determine normal vector in tangent space
		vs_out.tanLights.Normal = normalize(aNormal);
        vs_out.tanLights.FragPos = vs_out.FragPos;
		vs_out.tanLights.ViewPos = viewPos;
        vs_out.tanLights.dirLightDirection = dirLight.direction;
        		for (int i = 0; i < nPointLights; i++) {
			vs_out.tanLights.pointLightPositions[i] = pointLights[i].position;
		}
        for (int i = 0; i < nSpotLights; i++) {
			vs_out.tanLights.spotLightPositions[i] = spotLights[i].position;
			vs_out.tanLights.spotLightDirections[i] = spotLights[i].direction;
		}
    }
	else
	{
        // determine normal vector in tangent space
		vs_out.tanLights.Normal = normalize(normalModel * aNormal);
        // calculate tangent space matrix
		vec3 T = normalize(normalModel * aTangent);
		vec3 N = vs_out.tanLights.Normal;
		T = normalize(T - dot(T, N) * N); // re-orthogonalize T with respect to N
		vec3 B = cross(N, T); // get B, perpendicular to N and T
		mat3 TBNinv = transpose(mat3(T, B, N)); // orthogonal matrix => transpose = inverse

		// transform positions to the tangent space
		vs_out.tanLights.FragPos = TBNinv * vs_out.FragPos;
		vs_out.tanLights.ViewPos = TBNinv * viewPos;

		// directional light
		vs_out.tanLights.dirLightDirection = TBNinv * dirLight.direction;

		// point lights
		for (int i = 0; i < nPointLights; i++) {
			vs_out.tanLights.pointLightPositions[i] = TBNinv * pointLights[i].position;
		}

		// spot lights
		for (int i = 0; i < nSpotLights; i++) {
			vs_out.tanLights.spotLightPositions[i] = TBNinv * spotLights[i].position;
			vs_out.tanLights.spotLightDirections[i] = TBNinv * spotLights[i].direction;
		}
    }
    // // Calculate tangent space matrix
    // vec3 T = normalize(normalModel * aTangent);
    // vec3 N = vs_out.tanLights.Normal;
    // /*
    //     -Make sure T is perpendicular to N
    //     T is a combination of some T components.
    //         -T is tangent vector, Tt is component in T direction, T^ is T unit vector, Tt*T^ (seems to be the component perpendicular to N)
    //          Tn is the component in N direction(dot(T, N^)), N^ is N unit vector // Question: is is Tn = dot(T, N^) or TnN^ = dot(T, N)
    //     T = Tt * T^ + Tn * N^ 
    //     If we want only the component perpendicular to N (maybe Tt*T^)
    //     T - dot(T, N^) = Tt*T^              
    //     T = T - dot(T, N) * N
    //     dot(T, N) is the component of T that is parallel to N (the magnitude)
    //     and multiply by N to get the component of T that is parallel to N

    //     - To transforming world space coordinates into another space coordinates,
    //     we need to multiply it by the inverse of the space matrix (transformation matrix)
    // */
    // T = normalize(T - dot(T, N) * N); // Re-orthogonalize T with respect to N
    // vec3 B = cross(T, N);           // To make sure B is perpendicular to T and N
    // // TBN is the space of Tangent, Bitangent, Normal (tangent space)
    // mat3 TBNinv = transpose(mat3(T, B, N)); // Orthogonal matrix => transpose = inverse

    // // Transform (world?) positions to the tangent space
    // vs_out.tanLights.FragPos = TBNinv * vs_out.FragPos;
    // vs_out.tanLights.ViewPos = TBNinv * viewPos;

    // // Directional light
    // vs_out.tanLights.dirLightDirection = TBNinv * dirLight.direction;

    // // Point lights
    // // Takes pointLights from the UBO and transform them into tangent space
    // for(int i = 0; i < nPointLights; i++) {
    //     vs_out.tanLights.pointLightPositions[i] = TBNinv * pointLights[i].position;
    // }

    // // Spot lights
    // for(int i = 0; i < nSpotLights; i++) {
    //     vs_out.tanLights.spotLightPositions[i] = TBNinv * spotLights[i].position;
    //     vs_out.tanLights.spotLightDirections[i] = TBNinv * spotLights[i].direction;
    // }

    //fix

    // Set output for fragment shader
    gl_Position = projection * view * vec4(vs_out.FragPos, 1.0);
}


--- model render

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




=========================



Video 53 ,removed boundtypes parametrized and used only spheres

// Model::Model(std::string id, BoundTypes boundType, unsigned int maxNInstances, unsigned int flags)
//     :id (id), 
//     boundType(boundType), 
//     switches(flags),
//     currentNInstances(0), 
//     maxNInstances(maxNInstances),
//     collision(NULL) {}
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

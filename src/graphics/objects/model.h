#ifndef OBJECT_H
#define OBJECT_H

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

#include "mesh.h"

#include "../models/box.hpp"

#include "../../algorithms/bounds.h"

#include "../../physics/collisionmodel.h"
#include "../../physics/rigidbody.h"

#include "../../scene.h"

/*
    Model switches
    -DYNAMIC: Instances update every render (frame)
    -CONST_INSTANCES: VBO set at the beginning and stays forever so dont need to call glBufferSubdata all the time
    -NO_TEX: Bool on constructor if model is texture or material

*/

#define DYNAMIC			    (unsigned int)1     // 0b00000001       
#define CONST_INSTANCES	    (unsigned int)2     // 0b00000010
#define NO_TEX			    (unsigned int)4	    // 0b00000100       

/*
    Forward declaration so wont have circular dependency
*/
class CollisionModel;
class Scene;                                     

class Model {
    public:
        std::string id;                                 // ID of model in scene

        
        std::vector<Mesh> meshes;
        CollisionModel* collision;                      // Pointer to Collision model
        std::vector<BoundingRegion> boundingRegions;    // List of bounding regions (1 for each mesh)
        std::vector<RigidBody*> instances;              // For forces applied (collisions and such)

        int maxNInstances;                              // Max indices allowed
        int currentNInstances;

        unsigned int switches;                          // Combination of switches above

        /*
            Constructors
        */
        Model(std::string id, unsigned int maxNInstances, unsigned int flags = 0);

        /*
            Process functions

            These virtual functions are necessary due to Model generalization.
            Each model has its own way of loading and rendering (e.g. Sphere, Troll, etc.).
            So it will look for an overridden version of these methods.
        */
        virtual void init();
        void loadModel(std::string path);
        void enableCollisionModel();                    // Enable collision model
        void addMesh(Mesh* mesh);                        // Add mesh to the list
        virtual void render(
            Shader shader, 
            float dt, 
            Scene *scene
        );
        void cleanup();                                 // Free memory

        /*
            Instance methods
        */
        RigidBody* generateInstance(glm::vec3 size, float mass, glm::vec3 pos, glm::vec3 eRotation);

        void initInstances();                           // Initialize memory for instances
        void removeInstance(unsigned int idx);          // Remove instance at idx
        void removeInstance(std::string instanceId);    // Remove instance with id
        unsigned int getIdx(std::string id);            // Get index of instance with id
        
    protected:
        bool noTextures;
        std::string directory;
        // Textures already loaded, optimization so they arent loaded more than once.
        std::vector<Texture> textures_loaded;           

        /*
            Model loading functions (ASSIMP)
            
            Obs: _ai_ related to assimp lib stands for (a)ss(i)mp 

        */
        void processNode(aiNode *node, const aiScene *scene);   // Process node in object file
        Mesh processMesh(aiMesh *mesh, const aiScene *scene);   // Process mesh in object file
        Mesh processMesh(                                       // Process custom mesh
            BoundingRegion br,
            unsigned int nVertices, float *vertices,
            unsigned int nIndices, unsigned int *indices,
            bool calcTanVectors = true,
            unsigned int nCollisionPoints = 0, float *collisionPoints = NULL,
            unsigned int nCollisionFaces = 0, unsigned int *collisionIndices = NULL,
            bool pad = false
        );
        std::vector<Texture> loadTextures(aiMaterial *mat, aiTextureType type);

        /*
           VBOs for model matrices
        */
        BufferObject modelVBO;
        BufferObject normalModelVBO;
        static const BoundTypes boundType = BoundTypes::SPHERE;
};

#endif
#ifndef COLLISIONMODEL_H
#define COLLISIONMODEL_H

#include <vector>

#include "collisionmesh.h"

/*
    Forward declaration
*/
class Model;

class CollisionModel {
    public:
        Model* model;
        std::vector<CollisionMesh> meshes;

        CollisionModel(Model* model); 

};

#endif

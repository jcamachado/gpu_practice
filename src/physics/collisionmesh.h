#ifndef COLLISIONMESH_H
#define COLLISIONMESH_H

#include <vector>

#include "../algorithms/bounds.h"

/*
    Forward declarations
*/
class CollisionMesh;
class CollisionModel;
class RigidBody;

typedef struct Face {
    CollisionMesh* mesh;
    unsigned int i1, i2, i3;        // Indices for vertex list in the mesh

    glm::vec3 baseNormal;
    glm::vec3 norm;       // Transform normal, affected by rotation and scaling but not translation

    bool collidesWithFace(RigidBody* thisRB, Face& face, RigidBody* faceRB, glm::vec3& retNorm);
    bool collidesWithSphere(RigidBody* thisRB, BoundingRegion& br, glm::vec3& retNorm);
} Face;      

class CollisionMesh {
    public:
        CollisionModel* model;
        BoundingRegion br;

        std::vector<glm::vec3> points;
        std::vector<Face> faces;


        /*
            -nPoints: number of groups of three of coordinates, so the size of this float array nPoints*3
            -nFaces:  number of groups of three of coordinates which can be repeated. (indices I suppose, because...)
            -Putting in mesh terms, coordinates would go VBO and indices would go EBO


        */
        CollisionMesh(unsigned int nPoints, float* coordinates, unsigned int nFaces, unsigned int* indices);
};

#endif

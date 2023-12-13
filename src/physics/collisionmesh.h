#ifndef COLLISIONMESH_H
#define COLLISIONMESH_H

#include <vector>

#include "../algorithms/bounds.h"
#include "../algorithms/cmathematics/vec.h"             // Remember to distinguish between vec, glm::vec3 and std::vector

/*
    Forward declarations
*/
class CollisionModel;
class CollisionMesh;

typedef struct Face {
    CollisionMesh* mesh;
    unsigned int i1, i2, i3;        // Indices for vertex list in the mesh

    vec baseNormal;
    vec norm;       // Transform normal, affected by rotation and scaling but not translation

    bool collidesWith(struct Face& face);
} Face;      

class CollisionMesh {
    public:
        CollisionModel* model;
        BoundingRegion br;

        std::vector<vec> points;
        std::vector<Face> faces;


        /*
            -nPoints: number of groups of three of coordinates, so the size of this float array nPoints*3
            -nFaces:  number of groups of three of coordinates which can be repeated. (indices I suppose, because...)
            -Putting in mesh terms, coordinates would go VBO and indices would go EBO


        */
        CollisionMesh(unsigned int nPoints, float* coordinates, unsigned int nFaces, unsigned int* indices);
};

#endif

#include "collisionmesh.h"
#include "collisionmodel.h"

#include "../algorithms/cmathematics/matrix.h"

/*
    Line-plane intersection cases, checked sequentially:
    PI is the plane containing the vectors P1P2 and P1P3

    CASE 0
        line U1U2 lies in the plane PI
        t = 0 / 0

    CASE 1
        no planar intersection
        t = (R != 0) / 0    ; where R is a real number

        => intersection when                                ; tnum is t numerator, tden is t denominator
        !(tnum != 0 && tden == 0) =
        = tnum == 0 || tden != 0                            ; De Morgan's law
    
    CASE 2
        planar intersection, in between U1 and U2
        t = R / (R != 0) in the range [0, 1]

    CASE 3
        planar intersection, outside U1 and U2
        t = R / (R != 0) outside the range [0, 1]
*/
#define CASE0 (char)0
#define CASE1 (char)1
#define CASE2 (char)2
#define CASE3 (char)3

char linePlaneIntersection(vec P1, vec norm, vec U1, vec size, float& t){
    /*
        Calculate the parameter t of the line { U1 + side * t } at the point of intersection
        t = (N dot U1P1) / (N dot U1U2)..
    */
    vec U1P1 = vecSubtract(P1, U1);

    // Calculate the numerator and denominator of the t parameter
    float tnum = dot(norm, U1P1);    
    float tden = dot(norm, size);

    if (tden == 0.0f) {
        return tnum == 0.0f ? CASE0 : CASE1;

    }
    else {
        // Can divide
        t = tnum / tden;
        return t >= 0.0f && t <= 1.0f ? CASE2 : CASE3;
    }
}

bool Face::collidesWith(Face& face){
    // Transform coordinates so that the P1 is the origin
    vec P1 = this->mesh->points[this->i1];
    vec P2 = vecSubtract(this->mesh->points[this->i2], P1);
    vec P3 = vecSubtract(this->mesh->points[this->i3], P1);
    vec lines[3] = {
        P2,                     // P2 - P1 = A  
        P3,                     // P3 - P1 = B
        vecSubtract(P3, P2)     // (P3 - P1) - (P2 - P1) = P3 - P2 = C
    };

    vec U1 = vecSubtract(face.mesh->points[face.i1], P1);
    vec U2 = vecSubtract(face.mesh->points[face.i2], P1);
    vec U3 = vecSubtract(face.mesh->points[face.i3], P1);
    vec sideOrigins[3] = {
        U1,
        U1,
        U2
    };
    // This may change when integrating with the octree, and can be improved
    vec sides[3] = {
        vecSubtract(U2, U1),
        vecSubtract(U3, U1),
        vecSubtract(U3, U2)
    };
    
    // Set P1 as the origin (zero vector)
    P1.elements[0] = 0.0f; P1.elements[1] = 0.0f; P1.elements[2] = 0.0f;

    // Placeholders
    float c1, c2;
    
    /*
        Iterate through each bounding line of the target plane (face plane)
        Check intersections
    */
    for (int i = 0; i < 3; i++){
        // Get intersection with this plane
        float t = 0.0f;
        char currentCase = linePlaneIntersection(P1, this->norm, sideOrigins[i], sides[i], t);
        switch (currentCase) {
            case CASE0: {
                // line in the plane
                // determine the intersection with the 3 bounding lines of this face
                for (int j = 0; j < 3; j++){    // Iterate through the lines of this plane 
                    mat m = newColMat(3, 3, 
                        lines[j], 
                        vecScalarMultiplication(sides[i], -1.0f), 
                        sideOrigins[i]
                    );
                    // Do reduction RREF (reduced row echelon form)
                    rref(&m);
                    if (m.elements[2][2] != 0.0f) {
                        // No intersection with the lines
                        continue;
                    }

                    c1 = m.elements[0][2];
                    c2 = m.elements[1][2];

                    if (0.0f <= c1 && c1 <= 1.0f && 
                        0.0f <= c2 && c2 <= 1.0f) {
                        // Intersection with the lines
                        return true;
                    }
                }
                continue;
            }

            case CASE1:
                // No intersection with the plane -> no collision
                continue;
            
            case CASE2: {
                // Intersection with the plane, in range. 
                // Determine if inside this triangle (Bounded by P1, P2 and P3)

                // Get the intersection point
                vec intersection = vecAdd (
                                    sideOrigins[i], 
                                    vecScalarMultiplication(sides[i], t)
                                );
                printf("%f: ", t); printVec(intersection);
                // Represent the intersection point as a linear combination of P2 and P3
                // With float point precision erros, we add normal vector to be sure
                mat m = newColMat(3, 4, 
                    P2, 
                    P3, 
                    this->norm,
                    intersection
                );

                // RREF (Its broken for now)
                // rref(&m);

                // Obtain the coefficients of the linear combination
                // c3 ~= 0.0 because point is in the plane
                c1 = m.elements[0][2];
                c2 = m.elements[1][2];
                if (c1 >= 0.0f && c2 >= 0.0f &&
                    c1 + c2 <= 1.0f) 
                {
                    
                    // Intersection with the triangle setup by A, B and C
                    return true;
                }


                continue;
            }
            
            case CASE3:
                // Intersection with the plane, outside range. 
                // No collision
                continue;

        }
    }
    return false;
}

CollisionMesh::CollisionMesh(
    unsigned int nPoints, 
    float* coordinates, 
    unsigned int nFaces, 
    unsigned int* indices)
    : points(nPoints), faces(nFaces) {
    // Insert points into list
    for (unsigned int i = 0; i < nPoints; i++) {
        points[i] = newVector(                  // Like in Vertex::genlist
            3, 
            coordinates[i * 3 + 0],
            coordinates[i * 3 + 1],
            coordinates[i * 3 + 2]
        );   
    }

    // Calculate face normals
    for (unsigned int i = 0; i < nFaces; i++){
        unsigned int i1 = indices[i * 3 + 0];
        unsigned int i2 = indices[i * 3 + 1];
        unsigned int i3 = indices[i * 3 + 2];

        /*
            Collison Theory checkpoint:
            Given the 3 points point[0], point[1] and point[2] (P1, P2 and P3)): 
            A vector is resulted from  point[1] - point[0], B = point[2] - point[0], C = point[2] - point[1]
            When we have the indices, A, B and C from the points list, and through cross(A, B) we get the normal
        */
        vec A = vecSubtract(points[i2], points[i1]);      // A = point[1] - point[0]
        vec B = vecSubtract(points[i3], points[i1]);      // B = point[2] - point[0] 
        vec N = cross(A, B);                              // N = A x B

        faces[i] = {
            this,
            i1, i2, i3, // Indices making up the triangle
            N,          // Normal placeholder
            N           // Initial value for transformmed normal is the same as the base normal
        };  
    }
}

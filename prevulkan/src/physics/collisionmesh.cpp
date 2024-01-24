/*
    Because of forward declaration, including in the source file is desired
    (My understanding)
    since this file is included in the header file
    Like, compiles the header first, know that files exist, then compiles the source file
*/
#include "collisionmesh.h"
#include "collisionmodel.h"
#include "rigidbody.h"  
#include <iostream>
#include <limits>

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


char linePlaneIntersection(glm::vec3 P1, glm::vec3 norm, glm::vec3 U1, glm::vec3 side, float& t){
    /*
        Calculate the parameter t of the line { U1 + side * t } at the point of intersection
        t = (N dot U1P1) / (N dot U1U2)..
    */
    glm::vec3 U1P1 = P1 - U1;

    // Calculate the numerator and denominator of the t parameter
    float tnum = glm::dot(norm, U1P1);    
    float tden = glm::dot(norm, side);

    if (tden == 0.0f) {
        return tnum == 0.0f ? CASE0 : CASE1;

    }
    else {
        // Can divide
        t = tnum / tden;
        return t >= 0.0f && t <= 1.0f ? CASE2 : CASE3;
    }
}

template <int C, int R>
void rref(glm::mat<C, R, float>& m) {
	unsigned int currentRow = 0;
	for (unsigned int c = 0; c < C; c++) {
		unsigned int r = currentRow;
		if (r >= R)
		{
			// no more rows
			break;
		}

		// find nonzero entry
		for (; r < R; r++)
		{
			if (m[c][r] != 0.0f)
			{
				// non-zero value
				break;
			}
		}

		// didn't find a nonzero entry in column
		if (r == R)
		{
			continue;
		}

		// swap with proper row
		if (r != currentRow) {
			for (unsigned int i = 0; i < C; i++) {
				float tmp = m[i][currentRow];
				m[i][currentRow] = m[i][r];
				m[i][r] = tmp;
			}
		}

		// multiply row by 1 / value
		if (m[c][currentRow] != 0.0f) {
			float k = 1 / m[c][currentRow];
			m[c][currentRow] = 1.0f;
			for (unsigned int col = c + 1; col < C; col++)
			{
				m[col][currentRow] *= k;
			}
		}

		// clear out rows above and below
		for (r = 0; r < R; r++)
		{
			if (r == currentRow)
			{
				continue;
			}
			float k = -m[c][r];
			for (unsigned int i = 0; i < C; i++) {

				m[i][r] += k * m[i][currentRow];
			}
		}

		currentRow++;
	}
}

glm::vec3 mat4vec3mult(glm::mat4& m, glm::vec3& v) {
	glm::vec3 ret;
	for (int i = 0; i < 3; i++) {
		ret[i] = v[0] * m[0][i] + v[1] * m[1][i] + v[2] * m[2][i] + m[3][i];
	}
	return ret;
}

glm::vec3 linCombSolution(glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 point) {
	// represent the point as a linear combination of the 3 basis vectors
	glm::mat4x3 m(A, B, C, point);

	// do RREF
	rref(m);

	return m[3];
}

bool faceContainsPointRange(
    glm::vec3 A, glm::vec3 B, glm::vec3 N, 
    glm::vec3 point, 
    float radius
) {
	glm::vec3 c = linCombSolution(A, B, N, point);

	return c[0] >= -radius && c[1] >= -radius && c[0] + c[1] <= 1.0f + radius;
}

bool faceContainsPoint(
    glm::vec3 A, glm::vec3 B, glm::vec3 N, 
    glm::vec3 point
) {
	return faceContainsPointRange(A, B, N, point, 0.0f);
}

bool Face::collidesWithFace(
    RigidBody* thisRB, 
    Face& face, 
    RigidBody* faceRB, 
    glm::vec3& retNorm
){  
    if(thisRB != nullptr || faceRB != nullptr){
        return false;
    }

    // retNorm is the noram to return to be used in handleCollision
    // Transform coordinates so that the P1 is the origin
    glm::vec3 P1 = mat4vec3mult(thisRB->model, this->mesh->points[this->i1]);
	glm::vec3 P2 = mat4vec3mult(thisRB->model, this->mesh->points[this->i2]) - P1;
	glm::vec3 P3 = mat4vec3mult(thisRB->model, this->mesh->points[this->i3]) - P1;
	glm::vec3 lines[3] = {
		P2,
		P3,
		P3 - P2
	};

    // Model matrix transformations and normal cuts off translation since its only directions
    glm::vec3 thisNorm = glm::normalize(thisRB->normalModel * this->norm);
    // glm::vec3 thisNorm = thisRB->normalModel * this->norm;
    
    glm::vec3 U1 = mat4vec3mult(faceRB->model, face.mesh->points[face.i1]) - P1;
	glm::vec3 U2 = mat4vec3mult(faceRB->model, face.mesh->points[face.i2]) - P1;
	glm::vec3 U3 = mat4vec3mult(faceRB->model, face.mesh->points[face.i3]) - P1;

    retNorm = glm::normalize(faceRB->normalModel *  face.norm);
    // retNorm = faceRB->normalModel *  face.norm;

    // Set P1 as the origin (zero vector)
    P1[0] = 0.0f; P1[1] = 0.0f; P1[2] = 0.0f;

    // Placeholders
    float c1, c2, c3;

    glm::vec3 sideOrigins[3] = {
		U1,
		U1,
		U2
	};
	glm::vec3 sides[3] = {
		U2 - U1,
		U3 - U1,
		U3 - U2
	};
    
    /*
        Iterate through each bounding line of the target plane (face plane)
        Check intersections
    */
    for (int i = 0; i < 3; i++){
        // Get intersection with this plane
        float t = 0.0f;
        char currentCase = linePlaneIntersection(P1, thisNorm, sideOrigins[i], sides[i], t);
        switch (currentCase) {
            case CASE0: {
                // Check intersection of the 3 bounding lines of this face (line in the plane )
                for (int j = 0; j < 3; j++){    // Iterate through the lines of this plane 
                    glm::mat3 m(lines[j], -1.0f * sides[i], sideOrigins[i]);
                    
                    // Do reduction RREF (reduced row echelon form)
                    
                    rref(m);
                    if (m[2][2] != 0.0f) {
					    // no intersection
					    continue;
				    }

                    c1 = m[2][0];
				    c2 = m[2][1];

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
                glm::vec3 intersection = sideOrigins[i] + t * sides[i];
                
                if (faceContainsPoint(P2, P3, this->norm, intersection)) {
                    // Intersection with the plane, in range. 
                    // Determine if inside this triangle (Bounded by P1, P2 and P3)
                    return true;
                }
                else {
                    // Intersection with the plane, in range. 
                    // No collision
                    continue;
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

/*
    Gotta check if this works with particles as spheres
*/
bool Face::collidesWithSphere(RigidBody* thisRB, BoundingRegion& br, glm::vec3& retNorm) {
	// if (br.type != BoundTypes::SPHERE) {
	// 	return false;
	// }

	// apply model transformations
    if (thisRB == nullptr || br.isEmptyInstance) {
        return false;
    }
	glm::vec3 P1 = mat4vec3mult(thisRB->model, this->mesh->points[i1]);
	glm::vec3 P2 = mat4vec3mult(thisRB->model, this->mesh->points[i2]);
	glm::vec3 P3 = mat4vec3mult(thisRB->model, this->mesh->points[i3]);

    // Transform the normal
	glm::vec3 norm = thisRB->normalModel * this->norm;
	glm::vec3 unitN = norm / glm::length(norm); // Always = 1

    // Vector from P1 to the center of the sphere
	glm::vec3 distanceVec = br.center - P1;     
	float distance = glm::dot(distanceVec, unitN);

	if (abs(distance) < br.radius) {
		glm::vec3 circCenter = br.center + distance * unitN;
        retNorm = unitN;

		return faceContainsPointRange(P2 - P1, P3 - P1, norm, circCenter - P1, br.radius);
	}

	return false;
}

CollisionMesh::CollisionMesh(
    unsigned int nPoints, 
    float* coordinates, 
    unsigned int nFaces, 
    unsigned int* indices,
    BoundTypes type)
    : points(nPoints), faces(nFaces)
{
    /*
        THIS CONSTRUCTOR MAKES THE FORMAT OF THE VOLUME THAT
        WILL BE USER FOR THE COLLISION INTERACTIONS, NOT THE
        VISUALS
    */
    



    if (type == BoundTypes::SPHERE) {
        calcSphereCollMeshValues(this, nPoints, coordinates, nFaces, indices);
    }
    else if (type == BoundTypes::AABB) {
        calcAABBCollMeshValues(this, nPoints, coordinates, nFaces, indices);
    }
    
    // calcSphereCollMeshValues(this, nPoints, coordinates, nFaces, indices);
    // calcAABBCollMeshValues(this, nPoints, coordinates, nFaces, indices);

}

void CollisionMesh::calcSphereCollMeshValues(
    CollisionMesh *colMesh, unsigned int nPoints, 
    float* coordinates, 
    unsigned int nFaces, 
    unsigned int* indices)
{
    // Calculate the center of the sphere
    colMesh->points.resize(nPoints);
    colMesh->faces.resize(nFaces);
    glm::vec3 min(std::numeric_limits<float>::infinity());  // +infinity
    glm::vec3 max = -1.0f * min;                                   // -infinity
    // Insert points into list
    for (unsigned int i = 0; i < nPoints; i++) {
        colMesh->points[i] = {                  // Like in Vertex::genlist
            coordinates[i * 3 + 0],
            coordinates[i * 3 + 1],
            coordinates[i * 3 + 2]
        };

        for (int j = 0; j < 3; j++) {
            if (colMesh->points[i][j] < min[j]) {
                min[j] = colMesh->points[i][j];
            }

            if (colMesh->points[i][j] > max[j]) {
                max[j] = colMesh->points[i][j];
            }
        }
    }

    glm::vec3 center = (min + max) / 2.0f;
    float maxRadiusSquared = 0.0f;
    for (unsigned int i = 0; i < nPoints; i++) {
        float radiusSquared = 0.0f;
        for (int j = 0; j < 3; j++) {
            float dist = colMesh->points[i][j] - center[j];
            radiusSquared += dist * dist;
        }
        if (radiusSquared > maxRadiusSquared) {
            maxRadiusSquared = radiusSquared;
        }
    }

    colMesh->br = BoundingRegion(
        center, 
        sqrt(maxRadiusSquared)* DEFAULT_COLLMESH_SCALE_FACTOR
    );  // Bounding regions are still boxes
    colMesh->br.collisionMesh = colMesh;

    // Calculate face normals
    // TODO Needs to be revised about shared vertices OR ANYTHING ELSE
    // BROKEN, FACES/3 IS JUST TO NOT GET OUT OF BOUNDS
    for (unsigned int i = 0; i < nFaces/3; i++){ 
        unsigned int i1 = indices[i * 3 + 0];
        unsigned int i2 = indices[i * 3 + 1];
        unsigned int i3 = indices[i * 3 + 2];
        
        /*
            Collison Theory checkpoint:
            Given the 3 points point[0], point[1] and point[2] (P1, P2 and P3)): 
            A vector is resulted from  point[1] - point[0], B = point[2] - point[0], C = point[2] - point[1]
            When we have the indices, A, B and C from the points list, and through cross(A, B) we get the normal
        */
        glm::vec3 A = colMesh->points[i2] - colMesh->points[i1];  // A = P2 - P1
		glm::vec3 B = colMesh->points[i3] - colMesh->points[i1];  // B = P3 - P1 
        glm::vec3 N = glm::cross(A, B);         // N = A x B
        N = glm::normalize(N);

        colMesh->faces[i] = {
            colMesh,
            i1, i2, i3, // Indices making up the triangle
            N,          // Normal placeholder
            N           // Initial value for transformmed normal is the same as the base normal
        };  
    }
}
void CollisionMesh::calcAABBCollMeshValues(
    CollisionMesh *colMesh, unsigned int nPoints, 
    float* coordinates, 
    unsigned int nFaces, 
    unsigned int* indices)
{
    glm::vec3 min(std::numeric_limits<float>::infinity());  // +infinity
    glm::vec3 max = -1.0f * min;                                   // -infinity
    glm::vec3 center;
    glm::vec3 size;
    for (unsigned int i = 0; i < nPoints; i++) {
        colMesh->points[i] = {
            coordinates[i * 3 + 0],
            coordinates[i * 3 + 1],
            coordinates[i * 3 + 2]
        };

        for (int j = 0; j < 3; j++) {
            if (colMesh->points[i][j] < min[j]) {
                min[j] = colMesh->points[i][j];
            }

            if (colMesh->points[i][j] > max[j]) {
                max[j] = colMesh->points[i][j];
            }
        }
        center = (min + max) / 2.0f;
        size = max - min;
        min = center - size * DEFAULT_COLLMESH_SCALE_FACTOR / 2.0f;  // Scale the min point
        max = center + size * DEFAULT_COLLMESH_SCALE_FACTOR / 2.0f;
        colMesh->br = BoundingRegion(min, max);
    }

    // Calculate face normals
    for (unsigned int i = 0; i < nFaces; i++){
        unsigned int i1 = indices[i * 3 + 0];
        unsigned int i2 = indices[i * 3 + 1];
        unsigned int i3 = indices[i * 3 + 2];

        glm::vec3 A = colMesh->points[i2] - colMesh->points[i1];
        glm::vec3 B = colMesh->points[i3] - colMesh->points[i1];
        glm::vec3 N = glm::cross(A, B);
        N = glm::normalize(N);

        colMesh->faces[i] = {
            colMesh,
            i1, i2, i3,
            N,
            N
        };  
    }
}

// CollisionMesh::CollisionMesh(
//     unsigned int nPoints, 
//     float* coordinates, 
//     unsigned int nFaces, 
//     unsigned int* indices)
//     : points(nPoints), faces(nFaces) 
// {
    
//     glm::vec3 min(std::numeric_limits<float>::infinity());  // +infinity
//     glm::vec3 max = -1.0f * min;                                   // -infinity

//     // Insert points into list
//     for (unsigned int i = 0; i < nPoints; i++) {
//         points[i] = {                  // Like in Vertex::genlist
//             coordinates[i * 3 + 0],
//             coordinates[i * 3 + 1],
//             coordinates[i * 3 + 2]
//         };

//         for (int j = 0; j < 3; j++) {
//             if (points[i][j] < min[j]) {
//                 min[j] = points[i][j];
//             }

//             if (points[i][j] > max[j]) {
//                 max[j] = points[i][j];
//             }
//         }
//     }

//     glm::vec3 center = (min + max) / 2.0f;
//     float maxRadiusSquared = 0.0f;
//     for (unsigned int i = 0; i < nPoints; i++) {
//         float radiusSquared = 0.0f;
//         for (int j = 0; j < 3; j++) {
//             float dist = points[i][j] - center[j];
//             radiusSquared += dist * dist;
//         }
//         if (radiusSquared > maxRadiusSquared) {
//             maxRadiusSquared = radiusSquared;
//         }
//     }

//     this->br = BoundingRegion(center, sqrt(maxRadiusSquared));  // Bounding regions are still boxes
//     this->br.collisionMesh = this;

//     // Calculate face normals
//     for (unsigned int i = 0; i < nFaces; i++){
//         unsigned int i1 = indices[i * 3 + 0];
//         unsigned int i2 = indices[i * 3 + 1];
//         unsigned int i3 = indices[i * 3 + 2];

//         /*
//             Collison Theory checkpoint:
//             Given the 3 points point[0], point[1] and point[2] (P1, P2 and P3)): 
//             A vector is resulted from  point[1] - point[0], B = point[2] - point[0], C = point[2] - point[1]
//             When we have the indices, A, B and C from the points list, and through cross(A, B) we get the normal
//         */
//         glm::vec3 A = points[i2] - points[i1];  // A = P2 - P1
// 		glm::vec3 B = points[i3] - points[i1];  // B = P3 - P1 
//         glm::vec3 N = glm::cross(A, B);         // N = A x B
//         N = glm::normalize(N);

//         faces[i] = {
//             this,
//             i1, i2, i3, // Indices making up the triangle
//             N,          // Normal placeholder
//             N           // Initial value for transformmed normal is the same as the base normal
//         };  
//     }
// }

#ifndef BOUNDS_H
#define BOUNDS_H

#include <glm/glm.hpp> 

#include "../physics/rigidbody.h"

/*
    Forward declaration
*/
class RigidBody;
class CollisionMesh;

namespace Octree {
    class node;
}

enum class BoundTypes : unsigned char {
    //unsigned char to make it less memory intensive
    AABB = 0x00, // 0x00 = 0 Axis Aligned Bounding Box
    SPHERE = 0x01 // 0x01 = 1 
};

// Bounding box per mesh
class BoundingRegion {
    public:
        BoundTypes type;
        bool isEmptyInstance = false;

        /*
            Octree region values

            Pointers for quick access to instance and collision mesh
        */
        RigidBody* instance;
        CollisionMesh* collisionMesh;

        /*
            Octree node values
        */  
        Octree::node* cell; // Cell is a node, therefore, an octant

        /*
            Sphere values (Even though AABB will also have center and radius. (kinda))
            
            -ogCenter is the center of the region in the original octree
            -ogRadius is the radius of the region in the original octree
        */
        glm::vec3 center = glm::vec3(0.0f);
        float radius = 0.0f;
        glm::vec3 ogCenter = glm::vec3(0.0f);
        float ogRadius = 0.0f;


        /*
            AABB values
            Uses min max and not general coordinates to optimize and simplify calculations
            -ogMin is the minimum of the region in the original octree
            -ogMax is the maximum of the region in the original octree
        */
        glm::vec3 min = glm::vec3(0.0f);
        glm::vec3 max = glm::vec3(0.0f);
        glm::vec3 ogMin = glm::vec3(0.0f);
        glm::vec3 ogMax = glm::vec3(0.0f);

        /*
            Constructors
        */
        BoundingRegion(BoundTypes type = BoundTypes::AABB); // Default

        BoundingRegion(glm::vec3 center, float radius); //initialize as sphere
        
        BoundingRegion(glm::vec3 min, glm::vec3 max);   //initialize as aabb

        /*
            Calculating values for the region
        */
        void transform();                       //transform for instance

        glm::vec3 calculateCenter();
        glm::vec3 calculateDimensions();        //vectors that go from min and max of the box

        /*
            Testing methods
            Will be used for octree
        */
        bool containsPoint(glm::vec3 point);    //test if point is inside

        bool containsRegion(BoundingRegion br); //test if region is completely inside another region
        
        bool intersectsWith(BoundingRegion br); //test if region intersects with another region (partially contains)

        bool operator==(BoundingRegion br);     // Operator overload
};

#endif
#ifndef BOUNDS_H
#define BOUNDS_H

#include <glm/glm.hpp> 

#include "../physics/rigidbody.h"

enum class BoundTypes : unsigned char {
    //unsigned char to make it less memory intensive
    AABB = 0x00, // 0x00 = 0 Axis Aligned Bounding Box
    SPHERE = 0x01 // 0x01 = 1 
};

// Bounding box per mesh
class BoundingRegion {
    public:
        BoundTypes type;

        /*
            Octree region values

        */
        RigidBody* instance; //instance of the rigidbody that is inside the region

        /*
            Sphere values (Even though AABB will also have center and radius. (kinda))
            
            -ogCenter is the center of the region in the original octree
            -ogRadius is the radius of the region in the original octree
        */
        glm::vec3 center;
        float radius;
        glm::vec3 ogCenter;
        float ogRadius;


        /*
            AABB values
            Uses min max and not general coordinates to optimize and simplify calculations
            -ogMin is the minimum of the region in the original octree
            -ogMax is the maximum of the region in the original octree
        */
        glm::vec3 min;
        glm::vec3 max;
        glm::vec3 ogMin;
        glm::vec3 ogMax;

        /*
            Constructors
        */
        BoundingRegion(BoundTypes type = BoundTypes::AABB);

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
#ifndef BOUNDS_H
#define BOUNDS_H

#include <glm/glm.hpp> 

enum class BoundTypes : unsigned char {
    //unsigned char to make it less memory intensive
    AABB = 0x00, // 0x00 = 0 Axis Aligned Bounding Box
    SPHERE = 0x01 // 0x01 = 1 
};

// Bounding box per mesh
class BoundingRegion {
    public:
        BoundTypes type;

        //sphere values
        glm::vec3 center;
        float radius;

        //aabb values
        //Uses min max and not general coordinates to optimize and simplify calculations
        glm::vec3 min;
        glm::vec3 max;

        /*
            Constructors
        */

        //initialize with type
        BoundingRegion(BoundTypes type = BoundTypes::AABB);

        //initialize as sphere
        BoundingRegion(glm::vec3 center, float radius);

        //initialize as aabb
        BoundingRegion(glm::vec3 min, glm::vec3 max);

        /*
            Calculating values for the region
        */

        // center
        glm::vec3 calculateCenter();

        // calculate dimensions
        glm::vec3 calculateDimensions(); //vectors that go to min and max of the box

        /*
            Testing methods
        */

        //test if point is inside
        bool containsPoint(glm::vec3 point);   

        //test if region is completely inside of a region (necessario for octree)
        bool containsRegion(BoundingRegion br);
        
        //test if region intersects with another region
        bool intersectsWith(BoundingRegion br);

};

#endif
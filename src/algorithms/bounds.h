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
        BoundingRegion(BoundTypes type = BoundTypes::AABB);

        //initialize as sphere
        BoundingRegion(glm::vec3 center, float radius);

        //initialize as aabb
        BoundingRegion(glm::vec3 min, glm::vec3 max);

        /*
            Calculating values for the region
        */
        glm::vec3 calculateCenter();

        glm::vec3 calculateDimensions();        //vectors that go from min and max of the box

        /*
            Testing methods
        */
        bool containsPoint(glm::vec3 point);    //test if point is inside

        bool containsRegion(BoundingRegion br); //test if region is completely inside of region (necessario for octree)
        
        bool intersectsWith(BoundingRegion br); //test if region intersects with another region

        bool operator==(BoundingRegion br);     // Operator overload
};

#endif
#include "bounds.h"
#include <algorithm> // std::max, std::min compilation issue3

BoundingRegion::BoundingRegion(BoundTypes type)
    : type(type){}

BoundingRegion::BoundingRegion(glm::vec3 center, float radius)
    :type(BoundTypes::SPHERE), center(center), radius(radius) {}

BoundingRegion::BoundingRegion(glm::vec3 min, glm::vec3 max)
    : type(BoundTypes::AABB), min(min), max(max) {}

glm::vec3 BoundingRegion::calculateCenter() {
    return type == BoundTypes::AABB ? (min + max) / 2.0f : center;
}

glm::vec3 BoundingRegion::calculateDimensions() {
    return (type == BoundTypes::AABB) ? (max - min) : glm::vec3(2.0f * radius);
}

bool BoundingRegion::containsPoint(glm::vec3 point) {
    if (type == BoundTypes::AABB) {
        return (point.x >= min.x && point.x <= max.x) &&
               (point.y >= min.y && point.y <= max.y) &&
               (point.z >= min.z && point.z <= max.z);
    }
    else if (type == BoundTypes::SPHERE) {
        //sphere: distance must be less than radius
        // x^2 + y^2 + z^2 = r^2
        float distSquared = 0.0f;
        for (int i=0; i<3; i++) {
            distSquared += (point[i] - center[i]) * (point[i] - center[i]);
        }
        return distSquared < (radius * radius);
    }
}

// test if region is completely inside of another region
bool BoundingRegion::containsRegion(BoundingRegion br) {
    if (br.type == BoundTypes::AABB) {
    // if br is a box, just has to contain min and max, 
    // there is nothing below the minimum and nothing above the maximum
        return containsPoint(br.min) && containsPoint(br.max);
    }
    else if (type == BoundTypes::SPHERE && br.type == BoundTypes::SPHERE) {
    // if both are spheres, combination of centers and br.radius is less than (this) radius
        return glm::length(center - br.center) + br.radius < radius;
    }
    else{
    // if this is a box and br is a sphere, check if the sphere is inside the box
        if(!containsPoint(br.center)) {
            return false;
        }

        // center is inside the box
        /*
            for each axis (x, y, z)
                if the center to each side is smaller than the radius, return false
        */
        for (int i = 0; i < 3; i++) {
            if (abs(br.center[i] - min[i]) < br.radius || 
            abs(max[i] - br.center[i]) < br.radius) {
                return false;
            }
        }
    }
}

// Test if region intersects (partially contains)
bool BoundingRegion::intersectsWith(BoundingRegion br) {
    //overlap on all 3 axes
    
    if(type==BoundTypes::AABB && br.type==BoundTypes::AABB) {
        // if both are boxes
        glm::vec3 radiusThis = calculateDimensions() / 2.0f;    // "radius" of this box
        glm::vec3 radiusBr = br.calculateDimensions() / 2.0f;   // "radius" of br box

        glm::vec3 center = calculateCenter();                   // center of this box
        glm::vec3 centerBr = br.calculateCenter();              // center of br box

        glm::vec3 dist = glm::abs(center - centerBr);           // distance between centers

        // if the distance between centers is greater than the sum of the radii, return false
        for (int i = 0; i < 3; i++) {
            if (dist[i] > radiusThis[i] + radiusBr[i]) {
                return false;
            }
        }
        // failed to prove wrong on each axis
        return true;
    }
    else if (type == BoundTypes::SPHERE && br.type == BoundTypes::SPHERE){
    //both spheres - distance between centers is less than the sum of the radii
        return glm::length(center - br.center) < radius + br.radius;
    }
    else if (type == BoundTypes::SPHERE) {
        // this is a sphere, br is a box
        float distSquared = 0.0f;
        for (int i = 0; i < 3; i++) {
            // determine closest side (plane)
            float closestPt = std::max(br.min[i], std::min(center[i], br.max[i]));
            // add distance
            distSquared += (closestPt - center[i]) * (closestPt - center[i]);
        }
 
        return distSquared < (radius * radius);
    }
    else{
        // this is a box, br is a sphere
        // call algorith for br (defined in preceding else if block)
        return br.intersectsWith(*this);
     }
}
#ifndef OCTREE_HPP
#define OCTREE_HPP

#define N_CHILDREN 8
#define MIN_BOUNDS 0.5

#include <vector>
#include <queue>
#include <stack>

#include "list.hpp"
#include "states.hpp"
#include "bounds.h"

namespace Octree {
    enum class Octant : unsigned char{
        O1 = 0x01,  // = 0b00000001
        O2 = 0x02,  // = 0b00000010
        O3 = 0x04,  // = 0b00000100
        O4 = 0x08,  // = 0b00001000
        O5 = 0x10,  // = 0b00010000
        O6 = 0x20,  // = 0b00100000
        O7 = 0x40,  // = 0b01000000
        O8 = 0x80,  // = 0b10000000
    };

    /*
        Utility methods callbacks
    */

    // Calculate bounds of specified octant in bounding region
    void calculateBounds(BoundingRegion* out, Octant octant, BoundingRegion parentRegion);

    class node{
        public:
            node* parent;
            node* children[N_CHILDREN]; // 8 children, octree

            unsigned char activeOctants; // Bitmask for active octants

            bool hasChildren = false;

            bool treeReady = false;
            bool treeBuilt = false;

            std::vector<BoundingRegion> objects;
            std::queue<BoundingRegion> queue;   // Queue for objects to be added to tree

            BoundingRegion region;              // All will be of type AABB (Axis Aligned Bounding Box) for now

            node();

            node(BoundingRegion region);        // Constructor for root node

            // When iterating, objectlist is the list of objects that fit each of its octants
            node(BoundingRegion region, std::vector<BoundingRegion> objectList); 

            void build();

            void update();

            void processPending();

            bool insert(BoundingRegion object);

            void destroy();


    };
}

#endif
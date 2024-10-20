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

#include "../graphics/objects/model.h"


/*
    Forward declaration

*/
class Model;
class BoundingRegion;
class Box;

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
        // COMPLETE = 0xFF // = 0b11111111
    };

    /*
        Utility methods callbacks
    */

    // Calculate bounds of specified octant in bounding region
    void calculateBounds(BoundingRegion &out, Octant octant, BoundingRegion parentRegion);

    class node{
        public:
            node* parent = nullptr;
            // node* parent = 0;
            // node* parent;
            node* children[N_CHILDREN] = {0};  // 8 children, octree

            unsigned char activeOctants = 0x00; // Bitmask for active octants, mapping Octant enum to regions

            bool hasChildren = false;

            bool treeReady = false;
            bool treeBuilt = false;

            short maxLifespan = 8;              // Duration of empty node before it is destroyed
            short currentLifespan = -1;

            // initialize with empty vector
            std::vector<BoundingRegion> objects = {};
            std::queue<BoundingRegion> queue;   // Queue for objects to be added to tree

            BoundingRegion region;              // All will be of type AABB (Axis Aligned Bounding Box) for now

            node();

            node(BoundingRegion bounds);        // Constructor for root node

            // When iterating, objectlist is the list of objects that fit each of its octants
            node(BoundingRegion bounds, std::vector<BoundingRegion> objectList); 

            /*
                Add an instance to pending queue
                This function will take an instance and will add all of its models rigid bodies
                a new rigid body for the actual thing (words from the video)
            */
            void addToPending(RigidBody* instance, Model *model);

            void build();

            void update(Box &box);

            void processPending();

            bool insert(BoundingRegion obj);

            // Check collisions with all objects in the node
            void checkCollisionsSelf(BoundingRegion obj);
            // Check collisions with all objects in child node
            void checkCollisionsChildren(BoundingRegion obj);

            void destroy();
    };
}

#endif

/*
    Simplified algorithm for octree
    build tree          // Insert all objects into the tree    
    (some nodes may be pending by addToPending()
    process pending     // Before new frame, add queued objects to tree (if tree not built, add to objList)
    update tree         // Based on the movements, creation and death of objects and nodes

*/
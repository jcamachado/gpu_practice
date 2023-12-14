#include "octree.h"
#include "avl.h"
#include "../graphics/models/box.hpp"

// For each octant, a new bounding region is calculated and stored in out
// Out is a pointer to node region that will have its subregions calculated
void Octree::calculateBounds(BoundingRegion &out, Octant octant, BoundingRegion parentRegion){
    glm::vec3 center = parentRegion.calculateCenter();
    if (octant == Octant::O1) {
        out = BoundingRegion(center, parentRegion.max);
    }
    else if (octant == Octant::O2) {
        out = BoundingRegion(glm::vec3(parentRegion.min.x, center.y, center.z), glm::vec3(center.x, parentRegion.max.y, parentRegion.max.z));
    }
    else if (octant == Octant::O3) {
        out = BoundingRegion(glm::vec3(parentRegion.min.x, parentRegion.min.y, center.z), glm::vec3(center.x, center.y, parentRegion.max.z));
    }
    else if (octant == Octant::O4) {
        out = BoundingRegion(glm::vec3(center.x, parentRegion.min.y, center.z), glm::vec3(parentRegion.max.x, center.y, parentRegion.max.z));
    }
    else if (octant == Octant::O5) {
        out = BoundingRegion(glm::vec3(center.x, center.y, parentRegion.min.z), glm::vec3(parentRegion.max.x, parentRegion.max.y, center.z));
    }
    else if (octant == Octant::O6) {
        out = BoundingRegion(glm::vec3(parentRegion.min.x, center.y, parentRegion.min.z), glm::vec3(center.x, parentRegion.max.y, center.z));
    }
    else if (octant == Octant::O7) {
        out = BoundingRegion(parentRegion.min, center);
    }
    else if (octant == Octant::O8) {
        out = BoundingRegion(glm::vec3(center.x, parentRegion.min.y, parentRegion.min.z), glm::vec3(parentRegion.max.x, center.y, center.z));
    }
}

Octree::node::node()
    : region(BoundTypes::AABB) {}

Octree::node::node(BoundingRegion bounds)
    : region(bounds) {}

Octree::node::node(BoundingRegion bounds, std::vector<BoundingRegion> objectList)
    : region(bounds) {
        objects.insert(objects.end(), objectList.begin(), objectList.end());
}

void Octree::node::addToPending(RigidBody* instance, Model *model){
    // Get all the bounding regions of the models
    for (BoundingRegion br : model->boundingRegions){
        br.instance = instance;
        br.transform();
        queue.push(br);
    }
}

void Octree::node::build(){
    /*
        Variable declarations
        We have to declare some variables here because we are using goto statements
        Goto doesnt allow us to declare variables in the middle of the code block
    */
    glm::vec3 dimensions = region.calculateDimensions();
    BoundingRegion octants[N_CHILDREN];
    std::vector<BoundingRegion> octLists[N_CHILDREN]; // Array of list of objects in each octant

    /*
        Termination conditions
        - 1 or less objects (ie an empty leaf node)
        - dimensions are too small
    */
    // <= 1 objects

    if (objects.size() <= 1){
        goto setVars;
    }

    // Dimensions are too small
    for (int i = 0; i < 3; i++){
        if (dimensions[i] < MIN_BOUNDS){
            goto setVars;
        }
    }

    // Create regions
    for (int i = 0; i < N_CHILDREN; i++){
        calculateBounds(octants[i], (Octant)(1 << i), region);
    }

    // Determine which octants to place object in
    // If object doesnt fully belong to a region, it is added to the parent region
    // If objects overlap regions, they are in at least 1 region(parent), so they are not added to another region
    for (int i = 0, len = objects.size(); i < len; i++){
        BoundingRegion br = objects[i];

        for (int j = 0; j < N_CHILDREN; j++){
            if (octants[j].containsRegion(br)){
                octLists[j].push_back(br); // Found an octant to put it in, so it is not added to another octant
                // delList.push(i);        // Object has been placed on an octant, so it can be removed from the original list
                objects.erase(objects.begin() + i);
                i--;
                len--;
                break; 
            }

        }
    }

    // Populate octants
    for (int i = 0; i < N_CHILDREN; i++){
        if (octLists[i].size() != 0){
            children[i] = new node(octants[i], octLists[i]);
            States::activateIndex(&activeOctants, i);
            children[i]->parent = this;
            children[i]->build();
            hasChildren = true;
        }
    }

setVars:
    treeBuilt = true;
    treeReady = true;
    for (int i = 0; i < objects.size(); i++){
        objects[i].cell = this;
    }
}

void Octree::node::update(Box &box){    //build and update seems to be having segmenation faults
    if (treeBuilt && treeReady){
        // Countdown timer
        box.positions.push_back(region.calculateCenter());
        box.sizes.push_back(region.calculateDimensions());
        
        if(objects.size() == 0){
            if(!!hasChildren){
                if (currentLifespan==-1){
                    currentLifespan = maxLifespan;
                }
                else if (currentLifespan > 0){
                    currentLifespan--;
                }
            }
        }
        else{
            if (currentLifespan != -1){
                if(maxLifespan <= 64){
                    // Extend lifespan because "hotspot" 
                    maxLifespan <<= 2;
                }
            }
        }

        /*
            Remove objects that dont exist anymore

            012345678
            ABCDEFGHI          (list of objects),
            ABCDFGHI            if E dies (i=4), we want, everything will (has to) be left shifted
                                so we need to update _i_ and _listSize_
            
            since we do i++ with i--, we still get the next element F (i=4)
            i
        */
        for (int i = 0, listSize = objects.size(); i < listSize; i++){
            // Remove if on list of dead objects
            if (States::isActive(&objects[i].instance->state, INSTANCE_DEAD)){
                objects.erase(objects.begin() + i);
                // Update counter and size accordingly
                i--;
                listSize--;

            }
        }
        // Get moved objects that were in this leaf in previous frame
        std::stack<int> movedObjects;
        for (int i = 0, listSize = objects.size(); i < listSize; i++){
            if (States::isActive(&objects[i].instance->state, INSTANCE_MOVED)){
                objects[i].transform();
                movedObjects.push(i);
            }
            box.positions.push_back(objects[i].calculateCenter());
            box.sizes.push_back(objects[i].calculateDimensions());
        }

        /*
            Remove dead branches

            00110001        left shift until we get 00000000
        */
        unsigned char flags = activeOctants;
        for (int i = 0; 
            flags > 0; 
            flags >>= 1, i++) 
        {
            if (States::isIndexActive(&flags, 0)&&(children[i]->currentLifespan == 0)){
                // If this child is active and has no lifespan left
                if (children[i]->objects.size() > 0) {
                    // Branch is dead but has children, so reset lifespan
                    children[i]->currentLifespan = -1;
                }
                else {
                    // Branch is dead, remove it
                    children[i] = nullptr;
                    States::deactivateIndex(&activeOctants, i);
                }
            }
        }
        
        // Update child nodes
        if (children != nullptr){
            for (unsigned char flags = activeOctants, i = 0;
                flags > 0;
                flags >>= 1, i++){                      // Iterates over each bit in flags, each octant
                if (States::isIndexActive(&flags, 0)){
                    // Active octant
                    if (children[i] != nullptr){
                        children[i]->update(box);
                    }
                }
            }
        }
        // Move moved objects to new nodes
        BoundingRegion movedObj;
        while (movedObjects.size() != 0){
            /*
                For each moved object
                - Traverse up tree (start with current node) until find a node that completely encloses the object
                - Call insert (push object as far down as possible)
            */
            movedObj = objects[movedObjects.top()];       // Set to top object in stack
            node* current = this;                       // Placeholder

            while (!current->region.containsRegion(movedObj)){
                if (current->parent != nullptr){
                    current = current->parent;
                }
                else {
                    break;                  // If root, leave
                }
            }

            /*
                Once finished
                - Remove from objects list
                - remove from movedObjects stack
                - insert into found region
            */
            objects.erase(objects.begin() + movedObjects.top());
            movedObjects.pop();
            // current->insert(movedObj);
            current->insert(movedObj);

            // Collision detection  (We can use bruteforce for now, the number of objects per region is small)
            // Itself
            current=movedObj.cell;      // Current node might have changed node after previous code
            current->checkCollisionsSelf(movedObj);

            // Children
            current->checkCollisionsChildren(movedObj);

            // Parents
            while (current->parent){
            // while (current->parent){
                current = current->parent;
                current->checkCollisionsSelf(movedObj);
            }

        }
    }
    else {
        // Process pending results
        if (queue.size() > 0) {
            processPending();
        }
        // processPending();
    }
}

void Octree::node::processPending(){
    if(!treeBuilt){
        // Add objects to be sorted into branches when built
        while (queue.size() != 0){
            objects.push_back(queue.front());   // Place queued objects into objects list
            queue.pop();
        }
        build();                                // Place objects into its branches
    }
    else{
        // Insert objects immediately
        for (int i = 0, len = queue.size(); i<len ; i++){
            BoundingRegion br = queue.front();
            if (region.containsRegion(br)){
                insert(br);     // Insert object immediately
            }
            else{
                br.transform();
                queue.push(br); // Return to queue
            }
            queue.pop();
        }
    }
}

bool Octree::node::insert(BoundingRegion obj){
    /*
        Termination conditions
        - No objects (Empty leaf node)
        - Dimensions are less than MIN_BOUNDS

        In either case, we will push back the (new) region
        Because if there is no object, its gonna be the only object in the region and we
        dont want to divide it any further.
        And if the dimensions are too small, we dont want to divide it any further.

    */
    glm::vec3 dimensions = region.calculateDimensions();
    if (objects.size() == 0 || 
        dimensions.x < MIN_BOUNDS ||
        dimensions.y < MIN_BOUNDS ||
        dimensions.z < MIN_BOUNDS
    )
    {
        // Push back object
        obj.cell = this;
        objects.push_back(obj);
        return true;
    }

    // Safeguard if object doesnt fit in any octant, it is added to parent
    if (!region.containsRegion(obj)){
        return parent == nullptr ? false : parent->insert(obj);
    }

    // Create regions if not defined
    BoundingRegion octants[N_CHILDREN]; // Octants are the regions of the children
    for (int i = 0; i < N_CHILDREN; i++){
        if (children[i] != nullptr ){        // If child is not null, we want to use its region       
            octants[i] = children[i]->region;              
        }
        else{    
            calculateBounds(octants[i], (Octant)(1 << i), region); 
        }
    }

    objects.push_back(obj);

    // Determine which octants to put objects in
    std::vector<BoundingRegion> octLists[N_CHILDREN];    // array of list of objects in each octant
    for (int i = 0, len = objects.size(); i < len; i++) {
        objects[i].cell = this;
        for (int j = 0; j < N_CHILDREN; j++) {
            if (octants[j].containsRegion(objects[i])) {
                octLists[j].push_back(objects[i]);
                // Remove from objects list
                objects.erase(objects.begin() + i);
                i--;
                len--;
                break;
            }
        }
    }
    // Populate octants
    for (int i = 0; i < N_CHILDREN; i++) {
        if (octLists[i].size() != 0) {
            // Objects exist in this octant
            if (children[i]) {
             // Child octant exist
                for (BoundingRegion br : octLists[i]) {
                    children[i]->insert(br);
                }
            }
            else {  //broken on second iteration of recursion
                // Create new node, since it doesnt exist in this subregion
                children[i] = new node(octants[i], octLists[i]);
                children[i]->parent = this;
                States::activateIndex(&activeOctants, i);
                children[i]->build();
                hasChildren = true;
            }
        }
    }

    return true;
}

/*
    Collisions of objects in the node

    -Coarse check: Check if bounding regions intersect
    -Fine check: Check with actual mesh face (where they collided and how)

    4 cases of collision of objects A and B:
    - A and B dont have CollisionMesh
    - A has CollisionMesh, B doesnt
    - B has CollisionMesh, A doesnt
    - Both A and B have CollisionMesh
*/


void Octree::node::checkCollisionsSelf(BoundingRegion obj){ // CUDABLE?
    for (BoundingRegion br : objects){
        // Coarse check for bounding region intersection
        // Coarse  significa bruto, amarrotado, grosso, logo, uma verificação rápida
        if (br.instance == obj.instance) {
            continue; // Skip if same instance
        }

        if (br.intersectsWith(obj)){
            // Coarse check passed

            unsigned int nFacesBr = br.collisionMesh==nullptr ? br.collisionMesh->faces.size() : 0;
            unsigned int nFacesObj = obj.collisionMesh==nullptr ? obj.collisionMesh->faces.size() : 0;

            glm::vec3 norm;     // For handleCollision
            
            if(nFacesBr){
                if(obj.collisionMesh){      // Both have collision meshes
                    // Check all faces in br against all faces in obj. Quadratic hell O(n^2)
                    bool collisionFound = false;
                    for (unsigned int i = 0; i < nFacesBr && !collisionFound; i++){
                        for (unsigned int j = 0; j < nFacesObj && !collisionFound; j++){  //Cubic hell O(n^3)
                            if (br.collisionMesh->faces[i].collidesWithFace(
                                br.instance,
                                obj.collisionMesh->faces[j],
                                obj.instance,
                                norm
                            )){
                                std::cout << "Case 1: Instance " << br.instance->instanceId << 
                                "(" << br.instance->modelId << ") collided with instance " << obj.instance->instanceId << 
                                "(" << obj.instance->modelId << ")" << std::endl;
                                // collisionFound = true;
                                obj.instance->handleCollision(br.instance, norm);
                                break;
                            }
                        }
                    }
                }
                else {
                    // Br has collision mesh, obj doesnt
                    // Check all faces in br against objs sphere
                    for (unsigned int i = 0; i < nFacesBr; i++){
                        if (br.collisionMesh->faces[i].collidesWithSphere(
                            br.instance,
                            obj,
                            norm
                        )){
                            std::cout << "Case 2: Instance " << br.instance->instanceId << 
                            "(" << br.instance->modelId << ") collided with instance " << obj.instance->instanceId << 
                            "(" << obj.instance->modelId << ")" << std::endl;
                            obj.instance->handleCollision(br.instance, norm);
                            break;
                        }
                    }
                }
            }
            else {
                if (nFacesObj) {
                    // Obj has collision mesh, br doesnt
                    // Check all faces in obj against brs sphere
                    for (int i = 0; i < nFacesObj; i++){
                        if (obj.collisionMesh->faces[i].collidesWithSphere(
                            obj.instance,
                            br,
                            norm
                        )){
                            std::cout << "Case 3: Instance " << br.instance->instanceId << 
                            "(" << br.instance->modelId << ") collided with instance " << obj.instance->instanceId << 
                            "(" << obj.instance->modelId << ")" << std::endl;
                            obj.instance->handleCollision(br.instance, norm);
                            break;
                        }
                    }
                }
                else {
                    // Neither have collision mesh
                    // Coarse check passed (Teste collision between spheres)
                    // Check if spheres intersect
                    if (br.intersectsWith(obj)){
                        std::cout << "Case 4: Instance " << br.instance->instanceId << 
                        "(" << br.instance->modelId << ") collided with instance " << obj.instance->instanceId << 
                        "(" << obj.instance->modelId << ")" << std::endl;
                    }
                    norm = obj.center - br.center;
                    obj.instance->handleCollision(br.instance, norm);
                }
            }
        }
    }
}

void Octree::node::checkCollisionsChildren(BoundingRegion obj){
    if(children != nullptr){
        for (int flags = activeOctants, i = 0;
            flags > 0;
            flags >>= 1, i++){
            if (States::isIndexActive(&flags, 0) && children[i]){
                // If this child is active
                // if (children[i] != nullptr){ //by copilot
                    children[i]->checkCollisionsSelf(obj);
                    children[i]->checkCollisionsChildren(obj);
                // }
            }
        }
    }
}


void Octree::node::destroy() {
    // Clearing out children
    if (children != nullptr){
        for (int flags = activeOctants, i = 0;
            flags > 0;
            flags >>= 1, i++){
            if (States::isIndexActive(&flags, 0)){
                // If this child is active
                if (children[i] != nullptr){
                    children[i]->destroy();
                    children[i] = nullptr;
                }
            }
        }
    }

    // Clear this node
    objects.clear();
    while (queue.size() != 0){
        queue.pop();
    }
}

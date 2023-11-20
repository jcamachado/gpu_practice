#include "octree.h"

// For each octant, a new bounding region is calculated and stored in out
// Out is a pointer to node region that will have its subregions calculated
void Octree::calculateBounds(BoundingRegion* out, Octant octant, BoundingRegion parentRegion){
    glm::vec3 center = parentRegion.calculateCenter();
    if (octant == Octant::O1) {
        out = new BoundingRegion(center, parentRegion.max);
    }
    else if (octant == Octant::O2) {
        out = new BoundingRegion(glm::vec3(parentRegion.min.x, center.y, center.z), glm::vec3(center.x, parentRegion.max.y, parentRegion.max.z));
    }
    else if (octant == Octant::O3) {
        out = new BoundingRegion(glm::vec3(parentRegion.min.x, parentRegion.min.y, center.z), glm::vec3(center.x, center.y, parentRegion.max.z));
    }
    else if (octant == Octant::O4) {
        out = new BoundingRegion(glm::vec3(center.x, parentRegion.min.y, center.z), glm::vec3(parentRegion.max.x, center.y, parentRegion.max.z));
    }
    else if (octant == Octant::O5) {
        out = new BoundingRegion(glm::vec3(center.x, center.y, parentRegion.min.z), glm::vec3(parentRegion.max.x, parentRegion.max.y, center.z));
    }
    else if (octant == Octant::O6) {
        out = new BoundingRegion(glm::vec3(parentRegion.min.x, center.y, parentRegion.min.z), glm::vec3(center.x, parentRegion.max.y, center.z));
    }
    else if (octant == Octant::O7) {
        out = new BoundingRegion(parentRegion.min, center);
    }
    else if (octant == Octant::O8) {
        out = new BoundingRegion(glm::vec3(center.x, parentRegion.min.y, parentRegion.min.z), glm::vec3(parentRegion.max.x, center.y, center.z));
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

void Octree::node::build(){
    /*
        Termination conditions
        - 1 or less objects (ie an empty leaf node)
        - dimensions are too small
    */
    // <= 1 objects
    if (objects.size() <= 1){
        return;
    }

    // Dimensions are too small
    glm::vec3 dimensions = region.calculateDimensions();
    for (int i = 0; i < 3; i++){
        if (dimensions[i] < MIN_BOUNDS){
            return;
        }
    }

    // Create regions
    BoundingRegion octants[N_CHILDREN];
    for (int i = 0; i < N_CHILDREN; i++){
        calculateBounds(&octants[i], (Octant)(1 << i), region);
    }

    // Determine which octants to place object in
    std::vector<BoundingRegion> octLists[N_CHILDREN]; // Array of list of objects in each octant
    // If object doesnt fully belong to a region, it is added to the parent region
    std::stack<int> delList; // List of objects that have been placed

    // If objects overlap regions, they are in at least 1 region(parent), so they are not added to another region
    for (int i = 0, length = objects.size(); i < length; i++){
        BoundingRegion br = objects[i];
        for (int j = 0; j < N_CHILDREN; i++){
            if (octants[j].containsRegion(br)){
                octLists[j].push_back(br); // Found an octant to put it in, so it is not added to another octant
                delList.push(i);        // Object has been placed on an octant, so it can be removed from the original list
                break; 
            }
        }
    }

    // Remove objects on delList
    while (!delList.size() != 0){
        objects.erase(objects.begin() + delList.top());
        delList.pop();
    }

    // Populate octants
    for (int i = 0; i < N_CHILDREN; i++){
        if (octLists[i].size() > 0){
            children[i] = new node(octants[i], octLists[i]);
            States::activateIndex(&activeOctants, i);
            children[i]->build();
            hasChildren = true;
        }
    }

    treeBuilt = true;
    treeReady = true;

}

void Octree::node::update(){
    if (treeBuilt && treeReady){
        // Get moved objects that were in this leaf in previous frame
        std::vector<BoundingRegion> movedObjects(objects.size()); // Cap at size of objects

        // Remove objects that dont exist anymore
        for (int i = 0; i < objects.size(); i++){
            // Remove if on list of dead objects
            // TODO
        }

        // Update child nodes
        
        if (children != nullptr){
            for (unsigned char flags = activeOctants, i = 0;
                flags > 0;
                flags >>= 1, i++){                      // Iterates over each bit in flags, each octant
                    if (States::isIndexActive(&flags, 0)){
                        // Activate octant
                        if (children[i] == nullptr){

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
            movedObj = movedObjects[0];
            node* current = this;
            while (!current->region.containsRegion(movedObj)){
                if (current->parent != nullptr){
                    current = current->parent;
                }
                else {
                    break;                  // If root, leave
                }
            }

            // Remove first object, second object becomes first, so list behaves like a queue
            movedObjects.erase(movedObjects.begin());
            // getIndexOf returns the index of object, 
            // and it is the distance between the starting iterator and the iterator that represents the object
             
            objects.erase(objects.begin() + List::getIndexOf<BoundingRegion>(objects, movedObj));
            current->insert(movedObj);      // Inserts further down

            // Collision detection  (We can use bruteforce for now, the number of objects per region is small)
            // TODO
        }
    }
    else {
        // Process pending results
        if (queue.size() > 0){
            processPending();
        }
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
        while (queue.size() != 0){
            insert(queue.front());
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
        objects.push_back(obj);
        return true;
    }

    // Safeguard if object doesnt fit in any octant, it is added to parent
    if (!region.containsRegion(obj)){
        return parent == nullptr ? false : parent->insert(obj);
    }

    // Create regions if not defined
    BoundingRegion octants[N_CHILDREN];
    for (int i = 0; i < N_CHILDREN; i++){
        if (children[i] != nullptr){        // If child is not null, we want to use its region       
            octants[i] = children[i]->region;
        }
        else{                               // If child is null, we calculate the bounds based on calculateBounds
            calculateBounds(&octants[i], (Octant)(1 << i), region); 
        }

    }

    // Find region that fits item entirely
    for (int i = 0; i < N_CHILDREN; i++){
        if (octants[i].containsRegion(obj)){
            if (children[i] != nullptr){
                // Insert into child
                return children[i]->insert(obj);
            }
            else{
                // Create new node
                children[i] = new node(octants[i], {obj}); 
                States::activateIndex(&activeOctants, i);
                return true;
            }
        }
    }

    // Doesnt fit into any children, so add to parent
    objects.push_back(obj);
    return true;
}

void Octree::node::destroy() {
    // Clearing out children
    if (children != nullptr){
        for (int flags = activeOctants, i = 0;
            flags > 0;
            flags >> 1, i++){
            if (States::isActive(&flags, 0)){
                // If this child is active
                if (children[i] != nullptr){
                    children[i]->destroy();
                    children[i] = nullptr;
                }
            }
        }
    }

    // Clear this node
    while (queue.size() != 0){
        queue.pop();
    }
}
#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "../model.h"

class Sphere : public Model{
    public:
        Sphere(unsigned int maxNumInstances)
            : Model("sphere", BoundTypes::SPHERE, maxNumInstances, NO_TEX | DYNAMIC) { }

        void init(){
            loadModel("assets/models/sphere/scene.gltf");
        }
};

#endif

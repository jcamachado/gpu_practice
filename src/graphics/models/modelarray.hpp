#ifndef MODELARRAY_HPP
#define MODELARRAY_HPP

//The purpose of this file is to load the object once in memory
//and then use it as many times as we want (duplicate it) instead of
//loading it every time we want to use it.

#include "../model.h"

template <class T> 
class ModelArray{
    public:
        
        std::vector<RigidBody> instances;

        void init(){
            model.init();
        }

        void render(Shader shader, float dt){
            for(RigidBody &rb: instances){
                rb.update(dt);
                model.rb.pos = rb.pos;
                model.render(shader, dt);
            }
        }

        void setSize(glm::vec3 size){
            model.size = size;
        }

        void cleanup(){
            model.cleanup();
        }
    protected:
        T model;

};

#endif
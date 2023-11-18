#ifndef MODELARRAY_HPP
#define MODELARRAY_HPP

//The purpose of this file is to load the object once in memory
//and then use it as many times as we want (duplicate it) instead of
//loading it every time we want to use it.

#include "../model.h"
#include "box.hpp"

#define UPPER_BOUND 100 //3d vectors, max number of instances

//VAO is generated from each mesh
//VBO is generated from each vertex
template <class T> 

//many VAOs, one for each mesh, but only one VBO for each vertex(?)
class ModelArray{
    public:
        
        //instances mean that we only call the gpu once per mesh for each rendered item T
        std::vector<RigidBody> instances;

        void init(){
            model.init();

            //this is for whole objects
            //generate positions VBO
            posVBO = BufferObject(GL_ARRAY_BUFFER);
            posVBO.generate();
            posVBO.bind();
            posVBO.setData<glm::vec3>(UPPER_BOUND, NULL, GL_DYNAMIC_DRAW);
            //generate sizes VBO
            sizeVBO = BufferObject(GL_ARRAY_BUFFER);
            sizeVBO.generate();
            sizeVBO.bind();
            sizeVBO.setData<glm::vec3>(UPPER_BOUND, NULL, GL_DYNAMIC_DRAW);

            //set attribute pointers for each mesh
            for(unsigned int i=0, size = model.meshes.size(); i<size; i++){
                model.meshes[i].VAO.bind();
                // set vertex attribute pointers
                // positions
                posVBO.bind();
                // 1st param: 0, 1 and 2 are used for normal mesh (i believe he refers to mesh.cpp)
                // 2nd param and 3rd are related, so 3 GL_FLOATs
                // 4th param: stride is 1 glm::vec3 (related to template)
                // 6th param: reset every 1 instance
                posVBO.setAttrPointer<glm::vec3>(3, 3, GL_FLOAT, 1, 0, 1); 
                // size
                sizeVBO.bind();
                sizeVBO.setAttrPointer<glm::vec3>(4, 3, GL_FLOAT, 1, 0, 1);

                ArrayObject::clear();
            }
        }

        void render(Shader shader, float dt, Box *box, bool setLists = true){
            if(setLists){
                positions.clear();
                sizes.clear();
                for(RigidBody& rb: instances){
                    rb.update(dt);
                    positions.push_back(rb.pos);
                    sizes.push_back(model.size);
                }
            }
            //optimization, passing the calculations of translations and scaling into the vbo
            // so the gpu can do the calculations (the cuda cores)
            shader.setMat4("model", glm::mat4(1.0f)); //reset model matrix to identity matrix

            model.render(shader, dt, nullptr, false, false);

            //if more than UPPER_BOUND instances, only render UPPER_BOUND 
            int instances = std::min(UPPER_BOUND, (int)positions.size()); 
            
            //if instances exist, update data in VBOs
            if(instances!=0){
                // reset VBO data
                posVBO.bind();
                posVBO.updateData<glm::vec3>(0, instances, &positions[0]);

                sizeVBO.bind();
                sizeVBO.updateData<glm::vec3>(0, instances, &sizes[0]);

                sizeVBO.clear(); // it could be posVBO.clear() as well, since they have the same type of the array buffers
            }
            //render instanced data
            for(unsigned int i = 0, length = model.meshes.size(); i<length; i++){
                for (unsigned int j = 0; j < instances; j++){
                    box->addInstance(model.meshes[i].br, positions[j], sizes[j]);
                }

                model.meshes[i].VAO.bind();
                model.meshes[i].VAO.draw(GL_TRIANGLES, model.meshes[i].indices.size(), GL_UNSIGNED_INT, 0, instances);
                ArrayObject::clear();
            }
        }

        void setSize(glm::vec3 size){
            model.size = size;
        }

        void cleanup(){
            model.cleanup();
            posVBO.cleanup();
            sizeVBO.cleanup();
        }
    protected:
        T model;

        BufferObject posVBO;
        BufferObject sizeVBO;
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> sizes;

};

#endif
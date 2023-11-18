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
class ModelArray{
    public:
        
        //instances mean that we only call the gpu once per mesh for each rendered item T
        std::vector<RigidBody> instances;

        void init(){
            model.init();

            //this is for whole objects
            //generate positions VBO
            glGenBuffers(1, &posVBO);
            glBindBuffer(GL_ARRAY_BUFFER, posVBO);
            //allocate memory for positions VBO
            glBufferData(GL_ARRAY_BUFFER, UPPER_BOUND * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            //generate sizes VBO
            glGenBuffers(1, &sizeVBO);
            glBindBuffer(GL_ARRAY_BUFFER, sizeVBO);
            //allocate memory for sizes VBO
            glBufferData(GL_ARRAY_BUFFER, UPPER_BOUND * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            //set attribute pointers for each mesh
            for(unsigned int i=0, size = model.meshes.size(); i<size; i++){
                glBindVertexArray(model.meshes[i].VAO);
                
                //set attribute pointers for positions
                glBindBuffer(GL_ARRAY_BUFFER, posVBO);
                //Since normal and texcoord on mesh occupy the 1 and 2 positions in the shaders, 
                //we will keep counting from 3 and 4
                //since we have different vbos for positions and sizes, the offset will be 3
                //if they were combined, the offset would be 6 
                glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(3);

                //sizes
                glBindBuffer(GL_ARRAY_BUFFER, sizeVBO);
                glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(4);

                glBindBuffer(GL_ARRAY_BUFFER, 0);

                //different from before, now we want to get it on every instance of the object
                //not every vertex. We will say to openGL to draw the entire object and then get the next
                //model position and size

                glVertexAttribDivisor(3, 1); //reset _3rd_ attribute every _1_ instance
                glVertexAttribDivisor(4, 1); //reset _4th_ attribute every _1_ instance

                glBindVertexArray(0);
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

                glBindBuffer(GL_ARRAY_BUFFER, posVBO);
                //faster than glBufferData
                glBufferSubData(GL_ARRAY_BUFFER, 0, instances * 3 * sizeof(float), &positions[0]);
                
                glBindBuffer(GL_ARRAY_BUFFER, sizeVBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, instances * 3 * sizeof(float), &sizes[0]);

                glBindBuffer(GL_ARRAY_BUFFER, 0);
            }
            //render instanced data
            for(unsigned int i = 0, length = model.meshes.size(); i<length; i++){
                for (unsigned int j = 0; j < instances; j++){
                    box->addInstance(model.meshes[i].br, positions[j], sizes[j]);
                }

                glBindVertexArray(model.meshes[i].VAO);
                //same as glDrawElements but passes size = number of instances
                glDrawElementsInstanced(GL_TRIANGLES, model.meshes[i].indices.size(), GL_UNSIGNED_INT, 0, instances);
                glBindVertexArray(0);
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

        unsigned int posVBO;
        unsigned int sizeVBO;
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> sizes;

};

#endif
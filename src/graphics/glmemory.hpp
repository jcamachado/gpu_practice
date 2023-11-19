#ifndef GLMEMORY_HPP
#define GLMEMORY_HPP

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <map>
#include <iostream>

/*
    class for buffer objects
    - VBOs, EBOs, VAOs
*/

class BufferObject {
    public:
        // value/location of the buffer object
        GLuint val;
        // type of buffer object (GL_ARRAY_BUFFER || GL_ELEMENT_ARRAY_BUFFER || GL_VERTEX_ARRAY, etc.)
        GLenum type;

        BufferObject() {}
        BufferObject(GLenum type)
            : type(type)  {}

        // generate buffer object
        void generate() {
            glGenBuffers(1, &val);
        }

        // bind buffer object
        void bind() {
            glBindBuffer(type, val);
        }

        // set data (glBufferData)
        template<typename T>
        void setData(GLuint nElements, T* data, GLenum usage) {
            glBufferData(type, nElements*sizeof(T), data, usage);
        }

        // update data (glBufferSubData) (modelarrays, instances)
        template<typename T>
        void updateData(GLintptr offset, GLuint nElements, T* data) {
            //faster than glBufferData
            glBufferSubData(type, offset, nElements*sizeof(T), data);
        }

        // set attribute pointers
        // offset is the number of size in bytes away from the starting element that we pass into GLBufferdata
        template<typename T>
        void setAttrPointer(GLuint idx, GLint size, GLenum type, GLuint stride, GLuint offset, GLuint divisor=0) {
            glVertexAttribPointer(idx, size, type, GL_FALSE, stride*sizeof(T), (void*)(offset*sizeof(T)));
            glEnableVertexAttribArray(idx);
            if (divisor > 0) {
                // reset _idx_ attribute every _divisor_ iteration (instancing)
                glVertexAttribDivisor(idx, divisor);
            }
        }

        // clear buffer objects (bind 0)
        // called when you dont want a certain vbo bound anymore
        // once we are done loading the data, we unbind it and free that memory up from the GPU
        void clear() {
            glBindBuffer(type, 0);
        }

        // cleanup (deletion)
        void cleanup() {
            glDeleteBuffers(1, &val);
        }
};

/*
    class for array objects
    - VAOs
*/

class ArrayObject {
    public:
        // value/location
        GLuint val;

        // map of names to buffers
        std::map<const char*, BufferObject> buffers;

        // get buffer (override [])
        BufferObject& operator[](const char* key) {
            return buffers[key];
        }

        // generate object
        void generate() {
            glGenVertexArrays(1, &val);
        }

        // bind
        void bind() {
            glBindVertexArray(val);
        }

        // draw
        void draw(GLenum mode, GLuint count, GLenum type, GLint indices, GLuint instanceCount=1) {
            // video worked with glDrawElementsInstanced(mode, count, type, (void*)indices, instanceCount);
            // but here it is intermitently breaking the build
            // glDrawElementsInstanced(mode, count, type, (void*)indices, instanceCount);
            //same as glDrawElements but passes size = number of instances
            glDrawElementsInstanced(mode, count, type, (GLvoid*)(intptr_t)indices, instanceCount);
        }

        // cleanup (instead of only deleting the VAO1, we also delete its children, all the the VBOs and EBOs)
        void cleanup() {
            glDeleteVertexArrays(1, &val);
            // pair is a key value pair [key, value], where key is a const char* and value is a BufferObject 
            for (auto& pair : buffers) {
                // calls bufferobject cleanup
                pair.second.cleanup();
            }
        }

        // clear array object (bind 0)
        static void clear() {
            glBindVertexArray(0);
        }




};

#endif
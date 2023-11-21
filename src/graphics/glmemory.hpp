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

        /*
            .setAttrPointer<>(idx, size, type, stride, offset, divisor=0);
            _idx_: 0, 1 and 2 are used for normal mesh (I believe he refers to mesh.cpp)
            _size_ param and _type_ are related, so 3 GL_FLOATs
            _stride_: _stride_ means it is its value * the _template_ size (related to template)
            _offset_: is the number of the size in bytes away from the starting element that we pass into GLBufferdata
            _divisor_ : indicates after how many iterations the index should reset. (Instancing)
        */
        template<typename T>
        void setAttrPointer(GLuint idx, GLint size, GLenum type, GLuint stride, GLuint offset, GLuint divisor=0) {
            glVertexAttribPointer(idx, size, type, GL_FALSE, stride*sizeof(T), (void*)(offset*sizeof(T)));
            glEnableVertexAttribArray(idx);
            if (divisor > 0) {
                // Reset _idx_ attribute every _divisor_ iterations (instancing)
                glVertexAttribDivisor(idx, divisor);
            }
        }

        /*
            Clear buffer objects (bind 0)
            called when you dont want a certain vbo bound anymore
            once we are done loading the data, we unbind it and free that memory up from the GPU
        */
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
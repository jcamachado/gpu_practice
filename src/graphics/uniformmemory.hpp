#ifndef UNIFORMMEMORY_HPP
#define UNIFORMMEMORY_HPP

#include "../../lib/glad/glad.h"
#include <GLFW/glfw3.h>

#include <vector>
#include <string>

#include "vertexmemory.hpp"

/*
    Same structure as VertexMemory.hpp but using a different buffer type
*/

#define N 4     // Word size in bytes

namespace UBO {
    enum class Type : unsigned char {
        SCALAR = 0,     // Any 4 byte value. Integer, unsigned integer, float, etc.
        VEC2,
        VEC3,
        VEC4,
        ARRAY,
        STRUCT,
        INVALID
    };

    // Round up val to the next multiple of mul 2^n
    unsigned int roundUpPow2(unsigned int val, unsigned int n) {
        /*
            n=4
            pow2n = 2^4 = 16 = 0b00010000
            To get the next multiple of 16, we need to add 16 - remainder
            divisor = 0b00001111    // 1s after 1 of 16
            b & 1 = b -> 0 & 1 = 0, 1 & 1 = 1

            remainder = value & divisor = 0b00000000
            in value = 0b11010011 = 211
            so remainder = 0b00000011 = 3

            val += 16 - 3 = 224 -> 224/16 = 14
        */
        unsigned int pow2n = 0b1 << n;      // 1 << n -> 1 * 2^n = 2^n
        unsigned int divisor = pow2n-1;     // 2^n - 1 = 0b001...1111 (n 1s)
        unsigned int remainder = val & divisor;
        if (remainder) {
            val += pow2n - remainder;
        }

        return val;
    }

    /*
        -length: Length of the array or the number of elements in the struct. Allows us to store a value where we
        dont have to call the list method or a vector method to get the size that would be from std::vector<Element> list
        - list: For struc (list of sub-elements), or array (1st slot is the type)

    */
    typedef struct Element {    // Mapped with the UBO at vs file
        Type type;
        unsigned int baseAlign; // Base alignment of the element
        unsigned int length;
        std::vector<Element> list;


        std::string typeString(){
            switch (type) {
                case Type::SCALAR: return "scalar";
                case Type::VEC2: return "vec2";
                case Type::VEC3: return "vec3";
                case Type::VEC4: return "vec4";
                case Type::ARRAY: return "array<" + list[0].typeString() + ">";
                case Type::STRUCT: return "struct";
                default: return "invalid";
            }
        }

        unsigned int alignPow2() {
            switch (baseAlign) {
                case 2: return 1;
                case 4: return 2;
                case 8: return 3;
                case 16: return 4;
                default: return 0;
            }
        }

        unsigned int calcSize() {
            switch (type) {
                case Type::SCALAR: return N;
                case Type::VEC2: return 2*N;
                case Type::VEC3: return 3*N;
                case Type::VEC4: return 4*N;
                case Type::ARRAY:
                case Type::STRUCT:
                    return calcPaddedSize();
                default: return 0;
            }
        }

        unsigned int calcPaddedSize() {
            unsigned int offset = 0;
            switch (type) {
                case Type::ARRAY: 
                    return length * roundUpPow2(list[0].calcSize(), alignPow2());
                case Type::STRUCT: 
                    for (Element& e : list) {
                        offset += roundUpPow2(offset, e.alignPow2());
                        offset += e.calcSize();
                    }
                    return offset;
                case Type::SCALAR:
                case Type::VEC2:
                case Type::VEC3:
                case Type::VEC4:
                default: return calcSize();
            }
        }

        Element(Type type = Type::SCALAR)
            : type(type), length(0), list(0) {
                switch (type) {
                    case Type::SCALAR: baseAlign = N; break;
                    case Type::VEC2: baseAlign = 2*N; break;
                    case Type::VEC3:
                    case Type::VEC4: baseAlign = 4*N; break;
                    default: baseAlign = 0; break;
                }
            }

    } Element;

    inline Element newScalar() {
        return Element(Type::SCALAR);
    }

    inline Element newVec(unsigned char dim) {
        switch (dim) {
            case 2: return Type::VEC2;
            case 3: return Type::VEC3;
            case 4:
            default: return Type::VEC4;
        };
    }

    inline Element newArray(unsigned int length,Element arrElement) {
        Element ret(Type::ARRAY);
        ret.length = length;
        ret.list = { arrElement };
        ret.list.shrink_to_fit();

        ret.baseAlign = arrElement.type == Type::STRUCT ?
            arrElement.baseAlign : 
            roundUpPow2(arrElement.baseAlign, 4);

        return ret;
    }

    /*
        Matrices and Array of matrices
    */
    inline Element newColMat(unsigned char cols, unsigned char rows){
        return newArray(cols, newVec(rows));
    }

    inline Element newColMatArray(unsigned int nMatrices, unsigned char cols, unsigned char rows) {
        return newArray(nMatrices * cols, newVec(rows));
    }

    inline Element newRowMat(unsigned char rows, unsigned char cols) {
        return newArray(rows, newVec(cols));
    }

    inline Element newRowMatArray(unsigned int nMatrices, unsigned char rows, unsigned char cols) {
        return newArray(nMatrices * rows, newVec(cols));
    }

    /*
        Structs
    */
    inline Element newStruct(std::vector<Element> subelements) {
        Element ret(Type::STRUCT);
        ret.list.insert(ret.list.end(), subelements.begin(), subelements.end());
        ret.length = ret.list.size();

        // Base alignment is largest of its subelements
        if (subelements.size()) {
            for (Element& e : subelements) {
                if (e.baseAlign > ret.baseAlign){
                    ret.baseAlign = std::max(ret.baseAlign, e.baseAlign);
                }
            }
        }
            ret.baseAlign = roundUpPow2(ret.baseAlign, 4);
        return ret;

    }

    class UBO : public BufferObject {
        public:
            Element block;                      // Root element of the UBO (struct)
            unsigned int calculatedSize;        // Size of the UBO in bytes
            GLuint bindingPos;                  // Binding position of the UBO


            UBO(GLuint bindingPos) 
                : BufferObject(GL_UNIFORM_BUFFER), 
                block(newStruct({})), 
                calculatedSize(0), 
                bindingPos(bindingPos) {}
            
            UBO(GLuint bindingPos, std::vector<Element> elements)
                : BufferObject(GL_UNIFORM_BUFFER), 
                block(newStruct(elements)), 
                calculatedSize(0),
                bindingPos(bindingPos) {}
            
            void attachToShader(Shader shader, std::string name) {
                /*
                    name -> name of the block in the shader file.
                    ex: layout(std140) uniform Test { ... }
                    name = "Test"

                    There are binding points for UBOs. These are standard across all shaders.
                    0, 1, 2, indexes and so on.
                    But once a shader wants to use a UBO, it has to bind it to a binding point.
                    But the block indices for the shaders are not the same.
                    if shaderA uses 0 and 2 binding points, its block indices will follow its own order of 0 and 1.
                    shaderA(indices) = {0, 1} mapping to {0, 2} binding points
                */
                GLuint blockIndex = glGetUniformBlockIndex(shader.id, name.c_str()); // Different for each shader
                glUniformBlockBinding(shader.id, blockIndex, bindingPos);            // Same for all shaders
            }

            // Similar to setting VBO's memory
            void initNullData(GLenum usage) {
                if (!calculatedSize) {
                    calculatedSize = calcSize();
                }
                glBufferData(type, calculatedSize, NULL, usage);
            }

            void bindRange(GLuint offset = 0){
                if (!calculatedSize) {
                    calculatedSize = calcSize();
                }
                glBindBufferRange(type, bindingPos, val, offset, calculatedSize);
            }

            unsigned int calcSize() {
                return block.calcPaddedSize();
            }

            void addElement(Element element) {
                block.list.push_back(element);
                if (element.baseAlign > block.baseAlign) {
                    block.baseAlign = element.baseAlign;
                }
                block.length++;
            }
            /*
                Iteration
            */
            GLuint offset;
            GLuint poppedOffset;
            std::vector<std::pair<unsigned int, Element*>> indexStack;  // Stack to keep track of nested indices
            int currentDepth;                                           // Current depth of the stack -1

            // Initialize Iterator
            void startWrite() {
                currentDepth = 0;
                offset = 0;
                poppedOffset = 0;
                indexStack.clear();
                indexStack.push_back({0, &block});                      // Push root element
            }

            // Next element in iteration
            Element getNextElement() {
                // highest level struct popped, stack is empty
                if (currentDepth < 0) {
                    return Type::INVALID;
                }

                // Get current deepest array/struct (last element in the stack)
                Element* currentElement = indexStack[currentDepth].second;

                // Get element at the specified index within that iterable (iterables are not leaves)
                // Elements within the iterable are leaves
                if (currentElement->type == Type::STRUCT) {
                    currentElement = &currentElement->list[indexStack[currentDepth].first];
                }
                else { // Array
                    currentElement = &currentElement->list[0];
                }

                // Traverse down to deepest array/struct
                while (currentElement->type == Type::STRUCT || currentElement->type == Type::ARRAY) {
                    currentDepth++;
                    indexStack.push_back({ 0, currentElement });
                    currentElement = &currentElement->list[0];
                }

                /*
                    Now have current element (leaf, not iterable)
                    Pop from stack if necessary
                    When we pop, we need to realign and update the offset
                */
                poppedOffset = roundUpPow2(offset, currentElement->alignPow2()) + currentElement->calcSize();
                if (!pop()) {
                    // No items popped
                    poppedOffset = 0;
                }

                return *currentElement;
            }
        
        bool pop() {
            bool popped = false;

            for (int i = currentDepth; i >= 0; i--) {
                int advanceIdx = ++indexStack[i].first; // Move cursor forward in iterable
                if (advanceIdx >= indexStack[i].second->length) {
                    // If we are at the end of the iterable, pop iterable from stack
                    poppedOffset = roundUpPow2(poppedOffset, indexStack[i].second->alignPow2());    // Realignment
                    indexStack.erase(indexStack.begin() + i);
                    popped = true;
                    currentDepth--;
                }
                else {  // Nothing to pop
                    break;
                }
            }
            return popped;
        }

        template<typename T>
        void writeElement(T* data){
            Element element = getNextElement();
            offset = roundUpPow2(offset, element.alignPow2()) ;             // Round up to base alignment of current element
            // Offset need to be aligned here
            glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(T), data);    // Write element
            if (poppedOffset) {
                offset = poppedOffset;
            }
            else{
                offset += element.calcSize();                                      // Add to offset
            }
        }

        template <typename T>
        void writeArray(T* first, unsigned int n) {
            // Motivation: Pop wouldnt be called until the end of the array because arrays are not leaves
            for (int i = 0; i < n; i++){
                writeElement<T>(&first[i]);
            }
        }

        template <typename T, typename V>
        void writeArrayContainer(T *container, unsigned int n) {
        /*
            Container -> operator [],
            *container pointer to the first element of an array
            Motivation: Sometimes the array is not represented as an array of the element
            Ex: With a matrix glm::mat4 stores an array of 4 vec4s however we wont pass
            a glm::mat4 as a parameter element because we're not writing in glm::mat4s.
            We are writing in columns separately


            If we pass a glm::mat4 and we were saying i at 2, and called the pointer to
            a glm::mat4[2] that would assume we were looking for the second glm::mat4 in
            a array which is not what we want. We have to call the operator to get that.

        */
            for (int i = 0; i < n; i++) {
                writeElement<V>(&container->operator[](i)); // container[i] translates to container+i
            }
        }

        void advanceCursor(unsigned int n) {
            // skip number of elements
            for (int i = 0; i < n; i++) {
                Element element = getNextElement();
                offset = roundUpPow2(offset, element.alignPow2());
                if (poppedOffset) {
                    offset = poppedOffset;
                }
                else {
                    offset += element.calcSize();
                }
            }
        }

        /*
            This functions aims to skip arrays
        */
        void advanceArray(unsigned int nElements){ //Optimizable without for loop
            if (currentDepth < 0) {
                return;
            }

            Element* currentElement = indexStack[currentDepth].second;

            // Get the next array
            if (currentElement->type == Type::STRUCT) {
                currentElement = &currentElement->list[indexStack[currentDepth].first];

                unsigned int depthAddition = 0;
                // Elements that we would add to the stack
                std::vector<std::pair<unsigned int, Element*>> stackAddition;

                // Go to next array
                while (currentElement->type == Type::STRUCT){
                    depthAddition++;
                    stackAddition.push_back({ 0, currentElement });
                    currentElement = &currentElement->list[0];
                }

                if (currentElement->type != Type::ARRAY) {
                    // Did not find an array (reached primitive)
                    return;
                }//

                // Found array, apply changes
                currentDepth += depthAddition + 1; // + 1 for the array
                indexStack.insert(indexStack.end(), stackAddition.begin(), stackAddition.end());
                indexStack.push_back({ 0, currentElement });    // Push array to stack
            }

            // at an array, avance number of elements
            unsigned int finalIdx = indexStack[currentDepth].first + nElements;
            unsigned int advanceCount = nElements;
            if (finalIdx >= indexStack[currentDepth].second->length) {
                // advance to the end of array
                advanceCount = indexStack[currentDepth].second->length - indexStack[currentDepth].first;
            }

            // Calculate like array calPaddedSize
            offset += advanceCount * roundUpPow2(currentElement->list[0].calcSize(), currentElement->alignPow2());
			indexStack[currentDepth].first += advanceCount;
            // Pop from stack
            poppedOffset = offset;
            if (pop()){
                // Items popped
                offset = poppedOffset;
            }
        }

    };
}

#endif
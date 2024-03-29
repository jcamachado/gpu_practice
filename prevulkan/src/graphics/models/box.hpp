#ifndef BOX_HPP
#define BOX_HPP

//include glad
#include "../../../lib/glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include "../memory/vertexmemory.hpp"
#include "../rendering/shader.h"
#include "../../algorithms/bounds.h"

#define UPPER_BOUND 10000 //3d vectors, max number of instances
/*
    This class is just for lines, no meshes, so we cant reuse the Mesh class and 
    we have to generate the VAO, VBO and EBO

    Also, it is this class responsibility to draw the lines around the bounding regions
    So doesnt matter if it is a region from a model or a region from an octree cell. It will draw.
*/
class Box{
    public:
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> sizes;

        void init(){
            vertices = {
                // position				x	y	z	i
                0.5f,  0.5f,  0.5f, //	+	+	+	0
                -0.5f,  0.5f,  0.5f, //	-	+	+	1
                -0.5f, -0.5f,  0.5f, //	-	-	+	2
                0.5f, -0.5f,  0.5f, //	+	-	+	3
                0.5f,  0.5f, -0.5f, //	+	+	-	4
                -0.5f,  0.5f, -0.5f, //	-	+	-	5
                -0.5f, -0.5f, -0.5f, //	-	-	-	6
                0.5f, -0.5f, -0.5f	 //	+	-	-	7
            };

            indices = { // 12 lines
                // front face (+ve z)
                0, 1,
                1, 2,
                2, 3,
                3, 0,
                // back face (-ve z)
                4, 5,
                5, 6,
                6, 7,
                7, 4,
                // right face (+ve x)
                0, 4,
                3, 7,
                // left face (-ve x)
                1, 5,
                2, 6
            };
            VAO.generate();                                         // generate VAO
            VAO.bind();


            VAO["EBO"] = BufferObject(GL_ELEMENT_ARRAY_BUFFER);     // generate EBO
            VAO["EBO"].generate();
            VAO["EBO"].bind();
            VAO["EBO"].setData<GLuint>(indices.size(), &indices[0], GL_STATIC_DRAW);
            
            VAO["VBO"] = BufferObject(GL_ARRAY_BUFFER);             // generate VBO
            VAO["VBO"].generate();
            VAO["VBO"].bind();
            VAO["VBO"].setData<GLfloat>(vertices.size(), &vertices[0], GL_STATIC_DRAW);
            VAO["VBO"].setAttrPointer<GLfloat>(0, 3, GL_FLOAT, 3, 0);
            
            VAO["posVBO"] = BufferObject(GL_ARRAY_BUFFER);          // position VBO - Dynamic
            VAO["posVBO"].generate();
            VAO["posVBO"].bind();
            VAO["posVBO"].setData<glm::vec3>(UPPER_BOUND, NULL, GL_DYNAMIC_DRAW);
            VAO["posVBO"].setAttrPointer<glm::vec3>(1, 3, GL_FLOAT, 1, 0, 1);

            VAO["sizeVBO"] = BufferObject(GL_ARRAY_BUFFER);         // size VBO - Dynamic
            VAO["sizeVBO"].generate();
            VAO["sizeVBO"].bind();
            VAO["sizeVBO"].setData<glm::vec3>(UPPER_BOUND, NULL, GL_DYNAMIC_DRAW);
            VAO["sizeVBO"].setAttrPointer<glm::vec3>(2, 3, GL_FLOAT, 1, 0, 1);
            VAO["sizeVBO"].clear();

            ArrayObject::clear();
        }


        void render(Shader shader){
            shader.setMat4("model", glm::mat4(1.0f));
            
            //update data
            int instances = std::min(UPPER_BOUND, (int)positions.size()); 
            
            //if instances exist, update data in VBOs
            if(instances!=0){
                VAO["posVBO"].bind();
                VAO["posVBO"].updateData<glm::vec3>(0, instances, &positions[0]);

                VAO["sizeVBO"].bind();
                VAO["sizeVBO"].updateData<glm::vec3>(0, instances, &sizes[0]);
                
            }
            //render instanced data
            //GL_LINES is so it can draw in pairs of indices
            VAO.bind();
            VAO.draw(GL_LINES, indices.size(), GL_UNSIGNED_INT, 0, instances);
        }

        void addInstance(BoundingRegion br, glm::vec3 pos, glm::vec3 size){
            positions.push_back(br.calculateCenter() * size + pos); // position
            sizes.push_back(br.calculateDimensions() * size);
        }

        void cleanup(){
            VAO.cleanup();
        }

    private:
        ArrayObject VAO;

        std::vector<float> vertices;
        std::vector<unsigned int> indices;
};


#endif
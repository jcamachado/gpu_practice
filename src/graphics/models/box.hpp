#ifndef BOX_HPP
#define BOX_HPP

//include glad
#include "../../../lib/glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include "../shader.h"

#define UPPER_BOUND 100 //3d vectors, max number of instances

//This class is just for lines, no meshes
//So we cant reuse the Mesh class, and we have to generate the VAO, VBO and EBO
class Box{
    public:
        std::vector<glm::vec3> offsets;
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

            //generate VAO
            glGenVertexArrays(1, &VAO);
            glBindVertexArray(VAO);

            //generate vertices VBO
            glGenBuffers(1, &VBO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);

            //generate positions VBO
            glGenBuffers(1, &offsetVBO);
            glBindBuffer(GL_ARRAY_BUFFER, offsetVBO);
            //allocate memory for positions VBO
            glBufferData(GL_ARRAY_BUFFER, UPPER_BOUND * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            //generate sizes VBO
            glGenBuffers(1, &sizeVBO);
            glBindBuffer(GL_ARRAY_BUFFER, sizeVBO);
            //allocate memory for sizes VBO
            glBufferData(GL_ARRAY_BUFFER, UPPER_BOUND * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            //generate indices EBO
            glGenBuffers(1, &EBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

            //set attribute pointers for vertices
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0); //clear VBO

            // offsets
            glBindBuffer(GL_ARRAY_BUFFER, offsetVBO);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);

            //sizes
            glBindBuffer(GL_ARRAY_BUFFER, sizeVBO);
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(2);

            glVertexAttribDivisor(1, 1); //reset _1rd_ attribute every _1_ instance
            glVertexAttribDivisor(2, 1); //reset _2th_ attribute every _1_ instance

            //clear VAO
            glBindVertexArray(0);

            //generate dynamic VBO
        }


        void render(Shader shader){
            shader.setMat4("model", glm::mat4(1.0f));
            
            //update data
            int size = std::min(UPPER_BOUND, (int)offsets.size()); 
            
            //if instances exist, update data in VBOs
            if(size!=0){

                glBindBuffer(GL_ARRAY_BUFFER, offsetVBO);
                //faster than glBufferData
                glBufferSubData(GL_ARRAY_BUFFER, 0, size * 3 * sizeof(float), &offsets[0]);
                
                glBindBuffer(GL_ARRAY_BUFFER, sizeVBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, size * 3 * sizeof(float), &sizes[0]);

                glBindBuffer(GL_ARRAY_BUFFER, 0);
            }
            //render instanced data
            glBindVertexArray(VAO);

            //GL_LINES is so it can draw in pairs of indices
            glDrawElementsInstanced(GL_LINES, indices.size(), GL_UNSIGNED_INT, 0, size);
            glBindVertexArray(0);
        }
        void cleanup(){
            glDeleteVertexArrays(1, &VAO);
            glDeleteBuffers(1, &VBO);
            glDeleteBuffers(1, &offsetVBO);
            glDeleteBuffers(1, &sizeVBO);
        }

    private:
        unsigned int VAO, VBO, EBO;
        unsigned int offsetVBO, sizeVBO;

        std::vector<float> vertices;
        std::vector<unsigned int> indices;
};


#endif
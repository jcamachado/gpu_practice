#ifndef CUBE_HPP
#define CUBE_HPP

#include "../model.h"
#include "../material.h"

//does not use ebo because the normal vectors are not the same for each side
class Cube : public Model {
    public:
        Material material;

        Cube(unsigned int maxNumInstances, Material material = Material::red_plastic)
            : Model("cube", BoundTypes::AABB, maxNumInstances, CONST_INSTANCES | NO_TEX), material(material) {}
        void init(){
            int nVertices = 36;
             //each face of the cube have to have a texture
            float vertices[] = {
                // position					normal				texturecoord
                -0.5f, -0.5f, -0.5f,	 0.0f,  0.0f, -1.0f,	0.0f, 0.0f,
                 0.5f, -0.5f, -0.5f,	 0.0f,  0.0f, -1.0f,	1.0f, 0.0f,
                 0.5f,  0.5f, -0.5f,	 0.0f,  0.0f, -1.0f,	1.0f, 1.0f,
                 0.5f,  0.5f, -0.5f,	 0.0f,  0.0f, -1.0f,	1.0f, 1.0f,
                -0.5f,  0.5f, -0.5f,	 0.0f,  0.0f, -1.0f,	0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,	 0.0f,  0.0f, -1.0f,	0.0f, 0.0f,

                -0.5f, -0.5f,  0.5f,	 0.0f,  0.0f,  1.0f,	0.0f, 0.0f,
                 0.5f, -0.5f,  0.5f,	 0.0f,  0.0f,  1.0f,	1.0f, 0.0f,
                 0.5f,  0.5f,  0.5f,	 0.0f,  0.0f,  1.0f,	1.0f, 1.0f,
                 0.5f,  0.5f,  0.5f,	 0.0f,  0.0f,  1.0f,	1.0f, 1.0f,
                -0.5f,  0.5f,  0.5f,	 0.0f,  0.0f,  1.0f,	0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f,	 0.0f,  0.0f,  1.0f,	0.0f, 0.0f,

                -0.5f,  0.5f,  0.5f,	-1.0f,  0.0f,  0.0f,	1.0f, 0.0f,
                -0.5f,  0.5f, -0.5f,	-1.0f,  0.0f,  0.0f,	1.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,	-1.0f,  0.0f,  0.0f,	0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,	-1.0f,  0.0f,  0.0f,	0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f,	-1.0f,  0.0f,  0.0f,	0.0f, 0.0f,
                -0.5f,  0.5f,  0.5f,	-1.0f,  0.0f,  0.0f,	1.0f, 0.0f,

                 0.5f,  0.5f,  0.5f,	 1.0f,  0.0f,  0.0f,	1.0f, 0.0f,
                 0.5f,  0.5f, -0.5f,	 1.0f,  0.0f,  0.0f,	1.0f, 1.0f,
                 0.5f, -0.5f, -0.5f,	 1.0f,  0.0f,  0.0f,	0.0f, 1.0f,
                 0.5f, -0.5f, -0.5f,	 1.0f,  0.0f,  0.0f,	0.0f, 1.0f,
                 0.5f, -0.5f,  0.5f,	 1.0f,  0.0f,  0.0f,	0.0f, 0.0f,
                 0.5f,  0.5f,  0.5f,	 1.0f,  0.0f,  0.0f,	1.0f, 0.0f,

                -0.5f, -0.5f, -0.5f,	 0.0f, -1.0f,  0.0f,	0.0f, 1.0f,
                 0.5f, -0.5f, -0.5f,	 0.0f, -1.0f,  0.0f,	1.0f, 1.0f,
                 0.5f, -0.5f,  0.5f,	 0.0f, -1.0f,  0.0f,	1.0f, 0.0f,
                 0.5f, -0.5f,  0.5f,	 0.0f, -1.0f,  0.0f,	1.0f, 0.0f,
                -0.5f, -0.5f,  0.5f,	 0.0f, -1.0f,  0.0f,	0.0f, 0.0f,
                -0.5f, -0.5f, -0.5f,	 0.0f, -1.0f,  0.0f,	0.0f, 1.0f,

                -0.5f,  0.5f, -0.5f,	 0.0f,  1.0f,  0.0f,	0.0f, 1.0f,
                 0.5f,  0.5f, -0.5f,	 0.0f,  1.0f,  0.0f,	1.0f, 1.0f,
                 0.5f,  0.5f,  0.5f,	 0.0f,  1.0f,  0.0f,	1.0f, 0.0f,
                 0.5f,  0.5f,  0.5f,	 0.0f,  1.0f,  0.0f,	1.0f, 0.0f,
                -0.5f,  0.5f,  0.5f,	 0.0f,  1.0f,  0.0f,	0.0f, 0.0f,
                -0.5f,  0.5f, -0.5f,	 0.0f,  1.0f,  0.0f,	0.0f, 1.0f
            };
            std::vector<unsigned int> indices(nVertices);
            for(unsigned int i=0; i<nVertices; i++){
                indices[i] = i;
            }
            BoundingRegion br(glm::vec3(-0.5f), glm::vec3(0.5f));

            aiColor4D diff(
                material.diffuse.r, 
                material.diffuse.g, 
                material.diffuse.b, 
                1.0f
            );
            aiColor4D spec(
                material.specular.r, 
                material.specular.g, 
                material.specular.b, 
                1.0f
            );

            Mesh ret(br, diff, spec);
            ret.loadData(Vertex::genList(vertices, nVertices), indices);

            meshes.push_back(ret);
            boundingRegions.push_back(br);
        }
};

#endif
#include <GL/glut.h>
#include "glm/glm.hpp"
#include <vector>


struct Mesh
{
    GLuint PositionHandle;
    GLuint NormalHandle;
    GLuint IndexHandle;

    unsigned int IndexBufferLength;

    Mesh();
    Mesh(std::vector<float> const & Positions, std::vector<float> const & Normals, std::vector<unsigned short> const & Indices);
};

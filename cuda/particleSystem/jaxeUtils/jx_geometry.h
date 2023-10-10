// #include <GL/glut.h>
// #include "glm/glm.hpp"
// #include <vector>


// struct Mesh
// {
//     GLuint PositionHandle;
//     GLuint NormalHandle;
//     GLuint IndexHandle;

//     unsigned int IndexBufferLength;

//     Mesh();
//     Mesh(std::vector<float> const & Positions, std::vector<float> const & Normals, std::vector<unsigned short> const & Indices);
//     void Draw();
// };

//     Mesh * GeometryCreator::CreateCube(glm::vec3 const & Size)
// {
//     std::vector<float> Positions, Normals;
//     std::vector<unsigned short> Indices;

//     static float const CubePositions[] =
//     {
//         -0.5, -0.5, -0.5, // back face verts [0-3]
//         -0.5,  0.5, -0.5,
//          0.5,  0.5, -0.5,
//          0.5, -0.5, -0.5,
         
//         -0.5, -0.5,  0.5, // front face verts [4-7]
//         -0.5,  0.5,  0.5,
//          0.5,  0.5,  0.5,
//          0.5, -0.5,  0.5,
         
//         -0.5, -0.5,  0.5, // left face verts [8-11]
//         -0.5, -0.5, -0.5,
//         -0.5,  0.5, -0.5,
//         -0.5,  0.5,  0.5,
        
//          0.5, -0.5,  0.5, // right face verts [12-15]
//          0.5, -0.5, -0.5,
//          0.5,  0.5, -0.5,
//          0.5,  0.5,  0.5,
         
//         -0.5,  0.5,  0.5, // top face verts [16-19]
//         -0.5,  0.5, -0.5,
//          0.5,  0.5, -0.5,
//          0.5,  0.5,  0.5,
        
//         -0.5, -0.5,  0.5, // bottom face verts [20-23]
//         -0.5, -0.5, -0.5,
//          0.5, -0.5, -0.5,
//          0.5, -0.5,  0.5
//     };
//     Positions = std::vector<float>(CubePositions, CubePositions + 24 * 3);
//     int i = 0;
//     for (std::vector<float>::iterator it = Positions.begin(); it != Positions.end(); ++ it, ++ i)
//         * it *= Size[i %= 3];

//     static float const CubeNormals[] =
//     {
//          0,  0, -1, // back face verts [0-3]
//          0,  0, -1,
//          0,  0, -1,
//          0,  0, -1,
         
//          0,  0,  1, // front face verts [4-7]
//          0,  0,  1,
//          0,  0,  1,
//          0,  0,  1,
         
//         -1,  0,  0, // left face verts [8-11]
//         -1,  0,  0,
//         -1,  0,  0,
//         -1,  0,  0,
        
//          1,  0,  0, // right face verts [12-15]
//          1,  0,  0,
//          1,  0,  0,
//          1,  0,  0,
        
//          0,  1,  0, // top face verts [16-19]
//          0,  1,  0,
//          0,  1,  0,
//          0,  1,  0,
        
//          0, -1,  0, // bottom face verts [20-23]
//          0, -1,  0,
//          0, -1,  0,
//          0, -1,  0
//     };
//     Normals = std::vector<float>(CubeNormals, CubeNormals + 24 * 3);

//     static unsigned short const CubeIndices[] =
//     {
//          0,  1,  2, // back face verts [0-3]
//          2,  3,  0,
         
//          4,  7,  6, // front face verts [4-7]
//          6,  5,  4,
         
//          8, 11, 10, // left face verts [8-11]
//         10,  9,  8,
         
//         12, 13, 14, // right face verts [12-15]
//         14, 15, 12,
         
//         16, 19, 18, // top face verts [16-19]
//         18, 17, 16,
         
//         20, 21, 22, // bottom face verts [20-23]
//         22, 23, 20
//     };
//     Indices = std::vector<unsigned short>(CubeIndices, CubeIndices + 12 * 3);

//     return new Mesh(Positions, Normals, Indices);
// }


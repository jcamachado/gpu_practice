#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "../objects/model.h"

typedef struct {
	glm::vec3 pos;
	glm::vec3 norm;
	glm::vec2 texCoord;
} SphereVertex;

SphereVertex normalizeSphereVertex(glm::vec3 pos, float phi, float th, std::vector<float> *vertexListFloat) {
	glm::vec2 texCoord;
	texCoord.x = th / glm::two_pi<float>();
	texCoord.y = (phi + glm::half_pi<float>()) / glm::pi<float>();
	vertexListFloat->push_back(pos.x);
	vertexListFloat->push_back(pos.y);
	vertexListFloat->push_back(pos.z);
	vertexListFloat->push_back(pos.x);
	vertexListFloat->push_back(pos.y);
	vertexListFloat->push_back(pos.z);
	vertexListFloat->push_back(texCoord.x);
	vertexListFloat->push_back(texCoord.y);

	return { pos, pos, texCoord };
}

class Sphere : public Model {
public:
    Sphere(unsigned int maxNoInstances)
        : Model("sphere", maxNoInstances, NO_TEX | DYNAMIC) {}
	
	std::vector<float> vertexListFloat;


    void init() {
		std::vector<SphereVertex> vertices;
		std::vector<GLuint> indices;

		// generate vertices
		unsigned int res = 100; // number of rows and columns
		float circleStep = glm::two_pi<float>() / (float)res; // angle step between cells
		float heightStep = glm::pi<float>() / (float)res; // height of row

		int row = 0;
		int noVertices = 0;
		float phi = -glm::half_pi<float>();
		float y = glm::sin(phi);
		float radius;

		for (; phi < glm::half_pi<float>() + heightStep; phi += heightStep, row++) {
			y = glm::sin(phi);
			radius = glm::cos(phi);
			int cell = 0;
			for (float th = 0; th < glm::two_pi<float>(); th += circleStep, cell++) {
				vertices.push_back(normalizeSphereVertex(
					glm::vec3(radius * glm::cos(th), y, radius * glm::sin(th)),
					phi, th, &vertexListFloat
				));

				// add indices if not bottom row
				if (row)
				{
					int nextCell = (cell + 1) % res;
					indices.push_back(noVertices - res); // bottom left
					indices.push_back((row - 1) * res + nextCell); // bottom right
					indices.push_back(row * res + nextCell); // top right

					indices.push_back(noVertices - res); // bottom left
					indices.push_back(noVertices); // top left (this vertex)
					indices.push_back(row * res + nextCell); // top right
				}

				noVertices++;
			}
		}

		BoundingRegion br(glm::vec3(0.0f), 1.0f);
		float* arr = new float[vertexListFloat.size()];
		std::copy(vertexListFloat.begin(), vertexListFloat.end(), arr);

		Mesh ret = processMesh(
			br,
			(unsigned int)vertices.size(), &vertexListFloat[0],
			(unsigned int)indices.size(), &indices[0],
			true,
			(unsigned int)vertices.size(), &vertexListFloat[0],
			(unsigned int)indices.size(), &indices[0],
			true
		);
		// Mesh ret = processMesh(br,
		// 	(unsigned int)vertices.size(), &vertices[0].pos[0],
		// 	(unsigned int)indices.size(), &indices[0],
		// 	true);
		// colors
		ret.setupMaterial(Material::white_plastic);
		
		addMesh(&ret);

    }
};

#endif

/*

#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "../objects/model.h"

// typedef struct {
// 	glm::vec3 pos;
// 	glm::vec3 norm;
// 	glm::vec2 texCoord;
// } SphereVertex;

// std::vector<float> normalizeSphereVertex(glm::vec3 pos, float phi, float th) {
// 	glm::vec2 texCoord;
// 	texCoord.x = th / glm::two_pi<float>();
// 	texCoord.y = (phi + glm::half_pi<float>()) / glm::pi<float>();
// 	std::vector<float> ret = { pos.x, pos.y, pos.z, 
// 		pos.x, pos.y, pos.z, 
// 		texCoord.x, texCoord.y };
// 	return ret;
// }

Vertex normalizeSphereVertex(glm::vec3 pos, float phi, float th) {
	glm::vec2 texCoord;
	texCoord.x = th / glm::two_pi<float>();
	texCoord.y = (phi + glm::half_pi<float>()) / glm::pi<float>();
	return Vertex{ pos, pos, texCoord, pos};
}

class Sphere : public Model {
public:
    Sphere(unsigned int maxNoInstances)
        : Model("sphere", maxNoInstances, NO_TEX | DYNAMIC) {}

    void init() {
		std::vector<Vertex> vertices;		
		std::vector<unsigned int> indices;

		// generate vertices
		unsigned int res = 100; // number of rows and columns
		float circleStep = glm::two_pi<float>() / (float)res; // angle step between cells
		float heightStep = glm::pi<float>() / (float)res; // height of row

		int row = 0;
		int nVertices = 0;
		float phi = -glm::half_pi<float>();
		float y = glm::sin(phi);
		float radius;

		for (; phi < glm::half_pi<float>() + heightStep; phi += heightStep, row++) {
			y = glm::sin(phi);
			radius = glm::cos(phi);
			int cell = 0;
			for (float th = 0; th < glm::two_pi<float>(); th += circleStep, cell++) {
				vertices.push_back(normalizeSphereVertex(
					glm::vec3(radius * glm::cos(th), y, radius * glm::sin(th)),
					phi, th
				));

				// add indices if not bottom row
				if (row)
				{
					int nextCell = (cell + 1) % res;
					indices.push_back(nVertices - res); // bottom left
					indices.push_back((row - 1) * res + nextCell); // bottom right
					indices.push_back(row * res + nextCell); // top right

					indices.push_back(nVertices - res); // bottom left
					indices.push_back(nVertices); // top left (this vertex)
					indices.push_back(row * res + nextCell); // top right
				}

				nVertices++;
			}
		}
		std::vector<float> vertexList;	// list of the above vertices but in a float array
		vertexList = vertexArrayToFloatArray(vertices, true);	
		BoundingRegion br(glm::vec3(0.0f), 1.0f);

		// Mesh ret = processMesh(br,
		// 	(unsigned int)vertices.size(), &vertices[0].pos[0],
		// 	(unsigned int)indices.size(), &indices[0],
		// 	true);
		Mesh ret = processMesh(
			br,
			(unsigned int)vertices.size(), &vertexList[0],
			(unsigned int)indices.size(), &indices[0],
			true,
			(unsigned int)vertices.size(), &vertexList[0],
			(unsigned int)indices.size(), &indices[0],
			true
		);

		// colors
		ret.setupMaterial(Material::white_plastic);
		
		addMesh(&ret);

    }

	std::vector<float> vertexToFloatArray(Vertex v){
		return {
			v.pos.x, v.pos.y, v.pos.z, 
			v.normal.x, v.normal.y, v.normal.z,
			v.texCoord.x, v.texCoord.y,
			v.tangent.x, v.tangent.y, v.tangent.z
		};
	}
	std::vector<float> vertexArrayToFloatArray(std::vector<Vertex> vertices, bool simplified = false ){
		std::vector<float> v = {};
		int step = 1;
		if (simplified){
			step = 300;
		}
		for (int i = 0; i < vertices.size(); i+=step){
			std::vector<float> temp = vertexToFloatArray(vertices[i]);
			v.insert(v.end(), temp.begin(), temp.end());
		}
		return v;
	}
};

#endif
*/
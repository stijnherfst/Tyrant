#pragma once

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;

	Vertex() = default;
	Vertex(float x, float y, float z, float nx, float ny, float nz) : position( x, y, z ), normal(glm::vec3( nx, ny, nz )) {}
};

struct Triangle {
	unsigned index1;
	unsigned index2;
	unsigned index3;
	glm::vec3 color;
	glm::vec3 normal;
	bool fake;
	float _d, _d1, _d2, _d3;
	glm::vec3 _e1, _e2, _e3;
	glm::vec3 bottom;
	glm::vec3 top;
};

struct CudaMesh {
	unsigned verticesNo = 0;
	unsigned trianglesNo = 0;
	Vertex* vertices = nullptr;
	Triangle* triangles = nullptr;
};

struct BVHNode;
struct CacheFriendlyBVHNode;

extern BVHNode* bvh;
extern unsigned triangle_index_list_no;
extern int* triangle_index_list;
extern unsigned cache_bvh_no;
extern CacheFriendlyBVHNode* cache_bvh;

CudaMesh load_object(const std::string& source);
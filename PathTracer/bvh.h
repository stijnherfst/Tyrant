#pragma once

// The nice version of the BVH - a shallow hierarchy of inner and leaf nodes
struct BVHNode {
	glm::vec3 bottom;
	glm::vec3 top;
	virtual bool IsLeaf() = 0;
};

struct BVHInner : BVHNode {
	BVHNode* left;
	BVHNode* right;
	virtual bool IsLeaf() { 
		return false; 
	}
};

struct BVHLeaf : BVHNode {
	std::list<const Triangle*> triangles;
	virtual bool IsLeaf() { 
		return true; 
	}
};

struct AABB {
	glm::vec3 bottom;
	glm::vec3 top;

	glm::vec3 center;
	const Triangle* triangle_list;
	AABB() : bottom({ FLT_MAX, FLT_MAX, FLT_MAX }), top({ -FLT_MAX, -FLT_MAX, -FLT_MAX }), triangle_list(nullptr) {}
};

struct CacheFriendlyBVHNode {
	// AABB
	glm::vec3 bottom;
	glm::vec3 top;

	// parameters for leafnodes and innernodes occupy same space (union) to save memory
	// top bit discriminates between leafnode and innernode
	// no pointers, but indices (int): faster

	union {
		// inner node - stores indexes to array of CacheFriendlyBVHNode
		struct {
			unsigned idx_left;
			unsigned idx_right;
		} inner;
		// leaf node: stores triangle count and starting index in triangle list
		struct {
			unsigned count; // Top-most bit set, leafnode if set, innernode otherwise
			unsigned start_index_tri_list;
		} leaf;
	} u;
};

void CreateCFBVH(CudaMesh cuda_mesh);
void update_bvh(CudaMesh cuda_mesh, std::string filename);
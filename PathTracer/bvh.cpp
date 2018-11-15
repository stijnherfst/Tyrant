#include "stdafx.h"

BVHNode* bvh = nullptr;

unsigned triangle_index_list_no = 0;
int* triangle_index_list = nullptr;
unsigned cache_bvh_no = 0;
CacheFriendlyBVHNode* cache_bvh = nullptr;

// Based on http://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf
BVHNode* Recurse(std::vector<AABB>& work, int depth = 0) {
	if (work.size() < 4) {
		BVHLeaf *leaf = new BVHLeaf;
		for (auto&& i : work) {
			leaf->triangles.push_back(i.triangle_list);
		}

		return leaf;
	}

	glm::vec3 bottom = { FLT_MAX, FLT_MAX, FLT_MAX };
	glm::vec3 top = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

	for (auto&& i : work) {
		bottom = glm::min(bottom, i.bottom);
		top = glm::max(top, i.top);
	}

	// SAH, surface area heuristic calculation
	float side1 = top.x - bottom.x;
	float side2 = top.y - bottom.y;
	float side3 = top.z - bottom.z;

	// The current bbox has a cost of (number of triangles) * surfaceArea of C = N * SA
	float minimal_cost = work.size() * (side1 * side2 + side2 * side3 + side3 * side1);

	float best_split = FLT_MAX; // Best split along axis, will indicate no split with better cost found (below)

	int best_axis = -1;
	for (int axis = 0; axis < 3; axis++) {  // X, Y, Z
		// wDivide triangles based on current axis,
		float start = bottom[axis];
		float stop = top[axis];

		// In that axis, do the bounding boxes in the work queue "span" across, (meaning distributed over a reasonable distance)?
		// Or are they all already "packed" on the axis? Meaning that they are too close to each other
		if (abs(stop - start) < 1e-4)
			continue;

		// Binning: Try splitting at a uniform sampling (at equidistantly spaced planes) that gets smaller the deeper we go:
		float step = (stop - start) / (1024.f / (depth + 1.f));

		// For each bin (equally spaced bins of size "step"):
		for (float test_split = start + step; test_split < stop - step; test_split += step) {
			// Create left and right bounding box
			glm::vec3 lbottom = glm::vec3(FLT_MAX);
			glm::vec3 ltop = glm::vec3(-FLT_MAX);

			glm::vec3 rbottom = glm::vec3(FLT_MAX);
			glm::vec3 rtop = glm::vec3(-FLT_MAX);

			// Triangle count for left and right AABB
			int count_left = 0, count_right = 0;

			// Allocate triangles in remaining work list based on AABB centers
			for (auto&& i : work) {
				if (i.center[axis] < test_split) {
					lbottom = glm::min(lbottom, i.bottom);
					ltop = glm::max(ltop, i.top);
					count_left++;
				} else {
					rbottom = glm::min(rbottom, i.bottom);
					rtop = glm::max(rtop, i.top);
					count_right++;
				}
			}

			if (count_left <= 1 || count_right <= 1) {
				continue;
			}

			// Calculate surface areas
			float lside1 = ltop.x - lbottom.x;
			float lside2 = ltop.y - lbottom.y;
			float lside3 = ltop.z - lbottom.z;

			float rside1 = rtop.x - rbottom.x;
			float rside2 = rtop.y - rbottom.y;
			float rside3 = rtop.z - rbottom.z;

			// Calculate surface area
			float surface_left = lside1 * lside2 + lside2 * lside3 + lside3 * lside1;
			float surface_right = rside1 * rside2 + rside2 * rside3 + rside3 * rside1;

			// Calculate total cost
			float total_cost = surface_left * count_left + surface_right * count_right;

			// Track cheapest split
			if (total_cost < minimal_cost) {
				minimal_cost = total_cost;
				best_split = test_split;
				best_axis = axis;
			}
		}
	}

	if (best_axis == -1) {
		BVHLeaf *leaf = new BVHLeaf;
		for (auto&& i : work) {
			leaf->triangles.push_back(i.triangle_list);
		}
		return leaf;
	}

	// Otherwise, create BVH inner node with L and R child nodes, split with the optimal value we found above
	std::vector<AABB> left;
	std::vector<AABB> right;
	glm::vec3 lbottom = { FLT_MAX, FLT_MAX, FLT_MAX };
	glm::vec3 ltop = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	glm::vec3 rbottom = { FLT_MAX, FLT_MAX, FLT_MAX };
	glm::vec3 rtop = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

	// Bounding box center
	for (auto&& i : work) {
		if (i.center[best_axis] < best_split) {
			left.push_back(i);
			lbottom = glm::min(lbottom, i.bottom);
			ltop = glm::max(ltop, i.top);
		} else {
			right.push_back(i);
			rbottom = glm::min(rbottom, i.bottom);
			rtop = glm::max(rtop, i.top);
		}
	}

	// Create inner node
	BVHInner* inner = new BVHInner;

	inner->left = Recurse(left, depth + 1);
	inner->left->bottom = lbottom;
	inner->left->top = ltop;

	inner->right = Recurse(right, depth + 1);
	inner->right->bottom = rbottom;
	inner->right->top = rtop;

	return inner;
}

BVHNode* CreateBVH(CudaMesh cuda_mesh) {
	std::vector<AABB> work;
	glm::vec3 bottom = { FLT_MAX, FLT_MAX, FLT_MAX };
	glm::vec3 top = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

	for (unsigned j = 0; j < cuda_mesh.trianglesNo; j++) {
		const Triangle& triangle = cuda_mesh.triangles[j];

		AABB b;
		b.triangle_list = &triangle;

		// Pick smallest vertex for bottom of triangle bbox
		b.bottom = glm::min(b.bottom, cuda_mesh.vertices[triangle.index1].position);
		b.bottom = glm::min(b.bottom, cuda_mesh.vertices[triangle.index2].position);
		b.bottom = glm::min(b.bottom, cuda_mesh.vertices[triangle.index3].position);

		// Pick largest vertex for top of triangle bbox
		b.top = glm::max(b.top, cuda_mesh.vertices[triangle.index1].position);
		b.top = glm::max(b.top, cuda_mesh.vertices[triangle.index2].position);
		b.top = glm::max(b.top, cuda_mesh.vertices[triangle.index3].position);

		bottom = glm::min(bottom, b.bottom);
		top = glm::max(top, b.top);

		// Compute triangle bbox center
		b.center = (b.top + b.bottom) * 0.5f;

		work.push_back(b);
	}

	// Builds BVH and returns root node
	BVHNode* root = Recurse(work); 

	root->bottom = bottom;
	root->top = top;

	return root;
}

int count_boxes(BVHNode *root) {
	if (!root->IsLeaf()) {
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		return 1 + count_boxes(p->left) + count_boxes(p->right);
	} else
		return 1;
}

size_t count_triangles(BVHNode *root) {
	if (!root->IsLeaf()) {
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		return count_triangles(p->left) + count_triangles(p->right);
	} else {
		return dynamic_cast<BVHLeaf*>(root)->triangles.size();
	}
}

// recursively count depth
void calculate_max_depth(BVHNode *root, int depth, int& maxDepth) {
	if (maxDepth < depth)
		maxDepth = depth;
	if (!root->IsLeaf()) {
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		calculate_max_depth(p->left, depth + 1, maxDepth);
		calculate_max_depth(p->right, depth + 1, maxDepth);
	}
}

void create_cache_bvh(const Triangle *pFirstTriangle, BVHNode *root, unsigned& idxBoxes, unsigned &idxTriList) {
	unsigned currIdxBoxes = idxBoxes;
	cache_bvh[currIdxBoxes].bottom = root->bottom;
	cache_bvh[currIdxBoxes].top = root->top;

	// Depth first approach
	if (!root->IsLeaf()) { // inner node
		BVHInner *p = dynamic_cast<BVHInner*>(root);

		// Recursively populate left and right
		int index_left = ++idxBoxes;
		create_cache_bvh(pFirstTriangle, p->left, idxBoxes, idxTriList);
		int index_right = ++idxBoxes;
		create_cache_bvh(pFirstTriangle, p->right, idxBoxes, idxTriList);
		cache_bvh[currIdxBoxes].u.inner.idx_left = index_left;
		cache_bvh[currIdxBoxes].u.inner.idx_right = index_right;
	} else { // Leaf
		BVHLeaf *p = dynamic_cast<BVHLeaf*>(root);
		unsigned count = (unsigned)p->triangles.size();
		cache_bvh[currIdxBoxes].u.leaf.count = 0x80000000 | count;  // Highest bit set indicates leaf node, otherwise inner node
		cache_bvh[currIdxBoxes].u.leaf.start_index_tri_list = idxTriList;

		for (auto&& i : p->triangles) {
			triangle_index_list[idxTriList++] = i - pFirstTriangle;
		}
	}
}

void CreateCFBVH(CudaMesh cuda_mesh) {
	if (!bvh) {
		std::cout << "CreateCFBVH failed\n";
		exit(EXIT_FAILURE);
	}

	unsigned idxTriList = 0;
	unsigned idxBoxes = 0;

	triangle_index_list_no = count_triangles(bvh);
	triangle_index_list = new int[triangle_index_list_no];

	cache_bvh_no = count_boxes(bvh);
	cache_bvh = new CacheFriendlyBVHNode[cache_bvh_no];

	create_cache_bvh(&cuda_mesh.triangles[0], bvh, idxBoxes, idxTriList);

	if ((idxBoxes != cache_bvh_no - 1) || (idxTriList != triangle_index_list_no)) {
		std::cout << "CreateCFBVH failed\n";
		exit(EXIT_FAILURE);
	}

	int maxDepth = 0;
	calculate_max_depth(bvh, 0, maxDepth);
	if (maxDepth >= bvh_stack_size) {
		std::cout << "Increase bvh_stack_size\n";
		exit(EXIT_FAILURE);
	}
}

void update_bvh(CudaMesh cuda_mesh, std::string filename) {
	if (!bvh) {
		filename += ".bvh";
		FILE* fp = fopen(filename.c_str(), "rb");
		if (!fp) { // Create BVH
			bvh = CreateBVH(cuda_mesh);
			CreateCFBVH(cuda_mesh);

			fp = fopen(filename.c_str(), "wb");
			if (!fp)
				return;
			if (1 != fwrite(&cache_bvh_no, sizeof(unsigned), 1, fp)) 
				return;
			if (1 != fwrite(&triangle_index_list_no, sizeof(unsigned), 1, fp)) 
				return;
			if (cache_bvh_no != fwrite(cache_bvh, sizeof(CacheFriendlyBVHNode), cache_bvh_no, fp)) 
				return;
			if (triangle_index_list_no != fwrite(triangle_index_list, sizeof(int), triangle_index_list_no, fp)) 
				return;

			fclose(fp);
		} else { // Read existing BVH
			if (1 != fread(&cache_bvh_no, sizeof(unsigned), 1, fp)) 
				return;
			if (1 != fread(&triangle_index_list_no, sizeof(unsigned), 1, fp)) 
				return;

			cache_bvh = new CacheFriendlyBVHNode[cache_bvh_no];
			triangle_index_list = new int[triangle_index_list_no];

			if (cache_bvh_no != fread(cache_bvh, sizeof(CacheFriendlyBVHNode), cache_bvh_no, fp)) 
				return;
			if (triangle_index_list_no != fread(triangle_index_list, sizeof(int), triangle_index_list_no, fp)) 
				return;
			fclose(fp);
		}
	}
}

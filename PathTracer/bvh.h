#pragma once

#include "BBox.h"
#include "loader.h"
#include <algorithm>
#include <vector>

enum class PartitionAlgorithm { Middle,
								EqualCounts,
								SAH };
constexpr uint32_t MAX_PRIMITIVES_EqualCounts = 1;
//constexpr uint32_t MAX_PRIMITIVES_SAH;

class BVH {
public:
	BVH(std::vector<Triangle>& primitives, std::vector<BBox> primitivesBBoxes,
		PartitionAlgorithm partitionAlgo);
	struct BVHNode {
		void InitLeaf(int first, int n, const BBox& box);
		void initInterior(int axis, BVHNode* left, BVHNode* right);
		BBox bbox;
		BVHNode* children[2];
		int splitAxis, firstPrimOffset, nPrimitives;
	} * root;
	/*TODO: Implement destructor.*/
	~BVH() {}
	int nNodes = 0;
	const PartitionAlgorithm partitionAlgorithm;

private:
	//Number of buckets to split up axis for SAH
	static constexpr int bucket_number = 12;
	//Maximum number of primitives allowed in BVH leaf node for SAH
	static constexpr int max_prim_number = 4;

	//cost to traverse a node in the BVH. If INTERSECTION_COST is 1 then this is just percentage slower/faster
	static constexpr float TRAVERSAL_COST = 1.0f;

	//cost to intersect a triangle int the BVH. Leave this as "1" and change TRAVERSAL_COST instead
	static constexpr float INTERSECTION_COST = 1.0f;

	static_assert(INTERSECTION_COST == 1, "You can vary traversal_cost instead of intersection_cost");
	static_assert(bucket_number >= 2, "Buckets should be enough to split the space! At least 2 required");

	struct PrimitiveInfo {
		uint32_t primitiveNumber = {};
		BBox bbox = {};
		glm::vec3 centroid = {};
		PrimitiveInfo() = default;
		PrimitiveInfo(uint32_t primitiveNumber, BBox& bbox)
			: primitiveNumber(primitiveNumber)
			, bbox(bbox)
			, centroid(bbox.bounds[0] * 0.5f + bbox.bounds[1] * 0.5f) {}
	};

	//Calculate the bucket in which a primitive is to be placed for the SAH algorithm.
	//Bucket is computed by evenly splitting along a dimension the intervals in the centroid bounding box
	int computeBucket(PrimitiveInfo primitive, glm::vec3 centroidBottom, glm::vec3 centroidTop, int dim);

	BVHNode* recursiveBuild(int start, int end, int& nNodes,
							std::vector<PrimitiveInfo>& primitiveInfo,
							std::vector<Triangle>& orderedPrimitives,
							const std::vector<Triangle>& primitives);
};

class CachedBVH {
public:
	CachedBVH() = default;
	CachedBVH(int totalNodes, std::vector<Triangle> prims) {
		nodes = new CachedBVHNode[totalNodes];
	}
	struct CachedBVHNode {
		BBox bbox;
		union {
			int primitiveOffset;
			int secondChildOffset;
		};
		uint16_t primitiveCount;
		uint8_t axis;
		char pad[1];
	};
	static_assert(sizeof(CachedBVHNode) == 32, "Size is not correct");

	CachedBVHNode* nodes = nullptr;
	Triangle* primitives;

	__host__ int buildCachedBVH(BVH::BVHNode* node, int& offset);

	__device__ bool intersect(const Ray& ray, float& closestIntersection, int& primitiveIndex) {
		if (nodes == nullptr) {
			return false;
		}

		bool hit = false;
		glm::vec3 invDir = 1.f / ray.dir;
		int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
		// Follow ray through BVH nodes to find primitive intersections
		int toVisitOffset = 0, currentNodeIndex = 0;
		int nodesToVisit[64];
		while (true) {
			const CachedBVHNode* node = &nodes[currentNodeIndex];
			// Check ray against BVH node
			if (node->bbox.intersect(ray, invDir, dirIsNeg, closestIntersection)) {
				if (node->primitiveCount > 0) {
					// LEAF
					for (int i = 0; i < node->primitiveCount; ++i) {
						float intersection = primitives[node->primitiveOffset + i].intersect(ray);
						if (intersection > 0.00001f && intersection < closestIntersection && ((closestIntersection - intersection) > 0.00001f)) {
							primitiveIndex = node->primitiveOffset + i;
							closestIntersection = intersection;
							hit = true;
						}
					}
					if (toVisitOffset == 0)
						break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
				} else {
					// Choose which one to visit by looking at ray direction
					if (dirIsNeg[node->axis]) {
						nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
						currentNodeIndex = node->secondChildOffset;
					} else {
						nodesToVisit[toVisitOffset++] = node->secondChildOffset;
						currentNodeIndex = currentNodeIndex + 1;
					}
				}
			} else {
				if (toVisitOffset == 0)
					break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
		}
		return hit;
	}
};

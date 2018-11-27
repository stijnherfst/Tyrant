#pragma once

//#include <vector>
//#include "stdafx.h"

#include "BBox.h"
#include "loader.h"
#include <algorithm>
#include <vector>

enum class PartitionAlgorithm { Middle,
								EqualCounts,
								SAH };

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
	/*TODO: Implement this.*/
	~BVH() {}
	int nNodes = {};
	const PartitionAlgorithm partitionAlgorithm;

private:
	struct PrimitiveInfo {
		uint32_t primitiveNumber{};
		BBox bbox{};
		glm::vec3 centroid{};
		PrimitiveInfo() = default;
		PrimitiveInfo(uint32_t primitiveNumber, BBox& bbox)
			: primitiveNumber(primitiveNumber)
			, bbox(bbox)
			, centroid(bbox.bounds[0] * 0.5f + bbox.bounds[1] * 0.5f) {}
	};

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
						if (intersection > epsilon && intersection < closestIntersection && ((closestIntersection - intersection) > epsilon)) {
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

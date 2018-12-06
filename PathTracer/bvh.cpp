#include "stdafx.h"

BVH::BVH(std::vector<Triangle>& primitives, std::vector<BBox> primitivesBBoxes,
		 PartitionAlgorithm partitionAlgo)
	: partitionAlgorithm(partitionAlgo) {
	std::vector<PrimitiveInfo> primitiveInfo(primitives.size());
	for (size_t i = 0; i < primitives.size(); ++i) {
		primitiveInfo[i] = PrimitiveInfo(i, primitivesBBoxes[i]);
	}

	std::vector<Triangle> orderedPrimitives;
	orderedPrimitives.reserve(primitives.size());

	root = recursiveBuild(0, primitives.size(), nNodes, primitiveInfo,
						  orderedPrimitives, primitives);
	primitives = orderedPrimitives;
}

BVH::BVHNode* BVH::recursiveBuild(int start, int end, int& nNodes,
								  std::vector<PrimitiveInfo>& primitiveInfo,
								  std::vector<Triangle>& orderedPrimitives,
								  const std::vector<Triangle>& primitives) {
	assert(start != end && "Start == END");
	auto node = new BVHNode();
	++nNodes;
	/*Compute bbox for this node*/
	BBox nodeBBox;

	for (int i = start; i < end; ++i) {
		nodeBBox = Union(nodeBBox, primitiveInfo[i].bbox);
	}

	int nPrimitives = end - start;
	if (nPrimitives == 1) {
		/*Make leaf*/
		auto firstPrimOffset = orderedPrimitives.size();
		for (auto i = start; i < end; ++i) {
			int primitiveNumber = primitiveInfo[i].primitiveNumber;
			orderedPrimitives.push_back(primitives[primitiveNumber]);
		}
		node->InitLeaf(firstPrimOffset, nPrimitives, nodeBBox);
		return node;
	}
	/* Not leaf, get centroid bounds*/
	BBox centroidBBox;
	for (auto i = start; i < end; ++i)
		centroidBBox.addVertex(primitiveInfo[i].centroid);

	int dim = centroidBBox.largestExtent();

	/* Partition primitives into two sets and build children */
	int mid = (start + end) / 2;

	/*
          Avoid union type punning. Need float3 as arrray to handle "dim"
     without 'if' cases. Makes the code shorter.
  */
	float bottom[3];
	float top[3];
	memcpy(bottom, &centroidBBox.bounds[0], sizeof(float3));
	memcpy(top, &centroidBBox.bounds[1], sizeof(float3));

	/*
          Handle case of stacked bbboxes with same centroid
  */
	if (bottom[dim] == top[dim]) {
		// Create leaf _BVHBuildNode_
		int firstPrimOffset = orderedPrimitives.size();
		for (int i = start; i < end; ++i) {
			int primNum = primitiveInfo[i].primitiveNumber;
			orderedPrimitives.push_back(primitives[primNum]);
		}
		node->InitLeaf(firstPrimOffset, nPrimitives, nodeBBox);
		return node;
	} else {
		// Partition primitives based on _splitMethod_
		switch (partitionAlgorithm) {
		case PartitionAlgorithm::EqualCounts: {
			// Partition primitives into equally-sized subsets
			mid = (start + end) / 2;
			std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
							 &primitiveInfo[end - 1] + 1,
							 [dim](const PrimitiveInfo& a, const PrimitiveInfo& b) {
								 /*Not type punning through union, using this
                            * instead*/
								 float aVector[3];
								 float bVector[3];
								 memcpy(aVector, &a.centroid, sizeof(float3));
								 memcpy(bVector, &b.centroid, sizeof(float3));

								 return aVector[dim] < bVector[dim];
							 });
			break;
		}
		}
		node->initInterior(dim,
						   recursiveBuild(start, mid, nNodes, primitiveInfo,
										  orderedPrimitives, primitives),
						   recursiveBuild(mid, end, nNodes, primitiveInfo,
										  orderedPrimitives, primitives));
	}
	return node;
}

void BVH::BVHNode::InitLeaf(int first, int n, const BBox& box) {
	firstPrimOffset = first;
	bbox = box;
	children[0] = children[1] = nullptr;
	nPrimitives = n;
}

inline void BVH::BVHNode::initInterior(int axis, BVHNode* left,
									   BVHNode* right) {
	bbox = Union(left->bbox, right->bbox);
	children[0] = left;
	children[1] = right;
	nPrimitives = 0;
	splitAxis = axis;
}

int CachedBVH::buildCachedBVH(BVH::BVHNode* node, int& offset) {
	CachedBVHNode* cachedNode = &nodes[offset];

	int myOffset = offset; // Used in case it's second son of any node
	++offset;

	cachedNode->bbox = node->bbox;
	if (node->nPrimitives > 0) { // LEAF
		cachedNode->primitiveCount = node->nPrimitives;
		cachedNode->secondChildOffset = node->firstPrimOffset;
	} else { // INTERIOR
		cachedNode->primitiveCount = 0;
		cachedNode->axis = node->splitAxis;
		buildCachedBVH(node->children[0], offset);
		cachedNode->secondChildOffset = buildCachedBVH(node->children[1], offset);
	}
	return myOffset;
}


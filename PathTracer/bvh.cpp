#include "BVH.h"
#include "loader.h"
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>

BVH::BVH(std::vector<Triangle>& primitives, std::vector<BBox> primitivesBBoxes,
		 PartitionAlgorithm partitionAlgo)
	: partitionAlgorithm(partitionAlgo) {

	std::cout << "Creating BVH, total primitives: " << primitives.size() << "\n";
	if (primitives.size() == 0) {
		return;
	}

	std::vector<PrimitiveInfo> primitiveInfo(primitives.size());
	for (size_t i = 0; i < primitives.size(); ++i) {
		primitiveInfo[i] = PrimitiveInfo(i, primitivesBBoxes[i]);
	}

	//We'll hold contiguous leaves next to one another in orderedPrimitives
	std::vector<Triangle> orderedPrimitives;
	orderedPrimitives.reserve(primitives.size());

	root = recursiveBuild(0, primitives.size(), nNodes, primitiveInfo,
						  orderedPrimitives, primitives);
	primitives = orderedPrimitives;
	std::cout << "Created BVH, total nodes : " << nNodes << "\n";
}

int BVH::computeBucket(PrimitiveInfo primitive, glm::vec3 centroidBottom, glm::vec3 centroidTop, int dim) {
	//Get the distance from the start of the split in the axis;
	float distance = primitive.centroid[dim] - centroidBottom[dim];
	//Normalize the distance
	if (centroidTop[dim] > centroidBottom[dim]) {
		//TODO(Dan): Is this 'if' needed? 
		//Normalize to [0,1]
		distance = distance / (centroidTop[dim] - centroidBottom[dim]);
	}
	int bucket_idx = (int)(bucket_number * distance);
	if (bucket_idx == bucket_number) { //Can only happen if last primitive in axis has bottom == top
		bucket_idx--;
	}
	return bucket_idx;
}

//Recursively build a BVH node
BVH::BVHNode* BVH::recursiveBuild(int start, int end, int& nNodes,
								  std::vector<PrimitiveInfo>& primitiveInfo,
								  std::vector<Triangle>& orderedPrimitives,
								  const std::vector<Triangle>& primitives) {
	assert(start != end && "Start == END recursive build");

	//TODO(Dan): Use memory pool !!!
	auto node = new BVHNode();
	++nNodes;
	/*Compute bbox for this node*/

	BBox nodeBBox = {};
	for (int i = start; i < end; ++i) {
		nodeBBox = Union(nodeBBox, primitiveInfo[i].bbox);
	}

	int nPrimitives = end - start;

	if (nPrimitives == 1) { // LEAF
		int firstPrimOffset = orderedPrimitives.size();
		for (int i = start; i < end; ++i) {
			int primitiveNumber = primitiveInfo[i].primitiveNumber;
			orderedPrimitives.push_back(primitives[primitiveNumber]);
		}
		node->InitLeaf(firstPrimOffset, nPrimitives, nodeBBox);
		return node;
	}

	/* Not leaf, get centroid bounds*/
	BBox centroidBBox;
	for (int i = start; i < end; ++i)
		centroidBBox.addVertex(primitiveInfo[i].centroid);

	/* We split based on the largest axis*/
	int dim = centroidBBox.largestExtent();

	// Partition primitives into equally-sized subsets
	int mid = (start + end) / 2;

	const glm::vec3(&centroidBottom) = centroidBBox.bounds[0];
	const glm::vec3(&centroidTop) = centroidBBox.bounds[1];

	/* Handle case of stacked bbboxes with same centroid*/
	if (centroidBottom[dim] == centroidTop[dim]) {
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

			std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
							 &primitiveInfo[end - 1] + 1,
							 [dim](const PrimitiveInfo& a, const PrimitiveInfo& b) {
								 return a.centroid[dim] < b.centroid[dim];
							 });
			break;
		}
		case PartitionAlgorithm::SAH: {
			//Init buckets for binned SAH
			struct Bucket {
				int count = { 0 };
				BBox bounds;
			};
			Bucket buckets[bucket_number] = {};

			//Place all the primitives in the buckets
			glm::vec3 splitVector = centroidBBox.bounds[1] - centroidBBox.bounds[0];
			for (int i = start; i < end; ++i) {
				int bucket_idx = computeBucket(primitiveInfo[i], centroidBottom, centroidTop, dim);
				buckets[bucket_idx].count++;
				buckets[bucket_idx].bounds = Union(buckets[bucket_idx].bounds, primitiveInfo[i].bbox);
			}
			//Determine cost after splitting at each bucket
			//Split is [0,currentBucket] and (currentBucket, bucket_number - 2].
			//Splitting at last bucket would result in no actual split

			float cost[bucket_number] = {};
			float min_split_cost = FLT_MAX;
			int min_split_bucket = -1;
			for (int current_bucket = 0; current_bucket < bucket_number - 1; ++current_bucket) {
				int count_first_interval = 0;
				int count_second_interval = 0;
				BBox bbox_first_interval = {};
				BBox bbox_second_interval = {};

				//[0,current_bucket]
				for (int i = 0; i <= current_bucket; ++i) {
					bbox_first_interval = Union(bbox_first_interval, buckets[i].bounds);
					count_first_interval += buckets[i].count;
				}
				//(current_bucket, bucket_number - 1]
				for (int i = current_bucket + 1; i < bucket_number; ++i) {
					bbox_second_interval = Union(bbox_second_interval, buckets[i].bounds);
					count_second_interval += buckets[i].count;
				}
				//Compute SAH cost
				cost[current_bucket] = TRAVERSAL_COST + (count_first_interval * bbox_first_interval.surfaceArea() + count_second_interval * bbox_second_interval.surfaceArea()) / nodeBBox.surfaceArea();
				//Update min cost
				if (cost[current_bucket] < min_split_cost) {
					min_split_cost = cost[current_bucket];
					min_split_bucket = current_bucket;
				}
			}
			assert(min_split_bucket != -1);

			float leaf_cost = INTERSECTION_COST * nPrimitives;
			if (nPrimitives > max_prim_number || min_split_cost < leaf_cost) {
				PrimitiveInfo* pmid = std::partition(&primitiveInfo[start],
													 &primitiveInfo[end - 1] + 1,
													 [=](const PrimitiveInfo& pi) {
														 int bucketIndex = computeBucket(pi, centroidBottom, centroidTop, dim);
														 return bucketIndex <= min_split_bucket;
													 });
				//TODO(Dan): Is this technically undefined?
				mid = pmid - &primitiveInfo[0];
			} else {
				int firstPrimOffset = orderedPrimitives.size();
				for (int i = start; i < end; ++i) {
					int primNum = primitiveInfo[i].primitiveNumber;
					orderedPrimitives.push_back(primitives[primNum]);
				}
				node->InitLeaf(firstPrimOffset, nPrimitives, nodeBBox);
				return node;
			}
			break;
		}
		default: {
			std::cerr << "Error! No Valid partition algorithm selected!\n";
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

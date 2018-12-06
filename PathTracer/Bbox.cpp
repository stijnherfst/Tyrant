#include "stdafx.h"

__host__ BBox Union(const BBox& b1, const BBox& b2) {
	BBox temp = b1;
	temp.addVertex(b2.bounds[0]);
	temp.addVertex(b2.bounds[1]);
	return temp;
}

#include "stdafx.h"

void Scene::Load(const char path[]) {
	const aiScene* importer_scene = importer.ReadFile(
		path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices);
	StaticMesh mesh;
	mesh.load(importer_scene);
	for (auto& vertex : mesh.vertices) {
		glm::vec3 tempVertex = glm::vec3{ vertex.x, vertex.z, vertex.y };
		vertex = tempVertex;
	}
	//scale
	for (auto& vertex : mesh.vertices) {
		vertex = vertex * glm::vec3{ 1, 1, 1 };
	}
	meshes.push_back(mesh);

	/*Compute the primitives and bounding boxes needed for BVH from the meshes*/;
	for (const auto& mesh : meshes) {
		for (const auto& face : mesh.faces) {
			Triangle tempTriangle;
			glm::vec3 vertexes[3];
			BBox bbox;
			/*Load actual vertexes from mesh*/
			vertexes[0] = mesh.vertices[face.x];
			vertexes[1] = mesh.vertices[face.y];
			vertexes[2] = mesh.vertices[face.z];
			/*Compute bbox*/
			for (const auto& vertex : vertexes) {
				bbox.addVertex(vertex);
			}

			primitiveBBoxes.push_back(bbox);
			/*Get primitive in the format we like*/
			tempTriangle.vert = vertexes[0];
			tempTriangle.e1 = vertexes[1] - vertexes[0];
			tempTriangle.e2 = vertexes[2] - vertexes[0];
			// tempTriangle.materialType = mesh.reflectiveType;
			// tempTriangle.color = mesh.color;
			primitives.push_back(tempTriangle);
		}
	}

	if (primitives.size() == 0) {
		// TODO(Dan): Log error
		// Info("No primitives found in scene, loading scene without any");
		return;
	}
	BVH bvh(primitives, primitiveBBoxes, PartitionAlgorithm::EqualCounts);

	CachedBVH cachedBVH(bvh.nNodes, primitives);

	int offset = 0;
	cachedBVH.buildCachedBVH(bvh.root, offset);

	cuda(Malloc(&cachedBVH.primitives, primitives.size() * sizeof(Triangle)));
	cuda(Memcpy(&cachedBVH.primitives[0], &primitives[0],
				primitives.size() * sizeof(Triangle),
				cudaMemcpyKind::cudaMemcpyHostToDevice));

	CachedBVH::CachedBVHNode* tempNodes;
	cuda(Malloc(&tempNodes, bvh.nNodes * sizeof(CachedBVH::CachedBVHNode)));
	cuda(Memcpy(tempNodes, cachedBVH.nodes,
				bvh.nNodes * sizeof(CachedBVH::CachedBVHNode),
				cudaMemcpyKind::cudaMemcpyHostToDevice));

	delete[] cachedBVH.nodes;

	cachedBVH.nodes = tempNodes;
	gpuScene.CUDACachedBVH = cachedBVH;
}

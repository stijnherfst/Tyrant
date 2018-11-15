#pragma once

struct StaticMesh {
private:
	const aiScene* scene;
	Assimp::Importer importer;

public:

	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<unsigned int> indices;

	StaticMesh() {}
	StaticMesh(const std::string& path);

	int load(const std::string& path);
};
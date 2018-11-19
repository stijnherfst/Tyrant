#pragma once
#include <vector>
#include "BVH.h"
#include "Bbox.h"
#include "static_mesh.h"

class Scene {
 public:
  struct GPUScene {
    CachedBVH CUDACachedBVH;
  } gpuScene;

  void Load(const char path[]);

 private:
  // const aiScene *scene;
  Assimp::Importer importer;

  std::vector<Triangle> primitives;
  std::vector<BBox> primitiveBBoxes;
  // Sphere *spheres;
  std::vector<StaticMesh> meshes;
};
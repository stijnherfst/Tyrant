#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <stdbool.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "glad/glad.h"

#include "GLFW/glfw3.h"

#include "cuda.h"
#define GLM_FORCE_CUDA
//#define GLM_FORCE_PURE
#define GLM_FORCE_CXX17
#include "glm.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "assert_cuda.h"
#include "cuda_gl_interop.h"
#include "loader.h"
#include "static_mesh.h"
#include "variables.h"

//#include "Bbox.h"
//#include "BVH.h"

#include "camera.h"

#include "interop.h"
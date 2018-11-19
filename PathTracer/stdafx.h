#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <stdbool.h>
#include <stdlib.h>
#include <list>
#include <iostream>

#include "cuda.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#define GLM_FORCE_CUDA
//#define GLM_FORCE_PURE
#define GLM_FORCE_CXX17
#include "glm.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


#include "cuda_gl_interop.h"
#include "assert_cuda.h"
#include "variables.h"
#include "static_mesh.h"
#include "loader.h"

//#include "Bbox.h"
//#include "BVH.h"

#include "camera.h"

#include "interop.h"
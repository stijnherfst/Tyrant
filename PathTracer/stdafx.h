#pragma once

#include <stdbool.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "GLFW/glfw3.h"
#include "cuda.h"
#include "glad/glad.h"
#define GLM_FORCE_CUDA
//#define GLM_FORCE_PURE
#define GLM_FORCE_CXX17
#include "glm.hpp"

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include "assert_cuda.h"
#include "bvh.h"
#include "cuda_gl_interop.h"
#include "loader.h"
#include "static_mesh.h"
#include "variables.h"

#include "camera.h"

#include "interop.h"
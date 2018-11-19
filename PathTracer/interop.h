#pragma once

#include <cuda_runtime.h>
#include <stdbool.h>
#include "Scene.h"
struct pxl_interop {
  int width;
  int height;

  // GL buffers
  GLuint fb;
  GLuint rb;

  // CUDA resources
  cudaGraphicsResource_t cgr;
  cudaArray_t ca;

  cudaError_t set_size(const int width, const int height);
  void blit();
};

pxl_interop* pxl_interop_create();

void interop_destroy(struct pxl_interop* const interop);

cudaError_t launch_kernels(cudaArray_const_t array, glm::vec4* blit_buffer, Scene::GPUScene gpuScene);
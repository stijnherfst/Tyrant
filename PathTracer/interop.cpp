#include "stdafx.h"

pxl_interop* pxl_interop_create() {
  pxl_interop* interop = new pxl_interop();

  // allocate arrays

  // render buffer object w/a color buffer
  glCreateRenderbuffers(1, &interop->rb);

  // frame buffer object
  glCreateFramebuffers(1, &interop->fb);

  // attach rbo to fbo
  glNamedFramebufferRenderbuffer(interop->fb, GL_COLOR_ATTACHMENT0,
                                 GL_RENDERBUFFER, interop->rb);

  return interop;
}

void interop_destroy(pxl_interop* const interop) {
  cudaError_t cuda_err;

  // unregister CUDA resources
  if (interop->cgr != nullptr)
    cuda_err = cuda(GraphicsUnregisterResource(interop->cgr));

  glDeleteRenderbuffers(1, &interop->rb);
  glDeleteFramebuffers(1, &interop->fb);

  delete interop;
}

cudaError_t pxl_interop::set_size(const int width, const int height) {
  cudaError_t cuda_err = cudaSuccess;

  this->width = width;
  this->height = height;

  // unregister resource
  if (cgr != nullptr) cuda_err = cuda(GraphicsUnregisterResource(cgr));

  // resize rbo
  glNamedRenderbufferStorage(rb, GL_RGBA32F, width, height);

  // register rbo
  cuda_err =
      cuda(GraphicsGLRegisterImage(&cgr, rb, GL_RENDERBUFFER,
                                   cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                       cudaGraphicsRegisterFlagsWriteDiscard));

  // map graphics resources
  cuda_err = cuda(GraphicsMapResources(1, &cgr, 0));

  // get CUDA Array refernces
  cuda_err = cuda(GraphicsSubResourceGetMappedArray(&ca, cgr, 0, 0));

  // unmap graphics resources
  cuda_err = cuda(GraphicsUnmapResources(1, &cgr, 0));

  return cuda_err;
}

void pxl_interop::blit() {
  glBlitNamedFramebuffer(fb, 0, 0, 0, width, height, 0, height, width, 0,
                         GL_COLOR_BUFFER_BIT, GL_NEAREST);
}
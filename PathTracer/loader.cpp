#include "stdafx.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <cfloat>

#include <assert.h>
#include <string.h>

CudaMesh load_object(const std::string& source) {
  StaticMesh mesh(source);
  CudaMesh cuda_mesh;

  cuda_mesh.vertices = new Vertex[mesh.vertices.size()];
  cuda_mesh.verticesNo = mesh.vertices.size();

  Vertex* pCurrentVertex = cuda_mesh.vertices;
  for (int i = 0; i < mesh.vertices.size(); i++) {
    pCurrentVertex->position.x = mesh.vertices[i].x;
    pCurrentVertex->position.y = mesh.vertices[i].z;
    pCurrentVertex->position.z = mesh.vertices[i].y;
    pCurrentVertex->normal.x = mesh.normals[i].x;
    pCurrentVertex->normal.y = mesh.normals[i].y;
    pCurrentVertex->normal.z = mesh.normals[i].z;
    pCurrentVertex++;
  }

  cuda_mesh.triangles = new Triangle[mesh.indices.size() / 3];
  cuda_mesh.trianglesNo = mesh.indices.size() / 3;

  Triangle* pCurrentTriangle = cuda_mesh.triangles;
  for (int i = 0; i < cuda_mesh.trianglesNo; i++) {
    pCurrentTriangle->index1 = mesh.indices[i * 3];
    pCurrentTriangle->index2 = mesh.indices[i * 3 + 1];
    pCurrentTriangle->index3 = mesh.indices[i * 3 + 2];
    pCurrentTriangle->color.x = 255;
    pCurrentTriangle->color.y = 255;
    pCurrentTriangle->color.z = 255;
    pCurrentTriangle->normal = {0, 0, 0};
    pCurrentTriangle->bottom = {FLT_MAX, FLT_MAX, FLT_MAX};
    pCurrentTriangle->top = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    pCurrentTriangle++;
  }

  // Center scene at world's center
  glm::vec3 minp = {FLT_MAX, FLT_MAX, FLT_MAX};
  glm::vec3 maxp = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

  // Calculate scene bounding box
  for (unsigned i = 0; i < cuda_mesh.trianglesNo; i++) {
    minp = glm::min(minp,
                    cuda_mesh.vertices[cuda_mesh.triangles[i].index1].position);
    minp = glm::min(minp,
                    cuda_mesh.vertices[cuda_mesh.triangles[i].index2].position);
    minp = glm::min(minp,
                    cuda_mesh.vertices[cuda_mesh.triangles[i].index3].position);

    maxp = glm::max(maxp,
                    cuda_mesh.vertices[cuda_mesh.triangles[i].index1].position);
    maxp = glm::max(maxp,
                    cuda_mesh.vertices[cuda_mesh.triangles[i].index2].position);
    maxp = glm::max(maxp,
                    cuda_mesh.vertices[cuda_mesh.triangles[i].index3].position);
  }

  // Scene bounding box center before scaling and translating
  glm::vec3 origCenter = (maxp + minp) * 0.5f;
  minp -= origCenter;
  maxp -= origCenter;

  // Scale scene

  float maxi = 0;
  maxi = std::max(maxi, abs(minp.x));
  maxi = std::max(maxi, abs(minp.y));
  maxi = std::max(maxi, abs(minp.z));
  maxi = std::max(maxi, abs(maxp.x));
  maxi = std::max(maxi, abs(maxp.y));
  maxi = std::max(maxi, abs(maxp.z));

  // Center and scale vertices
  for (unsigned i = 0; i < cuda_mesh.verticesNo; i++) {
    cuda_mesh.vertices[i].position -= origCenter;
    cuda_mesh.vertices[i].position *= (20.f / maxi);
  }

  // Update triangle bounding boxes
  for (unsigned i = 0; i < cuda_mesh.trianglesNo; i++) {
    cuda_mesh.triangles[i].bottom =
        glm::min(cuda_mesh.triangles[i].bottom,
                 cuda_mesh.vertices[cuda_mesh.triangles[i].index1].position);
    cuda_mesh.triangles[i].bottom =
        glm::min(cuda_mesh.triangles[i].bottom,
                 cuda_mesh.vertices[cuda_mesh.triangles[i].index2].position);
    cuda_mesh.triangles[i].bottom =
        glm::min(cuda_mesh.triangles[i].bottom,
                 cuda_mesh.vertices[cuda_mesh.triangles[i].index3].position);
    cuda_mesh.triangles[i].top =
        glm::max(cuda_mesh.triangles[i].top,
                 cuda_mesh.vertices[cuda_mesh.triangles[i].index1].position);
    cuda_mesh.triangles[i].top =
        glm::max(cuda_mesh.triangles[i].top,
                 cuda_mesh.vertices[cuda_mesh.triangles[i].index2].position);
    cuda_mesh.triangles[i].top =
        glm::max(cuda_mesh.triangles[i].top,
                 cuda_mesh.vertices[cuda_mesh.triangles[i].index3].position);
  }

  // Roman Kuchkuda's triangle intersection.
  for (unsigned i = 0; i < cuda_mesh.trianglesNo; i++) {
    Triangle& triangle = cuda_mesh.triangles[i];

    // precompute edge vectors
    glm::vec3 vertex1 = cuda_mesh.vertices[triangle.index2].position -
                        cuda_mesh.vertices[triangle.index1].position;
    glm::vec3 vertex2 = cuda_mesh.vertices[triangle.index3].position -
                        cuda_mesh.vertices[triangle.index2].position;
    glm::vec3 vertex3 = cuda_mesh.vertices[triangle.index1].position -
                        cuda_mesh.vertices[triangle.index3].position;

    // plane of triangle, cross product of edge vectors vc1 and vc2
    triangle.normal = glm::cross(vertex1, vertex2);

    // choose longest alternative normal for maximum precision
    glm::vec3 alt1 = glm::cross(vertex2, vertex3);
    if (length(alt1) > length(triangle.normal))
      triangle.normal =
          alt1;  // higher precision when triangle has sharp angles

    glm::vec3 alt2 = cross(vertex3, vertex1);
    if (length(alt2) > length(triangle.normal)) triangle.normal = alt2;

    triangle.normal = normalize(triangle.normal);

    // precompute dot product between normal and first triangle vertex
    triangle._d =
        dot(triangle.normal, cuda_mesh.vertices[triangle.index1].position);

    // edge planes
    triangle._e1 = normalize(cross(triangle.normal, vertex1));
    triangle._d1 =
        dot(triangle._e1, cuda_mesh.vertices[triangle.index1].position);
    triangle._e2 = normalize(cross(triangle.normal, vertex2));
    triangle._d2 =
        dot(triangle._e2, cuda_mesh.vertices[triangle.index2].position);
    triangle._e3 = normalize(cross(triangle.normal, vertex3));
    triangle._d3 =
        dot(triangle._e3, cuda_mesh.vertices[triangle.index3].position);
  }

  return cuda_mesh;
}

#include "stdafx.h"
#include "sunsky.cuh"

#include "assert_cuda.h"
#include "cuda_surface_types.h"
#include "device_launch_parameters.h"
#include "surface_functions.h"
#include "variables.h"

surface<void, cudaSurfaceType2D> surf;
texture<float, cudaTextureTypeCubemap> skybox;

__device__ unsigned int RandomInt(unsigned int& seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}

__device__ float RandomFloat(unsigned int& seed) {
  return RandomInt(seed) * 2.3283064365387e-10f;
}

__device__ float RandomFloat2(unsigned int& seed) {
  return (RandomInt(seed) >> 16) / 65535.0f;
}

struct Ray {
  glm::vec3 orig;
  glm::vec3 dir;
  __device__ Ray(glm::vec3 origin, glm::vec3 direction)
      : orig(origin), dir(direction) {}
};

enum Refl_t { DIFF, SPEC, REFR };

inline __host__ __device__ float dot(const glm::vec4& v1, const glm::vec3& v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

struct Sphere {
  float radius;
  glm::vec3 position, color;
  Refl_t refl;

  __device__ float intersect(const Ray& r) const {
    glm::vec3 op = position - r.orig;
    float t;
    float b = glm::dot(op, r.dir);
    float disc = b * b - dot(op, op) + radius * radius;
    if (disc < 0)
      return 0;
    else
      disc = sqrtf(disc);
    return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
  }

  __device__ glm::vec3 random_point(unsigned int& seed) const {
    float u = RandomFloat(seed);
    float v = RandomFloat(seed);

    float cosPhi = 2.0f * u - 1.0f;
    float sinPhi = sqrt(1.0f - cosPhi * cosPhi);
    float theta = 2 * pi * v;

    float x = radius * sinPhi * sin(theta);
    float y = radius * cosPhi;
    float z = radius * sinPhi * cos(theta);

    return position + glm::vec3(x, y, z);
  }
};

__constant__ Sphere spheres[5];

// Branchless box intersection from
// https://tavianator.com/fast-branchless-raybounding-box-intersections/
__device__ inline bool RayBoxIntersection(const Ray& ray, int boxIdx,
                                          float* cudaBVHlimits) {
  glm::vec3 box_min = {cudaBVHlimits[6 * boxIdx + 0],
                       cudaBVHlimits[6 * boxIdx + 2],
                       cudaBVHlimits[6 * boxIdx + 4]};
  glm::vec3 box_max = {cudaBVHlimits[6 * boxIdx + 1],
                       cudaBVHlimits[6 * boxIdx + 3],
                       cudaBVHlimits[6 * boxIdx + 5]};

  glm::vec3 dir_inv = 1.f / ray.dir;

  float t1 = (box_min[0] - ray.orig[0]) * dir_inv[0];
  float t2 = (box_max[0] - ray.orig[0]) * dir_inv[0];

  float tmin = glm::min(t1, t2);
  float tmax = glm::max(t1, t2);

  for (int i = 1; i < 3; ++i) {
    t1 = (box_min[i] - ray.orig[i]) * dir_inv[i];
    t2 = (box_max[i] - ray.orig[i]) * dir_inv[i];

    tmin = glm::max(tmin, glm::min(t1, t2));
    tmax = glm::min(tmax, glm::max(t1, t2));
  }

  return tmax > glm::max(tmin, 0.f);
}

__device__ bool BVH_IntersectTriangles(int* cudaBVHindexesOrTrilists,
                                       const Ray& ray, unsigned avoidSelf,
                                       int& triangle_id, float& hit_distance,
                                       float* cudaBVHlimits,
                                       float* cudaTriangleIntersectionData,
                                       int* cudaTriIdxList) {
  triangle_id = -1;
  float bestTriDist = FLT_MAX;

  int stack[bvh_stack_size];

  int stackIdx = 0;
  stack[stackIdx++] = 0;
  glm::vec3 hitpoint;

  while (stackIdx) {
    int boxIdx = stack[stackIdx - 1];

    stackIdx--;

    if (!(cudaBVHindexesOrTrilists[4 * boxIdx + 0] & 0x80000000)) {
      if (RayBoxIntersection(ray, boxIdx, cudaBVHlimits)) {
        stack[stackIdx++] = cudaBVHindexesOrTrilists[4 * boxIdx + 1];
        stack[stackIdx++] = cudaBVHindexesOrTrilists[4 * boxIdx + 2];

        // Stack size of 32 works till about 1 million triangles?
        if (stackIdx > bvh_stack_size) {
          return false;
        }
      }
    } else {  // Leaf node
      for (unsigned i = cudaBVHindexesOrTrilists[4 * boxIdx + 3];
           i < cudaBVHindexesOrTrilists[4 * boxIdx + 3] +
                   (cudaBVHindexesOrTrilists[4 * boxIdx + 0] & 0x7fffffff);
           i++) {
        int idx = cudaTriIdxList[i];

        if (avoidSelf == idx) {
          continue;
        }

        glm::vec4 normal =
            glm::vec4(cudaTriangleIntersectionData[20 * idx + 4],
                      cudaTriangleIntersectionData[20 * idx + 5],
                      cudaTriangleIntersectionData[20 * idx + 6],
                      cudaTriangleIntersectionData[20 * idx + 7]);
        float k = dot(normal, ray.dir);
        if (k == 0.0f) continue;  // Parallel

        float s = (normal.w - dot(normal, ray.orig)) / k;
        if (s <= 0.0f || s <= epsilon)  // Behind the origin/epsilon
          continue;

        glm::vec3 hit = ray.orig + ray.dir * s;

        // Intersection is within triangle
        glm::vec4 ee1 = glm::vec4(cudaTriangleIntersectionData[20 * idx + 8],
                                  cudaTriangleIntersectionData[20 * idx + 9],
                                  cudaTriangleIntersectionData[20 * idx + 10],
                                  cudaTriangleIntersectionData[20 * idx + 11]);
        float kt1 = dot(ee1, hit) - ee1.w;
        if (kt1 < 0.0f) continue;

        glm::vec4 ee2 = glm::vec4(cudaTriangleIntersectionData[20 * idx + 12],
                                  cudaTriangleIntersectionData[20 * idx + 13],
                                  cudaTriangleIntersectionData[20 * idx + 14],
                                  cudaTriangleIntersectionData[20 * idx + 15]);
        float kt2 = dot(ee2, hit) - ee2.w;
        if (kt2 < 0.0f) continue;

        glm::vec4 ee3 = glm::vec4(cudaTriangleIntersectionData[20 * idx + 16],
                                  cudaTriangleIntersectionData[20 * idx + 17],
                                  cudaTriangleIntersectionData[20 * idx + 18],
                                  cudaTriangleIntersectionData[20 * idx + 19]);
        float kt3 = dot(ee3, hit) - ee3.w;
        if (kt3 < 0.0f) continue;

        // Take closest intersection
        float hitZ = dot(ray.orig - hit, ray.orig - hit);
        if (hitZ < bestTriDist) {
          bestTriDist = hitZ;
          triangle_id = idx;
        }
      }
    }
  }

  // Avoid expensive sqrt till the end
  hit_distance = sqrt(bestTriDist);

  return triangle_id != -1;
}

__device__ inline bool intersect_scene(const Ray& ray, float& t, int& id,
                                       int& geometry_type,
                                       int* cudaBVHindexesOrTrilists,
                                       float* cudaBVHlimits,
                                       float* cudaTriangleIntersectionData,
                                       int* cudaTriIdxList) {
  float n = sizeof(spheres) / sizeof(Sphere), d;
  float inf = 1e20f;
  t = 1e20f;

  for (int i = int(n); i--;) {
    if ((d = spheres[i].intersect(ray)) && d < t) {
      t = d;
      id = i;
      geometry_type = 1;
    }
  }

  int triangle_id = -1;
  glm::vec3 pointHitInWorldSpace;
  float hit_distance = 1e20f;

  // Intersect BVH
  if (BVH_IntersectTriangles(cudaBVHindexesOrTrilists, ray, -1, triangle_id,
                             hit_distance, cudaBVHlimits,
                             cudaTriangleIntersectionData, cudaTriIdxList)) {
    if (hit_distance < t) {
      t = hit_distance;
      geometry_type = 2;
      id = triangle_id;
    }
  }
  return t < inf;
}

__device__ glm::vec3 radiance(Ray& ray, unsigned int& seed,
                              int* cudaBVHindexesOrTrilists,
                              float* cudaBVHlimits,
                              float* cudaTriangleIntersectionData,
                              int* cudaTriIdxList, Triangle* triangles) {
  glm::vec3 color = {1.f, 1.f, 1.f};
  glm::vec3 direct = {0.f, 0.f, 0.f};

  int geometry_type = 0;
  int reflection_type;

  float distance;
  int id;
  for (int bounces = 0; bounces < 4; bounces++) {
    if (!intersect_scene(ray, distance, id, geometry_type,
                         cudaBVHindexesOrTrilists, cudaBVHlimits,
                         cudaTriangleIntersectionData, cudaTriIdxList)) {
      return direct + color * (bounces > 0 ? sky(ray.dir) : sunsky(ray.dir));
    }

    glm::vec3 position = ray.orig + ray.dir * distance;
    glm::vec3 normal;
    switch (geometry_type) {
      case 1:
        const Sphere& object = spheres[id];
        normal = (position - object.position) / object.radius;
        color *= object.color;
        reflection_type = object.refl;
        break;
      case 2:
        Triangle* triangle = &triangles[id];
        normal = triangle->normal;
        // color *= glm::vec3(0.75, 1, 0.75);
        reflection_type = DIFF;
        break;
    }

    bool outside = dot(normal, ray.dir) < 0;
    normal =
        outside
            ? normal
            : normal * -1.f;  // make n front facing is we are inside an object
    ray.orig = position + normal * epsilon;

    switch (reflection_type) {
      case DIFF: {
        // Random direction in hemisphere
        float r1 = 2.f * pi * RandomFloat(seed);
        float r2 = RandomFloat(seed);
        float r2s = sqrt(r2);

        // Transform to hemisphere coordinate system
        const glm::vec3 u = normalize(
            glm::cross((abs(normal.x) > .1f ? glm::vec3(0.f, 1.f, 0.f)
                                            : glm::vec3(1.f, 0.f, 0.f)),
                       normal));
        const glm::vec3 v = cross(normal, u);
        // Get sample on hemisphere
        const glm::vec3 d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s +
                                      normal * sqrt(1 - r2));

        glm::vec3 sunSampleDir =
            getConeSample(sunDirection, 1.0f - sunAngularDiameterCos, seed);
        float sunLight = dot(normal, sunSampleDir);

        Ray shadow_ray = Ray(position + normal * 0.01f, sunSampleDir);
        float shadow_ray_distance;
        int shadow_ray_id;

        if (sunLight > 0.0 &&
            !intersect_scene(shadow_ray, shadow_ray_distance, shadow_ray_id,
                             geometry_type, cudaBVHindexesOrTrilists,
                             cudaBVHlimits, cudaTriangleIntersectionData,
                             cudaTriIdxList)) {
          direct += color * sun(sunSampleDir) * sunLight * 1E-5f;
        }

        ray.dir = d;
        break;
      }
      case SPEC: {
        ray.dir = reflect(ray.dir, normal);
        break;
      }
      case REFR: {
        float n1 = outside ? 1.2f : 1.0f;
        float n2 = outside ? 1.0f : 1.2f;

        float r0 = (n1 - n2) / (n1 + n2);
        r0 *= r0;
        float fresnel =
            r0 + (1. - r0) * pow(1.0 - abs(dot(ray.dir, normal)), 5.);

        if (RandomFloat(seed) < fresnel) {
          ray.dir = reflect(ray.dir, normal);
        } else {
          ray.orig = position - normal * 2.f * epsilon;
          ray.dir = glm::refract(ray.dir, normal, n2 / n1);
        }
        break;
      }
    }
  }

  return direct;
}

__global__ void primary_rays(glm::vec3 camera_right, glm::vec3 camera_up,
                             glm::vec3 camera_direction, glm::vec3 O,
                             unsigned int frame, glm::vec4* blit_buffer,
                             int* cudaBVHindexesOrTrilists,
                             float* cudaBVHlimits,
                             float* cudaTriangleIntersectionData,
                             int* cudaTriIdxList, Triangle* triangles) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= render_width || y >= render_height) {
    return;
  }

  const float normalized_i = (x / (float)render_width) - 0.5f;
  const float normalized_j =
      ((render_height - y) / (float)render_height) - 0.5f;

  glm::vec3 direction =
      camera_direction + normalized_i * camera_right + normalized_j * camera_up;
  direction = normalize(direction);

  unsigned int seed = (frame * x * 147565741) * 720898027 * y;

  glm::vec3 r =
      radiance(Ray(O, direction), seed, cudaBVHindexesOrTrilists, cudaBVHlimits,
               cudaTriangleIntersectionData, cudaTriIdxList, triangles);

  const int index = y * render_width + x;
  blit_buffer[index] += glm::vec4(r.x, r.y, r.z, 1);
}

__global__ void blit_onto_framebuffer(glm::vec4* blit_buffer, unsigned frames) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= render_width || y >= render_height) {
    return;
  }

  const int index = y * render_width + x;
  glm::vec4 color = blit_buffer[index] / (float)frames;
  surf2Dwrite<glm::vec4>(
      glm::pow(color / (color + 1.f), glm::vec4(1.0f / 2.2f)), surf,
      x * sizeof(glm::vec4), y, cudaBoundaryModeZero);
}

bool first_time = true;
bool reset_buffer = false;
unsigned int frame = 0;
unsigned int hold_frame = 0;

cudaError_t launch_kernels(cudaArray_const_t array, glm::vec4* blit_buffer,
                           int* cudaBVHindexesOrTrilists, float* cudaBVHlimits,
                           float* cudaTriangleIntersectionData,
                           int* cudaTriIdxList, Triangle* triangles) {
  if (first_time) {
    first_time = false;

    Sphere sphere_data[5] = {{16.5, {0, 40, 16.5f}, {1, 1, 1}, DIFF},
                             {16.5, {40, 0, 16.5f}, {1, 1, 1}, REFR},
                             {16.5, {-40, 0, 16.5f}, {1, 1, 1}, SPEC},
                             {1e4f, {0, 0, -1e4f - 20}, {1, 1, 1}, DIFF},
                             {40, {0, -80, 18.0f}, {1.0, 0.0, 0.0}, DIFF}};

    cudaMemcpyToSymbol(spheres, sphere_data, 5 * sizeof(Sphere));

    float sun_angular = cos(sunSize * pi / 180.0);
    cuda(MemcpyToSymbol(sunAngularDiameterCos, &sun_angular, sizeof(float)));
  }

  cudaError_t cuda_err;
  static glm::vec3 last_pos;
  static glm::vec3 last_dir;

  cuda_err = cuda(BindSurfaceToArray(surf, array));

  if (cuda_err) {
    return cuda_err;
  }

  const glm::vec3 camera_right =
      glm::normalize(glm::cross(camera.direction, camera.up)) * 1.5f *
      ((float)render_width / render_height);
  const glm::vec3 camera_up =
      glm::normalize(glm::cross(camera_right, camera.direction)) * 1.5f;

  reset_buffer = last_pos != camera.position || last_dir != camera.direction;

  if (sun_position_changed) {
    sun_position_changed = false;
    reset_buffer = true;
    cuda(MemcpyToSymbol(SunPos, &sun_position, sizeof(glm::vec2)));
    glm::vec3 sun_direction = glm::normalize(fromSpherical(
        (sun_position - glm::vec2(0.0, 0.5)) * glm::vec2(6.28f, 3.14f)));
    cuda(MemcpyToSymbol(sunDirection, &sun_direction, sizeof(glm::vec3)));
  }

  if (reset_buffer) {
    reset_buffer = false;
    cudaMemset(blit_buffer, 0, render_width * render_height * sizeof(float4));
    hold_frame = 1;
  }

  dim3 threads(16, 16, 1);
  dim3 blocks(render_width / threads.x, render_height / threads.y, 1);
  primary_rays<<<blocks, threads>>>(
      camera_right, camera_up, camera.direction, camera.position, frame,
      blit_buffer, cudaBVHindexesOrTrilists, cudaBVHlimits,
      cudaTriangleIntersectionData, cudaTriIdxList, triangles);

  blit_onto_framebuffer<<<blocks, threads>>>(blit_buffer, hold_frame);

  frame++;
  hold_frame++;
  last_pos = camera.position;
  last_dir = camera.direction;

  return cudaSuccess;
}
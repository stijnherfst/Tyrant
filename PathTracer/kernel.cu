#include "stdafx.h"
#include "sunsky.cuh"

#include "Bbox.h"
#include "Rays.h"
#include "Scene.h"
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

__device__ inline bool intersect_scene(const Ray& ray, float& t, int& id,
                                       int& geometry_type,
                                       Scene::GPUScene sceneData) {
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
 // for (int i = 0; i < 12; ++i) {
	//float dist =  sceneData.CUDACachedBVH.primitives[i].intersect(ray);
 //   if (dist > epsilon) {
 //         if (dist < hit_distance) {
	//		hit_distance = dist;
 //           triangle_id = i;
	//	  }
	//}
 // }
 // Triangle tri;
 // tri.vert = glm::vec3{0, 0, 0};
 // glm::vec3 vert2 = glm::vec3{0, 0, 4};
 // glm::vec3 vert3 = glm::vec3{4, 0, 0};
 // tri.e1 = vert2 - tri.vert;
 // tri.e2 = vert3 - tri.vert;
	//float dist =  tri.intersect(ray);
 //   if (dist > epsilon) {
 //         if (dist < hit_distance) {
	//		hit_distance = dist;
	//	  }
	//}

  //if (hit_distance < t) {
  //  t = hit_distance;
  //  geometry_type = 2;
  //  id = triangle_id;
  //}
    if (sceneData.CUDACachedBVH.intersect(ray, t, id)) {
      geometry_type = 2;
    }
  // if (sceneData.CUDACachedBVH.intersect(rayToIntersect,
  // lowestIntersectT,hitIndex)){
  //	geometryType = GeomType::Triangle;
  //}
  //// Intersection is too far, stop recursion
  // if (lowestIntersectT >= HugeEpsilon){
  //	totalColor += (rayColorMask *
  //computeBackgroundColor(rayToIntersect.direction)); 	return totalColor;
  //}

  // if (geometryType == GeomType::Triangle) {
  //	const Triangle &triangle = sceneData.CUDACachedBVH.primitives[hitIndex];
  //	//refltype  = ReflectiveType::Coat;
  //	refltype  =static_cast<ReflectiveType>(triangle.materialType);
  //	albedo = triangle.color;
  //	intersectionPoint = originInWorldSpace + rayInWorldSpace *
  //lowestIntersectT; 	normal = cross(triangle.e1,triangle.e2); 	normal =
  //normalize(normal); 	orientedNormal = dot(normal, rayInWorldSpace) < 0 ?
  //normal : normal * -1;
  //	//albedo    = make_float3(1.0,0.8,0.1);
  //	emittance = make_float3(0);
  //	//totalColor += (rayColorMask * emittance);
  //}
  // Intersect BVH
  // if (BVH_IntersectTriangles(cudaBVHindexesOrTrilists, ray, -1, triangle_id,
  //                           hit_distance, cudaBVHlimits,
  //                           cudaTriangleIntersectionData, cudaTriIdxList)) {
  //  if (hit_distance < t) {
  //    t = hit_distance;
  //    geometry_type = 2;
  //    id = triangle_id;
  //  }
  //}
  return t < inf;
}

__device__ glm::vec3 radiance(Ray& ray, unsigned int& seed,
                              Scene::GPUScene sceneData) {
  glm::vec3 color = {1.f, 1.f, 1.f};
  glm::vec3 direct = {0.f, 0.f, 0.f};

  int geometry_type = 0;
  int reflection_type;

  float distance;
  int id;
  for (int bounces = 0; bounces < 4; bounces++) {
    if (!intersect_scene(ray, distance, id, geometry_type, sceneData)) {
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
        //return {255, 0, 0};
        Triangle* triangle = &(sceneData.CUDACachedBVH.primitives[id]);
        normal = glm::cross(triangle->e1, triangle->e2);
        normal = glm::normalize(normal);
        // color *= glm::vec3(1, 1, 1);
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
        const glm::vec3 u = (
            glm::cross((abs(normal.x) > .1f ? glm::vec3(0.f, 1.f, 0.f)
                                            : glm::vec3(1.f, 0.f, 0.f)),
                       normal));
        const glm::vec3 v = cross(normal, u);
        // Get sample on hemisphere
        const glm::vec3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s +
                                      normal * sqrt(1 - r2));

        glm::vec3 sunSampleDir =
            getConeSample(sunDirection, 1.0f - sunAngularDiameterCos, seed);
        float sunLight = dot(normal, sunSampleDir);

        Ray shadow_ray = Ray(position + normal * 0.01f, sunSampleDir);
        float shadow_ray_distance;
        int shadow_ray_id;

        if (sunLight > 0.0 &&
            !intersect_scene(shadow_ray, shadow_ray_distance, shadow_ray_id,
                             geometry_type, sceneData)) {
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
                             Scene::GPUScene sceneData) {
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

  glm::vec3 r = radiance(Ray(O, direction), seed, sceneData);

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
                           Scene::GPUScene sceneData) {
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

  dim3 threads(8, 8, 1);
  dim3 blocks(render_width / threads.x, render_height / threads.y, 1);
  primary_rays<<<blocks, threads>>>(camera_right, camera_up, camera.direction,
                                    camera.position, frame, blit_buffer,
                                    sceneData);
  threads  =  dim3(16, 16, 1);
  blocks = dim3(render_width / threads.x, render_height / threads.y, 1);
  blit_onto_framebuffer<<<blocks, threads>>>(blit_buffer, hold_frame);

  frame++;
  hold_frame++;
  last_pos = camera.position;
  last_dir = camera.direction;

  return cudaSuccess;
}
#include "stdafx.h"
#include "sunsky.cuh"

#include "assert_cuda.h"
#include "cuda_surface_types.h"
#include "device_launch_parameters.h"
#include "surface_functions.h"

#include "cuda_definitions.h"

constexpr int NUM_SPHERES = 7;
constexpr float VERY_FAR = 1e20f;
constexpr int MAX_BOUNCES = 5;

surface<void, cudaSurfaceType2D> surf;
texture<float, cudaTextureTypeCubemap> skybox;

//"Xorshift RNGs" by George Marsaglia
//http://excamera.com/sphinx/article-xorshift.html
__device__ unsigned int RandomInt(unsigned int& seed) {
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}

//Random float between [0,1).
__device__ float RandomFloat(unsigned int& seed) {
	return RandomInt(seed) * 2.3283064365387e-10f;
}

__device__ float RandomFloat2(unsigned int& seed) {
	return (RandomInt(seed) >> 16) / 65535.0f;
}

enum Refl_t { DIFF,
			  SPEC,
			  REFR,
			  PHONG };

inline __host__ __device__ float dot(const glm::vec4& v1, const glm::vec3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

struct Sphere {
	float radius;
	glm::vec3 position, color;
	glm::vec3 emmission;
	Refl_t refl;

	__device__ float intersect(const RayQueue& r) const {
		glm::vec3 op = position - r.origin;
		float t;
		float b = glm::dot(op, r.direction);
		float disc = b * b - dot(op, op) + radius * radius;
		if (disc < 0)
			return 0;

		disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}

	__device__ float intersect_simple(const ShadowQueue& r) const {
		glm::vec3 op = position - r.origin;
		float t;
		float b = glm::dot(op, r.direction);
		float disc = b * b - dot(op, op) + radius * radius;
		if (disc < 0)
			return 0;

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

__constant__ Sphere spheres[NUM_SPHERES];

__device__ inline bool intersect_scene(RayQueue& ray, Scene::GPUScene sceneData) {
	float d;
	ray.distance = VERY_FAR;

	for (int i = NUM_SPHERES; i--;) {
		//d = spheres[i].intersect(ray);
		if ((d = spheres[i].intersect(ray)) && d < ray.distance) {
			ray.distance = d;
			ray.identifier = i;
			ray.geometry_type = GeometryType::Sphere;
		}
	}

	if (sceneData.CUDACachedBVH.intersect(ray)) {
		ray.geometry_type = GeometryType::Triangle;
	}
	return ray.distance < VERY_FAR;
}

__device__ inline bool intersect_scene_simple(ShadowQueue& ray, Scene::GPUScene sceneData) {
	for (int i = NUM_SPHERES; i--;) {
		float d = spheres[i].intersect_simple(ray);
		if (d != 0) {
			return true;
		}
	}

	return sceneData.CUDACachedBVH.intersectSimple(ray);
}

__device__ unsigned int shadow_ray_cnt = 0;
__device__ unsigned int primary_ray_cnt = 0;
__device__ unsigned int start_position = 0;
__device__ unsigned int raynr_shade = 0;
__device__ unsigned int raynr_primary = 0;
__device__ unsigned int raynr_extend = 0;

__global__ void zero_variables() {
	shadow_ray_cnt = 0;
	start_position += ray_queue_buffer_size - primary_ray_cnt;
	start_position = start_position % (render_width * render_height);
	primary_ray_cnt = 0;
	raynr_shade = 0;
	raynr_primary = 0;
	raynr_extend = 0;
}

/// Generate primary rays
__global__ void primary_rays(RayQueue* queue, glm::vec3 camera_right, glm::vec3 camera_up, glm::vec3 camera_direction, glm::vec3 O) {

	while (true) {
		unsigned int index = atomicAdd(&raynr_primary, 1);

		if (primary_ray_cnt + index > ray_queue_buffer_size - 1) {
			return;
		}

		const int x = ((start_position + index) % render_width);
		const int y = ((start_position + index) / render_width) % render_height;

		const float normalized_i = (x / (float)render_width) - 0.5f;
		const float normalized_j = ((render_height - y) / (float)render_height) - 0.5f;

		glm::vec3 direction = camera_direction + normalized_i * camera_right + normalized_j * camera_up;
		direction = normalize(direction);

		queue[primary_ray_cnt + index] = { O, direction, { 1, 1, 1 }, 0, 0, 0, x, y };
	}
}

/// Advance the ray segments once
__global__ void __launch_bounds__(128, 8) extend(RayQueue* queue, Scene::GPUScene sceneData) {

	while (true) {
		unsigned int index = atomicAdd(&raynr_extend, 1);

		if (index > ray_queue_buffer_size - 1) {
			return;
		}

		RayQueue& ray = queue[index];

		ray.distance = VERY_FAR;
		//sceneData.CUDACachedBVH.intersect(ray);
		intersect_scene(ray, sceneData);
	}
}

/// Process collisions and spawn extension and shadow rays
__global__ void __launch_bounds__(128, 8) shade(RayQueue* queue, RayQueue* queue2, ShadowQueue* shadowQueue, Scene::GPUScene sceneData, glm::vec4* blit_buffer, unsigned int frame) {

	while (true) {
		unsigned int index = atomicAdd(&raynr_shade, 1);

		if (index > ray_queue_buffer_size - 1) {
			return;
		}

		int new_frame = 0;
		RayQueue& ray = queue[index];
		glm::vec3 color = glm::vec3(0.f);

		unsigned int seed = (frame * ray.x * 147565741) * 720898027 * index;
		int reflection_type = DIFF;

		if (ray.distance < VERY_FAR) {
			ray.origin += ray.direction * ray.distance;
			ray.lastSpecular = false;

			glm::vec3 normal;
			if (ray.geometry_type == GeometryType::Sphere) {
				const Sphere& object = spheres[ray.identifier];
				normal = (ray.origin - object.position) / object.radius;
				reflection_type = object.refl;
				color = color + (ray.direct * object.emmission);
				ray.direct *= object.color;
			} else {
				Triangle* triangle = &(sceneData.CUDACachedBVH.primitives[ray.identifier]);
				normal = glm::normalize(glm::cross(triangle->e1, triangle->e2));
				reflection_type = DIFF;
			}

			bool outside = dot(normal, ray.direction) < 0;
			normal = outside ? normal : normal * -1.f; // make n front facing is we are inside an object

			ray.origin += normal * epsilon;

			switch (reflection_type) {
			case DIFF: {

				// Generate new shadow ray
				glm::vec3 sunSampleDir = getConeSample(sunDirection, 1.0f - sunAngularDiameterCos, seed);
				float sunLight = dot(normal, sunSampleDir);

				// < 0.f means sun is behind the surface
				if (sunLight > 0.f) {
					unsigned shadow_index = atomicAdd(&shadow_ray_cnt, 1);
					//shadowQueue[shadow_index] = { ray.origin, sunSampleDir, sunLight, ray.y * render_width + ray.x };
					ShadowQueue rayy = { ray.origin, sunSampleDir, sunLight, ray.y * render_width + ray.x };

					if (!intersect_scene_simple(rayy, sceneData)) {
						color += ray.direct * (sun(rayy.direction) * rayy.sunlight * 1E-5f);
					}
				}

				if (ray.bounces < MAX_BOUNCES) {
					float r1 = 2.f * pi * RandomFloat(seed);
					float r2 = RandomFloat(seed);
					float r2s = sqrt(r2);

					// Transform to hemisphere coordinate system
					const glm::vec3 u = glm::normalize(glm::cross((abs(normal.x) > .1f ? glm::vec3(0.f, 1.f, 0.f) : glm::vec3(1.f, 0.f, 0.f)), normal));
					const glm::vec3 v = cross(normal, u);
					// Get sample on hemisphere
					const glm::vec3 d = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + normal * sqrt(1 - r2));
					ray.direction = d;
				}

				break;
			}
			case SPEC: {
				ray.lastSpecular = true;
				ray.direction = reflect(ray.direction, normal);
				break;
			}
			case REFR: {

				float n1 = outside ? 1.2f : 1.0f;
				float n2 = outside ? 1.0f : 1.2f;

				float r0 = (n1 - n2) / (n1 + n2);
				r0 *= r0;
				float fresnel = r0 + (1. - r0) * pow(1.0 - abs(dot(ray.direction, normal)), 5.);

				if (RandomFloat(seed) < fresnel) {
					ray.direction = reflect(ray.direction, normal);
				} else {
					ray.origin = ray.origin - normal * 2.f * epsilon;
					ray.direction = glm::refract(ray.direction, normal, n2 / n1);
				}
				break;
			}
			case PHONG: {
				// compute random perturbation of ideal reflection vector
				// the higher the phong exponent, the closer the perturbed vector
				// is to the ideal reflection direction
				float phi = 2 * pi * RandomFloat(seed);
				float r2 = RandomFloat(seed);
				float phongexponent = 25;
				float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
				float sinTheta = sqrtf(1 - cosTheta * cosTheta);

				/* 
				Create orthonormal basis uvw around reflection vector with 
				hitpoint as origin w is ray direction for ideal reflection
			 */
				glm::vec3 w;
				w = ray.direction - normal * 2.0f * dot(normal, ray.direction);
				w = normalize(w);

				// Transform to hemisphere coordinate system
				const glm::vec3 u = glm::normalize(
					glm::cross((abs(normal.x) > .9f ? glm::vec3(0.f, 1.f, 0.f)
													: glm::vec3(1.f, 0.f, 0.f)),
							   normal));
				const glm::vec3 v = cross(w, u);
				// Get sample on hemisphere
				// compute cosine weighted random ray direction on hemisphere

				glm::vec3 d = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
				d = normalize(d);

				glm::vec3 sunSampleDir = getConeSample(sunDirection, 1.0f - sunAngularDiameterCos, seed);
				float sunLight = dot(normal, sunSampleDir);

				//SunLight is cos of sampleDir to normal. For phong we weight it proportional to cos(theta) ^ phongExponent
				sunLight = powf(sunLight, phongexponent);
				if (sunLight > 0.f) {
					unsigned shadow_index = atomicAdd(&shadow_ray_cnt, 1);
					//shadowQueue[shadow_index] = { ray.origin, sunSampleDir, sunLight, ray.y * render_width + ray.x };
					ShadowQueue rayy = { ray.origin, sunSampleDir, sunLight, ray.y * render_width + ray.x };

					if (!intersect_scene_simple(rayy, sceneData)) {
						color += ray.direct * (sun(rayy.direction) * rayy.sunlight * 1E-5f);
					}
				}

				ray.origin = ray.origin + w * epsilon; // scene size dependent
				ray.direction = d;

				break;
			}
			}

			if (ray.bounces < MAX_BOUNCES) {
				ray.bounces++;

				unsigned primary_index = atomicAdd(&primary_ray_cnt, 1);
				queue2[primary_index] = ray;
			} else {

				new_frame++;
			}

		} else {
			// Don't generate new extended ray
			color += (ray.lastSpecular == false) ? ray.direct * sky(ray.direction) : ray.direct * sunsky(ray.direction);
			new_frame++;
		}

		atomicAdd(&blit_buffer[ray.y * render_width + ray.x].r, color.r);
		atomicAdd(&blit_buffer[ray.y * render_width + ray.x].g, color.g);
		atomicAdd(&blit_buffer[ray.y * render_width + ray.x].b, color.b);
		atomicAdd(&blit_buffer[ray.y * render_width + ray.x].a, new_frame);
	}
}

/// Proccess shadow rays
__global__ void connect(ShadowQueue* queue, Scene::GPUScene sceneData, glm::vec4* blit_buffer) {
	//const int index = blockIdx.x * blockDim.x + threadIdx.x;

	//if (index > shadow_ray_count - 1) {
	//	return;
	//}

	//ShadowQueue& ray = queue[index];

	//if (!sceneData.CUDACachedBVH.intersectSimple(ray)) {
	//	glm::vec3 color = sun(ray.direction) * ray.sunlight * 1E-5f;
	//primary_queue[ray.primary_index].direct += color;
	//	blit_buffer[ray.buffer_index] += glm::vec4(color, 0);
	//}
}

__global__ void blit_onto_framebuffer(glm::vec4* blit_buffer) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= render_width || y >= render_height) {
		return;
	}

	const int index = y * render_width + x;
	glm::vec4 color = blit_buffer[index];
	glm::vec4 cl = glm::vec4(color.r, color.g, color.b, 1) / color.a;
	cl.a = 1;

	surf2Dwrite<glm::vec4>(glm::pow(cl / (cl + 1.f), glm::vec4(1.0f / 2.2f)), surf, x * sizeof(glm::vec4), y, cudaBoundaryModeZero);
}

cudaError launch_kernels(cudaArray_const_t array, glm::vec4* blit_buffer, Scene::GPUScene sceneData, RayQueue* queue, RayQueue* queue2, ShadowQueue* shadow_queue) {
	static bool first_time = true;
	static bool reset_buffer = false;
	static unsigned int frame = 0;

	if (first_time) {
		first_time = false;

		Sphere sphere_data[NUM_SPHERES] = { { 16.5, { 0, 40, 16.5f }, { 1, 1, 1 }, { 0, 0, 0 }, DIFF },
											{ 16.5, { 40, 0, 16.5f }, { 1, 1, 1 }, { 0, 0, 0 }, REFR },
											{ 16.5, { -40, 0, 16.5f }, { 0.6, 0.5, 0.4 }, { 0, 0, 0 }, PHONG },
											{ 16.5, { -40, -50, 16.5f }, { 0.6, 0.5, 0.4 }, { 0, 0, 0 }, SPEC },
											{ 1e4f, { 0, 0, -1e4f - 20 }, { 1, 1, 1 }, { 0, 0, 0 }, PHONG },
											{ 20, { 0, -80, 20 }, { 1.0, 0.0, 0.0 }, { 0, 0, 0 }, DIFF },
											{ 30, { 0, -80, 120.0f }, { 0.0, 1.0, 0.0 }, { 2, 2, 2 }, DIFF } };
		cudaMemcpyToSymbol(spheres, sphere_data, NUM_SPHERES * sizeof(Sphere));

		float sun_angular = cos(sunSize * pi / 180.f);
		cuda(MemcpyToSymbol(sunAngularDiameterCos, &sun_angular, sizeof(float)));
	}

	cudaError cuda_err;
	static glm::vec3 last_pos;
	static glm::vec3 last_dir;

	cuda_err = cuda(BindSurfaceToArray(surf, array));

	if (cuda_err) {
		return cuda_err;
	}

	const glm::vec3 camera_right = glm::normalize(glm::cross(camera.direction, camera.up)) * 1.5f * ((float)render_width / render_height);
	const glm::vec3 camera_up = glm::normalize(glm::cross(camera_right, camera.direction)) * 1.5f;

	reset_buffer = last_pos != camera.position || last_dir != camera.direction;

	if (sun_position_changed) {
		sun_position_changed = false;
		reset_buffer = true;
		cuda(MemcpyToSymbol(SunPos, &sun_position, sizeof(glm::vec2)));
		glm::vec3 sun_direction = glm::normalize(fromSpherical((sun_position - glm::vec2(0.0, 0.5)) * glm::vec2(6.28f, 3.14f)));
		cuda(MemcpyToSymbol(sunDirection, &sun_direction, sizeof(glm::vec3)));
	}

	if (reset_buffer) {
		reset_buffer = false;
		cudaMemset(blit_buffer, 0, render_width * render_height * sizeof(float4));

		int new_value = 0;
		cuda(MemcpyToSymbol(primary_ray_cnt, &new_value, sizeof(int)));
	}

	primary_rays<<<40, 128>>>(queue, camera_right, camera_up, camera.direction, camera.position);
	zero_variables<<<1, 1>>>();
	extend<<<40, 128>>>(queue, sceneData);
	shade<<<40, 128>>>(queue, queue2, shadow_queue, sceneData, blit_buffer, frame);
	//connect<<<40, 128>>>(shadow_queue, sceneData, blit_buffer);

	dim3 threads = dim3(16, 16, 1);
	dim3 blocks = dim3(render_width / threads.x, render_height / threads.y, 1);
	blit_onto_framebuffer<<<blocks, threads>>>(blit_buffer);

	cuda(DeviceSynchronize());

	frame++;
	//hold_frame++;
	last_pos = camera.position;
	last_dir = camera.direction;

	return cudaSuccess;
}
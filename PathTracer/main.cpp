#include "stdafx.h"

Vertex* cudaVertices = nullptr;
Triangle* cudaTriangles = nullptr;
float* cudaTriangleIntersectionData = nullptr;
int* cudaTriIdxList = nullptr;
float* cudaBVHlimits = nullptr;
int* cudaBVHindexesOrTrilists = nullptr;

static void glfw_fps(GLFWwindow* window) {
	// Static fps counters
	static double previous_time = 0.0;
	static int frame_count = 0;

	const double current_time = glfwGetTime();
	const double elapsed = current_time - previous_time;

	if (elapsed > 0.5) {
		previous_time = current_time;

		const double fps = (double)frame_count / elapsed;

		int width, height;
		char tmp[64];

		glfwGetFramebufferSize(window, &width, &height);

		sprintf_s(tmp, 64, "(%u x %u) - FPS: %.2f ms: %.3f", width, height, fps, (elapsed / frame_count) * 1000);

		glfwSetWindowTitle(window, tmp);

		frame_count = 0;
	}

	frame_count++;
}

static void glfw_error_callback(int error, const char* description) {
	std::cout << description << "\n";
}

static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}

void upload_cuda_data(CudaMesh cuda_mesh) {
	// Store vertices in a GPU friendly format using float4

	float* vertices = new float[cuda_mesh.verticesNo * 8];
	for (unsigned i = 0; i < cuda_mesh.verticesNo; i++) {
		// first float4 stores vertex xyz position
		vertices[i * 8 + 0] = cuda_mesh.vertices[i].position.x;
		vertices[i * 8 + 1] = cuda_mesh.vertices[i].position.y;
		vertices[i * 8 + 2] = cuda_mesh.vertices[i].position.z;
		vertices[i * 8 + 3] = 0.f;
		// second float4 stores vertex normal xyz
		vertices[i * 8 + 4] = cuda_mesh.vertices[i].normal.x;
		vertices[i * 8 + 5] = cuda_mesh.vertices[i].normal.y;
		vertices[i * 8 + 6] = cuda_mesh.vertices[i].normal.z;
		vertices[i * 8 + 7] = 0.f;
	}

	// copy vertex data to CUDA global memory
	cudaMalloc(&cudaVertices, cuda_mesh.verticesNo * 8 * sizeof(float));
	cudaMemcpy(cudaVertices, vertices, cuda_mesh.verticesNo * 8 * sizeof(float), cudaMemcpyHostToDevice);

	// Store precomputed triangle intersection data in a GPU friendly format using float4
	float* triangle_intersections = new float[cuda_mesh.trianglesNo * 20];

	for (unsigned i = 0; i < cuda_mesh.trianglesNo; i++) {
		// Normal
		triangle_intersections[20 * i + 4] = cuda_mesh.triangles[i].normal.x;
		triangle_intersections[20 * i + 5] = cuda_mesh.triangles[i].normal.y;
		triangle_intersections[20 * i + 6] = cuda_mesh.triangles[i].normal.z;
		triangle_intersections[20 * i + 7] = cuda_mesh.triangles[i]._d;
		// Precomputed plane normal of triangle edge 1
		triangle_intersections[20 * i + 8] = cuda_mesh.triangles[i]._e1.x;
		triangle_intersections[20 * i + 9] = cuda_mesh.triangles[i]._e1.y;
		triangle_intersections[20 * i + 10] = cuda_mesh.triangles[i]._e1.z;
		triangle_intersections[20 * i + 11] = cuda_mesh.triangles[i]._d1;
		// Precomputed plane normal of triangle edge 2
		triangle_intersections[20 * i + 12] = cuda_mesh.triangles[i]._e2.x;
		triangle_intersections[20 * i + 13] = cuda_mesh.triangles[i]._e2.y;
		triangle_intersections[20 * i + 14] = cuda_mesh.triangles[i]._e2.z;
		triangle_intersections[20 * i + 15] = cuda_mesh.triangles[i]._d2;
		// Precomputed plane normal of triangle edge 3
		triangle_intersections[20 * i + 16] = cuda_mesh.triangles[i]._e3.x;
		triangle_intersections[20 * i + 17] = cuda_mesh.triangles[i]._e3.y;
		triangle_intersections[20 * i + 18] = cuda_mesh.triangles[i]._e3.z;
		triangle_intersections[20 * i + 19] = cuda_mesh.triangles[i]._d3;
	}

	cudaMalloc(&cudaTriangleIntersectionData, cuda_mesh.trianglesNo * 20 * sizeof(float));
	cudaMemcpy(cudaTriangleIntersectionData, triangle_intersections, cuda_mesh.trianglesNo * 20 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&cudaTriangles, cuda_mesh.trianglesNo * sizeof(Triangle));
	cudaMemcpy(cudaTriangles, cuda_mesh.triangles, cuda_mesh.trianglesNo * sizeof(Triangle), cudaMemcpyHostToDevice);

	// Allocate CUDA-side data (global memory for corresponding textures) for Bounding Volume Hierarchy data

	// Leaf nodes triangle lists (indices to global triangle list)
	cudaMalloc(&cudaTriIdxList, triangle_index_list_no * sizeof(int));
	cudaMemcpy(cudaTriIdxList, triangle_index_list, triangle_index_list_no * sizeof(int), cudaMemcpyHostToDevice);

	float* limits = new float[cache_bvh_no * 6];

	for (unsigned i = 0; i < cache_bvh_no; i++) {
		// Texture-wise:
		limits[6 * i + 0] = cache_bvh[i].bottom.x;
		limits[6 * i + 1] = cache_bvh[i].top.x;
		limits[6 * i + 2] = cache_bvh[i].bottom.y;
		limits[6 * i + 3] = cache_bvh[i].top.y;
		limits[6 * i + 4] = cache_bvh[i].bottom.z;
		limits[6 * i + 5] = cache_bvh[i].top.z;
	}

	cudaMalloc(&cudaBVHlimits, cache_bvh_no * 6 * sizeof(float));
	cudaMemcpy(cudaBVHlimits, limits, cache_bvh_no * 6 * sizeof(float), cudaMemcpyHostToDevice);

	int* indices_or_trilists = new int[cache_bvh_no * 4];

	for (unsigned i = 0; i < cache_bvh_no; i++) {
		indices_or_trilists[4 * i + 0] = cache_bvh[i].u.leaf.count;
		indices_or_trilists[4 * i + 1] = cache_bvh[i].u.inner.idx_right;
		indices_or_trilists[4 * i + 2] = cache_bvh[i].u.inner.idx_left;
		indices_or_trilists[4 * i + 3] = cache_bvh[i].u.leaf.start_index_tri_list;
	}

	// copy BVH node attributes to CUDA global memory
	cudaMalloc(&cudaBVHindexesOrTrilists, cache_bvh_no * 4 * sizeof(unsigned));
	cudaMemcpy(cudaBVHindexesOrTrilists, indices_or_trilists, cache_bvh_no * 4 * sizeof(unsigned), cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[]) {
	GLFWwindow* window;

	glfwSetErrorCallback(glfw_error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);
	glfwWindowHint(GLFW_RED_BITS, 32);
	glfwWindowHint(GLFW_GREEN_BITS, 32);
	glfwWindowHint(GLFW_BLUE_BITS, 32);
	glfwWindowHint(GLFW_ALPHA_BITS, 32);

	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(render_width, render_height, "CUDA Path Tracer", nullptr, nullptr);

	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);

	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	// ignore vsync
	glfwSwapInterval(0);

	// only copy r/g/b
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

	cudaError_t cuda_err;

	int gl_device_id;
	unsigned int gl_device_count;

	cuda_err = cuda(GLGetDevices(&gl_device_count, &gl_device_id, 1u, cudaGLDeviceListAll));

	int cuda_device_id = (argc > 1) ? atoi(argv[1]) : gl_device_id;
	cuda_err = cuda(SetDevice(cuda_device_id));

	const bool multi_gpu = gl_device_id != cuda_device_id;
	struct cudaDeviceProp props;

	cuda_err = cuda(GetDeviceProperties(&props, gl_device_id));
	printf("GL   : %-24s (%2d)\n", props.name, props.multiProcessorCount);

	cuda_err = cuda(GetDeviceProperties(&props, cuda_device_id));
	printf("CUDA : %-24s (%2d)\n", props.name, props.multiProcessorCount);

	pxl_interop* interop = pxl_interop_create();

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	cuda_err = interop->set_size(width, height);

	glfwSetKeyCallback(window, glfw_key_callback);

	CudaMesh cuda_mesh = load_object("Data/dragon.ply");
	update_bvh(cuda_mesh, "Data/dragon.ply");
	upload_cuda_data(cuda_mesh);

	// Allocate ray queue buffer
	RayQueue* ray_queue_buffer;
	cuda(Malloc(&ray_queue_buffer, ray_queue_buffer_size * sizeof(RayQueue)));

	glm::vec4* blit_buffer;
	cuda(Malloc(&blit_buffer, render_width * render_height * sizeof(glm::vec4)));

	double previous_time = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {
		double delta = glfwGetTime() - previous_time;
		previous_time = glfwGetTime();

		glfw_fps(window);

		if (glfwGetKey(window, GLFW_KEY_MINUS)) {
			sun_position += glm::vec2(0.05 * delta, 0.05 * delta);
			sun_position_changed = true;
		}

		if (glfwGetKey(window, GLFW_KEY_EQUAL)) {
			sun_position -= glm::vec2(0.05 * delta, 0.05 * delta);
			sun_position_changed = true;
		}
		camera.update(window, delta);

		launch_kernels(interop->ca, blit_buffer, cudaBVHindexesOrTrilists, cudaBVHlimits, cudaTriangleIntersectionData, cudaTriIdxList, cudaTriangles);

		interop->blit();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	interop_destroy(interop);
	glfwDestroyWindow(window);
	glfwTerminate();

	cuda(DeviceReset());

	exit(EXIT_SUCCESS);
}
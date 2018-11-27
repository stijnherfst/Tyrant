#include "Bbox.h"
#include "Scene.h"
#include "stdafx.h"

#include "IMGUI/imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

const char* glsl_version = "#version 450 core";

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

		sprintf_s(tmp, 64, "(%u x %u) - FPS: %.2f ms: %.3f", width, height, fps,
				  (elapsed / frame_count) * 1000);

		glfwSetWindowTitle(window, tmp);

		frame_count = 0;
	}

	frame_count++;
}

static void glfw_error_callback(int error, const char* description) {
	std::cout << description << "\n";
}

static void glfw_key_callback(GLFWwindow* window, int key, int scancode,
							  int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}

int main(int argc, char* argv[]) {

	// OpenGL and GLFW setup
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

	window = glfwCreateWindow(render_width, render_height, "CUDA Path Tracer",
							  nullptr, nullptr);

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

	// IMGUI setup
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Setup style
	ImGui::StyleColorsDark();

	// CUDA setup
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

	Scene scene;
	scene.Load("Data/dragon.ply");

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
		static double blit_time;
		blit_time = glfwGetTime();
		camera.update(window, delta);

		blit_time = glfwGetTime() - blit_time;
		launch_kernels(interop->ca, blit_buffer, scene.gpuScene);
		interop->blit();

		// IMGUI
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		{
			static std::vector<float> frame_times;

			frame_times.push_back(delta);

			if (frame_times.size() > 200) {
				frame_times.erase(frame_times.begin());
			}

			ImGui::Begin("Performance");
			ImGui::PlotHistogram("", frame_times.data(), frame_times.size(), 0, "Frametimes", 0, FLT_MAX, { 400, 100 });
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::Text("Blit time %f", blit_time);
			ImGui::End();
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	interop_destroy(interop);
	glfwDestroyWindow(window);
	glfwTerminate();

	cuda(DeviceReset());

	exit(EXIT_SUCCESS);
}
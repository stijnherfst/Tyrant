#pragma once

constexpr static float pi = 3.1415926535897932f;

constexpr static unsigned window_width = 720;
constexpr static unsigned window_height = 720;

constexpr static unsigned render_width = 720;
constexpr static unsigned render_height = 720;

constexpr static int bvh_stack_size = 32;

constexpr static float epsilon = 0.001f;

extern glm::vec2 sun_position;
extern bool sun_position_changed;

struct RayQueue {
	glm::vec3 origin;
	glm::vec3 direction;
	glm::vec3 direct;
	float distance;
	int identifier;
	int bounces;
	int x;
	int y;
};

struct ShadowQueue {
	glm::vec3 origin;
	glm::vec3 direction;
	float sunlight;
	int buffer_index;
};

const unsigned int ray_queue_buffer_size = 1'048'576;
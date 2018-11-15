void RenderPixel(uint x, uint y, UniformSampler *sampler) {
	Ray ray = m_scene->Camera->CalculateRayFromPixel(x, y, sampler);

	float3 color(0.0f);
	float3 throughput(1.0f);
	SurfaceInteraction interaction;

	// Bounce the ray around the scene
	const uint maxBounces = 15;
	for (uint bounces = 0; bounces < maxBounces; ++bounces) {
		m_scene->Intersect(ray);

		// The ray missed. Return the background color
		if (ray.GeomID == INVALID_GEOMETRY_ID) {
			color += throughput * m_scene->BackgroundColor;
			break;
		}

		// Fetch the material
		Material *material = m_scene->GetMaterial(ray.GeomID);
		// The object might be emissive. If so, it will have a corresponding light
		// Otherwise, GetLight will return nullptr
		Light *light = m_scene->GetLight(ray.GeomID);

		// If this is the first bounce or if we just had a specular bounce,
		// we need to add the emmisive light
		if ((bounces == 0 || (interaction.SampledLobe & BSDFLobe::Specular) != 0) && light != nullptr) {
			color += throughput * light->Le();
		}

		interaction.Position = ray.Origin + ray.Direction * ray.TFar;
		interaction.Normal = normalize(m_scene->InterpolateNormal(ray.GeomID, ray.PrimID, ray.U, ray.V));
		interaction.OutputDirection = normalize(-ray.Direction);


		// Calculate the direct lighting
		color += throughput * SampleLights(sampler, interaction, material->bsdf, light);


		// Get the new ray direction
		// Choose the direction based on the bsdf        
		material->bsdf->Sample(interaction, sampler);
		float pdf = material->bsdf->Pdf(interaction);

		// Accumulate the weight
		throughput = throughput * material->bsdf->Eval(interaction) / pdf;

		// Shoot a new ray

		// Set the origin at the intersection point
		ray.Origin = interaction.Position;

		// Reset the other ray properties
		ray.Direction = interaction.InputDirection;
		ray.TNear = 0.001f;
		ray.TFar = infinity;
	}

	m_scene->Camera->FrameBufferData.SplatPixel(x, y, color);
}



float3 SampleLights(UniformSampler *sampler, SurfaceInteraction interaction, BSDF *bsdf, Light *hitLight) const {
	std::size_t numLights = m_scene->NumLights();

	float3 L(0.0f);
	for (uint i = 0; i < numLights; ++i) {
		Light *light = &m_scene->Lights[i];

		// Don't let a light contribute light to itself
		if (light == hitLight) {
			continue;
		}

		L = L + EstimateDirect(light, sampler, interaction, bsdf);
	}

	return L;
}

float3 EstimateDirect(Light *light, UniformSampler *sampler, SurfaceInteraction &interaction, BSDF *bsdf) const {
	float3 directLighting = float3(0.0f);

	// Only sample if the BRDF is non-specular 
	if ((bsdf->SupportedLobes & ~BSDFLobe::Specular) != 0) {
		float pdf;
		float3 Li = light->SampleLi(sampler, m_scene, interaction, &pdf);

		// Make sure the pdf isn't zero and the radiance isn't black
		if (pdf != 0.0f && !all(Li)) {
			directLighting += bsdf->Eval(interaction) * Li / pdf;
		}
	}

	return directLighting;
}
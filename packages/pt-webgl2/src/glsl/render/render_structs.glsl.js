export const render_structs = /* glsl */`

	struct Ray {

		vec3 origin;
		vec3 direction;
		// Negative means ordinary production traversal. A non-negative value
		// selects the canonical watertight range traversal and rejects every
		// represented hit at or below this exact world-space t. The source face
		// is excluded independently so roundoff cannot immediately re-hit it.
		float minimumDistanceExclusive;
		uint ignoredFaceIndex;
		// 0 none, 1 face, 2 welded edge, 3 welded vertex. Edge/vertex tokens
		// carry exact represented world coordinates so duplicate-index UV seams
		// share the same one-segment self-exclusion without a new GPU binding.
		uint sourceFeatureKind;
		uint sourceBoundaryId;
		uint sourcePrimitiveInstanceId;
		vec3 sourceFeatureA;
		vec3 sourceFeatureB;

	};

	void setOrdinaryRayRange( inout Ray ray ) {

		ray.minimumDistanceExclusive = -1.0;
		ray.ignoredFaceIndex = 0xffffffffu;
		ray.sourceFeatureKind = 0u;
		ray.sourceBoundaryId = 0u;
		ray.sourcePrimitiveInstanceId = 0u;
		ray.sourceFeatureA = vec3( 0.0 );
		ray.sourceFeatureB = vec3( 0.0 );

	}

	void setExactRayRange(
		inout Ray ray, vec3 exactOrigin, uint sourceFaceIndex
	) {

		ray.origin = exactOrigin;
		ray.minimumDistanceExclusive = 0.0;
		ray.ignoredFaceIndex = sourceFaceIndex;
		ray.sourceFeatureKind = 1u;
		ray.sourceBoundaryId = 0u;
		ray.sourcePrimitiveInstanceId = 0u;
		ray.sourceFeatureA = vec3( 0.0 );
		ray.sourceFeatureB = vec3( 0.0 );

	}

	struct SurfaceHit {

		uvec4 faceIndices;
		vec3 barycoord;
		vec3 faceNormal;
		// Canonical represented-triangle point from the inclusive watertight
		// solve. Exact optical continuation anchors here instead of independently
		// rounding origin + direction * t across an adjacent one-ULP layer.
		vec3 point;
		float side;
		float dist;

	};

	struct RenderState {

		bool firstRay;
		bool transmissiveRay;
		bool isShadowRay;
		float accumulatedRoughness;
		int transmissiveTraversals;
		int traversals;
		uint depth;
		float wavelength;
		float wavelengthPdf;
                vec3 throughput;
                FogMaterial fogMaterial;
                MediumStack mediumStack;
		// D3 — envMapIntensity of the LAST shaded surface; scales the forward env
		// pickup after a bounce (the BSDF half of the env MIS estimator). 1.0 until
		// the first surface is shaded (camera-visible env is never material-scaled).
		float envMapIntensity;

	};

	RenderState initRenderState() {

		RenderState result;
		result.firstRay = true;
		result.transmissiveRay = true;
		result.isShadowRay = false;
		result.accumulatedRoughness = 0.0;
		result.transmissiveTraversals = 0;
		result.traversals = 0;
		result.wavelength = 550.0;
		result.wavelengthPdf = 1.0 / 400.0;
		result.throughput = vec3( 1.0 );
                result.depth = 0u;
		initFogMaterial( result.fogMaterial );
                initMediumStack( result.mediumStack );
		result.envMapIntensity = 1.0;
		return result;

	}

	void setFogSurfaceRecord( const in FogMaterial material, inout SurfaceRecord surf ) {

		vec3 normal = vec3( 0.0, 0.0, 1.0 );
		SurfaceRecord fogSurface;
		fogSurface.volumeParticle = true;
		fogSurface.faceNormal = normal;
		fogSurface.frontFace = true;
		fogSurface.normal = normal;
		fogSurface.normalBasis = getBasisFromNormal( normal );
		fogSurface.oppositeNormal = - normal;
		fogSurface.oppositeNormalBasis = getBasisFromNormal( - normal );
		fogSurface.eta = 1.0;
		fogSurface.f0 = 0.0;
		fogSurface.roughness = 1.0;
		fogSurface.filteredRoughness = 1.0;
		fogSurface.oppositeRoughness = 1.0;
		fogSurface.oppositeFilteredRoughness = 1.0;
		fogSurface.metalness = 0.0;
		fogSurface.color = material.color;
		fogSurface.rgbColor = material.color;
                // The incoming edge already divides by the exact RGB-mixture
                // (or scalar hero) collision density. Keep source and scatter
                // coefficients unfactored at the volume vertex.
                fogSurface.emission = material.emission;
		fogSurface.ior = 1.0;
		fogSurface.transmission = 0.0;
		fogSurface.thinFilm = false;
		fogSurface.thinFilmEnabled = 0.0;
		fogSurface.thinFilmLayerCount = 0.0;
		fogSurface.thinFilmIncidentIor = 1.0;
		fogSurface.thinFilmAngleDependent = false;
		fogSurface.dispersionStrength = 0.0;
		// Volume particles already represent the accepted free-flight collision;
		// they must take the HG branch, never the surface-SSS branch in renderMain.
		fogSurface.sssSigmaT = 0.0;
		fogSurface.sssAnisotropyG = material.anisotropy;
		fogSurface.sssSigmaS = material.sigmaS;
		fogSurface.hasSpectralAttenuation = material.hasSpectralAttenuation;
		fogSurface.activeLayerTransmission = vec3( 1.0 );
		fogSurface.hasActiveLayer = false;
		fogSurface.oppositeLayerTransmission = vec3( 1.0 );
		fogSurface.hasOppositeLayer = false;
		fogSurface.materialIndex = material.materialIndex;
		fogSurface.attenuationColor = material.attenuationColor;
		fogSurface.attenuationDistance = material.attenuationDistance;
		fogSurface.attenuationThickness = 0.0;
		fogSurface.hasAttenuationThickness = false;
		fogSurface.clearcoatNormal = normal;
		fogSurface.clearcoatBasis = fogSurface.normalBasis;
		fogSurface.clearcoat = 0.0;
		fogSurface.clearcoatRoughness = 1.0;
		fogSurface.filteredClearcoatRoughness = 1.0;
		fogSurface.sheen = 0.0;
		fogSurface.sheenColor = vec3( 0.0 );
		fogSurface.sheenRoughness = 1.0;
		fogSurface.iridescence = 0.0;
		fogSurface.iridescenceIor = 1.0;
		fogSurface.iridescenceThickness = 0.0;
		fogSurface.specularColor = vec3( 1.0 );
		fogSurface.specularIntensity = 0.0;
		fogSurface.anisotropy = 0.0;
		fogSurface.anisotropyRotation = 0.0;
                fogSurface.spectralReflectanceCoeffs = vec3( 0.0 );
                fogSurface.hasSpectralReflectance = false;
		// Medium scattering is not a material-local IBL override.
		fogSurface.envMapIntensity = 1.0;
		fogSurface.lobeMask = 1u;
		surf = fogSurface;

	}

`;

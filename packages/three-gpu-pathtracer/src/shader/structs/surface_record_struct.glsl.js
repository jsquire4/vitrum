export const surface_record_struct = /* glsl */`

	struct SurfaceRecord {

		// surface type
		bool volumeParticle;

		// geometry
		vec3 faceNormal;
		bool frontFace;
		vec3 normal;
		mat3 normalBasis;
		mat3 normalInvBasis;

		// cached properties
		float eta;
		float f0;

		// material
		float roughness;
		float filteredRoughness;
		float metalness;
		vec3 color;
		vec3 emission;

		// transmission
		float ior;
		float transmission;
		bool thinFilm;
		float thinFilmEnabled;
		float thinFilmLayerCount;
		float thinFilmIncidentIor;
		bool thinFilmAngleDependent;
		float dispersionStrength;
		float sssSigmaT;
		float sssAnisotropyG;
		vec3 sssAlbedo;
		bool hasSpectralAttenuation;
		vec3 activeLayerTransmission;
		float activeLayerRoughness;
		bool hasActiveLayer;
		uint materialIndex;
		vec3 attenuationColor;
		float attenuationDistance;

		// clearcoat
		vec3 clearcoatNormal;
		mat3 clearcoatBasis;
		mat3 clearcoatInvBasis;
		float clearcoat;
		float clearcoatRoughness;
		float filteredClearcoatRoughness;

		// sheen
		float sheen;
		vec3 sheenColor;
		float sheenRoughness;

		// iridescence
		float iridescence;
		float iridescenceIor;
		float iridescenceThickness;

		// specular
		vec3 specularColor;
		float specularIntensity;

		// Sprint 4: P1 — lobeMask bitfield for BSDF lobe skipping.
		// bit 0 = diffuse, bit 1 = specular/GGX, bit 2 = sheen,
		// bit 3 = clearcoat, bit 4 = iridescence, bit 5 = transmission.
		// Set in getSurfaceRecord; consumed by bsdfEval guards.
		uint lobeMask;

		// Sprint 4: P2 — lite BSDF flag for indirect bounces (depth > 1).
		// When true, bsdfEval skips sheen/clearcoat/iridescence and
		// replaces multiscatter GGX with single-scatter.
		// Respects forceFullBSDF material override.
		bool liteMode;
	};

	struct ScatterRecord {
		float specularPdf;
		float pdf;
		vec3 direction;
		float throughput;
	};

`;

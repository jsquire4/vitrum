export const util_functions = /* glsl */`

	// TODO: possibly this should be renamed something related to material or path tracing logic

	#ifndef RAY_OFFSET
	#define RAY_OFFSET 1e-4
	#endif

	// adjust the hit point by the surface normal by a factor of some offset and the
	// maximum component-wise value of the current point to accommodate floating point
	// error as values increase.
	vec3 stepRayOrigin( vec3 rayOrigin, vec3 rayDirection, vec3 offset, float dist ) {

		vec3 point = rayOrigin + rayDirection * dist;
		vec3 absPoint = abs( point );
		float maxPoint = max( absPoint.x, max( absPoint.y, absPoint.z ) );
		return point + offset * ( maxPoint + 1.0 ) * RAY_OFFSET;

	}

	// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_volume/README.md#attenuation
	vec3 transmissionAttenuation( float dist, vec3 attColor, float attDist ) {

		vec3 ot = - log( attColor ) / attDist;
		return exp( - ot * dist );

	}

	// Approximate hero-wavelength scalar from a RGB triplet parameterized as [R,G,B].
	float heroScalarFromRgb( vec3 rgb, float heroWavelength ) {
		float tB = 1.0 - smoothstep( 470.0, 530.0, heroWavelength );
		float tR = smoothstep( 570.0, 650.0, heroWavelength );
		float tG = clamp( 1.0 - tB - tR, 0.0, 1.0 );
		return max( dot( rgb, vec3( tR, tG, tB ) ), 0.0 );
	}

	float heroWeightFromRgb( vec3 rgb, float heroWavelength ) {
		return heroScalarFromRgb( rgb, heroWavelength );
	}

	// Packed spectral μ(λ) grid: MaterialsTexture.js texels 20..27 (32 floats),
	// uniform wavelength samples 380..780 nm (matches SPECTRAL_GRID_* in JS).
	float readSpectralAttenuationMu( sampler2D materialsTex, uint materialIndex, uint spectralIdx ) {

		const uint MATERIAL_PIXELS = 85u;
		const uint SPECTRAL_BASE_TEXEL = 20u;
		uint texelOffset = SPECTRAL_BASE_TEXEL + spectralIdx / 4u;
		uint comp = spectralIdx % 4u;
		vec4 v = texelFetch1D( materialsTex, materialIndex * MATERIAL_PIXELS + texelOffset );
		return v[ int( comp ) ];

	}

	float spectralAttenuationMuHero( sampler2D materialsTex, uint materialIndex, float heroWavelength ) {

		const float L0 = 380.0;
		const float L1 = 780.0;
		float t = clamp( ( heroWavelength - L0 ) / max( L1 - L0, 1e-6 ), 0.0, 1.0 );
		float fi = t * 31.0;
		uint i0 = uint( floor( fi ) );
		uint i1 = min( i0 + 1u, 31u );
		float w = fract( fi );
		float mu0 = readSpectralAttenuationMu( materialsTex, materialIndex, i0 );
		float mu1 = readSpectralAttenuationMu( materialsTex, materialIndex, i1 );
		return mix( mu0, mu1, w );

	}

	// Hero-path Beer-Lambert: spectral materials use packed μ(λ); otherwise RGB attenuation + hero projection.
	float transmissionAttenuationHero(
		sampler2D materialsTex,
		float dist,
		vec3 attColor,
		float attDist,
		bool hasSpectral,
		uint materialIndex,
		float heroWavelength
	) {

		if ( ! hasSpectral ) {

			return heroScalarFromRgb( transmissionAttenuation( dist, attColor, attDist ), heroWavelength );

		}

		float muLambda = spectralAttenuationMuHero( materialsTex, materialIndex, heroWavelength );
		return exp( - muLambda * dist );

	}

	vec3 getHalfVector( vec3 wi, vec3 wo, float eta ) {

		// get the half vector - assuming if the light incident vector is on the other side
		// of the that it's transmissive.
		vec3 h;
		if ( wi.z > 0.0 ) {

			h = normalize( wi + wo );

		} else {

			// Scale by the ior ratio to retrieve the appropriate half vector
			// From Section 2.2 on computing the transmission half vector:
			// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
			h = normalize( wi + wo * eta );

		}

		h *= sign( h.z );
		return h;

	}

	vec3 getHalfVector( vec3 a, vec3 b ) {

		return normalize( a + b );

	}

	// The discrepancy between interpolated surface normal and geometry normal can cause issues when a ray
	// is cast that is on the top side of the geometry normal plane but below the surface normal plane. If
	// we find a ray like that we ignore it to avoid artifacts.
	// This function returns if the direction is on the same side of both planes.
	bool isDirectionValid( vec3 direction, vec3 surfaceNormal, vec3 geometryNormal ) {

		bool aboveSurfaceNormal = dot( direction, surfaceNormal ) > 0.0;
		bool aboveGeometryNormal = dot( direction, geometryNormal ) > 0.0;
		return aboveSurfaceNormal == aboveGeometryNormal;

	}

	// ray sampling x and z are swapped to align with expected background view
	vec2 equirectDirectionToUv( vec3 direction ) {

		// from Spherical.setFromCartesianCoords
		vec2 uv = vec2( atan( direction.z, direction.x ), acos( direction.y ) );
		uv /= vec2( 2.0 * PI, PI );

		// apply adjustments to get values in range [0, 1] and y right side up
		uv.x += 0.5;
		uv.y = 1.0 - uv.y;
		return uv;

	}

	vec3 equirectUvToDirection( vec2 uv ) {

		// undo above adjustments
		uv.x -= 0.5;
		uv.y = 1.0 - uv.y;

		// from Vector3.setFromSphericalCoords
		float theta = uv.x * 2.0 * PI;
		float phi = uv.y * PI;

		float sinPhi = sin( phi );

		return vec3( sinPhi * cos( theta ), cos( phi ), sinPhi * sin( theta ) );

	}

	// power heuristic for multiple importance sampling
	float misHeuristic( float a, float b ) {

		float aa = a * a;
		float bb = b * b;
		return aa / ( aa + bb );

	}

	// tentFilter from Peter Shirley's 'Realistic Ray Tracing (2nd Edition)' book, pg. 60
	// erichlof/THREE.js-PathTracing-Renderer/
	float tentFilter( float x ) {

		return x < 0.5 ? sqrt( 2.0 * x ) - 1.0 : 1.0 - sqrt( 2.0 - ( 2.0 * x ) );

	}
`;

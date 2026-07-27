import {
	MATERIAL_PIXELS,
	MATERIAL_UV_SELECTOR_TEXEL_OFFSET,
} from '../structs/materialStride.js';

export const util_functions = /* glsl */`

	// General path-tracing utility helpers shared by material, BSDF, and ray code.
	// math_functions is composed later; declare the canonical fallback basis here
	// because getBasisFromSelectedUv is parsed before its definition.
	mat3 getBasisFromNormal( vec3 normal );

	// Resolve the scene-local attributesArray layer for one mapped-rich material
	// slot. Four integer-valued float selectors are packed per material texel.
	// Authored TextureRef.texCoord ids may be sparse/arbitrarily large; only this
	// dense layer id reaches the shader.
	int readMaterialMapUvLayer(
		sampler2D materialsTex, uint materialIndex, uint mapIndex
	) {

		const uint MATERIAL_PIXELS = ${MATERIAL_PIXELS}u;
		const uint UV_SELECTOR_BASE = ${MATERIAL_UV_SELECTOR_TEXEL_OFFSET}u;
		uint selectorTexel = UV_SELECTOR_BASE + mapIndex / 4u;
		uint selectorComponent = mapIndex % 4u;
		vec4 selectors = texelFetch1D(
			materialsTex, materialIndex * MATERIAL_PIXELS + selectorTexel
		);
		return int( round( selectors[ int( selectorComponent ) ] ) );

	}

	// Build the tangent frame for the exact UV layer selected by a material map.
	// Authored/CPU-derived ATTR_TANGENT is defined against UV0 and remains the
	// preferred Mikk-style smooth basis for that lane. For UV1 and arbitrary
	// scene-local layers, derive dP/du and dP/dv from this hit triangle so a map
	// never inherits UV0's orientation. Degenerate UVs safely fall back to the
	// authored tangent and finally to an arbitrary basis around the normal.
	mat3 getBasisFromSelectedUv(
		sampler2D positionAttr,
		sampler2DArray attributesArray,
		int uvLayer,
		uvec3 faceIndices,
		vec3 normal,
		vec4 uv0TangentSample
	) {

		vec3 n = length( normal ) > 1e-6
			? normalize( normal )
			: vec3( 0.0, 0.0, 1.0 );
		vec3 tangent = vec3( 0.0 );
		vec3 bitangentReference = vec3( 0.0 );
		float handedness = 1.0;
		bool haveTangent = false;

		// The core tangent contract is UV0-based. Preserve its smooth frame there;
		// every other selected lane must be reconstructed from that lane's UVs.
		if ( uvLayer == ATTR_UV && length( uv0TangentSample.xyz ) > 1e-6 ) {

			tangent = uv0TangentSample.xyz;
			handedness = uv0TangentSample.w < 0.0 ? -1.0 : 1.0;
			haveTangent = true;

		} else {

			vec3 p0 = texelFetch1D( positionAttr, faceIndices.x ).xyz;
			vec3 p1 = texelFetch1D( positionAttr, faceIndices.y ).xyz;
			vec3 p2 = texelFetch1D( positionAttr, faceIndices.z ).xyz;
			vec2 uv0 = texelFetch1D( attributesArray, uvLayer, faceIndices.x ).xy;
			vec2 uv1 = texelFetch1D( attributesArray, uvLayer, faceIndices.y ).xy;
			vec2 uv2 = texelFetch1D( attributesArray, uvLayer, faceIndices.z ).xy;
			vec3 edge1 = p1 - p0;
			vec3 edge2 = p2 - p0;
			vec2 delta1 = uv1 - uv0;
			vec2 delta2 = uv2 - uv0;
			float determinant = delta1.x * delta2.y - delta1.y * delta2.x;

			if ( abs( determinant ) > 1e-10 ) {

				float inverseDeterminant = 1.0 / determinant;
				tangent = ( edge1 * delta2.y - edge2 * delta1.y ) * inverseDeterminant;
				bitangentReference = ( edge2 * delta1.x - edge1 * delta2.x ) * inverseDeterminant;
				bool finiteFrame = all( lessThan( abs( tangent ), vec3( 1e30 ) ) ) &&
					all( lessThan( abs( bitangentReference ), vec3( 1e30 ) ) );
				haveTangent = finiteFrame && length( tangent ) > 1e-6 &&
					length( bitangentReference ) > 1e-6;

				if ( haveTangent ) {

					handedness = dot( cross( n, tangent ), bitangentReference ) < 0.0
						? -1.0
						: 1.0;

				}

			}

		}

		if ( ! haveTangent && length( uv0TangentSample.xyz ) > 1e-6 ) {

			tangent = uv0TangentSample.xyz;
			handedness = uv0TangentSample.w < 0.0 ? -1.0 : 1.0;
			haveTangent = true;

		}

		tangent -= n * dot( tangent, n );
		if ( ! haveTangent || length( tangent ) <= 1e-6 ) {

			return getBasisFromNormal( n );

		}

		tangent = normalize( tangent );
		vec3 bitangent = cross( n, tangent ) * handedness;
		if ( length( bitangent ) <= 1e-6 ) {

			return getBasisFromNormal( n );

		}

		return mat3( tangent, normalize( bitangent ), n );

	}

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

		const uint MATERIAL_PIXELS = ${MATERIAL_PIXELS}u;
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

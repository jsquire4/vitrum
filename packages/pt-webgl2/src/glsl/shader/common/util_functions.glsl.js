import {
	MATERIAL_PIXELS,
	MATERIAL_UV_SELECTOR_TEXEL_OFFSET,
} from '../structs/materialStride.js';

export const util_functions = /* glsl */`

	// General path-tracing utility helpers shared by material, BSDF, and ray code.
	// math_functions is composed later; declare the canonical fallback basis here
	// because getBasisFromSelectedUv is parsed before its definition.
	mat3 getBasisFromNormal( vec3 normal );

	// GLSL ES implementations do not have to encode the library's INFINITY
	// sentinel as IEEE +Inf. WebGL composes INFINITY as max finite binary32, so
	// transport code must classify both representations as an unbounded ray.
	bool vitrumIsInfiniteDistance( float distance ) {

		return isinf( distance ) || distance >= INFINITY;

	}

	bool vitrumFiniteNonZeroVec3( vec3 value ) {

		float scale = max( abs( value.x ), max( abs( value.y ), abs( value.z ) ) );
		return scale > 0.0 && scale <= 3.402823e38;

	}

	float vitrumFiniteNonNegativeRadianceLaneProduct( float left, float right ) {

		const float F32_MAX = 3.402823466e38;
		if (
			! ( left >= 0.0 ) || isnan( left ) || isinf( left ) ||
			! ( right >= 0.0 ) || isnan( right ) || isinf( right )
		) return 0.0;
		if ( left == 0.0 || right == 0.0 ) return 0.0;
		if ( left > F32_MAX / right ) return F32_MAX;
		float product = left * right;
		return product >= 0.0 && ! isnan( product ) && ! isinf( product )
			? product
			: 0.0;

	}

	vec3 vitrumFiniteNonNegativeRadianceProduct( vec3 left, vec3 right ) {

		return vec3(
			vitrumFiniteNonNegativeRadianceLaneProduct( left.x, right.x ),
			vitrumFiniteNonNegativeRadianceLaneProduct( left.y, right.y ),
			vitrumFiniteNonNegativeRadianceLaneProduct( left.z, right.z )
		);

	}

	float vitrumFiniteNonNegativeRadianceLaneSum( float left, float right ) {

		const float F32_MAX = 3.402823466e38;
		if (
			! ( left >= 0.0 ) || isnan( left ) || isinf( left ) ||
			! ( right >= 0.0 ) || isnan( right ) || isinf( right )
		) return 0.0;
		if ( left > F32_MAX - right ) return F32_MAX;
		float sum = left + right;
		return sum >= 0.0 && ! isnan( sum ) && ! isinf( sum ) ? sum : 0.0;

	}

	vec3 vitrumFiniteNonNegativeRadianceSum( vec3 left, vec3 right ) {

		return vec3(
			vitrumFiniteNonNegativeRadianceLaneSum( left.x, right.x ),
			vitrumFiniteNonNegativeRadianceLaneSum( left.y, right.y ),
			vitrumFiniteNonNegativeRadianceLaneSum( left.z, right.z )
		);

	}

	vec3 vitrumNormalizeVec3( vec3 value, vec3 fallback ) {

		float scale = max( abs( value.x ), max( abs( value.y ), abs( value.z ) ) );
		if ( ! ( scale > 0.0 ) || scale > 3.402823e38 ) {

			return fallback;

		}
		vec3 scaled = value / scale;
		return scaled / length( scaled );

	}

	float vitrumLengthVec3( vec3 value ) {

		float scale = max( abs( value.x ), max( abs( value.y ), abs( value.z ) ) );
		if ( ! ( scale > 0.0 ) || scale > 3.402823e38 ) return 0.0;
		float result = scale * length( value / scale );
		return result > 0.0 && result <= 3.402823e38 ? result : 0.0;

	}

	// Auxiliary buffers need a finite monotonic depth even when the Euclidean
	// norm is larger than binary32 can represent. Unlike vitrumLengthVec3,
	// which fails closed for transport math, this helper saturates invalid-high
	// magnitudes so a remote primary hit cannot inject Inf/NaN into denoisers.
	float vitrumSaturatedLengthVec3( vec3 value ) {

		if ( any( isnan( value ) ) ) return 3.402823e38;
		float scale = max( abs( value.x ), max( abs( value.y ), abs( value.z ) ) );
		if ( ! ( scale > 0.0 ) ) return 0.0;
		if ( scale > 3.402823e38 ) return 3.402823e38;
		float result = scale * length( value / scale );
		return result > 0.0 && result <= 3.402823e38
			? result
			: 3.402823e38;

	}

	float vitrumPositiveProductOverSquare(
		float first,
		float second,
		float distance
	) {

		if (
			! ( first > 0.0 ) || isnan( first ) || isinf( first ) ||
			! ( second > 0.0 ) || isnan( second ) || isinf( second ) ||
			! ( distance > 0.0 ) || isnan( distance ) || isinf( distance )
		) return 0.0;
		float result = exp2(
			log2( first ) + log2( second ) - 2.0 * log2( distance )
		);
		return result > 0.0 && ! isnan( result ) && ! isinf( result )
			? result
			: 0.0;

	}

	float vitrumInverseSquareDistance( vec3 delta ) {

		float distance = vitrumLengthVec3( delta );
		return vitrumPositiveProductOverSquare( 1.0, 1.0, distance );

	}

	struct VitrumAreaVectorMeasure {

		vec3 normal;
		float area;
		float edgeScale;
		bool valid;

	};

	// Shared finite-area geometry contract. Both axes are divided by one
	// max-component scale before forming their cross product; the unit normal is
	// recovered in that O(1) domain and coefficient*|u x v| is rescaled one
	// multiplication at a time. Exact collinearity and every non-finite or
	// unrepresentable result fail closed.
	VitrumAreaVectorMeasure vitrumMeasureAreaVector(
		vec3 u, vec3 v, float coefficient
	) {

		VitrumAreaVectorMeasure result;
		result.normal = vec3( 0.0, 1.0, 0.0 );
		result.area = 0.0;
		result.edgeScale = 0.0;
		result.valid = false;
		if (
			any( isnan( u ) ) || any( isinf( u ) ) ||
			any( isnan( v ) ) || any( isinf( v ) ) ||
			! ( coefficient > 0.0 ) || isnan( coefficient ) || isinf( coefficient )
		) {

			return result;

		}
		float edgeScale = max(
			max( abs( u.x ), max( abs( u.y ), abs( u.z ) ) ),
			max( abs( v.x ), max( abs( v.y ), abs( v.z ) ) )
		);
		if ( ! ( edgeScale > 0.0 ) || isinf( edgeScale ) ) return result;
		vec3 areaVector = cross( u / edgeScale, v / edgeScale );
		float crossScale = max(
			abs( areaVector.x ), max( abs( areaVector.y ), abs( areaVector.z ) )
		);
		if ( ! ( crossScale > 0.0 ) || isnan( crossScale ) || isinf( crossScale ) ) {

			return result;

		}
		vec3 areaDirection = areaVector / crossScale;
		float directionLength = length( areaDirection );
		if (
			! ( directionLength > 0.0 ) ||
			isnan( directionLength ) || isinf( directionLength )
		) {

			return result;

		}
		vec3 normal = areaDirection / directionLength;
		float area = coefficient * ( crossScale * directionLength );
		area = ( area * edgeScale ) * edgeScale;
		float inverseArea = 1.0 / area;
		if (
			! ( area > 0.0 ) || isnan( area ) || isinf( area ) ||
			! ( inverseArea > 0.0 ) || isnan( inverseArea ) || isinf( inverseArea ) ||
			any( isnan( normal ) ) || any( isinf( normal ) )
		) {

			return result;

		}
		result.normal = normal;
		result.area = area;
		result.edgeScale = edgeScale;
		result.valid = true;
		return result;

	}

		// Convert a uniform finite-area density to solid-angle measure without ever
		// forming distance² or multiplying independently tiny normalized terms.
		// Work in log2 space so an anisotropic-but-representable emitter cannot turn
		// both sides of the quotient into zero before their finite ratio is taken.
		// A valid result that cannot be represented as finite binary32 fails closed.
		float vitrumAreaToSolidAnglePdf(
			float distance,
			float cosine,
			VitrumAreaVectorMeasure measure
	) {

		if (
			! ( distance > 0.0 ) || isnan( distance ) || isinf( distance ) ||
				! ( cosine > 0.0 ) || isnan( cosine ) || isinf( cosine ) ||
				! measure.valid ||
				! ( measure.area > 0.0 ) ||
				isnan( measure.area ) || isinf( measure.area )
			) {

				return 0.0;

			}
			float logResult =
				2.0 * log2( distance ) -
				log2( measure.area ) -
				log2( cosine );
			if ( isnan( logResult ) || isinf( logResult ) ) return 0.0;
			float result = exp2( logResult );
			return result > 0.0 && ! isnan( result ) && ! isinf( result )
				? result
				: 0.0;

	}

	// Exact affine coordinates from the dominant 2D projection. The returned z
	// lane is 1 for success and 0 for failure.
	vec3 vitrumAreaVectorCoordinates(
		vec3 u, vec3 v, vec3 relative, VitrumAreaVectorMeasure measure
	) {

		if ( ! measure.valid ) return vec3( 0.0 );
		vec3 scaledU = u / measure.edgeScale;
		vec3 scaledV = v / measure.edgeScale;
		vec3 scaledRelative = relative / measure.edgeScale;
		vec3 absNormal = abs( measure.normal );
		float determinant;
		float uNumerator;
		float vNumerator;
		if ( absNormal.x >= absNormal.y && absNormal.x >= absNormal.z ) {

			determinant = scaledU.y * scaledV.z - scaledU.z * scaledV.y;
			uNumerator = scaledRelative.y * scaledV.z - scaledRelative.z * scaledV.y;
			vNumerator = scaledU.y * scaledRelative.z - scaledU.z * scaledRelative.y;

		} else if ( absNormal.y >= absNormal.z ) {

			determinant = scaledU.z * scaledV.x - scaledU.x * scaledV.z;
			uNumerator = scaledRelative.z * scaledV.x - scaledRelative.x * scaledV.z;
			vNumerator = scaledU.z * scaledRelative.x - scaledU.x * scaledRelative.z;

		} else {

			determinant = scaledU.x * scaledV.y - scaledU.y * scaledV.x;
			uNumerator = scaledRelative.x * scaledV.y - scaledRelative.y * scaledV.x;
			vNumerator = scaledU.x * scaledRelative.y - scaledU.y * scaledRelative.x;

		}
		if ( determinant == 0.0 || isnan( determinant ) || isinf( determinant ) ) {

			return vec3( 0.0 );

		}
		vec2 coordinates = vec2( uNumerator, vNumerator ) / determinant;
		if ( any( isnan( coordinates ) ) || any( isinf( coordinates ) ) ) {

			return vec3( 0.0 );

		}
		return vec3( coordinates, 1.0 );

	}

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

		vec3 n = vitrumNormalizeVec3( normal, vec3( 0.0, 0.0, 1.0 ) );
		vec3 tangent = vec3( 0.0 );
		vec3 bitangentReference = vec3( 0.0 );
		float handedness = 1.0;
		bool haveTangent = false;

		// The core tangent contract is UV0-based. Preserve its smooth frame there;
		// every other selected lane must be reconstructed from that lane's UVs.
		if ( uvLayer == ATTR_UV && vitrumFiniteNonZeroVec3( uv0TangentSample.xyz ) ) {

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
			float uvScale = max(
				max( abs( delta1.x ), abs( delta1.y ) ),
				max( abs( delta2.x ), abs( delta2.y ) )
			);
			bool validUvScale = uvScale > 0.0 && uvScale <= 3.402823e38;
			float inverseUvScale = validUvScale ? 1.0 / uvScale : 0.0;
			vec2 normalizedDelta1 = delta1 * inverseUvScale;
			vec2 normalizedDelta2 = delta2 * inverseUvScale;
			float determinant =
				normalizedDelta1.x * normalizedDelta2.y -
				normalizedDelta1.y * normalizedDelta2.x;

			if (
				validUvScale &&
				abs( determinant ) > 1e-7
			) {

				float inverseDeterminant = 1.0 / determinant;
				tangent =
					( edge1 * normalizedDelta2.y - edge2 * normalizedDelta1.y ) *
					inverseDeterminant;
				bitangentReference =
					( edge2 * normalizedDelta1.x - edge1 * normalizedDelta2.x ) *
					inverseDeterminant;
				haveTangent =
					vitrumFiniteNonZeroVec3( tangent ) &&
					vitrumFiniteNonZeroVec3( bitangentReference );

				if ( haveTangent ) {

					handedness = dot( cross( n, tangent ), bitangentReference ) < 0.0
						? -1.0
						: 1.0;

				}

			}

		}

		if ( ! haveTangent && vitrumFiniteNonZeroVec3( uv0TangentSample.xyz ) ) {

			tangent = uv0TangentSample.xyz;
			handedness = uv0TangentSample.w < 0.0 ? -1.0 : 1.0;
			haveTangent = true;

		}

		tangent -= n * dot( tangent, n );
		if ( ! haveTangent || ! vitrumFiniteNonZeroVec3( tangent ) ) {

			return getBasisFromNormal( n );

		}

		tangent = vitrumNormalizeVec3( tangent, vec3( 1.0, 0.0, 0.0 ) );
		vec3 bitangent = cross( n, tangent ) * handedness;
		if ( ! vitrumFiniteNonZeroVec3( bitangent ) ) {

			return getBasisFromNormal( n );

		}

		return mat3(
			tangent,
			vitrumNormalizeVec3( bitangent, vec3( 0.0, 1.0, 0.0 ) ),
			n
		);

	}

	#ifndef RAY_OFFSET
	#define RAY_OFFSET 1.175494351e-38
	#endif

	// adjust the hit point by the surface normal by a factor of some offset and the
	// maximum component-wise value of the current point to accommodate floating point
	// error as values increase. RAY_OFFSET is scene-relative; the coordinate term is
	// four binary32 ULPs and therefore translation-safe without multiplying the two.
	vec3 stepRayOrigin( vec3 rayOrigin, vec3 rayDirection, vec3 offset, float dist ) {

		vec3 point = rayOrigin + rayDirection * dist;
		vec3 absPoint = abs( point );
		float maxPoint = max( absPoint.x, max( absPoint.y, absPoint.z ) );
		float coordinateStep = maxPoint * ( 4.0 * 1.192092896e-7 );
		float step = max( RAY_OFFSET, max( coordinateStep, 1.175494351e-38 ) );
		return point + offset * step;

	}

	// Evaluate exp( - extinction * dist ) without ever forming 0 * infinity.
	// This is also the defensive boundary for malformed transport inputs: a NaN,
	// negative distance, or negative extinction fails closed instead of poisoning
	// every later path-throughput operation.
	float extinctionTransmittance( float extinction, float dist ) {

		if ( isnan( extinction ) || extinction < 0.0 || isnan( dist ) || dist < 0.0 ) {

			return 0.0;

		}
		if ( dist == 0.0 ) return 1.0;
		if ( vitrumIsInfiniteDistance( dist ) ) {
			return extinction == 0.0 ? 1.0 : 0.0;
		}
		if ( isinf( extinction ) ) return 0.0;
		return exp( - extinction * dist );

	}

	vec3 extinctionTransmittance( vec3 extinction, float dist ) {

		return vec3(
			extinctionTransmittance( extinction.x, dist ),
			extinctionTransmittance( extinction.y, dist ),
			extinctionTransmittance( extinction.z, dist )
		);

	}

	// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_volume/README.md#attenuation
	vec3 transmissionAttenuation( float dist, vec3 attColor, float attDist ) {

		if (
			isnan( attDist ) || attDist <= 0.0 ||
			any( isnan( attColor ) ) || any( isinf( attColor ) ) ||
			any( lessThan( attColor, vec3( 0.0 ) ) ) ||
			any( greaterThan( attColor, vec3( 1.0 ) ) )
		) {

			return vec3( 0.0 );

		}
		if ( vitrumIsInfiniteDistance( attDist ) ) {

			return extinctionTransmittance( vec3( 0.0 ), dist );

		}
		vec3 ot = - log( attColor ) / attDist;
		return extinctionTransmittance( ot, dist );

	}

	// Approximate hero-wavelength scalar from a RGB triplet parameterized as [R,G,B].
	float heroScalarFromRgb( vec3 rgb, float heroWavelength ) {
		float tB = 1.0 - smoothstep( 470.0, 530.0, heroWavelength );
		float tR = smoothstep( 570.0, 650.0, heroWavelength );
		float tG = clamp( 1.0 - tB - tR, 0.0, 1.0 );
		return max( dot( rgb, vec3( tR, tG, tB ) ), 0.0 );
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
		return extinctionTransmittance( muLambda, dist );

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

		if (
			! ( a >= 0.0 ) || ! ( b >= 0.0 ) ||
			isinf( a ) || isinf( b )
		) return 0.0;
		float pdfScale = max( a, b );
		if ( ! ( pdfScale > 0.0 ) ) return 0.0;
		float scaledA = a / pdfScale;
		float scaledB = b / pdfScale;
		float aa = scaledA * scaledA;
		float bb = scaledB * scaledB;
		return aa / ( aa + bb );

	}

	// tentFilter from Peter Shirley's 'Realistic Ray Tracing (2nd Edition)' book, pg. 60
	// erichlof/THREE.js-PathTracing-Renderer/
	float tentFilter( float x ) {

		return x < 0.5 ? sqrt( 2.0 * x ) - 1.0 : 1.0 - sqrt( 2.0 - ( 2.0 * x ) );

	}
`;

/**
 * No-path-loop next-event resolve.
 *
 * The replay pass stores one uniformly selected, visibility-tested NEE vertex in
 * four RGBA32F attachments. This shader re-hits that one incoming ray, rebuilds
 * the exact mapped/layered SurfaceRecord, evaluates the authored BSDF/PDF, and
 * applies the Horvitz-Thompson K factor. It deliberately contains no path-bounce
 * loop; that separation is required for bounded execution on ANGLE/SwiftShader.
 */
export const NEE_RESOLVE_MAIN = /* glsl */ `

	uniform sampler2D uNeeCandidate0;
	uniform sampler2D uNeeCandidate1;
	uniform sampler2D uNeeCandidate2;
	uniform sampler2D uNeeCandidate3;

	bool neeFiniteFloat( float value ) {

		return ! isnan( value ) && ! isinf( value );

	}

        bool neeFiniteVec3( vec3 value ) {

		return ! any( isnan( value ) ) && ! any( isinf( value ) );

        }

        vec3 neeOctDecodeDirection( vec2 encoded ) {

                vec3 n = vec3(
                        encoded,
                        1.0 - abs( encoded.x ) - abs( encoded.y )
                );
                if ( n.z < 0.0 ) {

                        vec2 signNotZero = vec2(
                                n.x >= 0.0 ? 1.0 : -1.0,
                                n.y >= 0.0 ? 1.0 : -1.0
                        );
                        n.xy = ( 1.0 - abs( n.yx ) ) * signNotZero;

                }
                return normalize( n );

        }

	float neeHeroWavelengthPdf( float wavelength ) {

		if ( uSpectralRendering == 0 ) return 1.0;
		float tablePosition = clamp( ( wavelength - 380.0 ) / 5.0, 0.0, 80.0 );
		int lo = min( int( floor( tablePosition ) ), 80 );
		float t = tablePosition - float( lo );
		return misMixturePdf( lo, t );

	}

	void main() {

		ivec2 pixel = ivec2( gl_FragCoord.xy );
		vec4 candidate0 = texelFetch( uNeeCandidate0, pixel, 0 );
		vec4 candidate1 = texelFetch( uNeeCandidate1, pixel, 0 );
		vec4 candidate2 = texelFetch( uNeeCandidate2, pixel, 0 );
		vec4 candidate3 = texelFetch( uNeeCandidate3, pixel, 0 );
		uint flags = floatBitsToUint( candidate3.w );
		bool valid = ( flags & 1u ) != 0u;
		bool fogPreResolved = ( flags & 2u ) != 0u;
                bool deltaLight = ( flags & 4u ) != 0u;
                uint pathDepth = ( flags >> 3u ) & 127u;
                uint candidateCount = ( flags >> 10u ) & 127u;
                vec3 lightDirection = neeOctDecodeDirection( candidate3.xy );
                float bdptCrossFamilyMisWeight = candidate3.z;

		pc_fragColor = vec4( 0.0 );
		if ( ! valid || candidateCount == 0u || candidateCount > 64u ) return;
		if (
			! neeFiniteVec3( candidate0.xyz ) ||
			! neeFiniteFloat( candidate0.w ) ||
			candidate0.w < 0.0 ||
			! neeFiniteVec3( candidate1.xyz ) ||
			length( candidate1.xyz ) < 1e-8 ||
			! neeFiniteFloat( candidate1.w ) ||
			candidate1.w < 380.0 || candidate1.w > 780.0 ||
			! neeFiniteVec3( candidate2.xyz ) ||
			! neeFiniteFloat( candidate2.w ) ||
                        ! neeFiniteVec3( lightDirection ) ||
                        length( lightDirection ) < 1e-8 ||
                        ! neeFiniteFloat( bdptCrossFamilyMisWeight ) ||
                        bdptCrossFamilyMisWeight < 0.0 ||
                        bdptCrossFamilyMisWeight > 1.0
		) return;

		float htScale = float( candidateCount );
		if ( fogPreResolved ) {

			pc_fragColor = vec4( candidate2.rgb * htScale, 0.0 );
			return;

		}

		float lightPdf = candidate2.w;
		if ( lightPdf <= 0.0 ) return;

		Ray incomingRay;
		incomingRay.origin = candidate0.xyz;
		incomingRay.direction = normalize( candidate1.xyz );
		FogMaterial noFog;
		noFog.fogVolume = false;
		SurfaceHit surfaceHit;
		int hitType = traceScene( incomingRay, noFog, surfaceHit );
		// A failed/mismatched re-hit is a zero-valued proposal. Never resample or
		// condition on it: doing so would change the original proposal density.
		if ( hitType != SURFACE_HIT ) return;

		uint materialIndex = uTexelFetch1D(
			materialIndexAttribute,
			surfaceHit.faceIndices.w
		).r;
		SurfaceRecord surf;
		if (
			getSurfaceRecord(
				materialIndex,
				surfaceHit,
				attributesArray,
				candidate0.w,
				int( pathDepth ),
				candidate1.w,
				true,
				surf
			) != HIT_SURFACE
		) return;

		vec3 sampleColor;
		float bsdfPdf = bsdfResult(
                        - incomingRay.direction,
                        lightDirection,
			surf,
			candidate1.w,
			sampleColor
		);
		if (
			bsdfPdf <= 0.0 ||
			! neeFiniteFloat( bsdfPdf ) ||
			! neeFiniteVec3( sampleColor ) ||
			! all( greaterThanEqual( sampleColor, vec3( 0.0 ) ) )
		) return;

		float wavelengthPdf = neeHeroWavelengthPdf( candidate1.w );
		if ( wavelengthPdf <= 0.0 || ! neeFiniteFloat( wavelengthPdf ) ) return;
                #if FEATURE_BDPT
                float misWeight = bdptCrossFamilyMisWeight;
                #else
                float misWeight = deltaLight
                        ? 1.0
                        : misHeuristic( lightPdf, bsdfPdf );
                #endif
		vec3 spectralContribution =
			candidate2.rgb *
			pathThroughputFromRgb( sampleColor, candidate1.w );
		vec3 resolved = wavelengthToRGB(
			candidate1.w,
			spectralContribution,
			wavelengthPdf
            ) * misWeight * htScale / lightPdf;
		if ( ! neeFiniteVec3( resolved ) ) return;
		pc_fragColor = vec4( resolved, 0.0 );

	}
`;

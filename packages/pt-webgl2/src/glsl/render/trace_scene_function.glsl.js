export const trace_scene_function = /* glsl */`

	#define NO_HIT 0
	#define SURFACE_HIT 1
	#define LIGHT_HIT 2
	#define FOG_HIT 3
	#define INVALID_HIT -1

	// Passing the global variable 'lights' into this function caused shader program errors.
	// So global variables like 'lights' and 'bvh' were moved out of the function parameters.
	// For more information, refer to: https://github.com/gkjohnson/three-gpu-pathtracer/pull/457
	int traceScene(
		Ray ray, const in MediumStack mediumStack,
		const in FogMaterial fogMaterial, float heroWavelength,
		inout SurfaceHit surfaceHit
		) {

			int result = NO_HIT;
			// BVH traversal writes dist only when it accepts a triangle. Fog
			// free-flight comparison must therefore start from an explicit miss
			// distance rather than reading an uninitialized SurfaceHit field.
			surfaceHit.dist = INFINITY;
			bool invalidRange = false;
		#if ADVANCED_OPTICAL_TRANSPORT
			bool hit = ray.minimumDistanceExclusive >= 0.0
				? bvhIntersectExactRangeFirstHit( ray, surfaceHit, invalidRange )
				: bvhIntersectCanonicalInitialFirstHit(
					ray, surfaceHit, invalidRange
				);
		#else
			bool hit = ray.minimumDistanceExclusive >= 0.0
				? bvhIntersectExactRangeFirstHit( ray, surfaceHit, invalidRange )
				: bvhIntersectFirstHit(
					bvh, ray.origin, ray.direction,
					surfaceHit.faceIndices, surfaceHit.faceNormal,
					surfaceHit.barycoord, surfaceHit.side, surfaceHit.dist
				);
		#endif
			if ( invalidRange ) return INVALID_HIT;

		#if FEATURE_FOG

		if ( fogMaterial.fogVolume ) {

			// Censor free flight at both the next geometric surface and the
			// remaining KHR volume-distance budget. A collision wins only on a
			// strict interior distance; an exact tie belongs to the surface/cap
			// survival atom. This intentionally has no geometric clearance term:
			// rejecting collisions within RAY_OFFSET of a surface biases the law.
			float segmentLimit = mediumEffectiveSegmentDistance(
				mediumStack, max( surfaceHit.dist, 0.0 )
			);
			if ( segmentLimit < 0.0 ) return INVALID_HIT;
			float particleDist = fogFreeFlightSampleDistance(
				materials, fogMaterial, heroWavelength, rand2( 1 )
			);
			if ( particleDist < segmentLimit ) {

				surfaceHit.side = 1.0;
				surfaceHit.faceNormal = normalize( - ray.direction );
				surfaceHit.dist = particleDist;
				return FOG_HIT;

			}

		}

		#endif

		if ( hit ) {

			result = SURFACE_HIT;

		}

		return result;

	}

`;

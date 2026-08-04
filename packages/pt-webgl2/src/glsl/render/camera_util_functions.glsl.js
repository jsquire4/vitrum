export const camera_util_functions = /* glsl */`

	vec3 ndcToRayOrigin( vec2 coord ) {

		vec4 rayOrigin4 = cameraWorldMatrix * invProjectionMatrix * vec4( coord, - 1.0, 1.0 );
		return rayOrigin4.xyz / rayOrigin4.w;
	}

	Ray getCameraRay() {

		vec2 ssd = vec2( 1.0 ) / resolution;
		vec3 cameraForward = vitrumNormalizeVec3(
			( cameraWorldMatrix * vec4( 0.0, 0.0, - 1.0, 0.0 ) ).xyz,
			vec3( 0.0, 0.0, - 1.0 )
		);

		// Jitter the camera ray by finding a uv coordinate at a random sample
		// around this pixel's UV coordinate for AA
			vec2 ruv = rand2( 0 );
			vec2 logicalUv = ( gl_FragCoord.xy + uTileOrigin ) / resolution;
			vec2 jitteredUv = logicalUv + vec2( tentFilter( ruv.x ) * ssd.x, tentFilter( ruv.y ) * ssd.y );
		Ray ray;

		#if CAMERA_TYPE == 2

			// Equirectangular projection
			vec4 rayDirection4 = vec4( equirectUvToDirection( jitteredUv ), 0.0 );
			vec4 rayOrigin4 = vec4( 0.0, 0.0, 0.0, 1.0 );

			rayDirection4 = cameraWorldMatrix * rayDirection4;
			rayOrigin4 = cameraWorldMatrix * rayOrigin4;

			ray.direction = vitrumNormalizeVec3( rayDirection4.xyz, cameraForward );
			ray.origin = rayOrigin4.xyz / rayOrigin4.w;

		#else

			// get [- 1, 1] normalized device coordinates
			vec2 ndc = 2.0 * jitteredUv - vec2( 1.0 );
			ray.origin = ndcToRayOrigin( ndc );

			#if CAMERA_TYPE == 1

				// Orthographic projection
				ray.direction = ( cameraWorldMatrix * vec4( 0.0, 0.0, - 1.0, 0.0 ) ).xyz;
				ray.direction = vitrumNormalizeVec3( ray.direction, cameraForward );

			#else

				// Perspective projection
				ray.direction = vitrumNormalizeVec3(
					mat3( cameraWorldMatrix ) *
						( invProjectionMatrix * vec4( ndc, 0.0, 1.0 ) ).xyz,
					cameraForward
				);

			#endif

		#endif

		vec3 baseDirection = vitrumNormalizeVec3( ray.direction, cameraForward );
		ray.direction = baseDirection;

		#if FEATURE_DOF
		if ( physicalCamera.bokehSize > 0.0 ) {

			// Get the aperture sample. A zero aperture deliberately bypasses this
			// block, preserving the pinhole ray and the pinhole RNG sequence.
			// if blades === 0 then we assume a circle
			vec3 shapeUVW= rand3( 1 );
			int blades = physicalCamera.apertureBlades;
			float anamorphicRatio = physicalCamera.anamorphicRatio;
			vec2 apertureSample = sampleAperture( blades, shapeUVW );
			apertureSample *= physicalCamera.bokehSize * 0.5 * 1e-3;

			// Build the saturated anamorphic factors without ever evaluating
			// 1.0 / ratio for a positive subnormal ratio. A ternary/mix is not
			// used because implementations may eagerly evaluate both operands.
			vec2 anamorphicScale;
			if ( anamorphicRatio < 1.0 ) {

				anamorphicScale = vec2( anamorphicRatio, 1.0 );

			} else {

				anamorphicScale = vec2( 1.0, 1.0 / anamorphicRatio );

			}

			// rotate the aperture shape
			apertureSample =
				rotateVector( apertureSample, physicalCamera.apertureRotation ) *
				anamorphicScale;

			// Build the focus vector in camera-relative geometry. Constructing a
			// world-space focal point and then subtracting the shifted ray origin
			// loses the entire focus vector when the camera translation is large.
			vec3 apertureWorldOffset =
				( cameraWorldMatrix * vec4( apertureSample, 0.0, 0.0 ) ).xyz;
			vec3 candidateOrigin = ray.origin + apertureWorldOffset;
			float apertureOffsetScale = max(
				abs( apertureWorldOffset.x ),
				max( abs( apertureWorldOffset.y ), abs( apertureWorldOffset.z ) )
			);
			float relativeFocusScale = max(
				physicalCamera.focusDistance,
				apertureOffsetScale
			);

			// Divide both terms by one positive common scale before subtraction.
			// The direction is unchanged mathematically, while each operand stays
			// within unit magnitude for the complete positive-float focus domain.
			vec3 relativeFocusDirection =
				baseDirection *
					( physicalCamera.focusDistance / relativeFocusScale ) -
				apertureWorldOffset / relativeFocusScale;
			bool candidateOriginFinite =
				! any( isnan( candidateOrigin ) ) &&
				! any( isinf( candidateOrigin ) );

			// Apply aperture translation and refocusing atomically. If either
			// compound operation is invalid, or the rounded scale-equivalent
			// terms are exactly identical, retain the original pinhole ray. The
			// latter is the only numerically unresolvable direction; camera-space
			// local Z is disjoint from the aperture's local XY plane.
			if (
				candidateOriginFinite &&
				vitrumFiniteNonZeroVec3( relativeFocusDirection )
			) {

				ray.origin = candidateOrigin;
				ray.direction = vitrumNormalizeVec3(
					relativeFocusDirection,
					baseDirection
				);

			}

		}
		#endif

		ray.direction = vitrumNormalizeVec3( ray.direction, baseDirection );
		setOrdinaryRayRange( ray );

		return ray;

	}

`;

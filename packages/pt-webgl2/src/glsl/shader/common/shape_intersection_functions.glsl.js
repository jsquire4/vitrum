export const shape_intersection_functions = /* glsl */`

	bool areaLightPlaneCoordinates(
		vec3 center,
		vec3 normal,
		vec3 u,
		vec3 v,
		vec3 rayOrigin,
		vec3 rayDirection,
		out float planeDist,
		out vec2 coordinates
	) {

		planeDist = 0.0;
		coordinates = vec2( 0.0 );

		VitrumAreaVectorMeasure areaMeasure =
			vitrumMeasureAreaVector( u, v, 1.0 );
		if ( ! areaMeasure.valid ) return false;
		float rayDenominator = dot( rayDirection, areaMeasure.normal );
		if ( rayDenominator == 0.0 || isnan( rayDenominator ) || isinf( rayDenominator ) ) {

			return false;

		}

		float t = dot( center - rayOrigin, areaMeasure.normal ) / rayDenominator;
		if (
			! ( t > max( RAY_OFFSET * 0.01, 1.175494351e-38 ) ) ||
			isnan( t ) || isinf( t )
		) {

			return false;

		}

		vec3 relative = rayOrigin + rayDirection * t - center;
		vec3 areaCoordinates = vitrumAreaVectorCoordinates(
			u, v, relative, areaMeasure
		);
		if ( areaCoordinates.z == 0.0 ) {

			return false;

		}

		planeDist = t;
		coordinates = areaCoordinates.xy;
		return true;

	}

	// Finds the point where the ray intersects the plane defined by u and v and checks if this point
	// falls in the bounds of the rectangle on that same plane.
	// Plane intersection: https://lousodrome.net/blog/light/2020/07/03/intersection-of-a-ray-and-a-plane/
	bool intersectsRectangle( vec3 center, vec3 normal, vec3 u, vec3 v, vec3 rayOrigin, vec3 rayDirection, inout float dist ) {

		float planeDist;
		vec2 coordinates;
		if (
			areaLightPlaneCoordinates(
				center, normal, u, v, rayOrigin, rayDirection, planeDist, coordinates
			) &&
			all( lessThanEqual( abs( coordinates ), vec2( 0.5 ) ) )
		) {

			dist = planeDist;
			return true;

		}

		return false;

	}

	// Finds the point where the ray intersects the plane defined by u and v and checks if this point
	// falls in the bounds of the circle on that same plane. See above URL for a description of the plane intersection algorithm.
	bool intersectsCircle( vec3 position, vec3 normal, vec3 u, vec3 v, vec3 rayOrigin, vec3 rayDirection, inout float dist ) {

		float planeDist;
		vec2 coordinates;
		if (
			areaLightPlaneCoordinates(
				position, normal, u, v, rayOrigin, rayDirection, planeDist, coordinates
			) &&
			dot( coordinates, coordinates ) <= 0.25
		) {

			dist = planeDist;
			return true;

		}

		return false;

	}

`;

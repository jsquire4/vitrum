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

		float rayDenominator = dot( rayDirection, normal );
		if ( rayDenominator == 0.0 || isnan( rayDenominator ) || isinf( rayDenominator ) ) {

			return false;

		}

		float t = dot( center - rayOrigin, normal ) / rayDenominator;
		if ( ! ( t > EPSILON ) || isnan( t ) || isinf( t ) ) {

			return false;

		}

		vec3 relative = rayOrigin + rayDirection * t - center;
		float uLengthSquared = dot( u, u );
		float vLengthSquared = dot( v, v );
		float axisDot = dot( u, v );
		vec3 axisCross = cross( u, v );
		float gramDeterminant = dot( axisCross, axisCross );
		if (
			! ( uLengthSquared > 0.0 ) ||
			! ( vLengthSquared > 0.0 ) ||
			! ( gramDeterminant > 0.0 ) ||
			isnan( uLengthSquared ) || isinf( uLengthSquared ) ||
			isnan( vLengthSquared ) || isinf( vLengthSquared ) ||
			isnan( axisDot ) || isinf( axisDot ) ||
			isnan( gramDeterminant ) || isinf( gramDeterminant )
		) {

			return false;

		}

		float relativeU = dot( relative, u );
		float relativeV = dot( relative, v );
		float uCoordinate =
			( relativeU * vLengthSquared - relativeV * axisDot ) / gramDeterminant;
		float vCoordinate =
			( relativeV * uLengthSquared - relativeU * axisDot ) / gramDeterminant;
		if (
			isnan( uCoordinate ) || isinf( uCoordinate ) ||
			isnan( vCoordinate ) || isinf( vCoordinate )
		) {

			return false;

		}

		planeDist = t;
		coordinates = vec2( uCoordinate, vCoordinate );
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

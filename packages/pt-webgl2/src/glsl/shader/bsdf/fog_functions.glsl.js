export const fog_functions = /* glsl */`

	// returns the hit distance given the material density
	float intersectFogVolume( float extinction, float u ) {

		// https://raytracing.github.io/books/RayTracingTheNextWeek.html#volumes/constantdensitymediums
		if (
			isnan( extinction ) || extinction < 0.0 ||
			isnan( u ) || isinf( u ) || u < 0.0 || u > 1.0 ||
			u == 0.0
		) return INFINITY;
		// A zero-transmittance channel is an exact absorbing atom at t=0,
		// not a finite exponential whose coefficient can be packed in f32.
		if ( isinf( extinction ) ) return 0.0;
		if ( extinction == 0.0 ) return INFINITY;
		if ( u == 1.0 ) return 0.0;
		return - log( u ) / extinction;

	}

`;

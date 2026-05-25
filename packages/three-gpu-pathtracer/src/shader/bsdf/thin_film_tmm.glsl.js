export const thin_film_tmm = /* glsl */`

	// Sprint 14 (RFE-14): 35-layer thin-film evaluator (TE approximation).
	#define N_THIN_FILM_LAYERS 35
	const uint MATERIAL_PIXELS = 85u;
	const uint THIN_FILM_SAMPLE_OFFSET = 28u;

	float getMaterialStackScalar( uint materialIndex, uint scalarOffset ) {
		uint sampleIdx = THIN_FILM_SAMPLE_OFFSET + scalarOffset / 4u;
		vec4 s = texelFetch1D( materials, materialIndex * MATERIAL_PIXELS + sampleIdx );
		uint c = scalarOffset % 4u;
		return c == 0u ? s.x : ( c == 1u ? s.y : ( c == 2u ? s.z : s.w ) );
	}

	vec2 cMul( vec2 a, vec2 b ) {
		return vec2( a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x );
	}

	vec2 cDiv( vec2 a, vec2 b ) {
		float d = max( dot( b, b ), 1e-8 );
		return vec2(
			( a.x * b.x + a.y * b.y ) / d,
			( a.y * b.x - a.x * b.y ) / d
		);
	}

	vec2 cSin( vec2 z ) {
		float a = z.x;
		float b = z.y;
		return vec2(
			sin( a ) * cosh( b ),
			cos( a ) * sinh( b )
		);
	}

	vec2 cCos( vec2 z ) {
		float a = z.x;
		float b = z.y;
		return vec2(
			cos( a ) * cosh( b ),
			- sin( a ) * sinh( b )
		);
	}

	// Returns vec2(R, T) for the hero wavelength.
	vec2 thinFilmTMM(
		uint materialIndex,
		int thinFilmLayerCount,
		float lambdaNm,
		float substrateIor,
		float incidentIor,
		float viewCosTheta
	) {
		if ( thinFilmLayerCount <= 0 ) {
			return vec2( 0.0, 1.0 );
		}

		float lambdaUm = max( lambdaNm * 0.001, 1e-6 );
		float eta0 = max( incidentIor, 1.0 );
		float etaS = max( substrateIor, 1.0 );
		float angleScale = clamp( viewCosTheta, 0.05, 1.0 );

		vec2 m11 = vec2( 1.0, 0.0 );
		vec2 m12 = vec2( 0.0, 0.0 );
		vec2 m21 = vec2( 0.0, 0.0 );
		vec2 m22 = vec2( 1.0, 0.0 );

		for ( int i = 0; i < N_THIN_FILM_LAYERS; i ++ ) {
			if ( i >= thinFilmLayerCount ) break;

			uint layerBase = uint( i ) * 3u;
			float nj = max( getMaterialStackScalar( materialIndex, layerBase ), 1.0 );
			float djUm = max( getMaterialStackScalar( materialIndex, layerBase + 1u ) * 0.001, 0.0 );
			float kj = max( getMaterialStackScalar( materialIndex, layerBase + 2u ), 0.0 );
			vec2 etaJ = vec2( nj, - kj );
			vec2 delta = etaJ * ( 2.0 * PI * djUm * angleScale / lambdaUm );

			vec2 sinDelta = cSin( delta );
			vec2 cosDelta = cCos( delta );
			vec2 minusI = vec2( 0.0, -1.0 );

			vec2 a11 = cosDelta;
			vec2 a12 = cMul( minusI, cDiv( sinDelta, etaJ ) );
			vec2 a21 = cMul( minusI, cMul( etaJ, sinDelta ) );
			vec2 a22 = cosDelta;

			vec2 nm11 = cMul( m11, a11 ) + cMul( m12, a21 );
			vec2 nm12 = cMul( m11, a12 ) + cMul( m12, a22 );
			vec2 nm21 = cMul( m21, a11 ) + cMul( m22, a21 );
			vec2 nm22 = cMul( m21, a12 ) + cMul( m22, a22 );
			m11 = nm11;
			m12 = nm12;
			m21 = nm21;
			m22 = nm22;
		}

		vec2 eta0m11 = m11 * eta0;
		vec2 eta0etaSm12 = m12 * ( eta0 * etaS );
		vec2 etaSm22 = m22 * etaS;
		vec2 den = eta0m11 + eta0etaSm12 + m21 + etaSm22;
		vec2 numR = eta0m11 + eta0etaSm12 - m21 - etaSm22;
		vec2 r = cDiv( numR, den );
		vec2 t = cDiv( vec2( 2.0 * eta0, 0.0 ), den );

		float R = dot( r, r );
		float T = ( etaS / eta0 ) * dot( t, t );
		return vec2( clamp( R, 0.0, 1.0 ), clamp( T, 0.0, 1.0 ) );
	}

`;

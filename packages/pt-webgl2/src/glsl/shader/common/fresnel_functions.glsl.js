export const fresnel_functions = /* glsl */`

	bool totalInternalReflection( float cosTheta, float eta ) {

		if ( eta == 0.0 ) return false;
		if (
			isnan( cosTheta ) || isinf( cosTheta ) ||
			isnan( eta ) || isinf( eta ) || eta < 0.0
		) return true;
		float boundedCosTheta = clamp( abs( cosTheta ), 0.0, 1.0 );
		float sinTheta = sqrt( max(
			1.0 - boundedCosTheta * boundedCosTheta, 0.0
		) );
		return eta * sinTheta > 1.0;

	}

	// https://google.github.io/filament/Filament.md.html#materialsystem/diffusebrdf
	float schlickFresnel( float cosine, float f0 ) {

		if (
			isnan( cosine ) || isinf( cosine ) ||
			isnan( f0 ) || isinf( f0 )
		) return 1.0;
		float boundedCosine = clamp( abs( cosine ), 0.0, 1.0 );
		float boundedF0 = clamp( f0, 0.0, 1.0 );
		float oneMinusCosine = 1.0 - boundedCosine;
		float weight = oneMinusCosine * oneMinusCosine;
		weight *= weight * oneMinusCosine;
		return clamp(
			boundedF0 + ( 1.0 - boundedF0 ) * weight,
			0.0, 1.0
		);

	}

	vec3 schlickFresnel( float cosine, vec3 f0 ) {

		if (
			isnan( cosine ) || isinf( cosine ) ||
			any( isnan( f0 ) ) || any( isinf( f0 ) )
		) return vec3( 1.0 );
		float boundedCosine = clamp( abs( cosine ), 0.0, 1.0 );
		vec3 boundedF0 = clamp( f0, vec3( 0.0 ), vec3( 1.0 ) );
		float oneMinusCosine = 1.0 - boundedCosine;
		float weight = oneMinusCosine * oneMinusCosine;
		weight *= weight * oneMinusCosine;
		return clamp(
			boundedF0 + ( vec3( 1.0 ) - boundedF0 ) * weight,
			vec3( 0.0 ), vec3( 1.0 )
		);

	}

	vec3 schlickFresnel( float cosine, vec3 f0, vec3 f90 ) {

		if (
			isnan( cosine ) || isinf( cosine ) ||
			any( isnan( f0 ) ) || any( isinf( f0 ) ) ||
			any( isnan( f90 ) ) || any( isinf( f90 ) )
		) return vec3( 1.0 );
		float boundedCosine = clamp( abs( cosine ), 0.0, 1.0 );
		vec3 boundedF0 = clamp( f0, vec3( 0.0 ), vec3( 1.0 ) );
		vec3 boundedF90 = clamp( f90, vec3( 0.0 ), vec3( 1.0 ) );
		float oneMinusCosine = 1.0 - boundedCosine;
		float weight = oneMinusCosine * oneMinusCosine;
		weight *= weight * oneMinusCosine;
		return clamp(
			boundedF0 + ( boundedF90 - boundedF0 ) * weight,
			vec3( 0.0 ), vec3( 1.0 )
		);

	}

	float dielectricFresnel( float cosThetaI, float eta ) {

		// eta==0 is the engine's finite-arithmetic representation of the
		// infinite-IOR compatibility limit. Its exact limiting reflectance is one,
		// including grazing angles where the algebra below would form 0 / 0.
		if ( eta == 0.0 ) return 1.0;
		if (
			isnan( cosThetaI ) || isinf( cosThetaI ) ||
			isnan( eta ) || isinf( eta ) || eta < 0.0
		) return 1.0;
		if ( eta == 1.0 ) return 0.0;
		cosThetaI = clamp( abs( cosThetaI ), 0.0, 1.0 );

		// https://schuttejoe.github.io/post/disneybsdf/
		float ni = eta;
		float nt = 1.0;

		// Check for total internal reflection
		float sinThetaISq = max( 1.0f - cosThetaI * cosThetaI, 0.0 );
		float sinThetaTSq = eta * eta * sinThetaISq;
		if( sinThetaTSq >= 1.0 ) {

			return 1.0;

		}

		float sinThetaT = sqrt( sinThetaTSq );

		float cosThetaT = sqrt( max( 0.0, 1.0f - sinThetaT * sinThetaT ) );
		float parallelDenom =
			( nt * cosThetaI ) + ( ni * cosThetaT );
		float perpendicularDenom =
			( ni * cosThetaI ) + ( nt * cosThetaT );
		if (
			! ( abs( parallelDenom ) > 1e-20 ) ||
			! ( abs( perpendicularDenom ) > 1e-20 )
		) return 1.0;
		float rParallel =
			( ( nt * cosThetaI ) - ( ni * cosThetaT ) ) /
			parallelDenom;
		float rPerpendicular =
			( ( ni * cosThetaI ) - ( nt * cosThetaT ) ) /
			perpendicularDenom;
		float result =
			( rParallel * rParallel + rPerpendicular * rPerpendicular ) /
			2.0;
		return isnan( result ) || isinf( result )
			? 1.0
			: clamp( result, 0.0, 1.0 );

	}

	// https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/schlickapproximation
	float iorRatioToF0( float eta ) {

		if ( eta == 0.0 ) return 1.0;
		if ( isnan( eta ) || isinf( eta ) || eta < 0.0 ) return 1.0;
		float ratio = ( 1.0 - eta ) / ( 1.0 + eta );
		return clamp( ratio * ratio, 0.0, 1.0 );

	}

	// evaluateFresnel — Schlick Fresnel with total-internal-reflection guard.
	// Used by specularEval (bsdf_functions.glsl.js) for coloured specular lobes.
	//
	// The former "blown out pixels" concern (evaluateFresnelWeight) was a
	// pre-B9 artefact: the single-scatter specular lobe was not energy-compensated,
	// so rough metals/dielectrics drained energy and compensating heuristics were
	// needed.  Since B9 (Kulla-Conty ggxMultiscatter), specularEval adds the
	// multi-bounce lobe back in; the base Schlick evaluation is now correct with no
	// overbright. The getLobeWeights branch that picks between diffuse/specular/
	// transmission uses disneyFresnel (below) rather than evaluateFresnel, so the
	// two code paths remain independent and consistent.
	vec3 evaluateFresnel( float cosTheta, float eta, vec3 f0, vec3 f90 ) {

		if ( totalInternalReflection( cosTheta, eta ) ) {

			return f90;

		}

		return schlickFresnel( cosTheta, f0, f90 );

	}

	// https://schuttejoe.github.io/post/disneybsdf/
	float disneyFresnel( vec3 wo, vec3 wi, vec3 wh, float f0, float eta, float metalness ) {

		if (
			any( isnan( wo ) ) || any( isinf( wo ) ) ||
			any( isnan( wi ) ) || any( isinf( wi ) ) ||
			any( isnan( wh ) ) || any( isinf( wh ) ) ||
			isnan( metalness ) || isinf( metalness )
		) return 1.0;
		float dotHV = clamp( abs( dot( wo, wh ) ), 0.0, 1.0 );
		if ( totalInternalReflection( dotHV, eta ) ) {

			return 1.0;

		}

		float dotHL = clamp( abs( dot( wi, wh ) ), 0.0, 1.0 );
		float dielectricTerm = dielectricFresnel( dotHV, eta );
		float metallicFresnel = schlickFresnel( dotHL, f0 );

		return clamp(
			mix(
				dielectricTerm,
				metallicFresnel,
				clamp( metalness, 0.0, 1.0 )
			),
			0.0, 1.0
		);

	}

`;

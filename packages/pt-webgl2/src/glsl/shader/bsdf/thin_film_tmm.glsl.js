import { MATERIAL_PIXELS } from '../structs/materialStride.js';

export const thin_film_tmm = /* glsl */`

	// Canonical coherent thin-film transfer used by the WebGPU peer. Each
	// polarization walks the complete incident -> layers -> substrate network,
	// with Snell propagation and optical admittance evaluated in every medium.
	#define N_THIN_FILM_LAYERS 35
	const uint MATERIAL_PIXELS = ${MATERIAL_PIXELS}u;
	const uint THIN_FILM_SAMPLE_OFFSET = 28u;

	float getMaterialStackScalar( uint materialIndex, uint scalarOffset ) {
		uint sampleIdx = THIN_FILM_SAMPLE_OFFSET + scalarOffset / 4u;
		vec4 s = texelFetch1D(
			materials, materialIndex * MATERIAL_PIXELS + sampleIdx
		);
		uint c = scalarOffset % 4u;
		return c == 0u ? s.x : ( c == 1u ? s.y : ( c == 2u ? s.z : s.w ) );
	}

	vec2 cMul( vec2 a, vec2 b ) {
		return vec2( a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x );
	}

	vec2 cDiv( vec2 a, vec2 b ) {
		float d = dot( b, b );
		if ( ! ( d > 1e-20 ) || isnan( d ) || isinf( d ) ) {
			return vec2( 0.0 );
		}
		return vec2(
			( a.x * b.x + a.y * b.y ) / d,
			( a.y * b.x - a.x * b.y ) / d
		);
	}

	vec2 cSqrtPhysical( vec2 z ) {
		float radius = sqrt( max( dot( z, z ), 0.0 ) );
		float re = sqrt( max( 0.0, 0.5 * ( radius + z.x ) ) );
		float imMagnitude = sqrt( max( 0.0, 0.5 * ( radius - z.x ) ) );
		return vec2( re, z.y < 0.0 ? - imMagnitude : imMagnitude );
	}

	vec2 cExpI( vec2 z ) {
		float amplitude = exp( clamp( - z.y, -80.0, 0.0 ) );
		return amplitude * vec2( cos( z.x ), sin( z.x ) );
	}

	bool thinFilmFinite( float value ) {
		return ! isnan( value ) && ! isinf( value ) && abs( value ) < 1e30;
	}

	struct ThinFilmScatter {
		vec2 rL;
		vec2 tLR;
		vec2 rR;
		vec2 tRL;
	};

	ThinFilmScatter thinFilmCascade(
		ThinFilmScatter a, ThinFilmScatter b
	) {
		vec2 inv = cDiv(
			vec2( 1.0, 0.0 ),
			vec2( 1.0, 0.0 ) - cMul( a.rR, b.rL )
		);
		ThinFilmScatter result;
		result.rL = a.rL + cMul(
			cMul( cMul( a.tRL, b.rL ), inv ), a.tLR
		);
		result.tLR = cMul( cMul( b.tLR, inv ), a.tLR );
		result.rR = b.rR + cMul(
			cMul( cMul( b.tLR, a.rR ), inv ), b.tRL
		);
		result.tRL = cMul( cMul( a.tRL, inv ), b.tRL );
		return result;
	}

	vec2 thinFilmLayerN( uint materialIndex, int layerIndex ) {
		uint base = uint( layerIndex ) * 3u;
		return vec2(
			max( getMaterialStackScalar( materialIndex, base ), 1.0 ),
			max( getMaterialStackScalar( materialIndex, base + 2u ), 0.0 )
		);
	}

	float thinFilmLayerThicknessNm( uint materialIndex, int layerIndex ) {
		return max( getMaterialStackScalar(
			materialIndex, uint( layerIndex ) * 3u + 1u
		), 0.0 );
	}

	vec2 thinFilmMediumN(
		uint materialIndex,
		int mediumIndex,
		int layerCount,
		float incidentIor,
		float substrateIor,
		bool frontFace
	) {
		if ( mediumIndex == 0 ) {
			return vec2( frontFace ? incidentIor : substrateIor, 0.0 );
		}
		if ( mediumIndex == layerCount + 1 ) {
			return vec2( frontFace ? substrateIor : incidentIor, 0.0 );
		}
		int authoredIndex = frontFace
			? mediumIndex - 1
			: layerCount - mediumIndex;
		return thinFilmLayerN( materialIndex, authoredIndex );
	}

	float thinFilmMediumThicknessNm(
		uint materialIndex,
		int mediumIndex,
		int layerCount,
		bool frontFace
	) {
		int authoredIndex = frontFace
			? mediumIndex - 1
			: layerCount - mediumIndex;
		return thinFilmLayerThicknessNm( materialIndex, authoredIndex );
	}

	vec2 thinFilmPhysicalCosine( vec2 n, float transverseIndex ) {
		vec2 ratio = cDiv( vec2( transverseIndex, 0.0 ), n );
		vec2 cosine = cSqrtPhysical(
			vec2( 1.0, 0.0 ) - cMul( ratio, ratio )
		);
		vec2 kz = cMul( n, cosine );
		if ( kz.y < -1e-7 || ( abs( kz.y ) <= 1e-7 && kz.x < 0.0 ) ) {
			cosine = - cosine;
		}
		return cosine;
	}

	vec2 thinFilmAdmittance( vec2 n, vec2 cosine, bool pPolarized ) {
		return pPolarized ? cDiv( cosine, n ) : cMul( n, cosine );
	}

	vec2 thinFilmPolarizedRt(
		uint materialIndex,
		int layerCountRaw,
		float wavelengthNm,
		float substrateIorRaw,
		float incidentIorRaw,
		bool angleDependent,
		float microfacetCos,
		bool frontFace,
		bool pPolarized
	) {
		int layerCount = clamp( layerCountRaw, 0, N_THIN_FILM_LAYERS );
		float substrateIor = max( substrateIorRaw, 1.0 );
		float incidentIor = max( incidentIorRaw, 1.0 );
		float etaIncident = frontFace ? incidentIor : substrateIor;
		float cos0 = angleDependent
			? clamp( microfacetCos, 0.0, 1.0 )
			: 1.0;
		float transverseIndex = etaIncident * sqrt(
			max( 0.0, 1.0 - cos0 * cos0 )
		);

		ThinFilmScatter network;
		network.rL = vec2( 0.0 );
		network.tLR = vec2( 1.0, 0.0 );
		network.rR = vec2( 0.0 );
		network.tRL = vec2( 1.0, 0.0 );

		for ( int interfaceIndex = 0;
			interfaceIndex < N_THIN_FILM_LAYERS + 1;
			interfaceIndex ++ ) {
			if ( interfaceIndex > layerCount ) break;
			vec2 n0 = thinFilmMediumN(
				materialIndex, interfaceIndex, layerCount,
				incidentIor, substrateIor, frontFace
			);
			vec2 n1 = thinFilmMediumN(
				materialIndex, interfaceIndex + 1, layerCount,
				incidentIor, substrateIor, frontFace
			);
			vec2 cos0Layer = thinFilmPhysicalCosine( n0, transverseIndex );
			vec2 cos1Layer = thinFilmPhysicalCosine( n1, transverseIndex );
			vec2 q0 = thinFilmAdmittance( n0, cos0Layer, pPolarized );
			vec2 q1 = thinFilmAdmittance( n1, cos1Layer, pPolarized );
			vec2 qSum = q0 + q1;
			vec2 rL = cDiv( q0 - q1, qSum );
			ThinFilmScatter boundary;
			boundary.rL = rL;
			boundary.tLR = cDiv( 2.0 * q0, qSum );
			boundary.rR = - rL;
			boundary.tRL = cDiv( 2.0 * q1, qSum );
			network = thinFilmCascade( network, boundary );

			if ( interfaceIndex < layerCount ) {
				float thicknessNm = thinFilmMediumThicknessNm(
					materialIndex, interfaceIndex + 1,
					layerCount, frontFace
				);
				vec2 phase = cMul( n1, cos1Layer ) *
					( 2.0 * PI * thicknessNm / max( wavelengthNm, 1e-4 ) );
				vec2 propagationFactor = cExpI( phase );
				ThinFilmScatter propagation;
				propagation.rL = vec2( 0.0 );
				propagation.tLR = propagationFactor;
				propagation.rR = vec2( 0.0 );
				propagation.tRL = propagationFactor;
				network = thinFilmCascade( network, propagation );
			}
		}

		vec2 nIncident = thinFilmMediumN(
			materialIndex, 0, layerCount,
			incidentIor, substrateIor, frontFace
		);
		vec2 nTransmitted = thinFilmMediumN(
			materialIndex, layerCount + 1, layerCount,
			incidentIor, substrateIor, frontFace
		);
		vec2 qIncident = thinFilmAdmittance(
			nIncident,
			thinFilmPhysicalCosine( nIncident, transverseIndex ),
			pPolarized
		);
		vec2 qTransmitted = thinFilmAdmittance(
			nTransmitted,
			thinFilmPhysicalCosine( nTransmitted, transverseIndex ),
			pPolarized
		);
		return vec2(
			dot( network.rL, network.rL ),
			max( qTransmitted.x, 0.0 ) / max( qIncident.x, 1e-8 ) *
				dot( network.tLR, network.tLR )
		);
	}

	// Returns complete unpolarized stack (R, T, absorbed) energy. This includes
	// the final incident-medium/substrate interface even for a zero-layer stack.
	vec3 thinFilmTMM(
		uint materialIndex,
		int thinFilmLayerCount,
		float wavelengthNm,
		float substrateIor,
		float incidentIor,
		bool angleDependent,
		float microfacetCos,
		bool frontFace
	) {
		vec2 rtS = thinFilmPolarizedRt(
			materialIndex, thinFilmLayerCount, wavelengthNm,
			substrateIor, incidentIor, angleDependent,
			microfacetCos, frontFace, false
		);
		vec2 rtP = thinFilmPolarizedRt(
			materialIndex, thinFilmLayerCount, wavelengthNm,
			substrateIor, incidentIor, angleDependent,
			microfacetCos, frontFace, true
		);
		float reflectance = max( 0.0, 0.5 * ( rtS.x + rtP.x ) );
		float transmittance = max( 0.0, 0.5 * ( rtS.y + rtP.y ) );
		if (
			! thinFilmFinite( reflectance ) ||
			! thinFilmFinite( transmittance ) ||
			reflectance + transmittance > 1.0001
		) {
			// Structural/numeric corruption loses the sample as absorption. It must
			// never synthesize a bright mirror that was not authored.
			return vec3( 0.0, 0.0, 1.0 );
		}
		if ( reflectance + transmittance > 1.0 ) {
			float invSum = 1.0 / ( reflectance + transmittance );
			reflectance *= invSum;
			transmittance *= invSum;
		}
		return vec3(
			reflectance,
			transmittance,
			max( 0.0, 1.0 - reflectance - transmittance )
		);
	}

	struct ThinFilmRgb {
		vec3 reflectance;
		vec3 transmittance;
	};

	ThinFilmRgb thinFilmTMMRgb(
		uint materialIndex,
		int thinFilmLayerCount,
		float heroWavelengthNm,
		float substrateIor,
		float incidentIor,
		bool angleDependent,
		float microfacetCos,
		bool frontFace
	) {
		ThinFilmRgb result;
		if ( uSpectralRendering != 0 ) {
			vec3 rt = thinFilmTMM(
				materialIndex, thinFilmLayerCount, heroWavelengthNm,
				substrateIor, incidentIor, angleDependent,
				microfacetCos, frontFace
			);
			result.reflectance = vec3( rt.x );
			result.transmittance = vec3( rt.y );
			return result;
		}

		vec3 red = thinFilmTMM(
			materialIndex, thinFilmLayerCount, 650.0,
			substrateIor, incidentIor, angleDependent,
			microfacetCos, frontFace
		);
		vec3 green = thinFilmTMM(
			materialIndex, thinFilmLayerCount, 510.0,
			substrateIor, incidentIor, angleDependent,
			microfacetCos, frontFace
		);
		vec3 blue = thinFilmTMM(
			materialIndex, thinFilmLayerCount, 475.0,
			substrateIor, incidentIor, angleDependent,
			microfacetCos, frontFace
		);
		result.reflectance = vec3( red.x, green.x, blue.x );
		result.transmittance = vec3( red.y, green.y, blue.y );
		return result;
	}

	struct BsdfLayeredInterfaceResponse {
		vec3 reflectance;
		vec3 baseTransmittance;
	};

	// A coherent stack replaces the authored bare interface. TMM already owns
	// the substrate boundary, so adding R_stack to authored Fresnel (or
	// multiplying T_stack by 1-F again) would count that interface twice. The
	// bounded coated-vs-bare odds replacement preserves artistic F0 while keeping
	// the exact surviving stack energy and the zero-layer bare-interface limit.
	BsdfLayeredInterfaceResponse bsdfLayeredInterfaceResponse(
		vec3 baseFresnel,
		const in SurfaceRecord surf,
		float microfacetCos,
		float heroWavelength
	) {
		BsdfLayeredInterfaceResponse response;
		vec3 baseF = clamp( baseFresnel, vec3( 0.0 ), vec3( 1.0 ) );
		vec3 baseT = vec3( 1.0 ) - baseF;
		response.reflectance = baseF;
		response.baseTransmittance = baseT;
		if ( ! ( surf.thinFilmEnabled > 0.5 ) || surf.thinFilmLayerCount < 0.5 ) {
			return response;
		}

		float interfaceCos = surf.thinFilmAngleDependent
			? clamp( abs( microfacetCos ), 0.0, 1.0 )
			: 1.0;
		ThinFilmRgb filmRt = thinFilmTMMRgb(
			surf.materialIndex,
			int( surf.thinFilmLayerCount + 0.5 ),
			heroWavelength,
			max( surf.ior, 1.0 ),
			surf.thinFilmIncidentIor,
			surf.thinFilmAngleDependent,
			interfaceCos,
			surf.frontFace
		);
		float etaIncident = surf.frontFace
			? surf.thinFilmIncidentIor
			: max( surf.ior, 1.0 );
		float etaTransmitted = surf.frontFace
			? max( surf.ior, 1.0 )
			: surf.thinFilmIncidentIor;
		float bareR = dielectricFresnel(
			interfaceCos,
			etaIncident / max( etaTransmitted, 1e-8 )
		);
		float bareT = 1.0 - bareR;
		vec3 reflectedWeight =
			baseF * filmRt.reflectance / max( bareR, 1e-6 );
		vec3 transmittedWeight =
			baseT * filmRt.transmittance / max( bareT, 1e-6 );
		vec3 weightSum = reflectedWeight + transmittedWeight;
		vec3 reflectedFraction = clamp(
			mix(
				baseF,
				reflectedWeight / max( weightSum, vec3( 1e-20 ) ),
				greaterThan( weightSum, vec3( 1e-20 ) )
			),
			vec3( 0.0 ), vec3( 1.0 )
		);
		reflectedFraction = mix(
			reflectedFraction,
			vec3( 1.0 ),
			bvec3(
				filmRt.transmittance.r <= 1e-20 && filmRt.reflectance.r > 1e-20,
				filmRt.transmittance.g <= 1e-20 && filmRt.reflectance.g > 1e-20,
				filmRt.transmittance.b <= 1e-20 && filmRt.reflectance.b > 1e-20
			)
		);
		reflectedFraction = mix(
			reflectedFraction,
			vec3( 0.0 ),
			bvec3(
				filmRt.reflectance.r <= 1e-20 && filmRt.transmittance.r > 1e-20,
				filmRt.reflectance.g <= 1e-20 && filmRt.transmittance.g > 1e-20,
				filmRt.reflectance.b <= 1e-20 && filmRt.transmittance.b > 1e-20
			)
		);
		vec3 survivingEnergy = clamp(
			filmRt.reflectance + filmRt.transmittance,
			vec3( 0.0 ), vec3( 1.0 )
		);
		response.reflectance = survivingEnergy * reflectedFraction;
		response.baseTransmittance =
			survivingEnergy * ( vec3( 1.0 ) - reflectedFraction );
		return response;
	}

`;

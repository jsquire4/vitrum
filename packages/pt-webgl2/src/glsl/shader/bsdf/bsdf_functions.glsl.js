/*
wi     : incident vector or light vector (pointing toward the light)
wo     : outgoing vector or view vector (pointing towards the camera)
wh     : computed half vector from wo and wi
Eval   : Get the color and pdf for a direction
Sample : Get the direction, color, and pdf for a sample
eta    : Greek character used to denote the "ratio of ior"
f0     : Amount of light reflected when looking at a surface head on - "fresnel 0"
f90    : Amount of light reflected at grazing angles
*/

/** @public — dynamic-access test-load-bearing; accessed via namespace import in b9Multiscatter.test.ts */
export const bsdf_functions = /* glsl */`

	// RFE-03 / Sprint 14 — defined before bsdfEval since GLSL parses top-down and
	// some drivers don't accept forward-declarations whose params reference user structs.
	vec3 pathThroughputFromRgb( vec3 rgb, float heroWavelength ) {
		if ( uSpectralRendering == 0 ) return max( rgb, vec3( 0.0 ) );
		return vec3( heroScalarFromRgb( rgb, heroWavelength ) );
	}

	vec3 activeLayerThroughput( const in SurfaceRecord surf, float heroWavelength ) {
		if ( ! surf.hasActiveLayer ) return vec3( 1.0 );
		// BSDF evaluators return RGB-domain f*cos. Conversion to the hero-path
		// scalar happens once at the estimator boundary; converting a layer here
		// and the complete response again would project it twice.
		return max( surf.activeLayerTransmission, vec3( 0.0 ) );
	}

	vec3 oppositeLayerThroughput( const in SurfaceRecord surf, float heroWavelength ) {
		if ( ! surf.hasOppositeLayer ) return vec3( 1.0 );
		return max( surf.oppositeLayerTransmission, vec3( 0.0 ) );
	}

	SurfaceRecord oppositeFacingSurface(
		const in SurfaceRecord surf,
		bool reciprocalEta
	) {
		SurfaceRecord result = surf;
		result.frontFace = ! surf.frontFace;
		result.normal = surf.oppositeNormal;
		result.normalBasis = surf.oppositeNormalBasis;
		result.oppositeNormal = surf.normal;
		result.oppositeNormalBasis = surf.normalBasis;
		result.roughness = surf.oppositeRoughness;
		result.filteredRoughness = surf.oppositeFilteredRoughness;
		result.oppositeRoughness = surf.roughness;
		result.oppositeFilteredRoughness = surf.filteredRoughness;
		result.activeLayerTransmission = surf.oppositeLayerTransmission;
		result.hasActiveLayer = surf.hasOppositeLayer;
		result.oppositeLayerTransmission = surf.activeLayerTransmission;
		result.hasOppositeLayer = surf.hasActiveLayer;
		result.clearcoatNormal = result.normal;
		result.clearcoatBasis = result.normalBasis;
		if ( reciprocalEta ) {
			result.eta = surf.eta == 0.0 ? 0.0 : 1.0 / surf.eta;
			result.f0 = iorRatioToF0( result.eta );
		}
		return result;
	}

	float surfaceEtaForOutgoingDirection(
		vec3 wo, const in SurfaceRecord surf
	) {
		if ( surf.eta == 0.0 ) return 0.0;
		return wo.z >= 0.0 ? surf.eta : 1.0 / surf.eta;
	}

	vec3 surfaceAuthoredInterfaceFresnel(
		float microfacetCos,
		vec3 wo,
		const in SurfaceRecord surf
	) {
		float eta = surfaceEtaForOutgoingDirection( wo, surf );
		vec3 f0Color = mix(
			surf.f0 * surf.specularColor * surf.specularIntensity,
			surf.color,
			surf.metalness
		);
		vec3 f90Color = vec3(
			mix( surf.specularIntensity, 1.0, surf.metalness )
		);
		float boundedCos = clamp( abs( microfacetCos ), 0.0, 1.0 );
		vec3 fresnel = evaluateFresnel(
			boundedCos, eta, f0Color, f90Color
		);
		// A coherent ThinFilmStack supersedes the single-layer artistic
		// iridescence approximation; both represent the same interface layer.
		if (
			( surf.lobeMask & 16u ) != 0u &&
			! ( surf.thinFilmEnabled > 0.5 )
		) {
			vec3 iridescenceF = evalIridescence(
				1.0,
				surf.iridescenceIor,
				boundedCos,
				surf.iridescenceThickness,
				f0Color
			);
			fresnel = mix( fresnel, iridescenceF, surf.iridescence );
		}
		return clamp( fresnel, vec3( 0.0 ), vec3( 1.0 ) );
	}

	BsdfLayeredInterfaceResponse surfaceLayeredInterfaceResponse(
		float microfacetCos,
		vec3 wo,
		const in SurfaceRecord surf,
		float heroWavelength
	) {
		SurfaceRecord responseSurf = surf;
		// Reversing one physical coated interface keeps its microfacet
		// distribution but traverses the coherent layer order substrate-first.
		if ( wo.z < 0.0 ) responseSurf.frontFace = ! surf.frontFace;
		return bsdfLayeredInterfaceResponse(
			surfaceAuthoredInterfaceFresnel(
				microfacetCos, wo, surf
			),
			responseSurf,
			microfacetCos,
			heroWavelength
		);
	}

		float hg_phase( float cosTheta, float g ) {
			float gg = clamp( g, -0.999999, 0.999999 );
			float a = abs( gg );
			float clampedCos = clamp( cosTheta, -1.0, 1.0 );
			float alignedCos = gg >= 0.0 ? clampedCos : - clampedCos;
			float oneMinusA = 1.0 - a;
			float denom =
				oneMinusA * oneMinusA +
				2.0 * a * ( 1.0 - alignedCos );
			return
				( oneMinusA * ( 1.0 + a ) ) /
				( 4.0 * PI * denom * sqrt( denom ) );
		}

		// Algebraically exact HG inverse for volume-particle sampling. Mirrors
		// @vitrum/shared-samplers::sampleHG, including the
		// cancellation-safe rational form around isotropy.
		float sampleHgCosTheta( float u, float g ) {
			float gg = clamp( g, -0.999999, 0.999999 );
			float q = 1.0 - 2.0 * u;
			float cosTheta;
			if ( abs( gg ) < 0.125 ) {
				float d = 1.0 + gg * q;
				float numerator =
					2.0 * q +
					gg * ( q * q + 3.0 ) +
					2.0 * gg * gg * q +
					gg * gg * gg * ( q * q - 1.0 );
				cosTheta = numerator / ( 2.0 * d * d );
			} else {
				float ratio =
					( 1.0 - gg * gg ) /
					( 1.0 + gg * q );
				cosTheta =
					( 1.0 + gg * gg - ratio * ratio ) /
					( 2.0 * gg );
			}
			return clamp( cosTheta, -1.0, 1.0 );
		}

	// diffuse
	float diffuseEval(
		vec3 wo, vec3 wi, vec3 wh,
		const in SurfaceRecord surf,
		float heroWavelength,
		inout vec3 color
	) {

		// https://schuttejoe.github.io/post/disneybsdf/
		float fl = schlickFresnel( wi.z, 0.0 );
		float fv = schlickFresnel( wo.z, 0.0 );

		float metalFactor = ( 1.0 - surf.metalness );
		// Physical diffuse energy is reduced by transmission exactly here. The
		// same factor in getLobeWeights is a sampling probability/PDF term, not
		// another BRDF attenuation.
		float transFactor = ( 1.0 - surf.transmission );
		float rr = 0.5 + 2.0 * surf.roughness * fl * fl;
		float retro = rr * ( fl + fv + fl * fv * ( rr - 1.0f ) );
		float lambert = ( 1.0f - 0.5f * fl ) * ( 1.0f - 0.5f * fv );

		// Subsurface scattering is owned by explicit bulk free-flight and HG medium
		// vertices. A diffuse subsurface approximation here would double-count that
		// transport, so the boundary retains the ordinary surface diffuse lobe.
		BsdfLayeredInterfaceResponse interfaceResponse =
			surfaceLayeredInterfaceResponse(
				dot( wo, wh ), wo, surf, heroWavelength
			);
		color = interfaceResponse.baseTransmittance *
			transFactor * metalFactor * wi.z * surf.color *
			( retro + lambert ) / PI;

		return wi.z / PI;

	}

	vec3 diffuseDirection( vec3 wo, const in SurfaceRecord surf ) {

		vec3 lightDirection = sampleSphere( rand2( 11 ) );
		lightDirection.z += 1.0;
		lightDirection = normalize( lightDirection );

		return lightDirection;

	}

	vec3 rotateAnisotropyFrame( vec3 v, float angle ) {

		float c = cos( angle );
		float s = sin( angle );
		return vec3( c * v.x + s * v.y, - s * v.x + c * v.y, v.z );

	}

	vec3 unrotateAnisotropyFrame( vec3 v, float angle ) {

		float c = cos( angle );
		float s = sin( angle );
		return vec3( c * v.x - s * v.y, s * v.x + c * v.y, v.z );

	}

        vec2 anisotropicRoughnessAxes( const in SurfaceRecord surf ) {

                float roughness = clamp( surf.filteredRoughness, 0.0, 1.0 );
                // A positive anisotropy request deliberately stays on the
                // anisotropic continuous path. Only the degenerate exact-zero
                // width needs a finite representation; every authored positive
                // roughness, however small, is preserved verbatim.
                if ( roughness == 0.0 ) roughness = 0.001;
                float aspect = sqrt( max( 1.0 - 0.9 * clamp( surf.anisotropy, 0.0, 1.0 ), 0.1 ) );
                return min( vec2( roughness / aspect, roughness * aspect ), vec2( 1.0 ) );

	}

	float ggxDistributionAnisotropic( vec3 halfVector, vec2 roughness ) {

		if ( halfVector.z <= 0.0 ) return 0.0;
                float ax = roughness.x;
                float ay = roughness.y;
                if ( ! ( ax > 0.0 ) || ! ( ay > 0.0 ) ) return 0.0;
                float hx = halfVector.x / ax;
                float hy = halfVector.y / ay;
                float denom = hx * hx + hy * hy + halfVector.z * halfVector.z;
                float normalization = PI * ax * ay * denom * denom;
                if ( ! ( normalization > 0.0 ) ) return 0.0;
                return 1.0 / normalization;

	}

	float ggxLambdaAnisotropic( vec3 w, vec2 roughness ) {

		float z2 = max( w.z * w.z, 1e-6 );
                float ax = max( roughness.x, 0.0 );
                float ay = max( roughness.y, 0.0 );
		float a2tan2 = ( ax * ax * w.x * w.x + ay * ay * w.y * w.y ) / z2;
		return ( - 1.0 + sqrt( 1.0 + a2tan2 ) ) * 0.5;

	}

	float ggxShadowMaskG1Anisotropic( vec3 w, vec2 roughness ) {

		return 1.0 / ( 1.0 + ggxLambdaAnisotropic( w, roughness ) );

	}

	float ggxShadowMaskG2Anisotropic( vec3 wi, vec3 wo, vec2 roughness ) {

		return 1.0 / ( 1.0 + ggxLambdaAnisotropic( wi, roughness ) + ggxLambdaAnisotropic( wo, roughness ) );

	}

	float ggxDistributionForSurface( vec3 halfVector, const in SurfaceRecord surf ) {

            if ( surf.anisotropy <= 0.0 ) return ggxDistribution( halfVector, surf.filteredRoughness );
		vec3 h = rotateAnisotropyFrame( halfVector, surf.anisotropyRotation );
		return ggxDistributionAnisotropic( h, anisotropicRoughnessAxes( surf ) );

	}

	float ggxShadowMaskG1ForSurface( vec3 w, const in SurfaceRecord surf ) {

            if ( surf.anisotropy <= 0.0 ) return ggxShadowMaskG1( acos( w.z ), surf.filteredRoughness );
		vec3 wa = rotateAnisotropyFrame( w, surf.anisotropyRotation );
		return ggxShadowMaskG1Anisotropic( wa, anisotropicRoughnessAxes( surf ) );

	}

	float ggxShadowMaskG2ForSurface( vec3 wi, vec3 wo, const in SurfaceRecord surf ) {

            if ( surf.anisotropy <= 0.0 ) return ggxShadowMaskG2( wi, wo, surf.filteredRoughness );
		vec3 wia = rotateAnisotropyFrame( wi, surf.anisotropyRotation );
		vec3 woa = rotateAnisotropyFrame( wo, surf.anisotropyRotation );
		return ggxShadowMaskG2Anisotropic( wia, woa, anisotropicRoughnessAxes( surf ) );

	}

	float ggxPdfForSurface( vec3 wi, vec3 halfVector, const in SurfaceRecord surf ) {

            if ( surf.anisotropy <= 0.0 ) return ggxPDF( wi, halfVector, surf.filteredRoughness );
                float D = ggxDistributionForSurface( halfVector, surf );
                float G1 = ggxShadowMaskG1ForSurface( wi, surf );
                float projectedView = abs( wi.z );
                if ( ! ( projectedView > 0.0 ) ) return 0.0;
                return D * G1 * abs( dot( wi, halfVector ) ) / projectedView;

	}

	vec3 ggxDirectionForSurface( vec3 wo, const in SurfaceRecord surf, vec2 uv ) {

            if ( surf.anisotropy <= 0.0 ) return ggxDirection( wo, vec2( surf.filteredRoughness ), uv );
		vec3 woAniso = rotateAnisotropyFrame( wo, surf.anisotropyRotation );
		vec3 hAniso = ggxDirection( woAniso, anisotropicRoughnessAxes( surf ), uv );
		return normalize( unrotateAnisotropyFrame( hAniso, surf.anisotropyRotation ) );

	}

	// The core contract defines material.ior at the Fraunhofer d line and
	// dispersionAbbeNumber through the two-term Cauchy approximation. The
	// material payload stores the derived B coefficient in nm^2, so evaluate it
	// directly in nm. This keeps n(d) exactly equal to the authored base IOR.
	const float FRAUNHOFER_D_NM = 589.3;

	float cauchyIORFromDLine( float lambdaNm, float iorAtD, float bNm2 ) {
		float lambda2 = lambdaNm * lambdaNm;
		if ( ! ( lambda2 > 0.0 ) || isnan( lambda2 ) || isinf( lambda2 ) ) return iorAtD;
		float dLine2 = FRAUNHOFER_D_NM * FRAUNHOFER_D_NM;
		float result = iorAtD + bNm2 * ( 1.0 / lambda2 - 1.0 / dLine2 );
		return isnan( result ) || isinf( result ) ? iorAtD : max( 1.0, result );
	}

	bool cauchyDispersionEnabled( const in SurfaceRecord surf ) {

		return uSpectralRendering != 0 && surf.dispersionStrength > 0.0;

	}

	float surfaceIorAtHero( const in SurfaceRecord surf, float heroWavelength ) {

		if ( surf.ior == 0.0 ) return 0.0;

		if ( ! cauchyDispersionEnabled( surf ) ) {

			return surf.ior;

		}

		return cauchyIORFromDLine( heroWavelength, surf.ior, surf.dispersionStrength );

	}

	float opticalMaterialIorAtHero(
		const in FogMaterial material,
		float heroWavelength
	) {

		if ( material.ior == 0.0 ) return 0.0;
		if ( uSpectralRendering == 0 || material.dispersionStrength <= 0.0 ) {
			return material.ior;
		}
		return cauchyIORFromDLine(
			heroWavelength, material.ior, material.dispersionStrength
		);

	}

	bool configureSurfaceOpticalInterface(
		const in MediumStack incidentStack,
		uint surfaceBoundaryId,
		const in MaterialControl control,
		float heroWavelength,
		inout SurfaceRecord surf
	) {

		float incidentIor = 1.0;
		if ( incidentStack.count > 0 ) {
			FogMaterial incidentMedium = readFogMaterialInfo(
				materials,
				incidentStack.materialIds[ incidentStack.count - 1 ]
			);
			incidentIor = opticalMaterialIorAtHero(
				incidentMedium, heroWavelength
			);
		}

		float transmittedIor;
		if ( control.opticalVolume && ! surf.frontFace ) {
			if (
				incidentStack.count <= 0 ||
				incidentStack.boundaryIds[ incidentStack.count - 1 ] !=
					surfaceBoundaryId ||
				incidentStack.materialIds[ incidentStack.count - 1 ] !=
					surf.materialIndex
			) return false;
			if ( incidentStack.count > 1 ) {
				FogMaterial enclosingMedium = readFogMaterialInfo(
					materials,
					incidentStack.materialIds[ incidentStack.count - 2 ]
				);
				transmittedIor = opticalMaterialIorAtHero(
					enclosingMedium, heroWavelength
				);
			} else {
				transmittedIor = 1.0;
			}
		} else {
			// Entering bulk interfaces, virtual thin sheets, and ordinary dielectric
			// surfaces all use the material on the opposite side of this interface.
			transmittedIor = surfaceIorAtHero( surf, heroWavelength );
		}

		if (
			incidentIor < 0.0 || transmittedIor < 0.0 ||
			isnan( incidentIor ) || isinf( incidentIor ) ||
			isnan( transmittedIor ) || isinf( transmittedIor )
		) return false;
		bool incidentInfiniteIor = incidentIor == 0.0;
		bool transmittedInfiniteIor = transmittedIor == 0.0;
		if ( incidentInfiniteIor || transmittedInfiniteIor ) {
			// Equal compatibility limits have no interface contrast. Otherwise use
			// eta=0 as a stable, orientation-independent perfect-reflector sentinel:
			// F0 is exactly one and every transmission contribution is exactly zero.
			surf.eta = incidentInfiniteIor && transmittedInfiniteIor ? 1.0 : 0.0;
		} else {
			surf.eta = incidentIor / transmittedIor;
		}
		if ( surf.eta < 0.0 || isnan( surf.eta ) || isinf( surf.eta ) ) {
			return false;
		}
		surf.f0 = surf.eta == 0.0 ? 1.0 : iorRatioToF0( surf.eta );
		return true;

	}

	float transmissionEtaAtHero(
		const in SurfaceRecord surf,
		float heroWavelength,
		vec3 localWo
	) {

		// surf.eta is already resolved from the incident optical stack at the
		// active hero wavelength. Reverse-density evaluation sees wo below the
		// stored normal and must use the reciprocal interface ratio.
		if ( surf.eta == 0.0 ) return 0.0;
		return localWo.z >= 0.0 ? surf.eta : 1.0 / surf.eta;

	}

	// specular
	float specularEval( vec3 wo, vec3 wi, vec3 wh, const in SurfaceRecord surf, float heroWavelength, inout vec3 color ) {

		// if roughness is set to 0 then D === NaN which results in black pixels
		float roughness = surf.filteredRoughness;

		vec3 f0Color = mix(
			surf.f0 * surf.specularColor * surf.specularIntensity,
			surf.color,
			surf.metalness
		);
		BsdfLayeredInterfaceResponse interfaceResponse =
			surfaceLayeredInterfaceResponse(
				dot( wo, wh ), wo, surf, heroWavelength
			);
		vec3 F = interfaceResponse.reflectance;

		// PDF
		// See 14.1.1 Microfacet BxDFs in https://www.pbr-book.org/
		float G = ggxShadowMaskG2ForSurface( wi, wo, surf );
		float D = ggxDistributionForSurface( wh, surf );
		float ggxPdf = ggxPdfForSurface( wo, wh, surf );

		color = wi.z * F * G * D / ( 4.0 * abs( wi.z * wo.z ) );

		// B9 — Kulla-Conty multiscatter energy compensation. The single-scatter
		// lobe above drops the multi-bounce microfacet inter-reflections, so rough
		// metals/speculars read dark; add the multiscatter lobe that restores them.
		// Favg ≈ f0 + (1 − f0)/21 (Fdez-Agüera 2019 cosine-weighted average Fresnel).
		vec3 coatedF0 = bsdfLayeredInterfaceResponse(
			f0Color, surf, abs( wo.z ), heroWavelength
		).reflectance;
		vec3 Favg = coatedF0 + ( vec3( 1.0 ) - coatedF0 ) * ( 1.0 / 21.0 );
		color += ggxMultiscatter( roughness, abs( wo.z ), abs( wi.z ), Favg );

		return ggxPdf / ( 4.0 * dot( wo, wh ) );

	}

	vec3 specularDirection( vec3 wo, const in SurfaceRecord surf ) {

		// sample ggx vndf distribution which gives a new normal
		vec3 halfVector = ggxDirectionForSurface( wo, surf, rand2( 12 ) );

		// apply to new ray by reflecting off the new normal
		return - reflect( wo, halfVector );

	}


	// Transmission / refraction (GGX microfacet BTDF).
	// PDF follows Walter et al., EGSR07 §4.2 — consistent with half-vector Jacobians
	// for the exact VNDF half-vector sampled by transmissionDirection().
	float transmissionEval( vec3 wo, vec3 wi, vec3 wh, const in SurfaceRecord surf, float heroWavelength, inout vec3 color ) {

		if ( surf.thinFilm ) {
			color = vec3( 0.0 );
			return 0.0;
		}
		float eta = transmissionEtaAtHero( surf, heroWavelength, wo );
		if ( ! ( eta > 0.0 ) || isnan( eta ) || isinf( eta ) ) {
			color = vec3( 0.0 );
			return 0.0;
		}
		vec3 interfaceWo = wo;
		float orientation = wo.z < 0.0 ? -1.0 : 1.0;
		wo *= orientation;
		wi *= orientation;
		wh = getHalfVector( wi, wo, eta );
		if ( wh.z < 0.0 ) wh = - wh;
		float woDotH = dot( wo, wh );
		float wiDotH = dot( wi, wh );
		if ( woDotH * wiDotH >= 0.0 || abs( wo.z ) <= EPSILON || abs( wi.z ) <= EPSILON ) {
			color = vec3( 0.0 );
			return 0.0;
		}

		// Our transmission half-vector is normalize(wi + eta * wo), equivalent
		// to Walter's normalize(wo + eta_t/eta_i * wi). The solid-angle
		// Jacobian for mapping the sampled microfacet normal to wi is therefore:
		//   dwh/dwi = |wi·wh| / (wi·wh + eta * wo·wh)^2
		float sqrtDenom = wiDotH + eta * woDotH;
		float denom = sqrtDenom * sqrtDenom;
		if ( denom <= 1e-12 ) {
			color = vec3( 0.0 );
			return 0.0;
		}
		float pdfWh = ggxPdfForSurface( wo, wh, surf );
		float pdfWi = pdfWh * abs( wiDotH ) / denom;
		if ( pdfWi <= 0.0 ) {
			color = vec3( 0.0 );
			return 0.0;
		}

		BsdfLayeredInterfaceResponse interfaceResponse =
			surfaceLayeredInterfaceResponse(
				abs( woDotH ), interfaceWo, surf, heroWavelength
			);
		float D = ggxDistributionForSurface( wh, surf );
		float G = ggxShadowMaskG2ForSurface( wi, wo, surf );
		// Walter et al. rough-dielectric BTDF in radiance-transport mode,
		// multiplied by |n·wi| because ScatterRecord.throughput stores f*cos.
		// eta^2 is 1/(eta_t/eta_i)^2 for this shader's eta convention.
		float btdfCos =
			D * G * abs( wiDotH * woDotH ) * eta * eta /
			max( abs( wo.z ) * denom, 1e-12 );
		color = surf.transmission * surf.color *
			interfaceResponse.baseTransmittance * btdfCos;

		return pdfWi;

	}

	// Conditional density and f*cos of one interface, without material color,
	// face-layer absorption, or lobe-selection probability. Coherent-stack
	// transmittance is part of this physical interface response (and reverses its
	// layer order with wo); this is the exact factor used by the augmented sheet.
	float transmissionInterfaceEval(
		vec3 wo,
		vec3 wi,
		const in SurfaceRecord surf,
		float heroWavelength,
		out vec3 interfaceCos
	) {
		interfaceCos = vec3( 0.0 );
		if ( wo.z * wi.z >= 0.0 ) return 0.0;
		float eta = transmissionEtaAtHero( surf, heroWavelength, wo );
		if ( ! ( eta > 0.0 ) || isnan( eta ) || isinf( eta ) ) return 0.0;
		float orientation = wo.z < 0.0 ? -1.0 : 1.0;
		vec3 canonicalWo = wo * orientation;
		vec3 canonicalWi = wi * orientation;
		if ( canonicalWo.z <= EPSILON || canonicalWi.z >= - EPSILON ) {
			return 0.0;
		}
		vec3 wh = getHalfVector( canonicalWi, canonicalWo, eta );
		if ( wh.z < 0.0 ) wh = - wh;
		float woDotH = dot( canonicalWo, wh );
		float wiDotH = dot( canonicalWi, wh );
		float sqrtDenom = wiDotH + eta * woDotH;
		float denom = sqrtDenom * sqrtDenom;
		if ( woDotH * wiDotH >= 0.0 || ! ( denom > 1e-12 ) ) return 0.0;
		float pdfWi = ggxPdfForSurface( canonicalWo, wh, surf ) *
			abs( wiDotH ) / denom;
		if ( ! ( pdfWi > 0.0 ) || isnan( pdfWi ) || isinf( pdfWi ) ) {
			return 0.0;
		}
		BsdfLayeredInterfaceResponse interfaceResponse =
			surfaceLayeredInterfaceResponse(
				abs( woDotH ), wo, surf, heroWavelength
			);
		float D = ggxDistributionForSurface( wh, surf );
		float G = ggxShadowMaskG2ForSurface(
			canonicalWi, canonicalWo, surf
		);
		interfaceCos = interfaceResponse.baseTransmittance *
			D * G * abs( wiDotH * woDotH ) * eta * eta /
			max( canonicalWo.z * denom, 1e-12 );
		if (
			any( lessThan( interfaceCos, vec3( 0.0 ) ) ) ||
			any( isnan( interfaceCos ) ) || any( isinf( interfaceCos ) )
		) {
			interfaceCos = vec3( 0.0 );
			return 0.0;
		}
		return pdfWi;
	}

	vec3 sampleTransmissionInterfaceDirection(
		vec3 wo,
		const in SurfaceRecord surf,
		float heroWavelength,
		vec2 uv
	) {
		float eta = transmissionEtaAtHero( surf, heroWavelength, wo );
		if ( ! ( eta > 0.0 ) || isnan( eta ) || isinf( eta ) ) {
			return vec3( 0.0 );
		}
		float orientation = wo.z < 0.0 ? -1.0 : 1.0;
		vec3 canonicalWo = wo * orientation;
		vec3 halfVector = ggxDirectionForSurface( canonicalWo, surf, uv );
		vec3 canonicalWi = refract(
			normalize( - canonicalWo ), halfVector, eta
		);
		if ( ! ( dot( canonicalWi, canonicalWi ) > 1e-16 ) ) {
			return vec3( 0.0 );
		}
		return normalize( canonicalWi ) * orientation;
	}

	vec3 transmissionDirection( vec3 wo, const in SurfaceRecord surf ) {
		return sampleTransmissionInterfaceDirection(
			wo, surf, 0.0, rand2( 13 )
		);
	}

	// ── Sprint 8: Chromatic dispersion via Cauchy formula + Jakob+Hanika rider ──
	//
	// evalSpectrum: 6-instruction sigmoid polynomial evaluation.
	// GLSL mirror of @vitrum/shared-samplers/src/jakobHanika.ts::evaluateSpectrum.
	// sigmoid(x) = 0.5 + x * inversesqrt(1 + x²) * 0.5
	//
	// coeffs = (c0, c1, c2) from host-side rgbToSpectralCoefficients.
	// lambda  = wavelength in nm.
	float evalSpectrum( vec3 coeffs, float lambda ) {
		float x = coeffs.x + coeffs.y * lambda + coeffs.z * lambda * lambda;
		return 0.5 + x * inversesqrt( 1.0 + x * x ) * 0.5;
	}

        vec3 attenuationSigmaA( vec3 attColor, float attDist ) {
                if ( attDist <= 0.0 || vitrumIsInfiniteDistance( attDist ) ) {
					return vec3( 0.0 );
				}
                vec3 transmittance = min( attColor, vec3( 1.0 ) );
                return max( - log( transmittance ) / attDist, vec3( 0.0 ) );
        }

        vec3 transmissionAttenuationThroughput(
		sampler2D materialsTex,
		float dist,
		vec3 attColor,
		float attDist,
		bool hasSpectral,
		uint materialIndex,
		float heroWavelength
	) {
		if ( uSpectralRendering == 0 ) {
			return transmissionAttenuation( dist, attColor, attDist );
		}
		return vec3( transmissionAttenuationHero(
			materialsTex,
			dist,
			attColor,
			attDist,
			hasSpectral,
			materialIndex,
			heroWavelength
                ) );
        }

        vec3 fogTrueExtinction(
                sampler2D materialsTex,
                const in FogMaterial fog,
                float heroWavelength
        ) {
                vec3 sigmaA = attenuationSigmaA(
                        fog.attenuationColor, fog.attenuationDistance
                );
                if ( uSpectralRendering == 0 ) return sigmaA + fog.sigmaS;
                float sigmaAHero = fog.hasSpectralAttenuation
                        ? spectralAttenuationMuHero(
                                materialsTex, fog.materialIndex, heroWavelength
                        )
                        : heroScalarFromRgb( sigmaA, heroWavelength );
                float sigmaSHero = heroScalarFromRgb(
                        fog.sigmaS, heroWavelength
                );
                return vec3( sigmaAHero + sigmaSHero );
        }

        vec3 fogSegmentTransmittance(
                sampler2D materialsTex,
                const in FogMaterial fog,
                float dist,
                float heroWavelength
        ) {
                return extinctionTransmittance(
                        fogTrueExtinction(
                                materialsTex, fog, heroWavelength
                        ),
                        dist
                );
        }


	float fogFreeFlightSampleDistance(
		sampler2D materialsTex,
		const in FogMaterial fog,
		float heroWavelength,
		vec2 u
	) {
		vec3 sigmaT = fogTrueExtinction(
			materialsTex, fog, heroWavelength
		);
		float sampledExtinction = sigmaT.x;
		if ( uSpectralRendering == 0 ) {
			vec3 channelProbability = representedEqualThreeWayProbabilities();
			int channel = u.x < channelProbability.x
				? 0
				: u.x < channelProbability.x + channelProbability.y
					? 1
					: 2;
			sampledExtinction = sigmaT[ channel ];
		}
		return intersectFogVolume( sampledExtinction, u.y );
	}

	float fogExtinctionCollisionDensity(
		float extinction,
		float dist
	) {
		if (
			isnan( extinction ) || extinction < 0.0 ||
			isnan( dist ) || dist < 0.0
		) return 0.0;
		if ( dist == 0.0 && isinf( extinction ) ) return INFINITY;
		if ( isinf( extinction ) || vitrumIsInfiniteDistance( dist ) ) return 0.0;
		return extinction * extinctionTransmittance( extinction, dist );
	}

	float fogProposalSurvival(
		sampler2D materialsTex,
		const in FogMaterial fog,
		float dist,
		float heroWavelength
	) {
		vec3 survival = fogSegmentTransmittance(
			materialsTex, fog, dist, heroWavelength
		);
		vec3 channelProbability = representedEqualThreeWayProbabilities();
		return uSpectralRendering != 0
			? survival.x
			: dot( channelProbability, survival );
	}

	float fogProposalCollisionDensity(
		sampler2D materialsTex,
		const in FogMaterial fog,
		float dist,
		float heroWavelength
	) {
		vec3 sigmaT = fogTrueExtinction(
			materialsTex, fog, heroWavelength
		);
		vec3 density = vec3(
			fogExtinctionCollisionDensity( sigmaT.x, dist ),
			fogExtinctionCollisionDensity( sigmaT.y, dist ),
			fogExtinctionCollisionDensity( sigmaT.z, dist )
		);
		vec3 channelProbability = representedEqualThreeWayProbabilities();
		return uSpectralRendering != 0
			? density.x
			: dot( channelProbability, density );
	}

	// RGB transport samples the nearest RNG-representable equal-channel mixture
	// of the three authored exponential extinction laws. Surface weights remain
	// bounded by the reciprocal of the smallest represented channel weight, and a
	// zero-color (+Infinity) lane cannot contaminate either finite lane. Hero-
	// wavelength transport is the scalar specialization of the same estimator.
	vec3 fogFreeFlightSurvivalWeight(
		sampler2D materialsTex,
		const in FogMaterial fog,
		float dist,
		float heroWavelength
	) {
		vec3 survival = fogSegmentTransmittance(
			materialsTex, fog, dist, heroWavelength
		);
		vec3 channelProbability = representedEqualThreeWayProbabilities();
		float proposalSurvival = uSpectralRendering != 0
			? survival.x
			: dot( channelProbability, survival );
		return proposalSurvival > 0.0 && ! isinf( proposalSurvival )
			? survival / proposalSurvival
			: vec3( 0.0 );
	}

	vec3 fogFreeFlightCollisionWeight(
		sampler2D materialsTex,
		const in FogMaterial fog,
		float dist,
		float heroWavelength
	) {
		vec3 survival = fogSegmentTransmittance(
			materialsTex, fog, dist, heroWavelength
		);
		float proposalDensity = fogProposalCollisionDensity(
			materialsTex, fog, dist, heroWavelength
		);
		return proposalDensity > 0.0 && ! isinf( proposalDensity )
			? survival / proposalDensity
			: vec3( 0.0 );
	}

	vec3 opticalSolidSegmentTransmittance(
		sampler2D materialsTex,
		const in MediumStack stack,
		const in FogMaterial medium,
		float dist,
		float heroWavelength
	) {

		if ( isnan( dist ) || dist < 0.0 ) return vec3( 0.0 );
		if ( stack.count <= 0 || dist == 0.0 ) return vec3( 1.0 );
		float attenuationDist = dist;
		int top = stack.count - 1;
		if ( stack.hasAttenuationThicknesses[ top ] ) {
			attenuationDist = min(
				attenuationDist,
				max( stack.attenuationThicknesses[ top ], 0.0 )
			);
		} else if ( medium.hasAttenuationThickness ) {
			attenuationDist = min(
				attenuationDist, medium.attenuationThickness
			);
		}
		return transmissionAttenuationThroughput(
			materialsTex,
			attenuationDist,
			medium.attenuationColor,
			medium.attenuationDistance,
			medium.hasSpectralAttenuation,
			medium.materialIndex,
			heroWavelength
		);

	}

	vec3 opticalPathSegmentThroughput(
		sampler2D materialsTex,
		const in MediumStack stack,
		const in FogMaterial medium,
		float dist,
		float heroWavelength
	) {

		if ( isnan( dist ) || dist < 0.0 ) return vec3( 0.0 );
		if ( stack.count <= 0 || dist == 0.0 ) return vec3( 1.0 );
		if ( medium.fogVolume ) {
			return fogFreeFlightSurvivalWeight(
				materialsTex, medium, dist, heroWavelength
			);
		}
		return opticalSolidSegmentTransmittance(
			materialsTex,
			stack,
			medium,
			dist,
			heroWavelength
		);

	}

	vec3 opticalVisibilitySegmentTransmittance(
		sampler2D materialsTex,
		const in MediumStack stack,
		const in FogMaterial medium,
		float dist,
		float heroWavelength
	) {

		if ( isnan( dist ) || dist < 0.0 ) return vec3( 0.0 );
		if ( stack.count <= 0 || dist == 0.0 ) return vec3( 1.0 );
		if ( medium.fogVolume ) {
			return fogSegmentTransmittance(
				materialsTex, medium, dist, heroWavelength
			);
		}
		return opticalSolidSegmentTransmittance(
			materialsTex,
			stack,
			medium,
			dist,
			heroWavelength
		);

	}

	        float mediumPhasePdf( vec3 worldWo, vec3 worldWi, float g ) {
                float cosTheta = clamp(
                        dot( - normalize( worldWo ), normalize( worldWi ) ),
                        -1.0,
                        1.0
                );
                return hg_phase( cosTheta, g );
        }

        vec3 sampleMediumPhase( vec3 worldWo, float g, vec2 uv ) {
                float cosTheta = sampleHgCosTheta( uv.x, g );
                float sinTheta = sqrt( max( 1.0 - cosTheta * cosTheta, 0.0 ) );
                float phi = 2.0 * PI * uv.y;
                vec3 localDirection = vec3(
                        sinTheta * cos( phi ),
                        sinTheta * sin( phi ),
                        cosTheta
                );
                return normalize(
                        getBasisFromNormal( - normalize( worldWo ) ) * localDirection
                );
        }

	// Sprint 12: dielectric transmission with per-material, hero-wavelength Cauchy IOR.
	vec3 dispersionTransmissionDirection( vec3 wo, const in SurfaceRecord surf, float heroWavelength ) {
		return sampleTransmissionInterfaceDirection(
			wo, surf, heroWavelength, rand2( 13 )
		);
	}

	// clearcoat
	float clearcoatEval( vec3 wo, vec3 wi, vec3 wh, const in SurfaceRecord surf, inout vec3 color ) {

		float ior = 1.5;
		float f0 = iorRatioToF0( ior );
		bool frontFace = surf.frontFace;
		float roughness = surf.filteredClearcoatRoughness;

		float eta = frontFace ? 1.0 / ior : ior;
		float G = ggxShadowMaskG2( wi, wo, roughness );
		float D = ggxDistribution( wh, roughness );
		float F = schlickFresnel( dot( wi, wh ), f0 );

		float fClearcoat = F * D * G / ( 4.0 * abs( wi.z * wo.z ) );
		color = color * ( 1.0 - surf.clearcoat * F ) + fClearcoat * surf.clearcoat * wi.z;

		// PDF
		// See equation (27) in http://jcgt.org/published/0003/02/03/
		return ggxPDF( wo, wh, roughness ) / ( 4.0 * dot( wi, wh ) );

	}

	vec3 clearcoatDirection( vec3 wo, const in SurfaceRecord surf ) {

		// sample ggx vndf distribution which gives a new normal
		float roughness = surf.filteredClearcoatRoughness;
		vec3 halfVector = ggxDirection(
			wo,
			vec2( roughness ),
			rand2( 14 )
		);

		// apply to new ray by reflecting off the new normal
		return - reflect( wo, halfVector );

	}

	// sheen
	vec3 sheenColor( vec3 wo, vec3 wi, vec3 wh, const in SurfaceRecord surf ) {

		float cosThetaO = saturateCos( wo.z );
		float cosThetaI = saturateCos( wi.z );
		float cosThetaH = wh.z;

		float D = velvetD( cosThetaH, surf.sheenRoughness );
		float G = velvetG( cosThetaO, cosThetaI, surf.sheenRoughness );

		// See equation (1) in http://www.aconty.com/pdf/s2017_pbs_imageworks_sheen.pdf
		vec3 color = surf.sheenColor;
		color *= D * G / ( 4.0 * abs( cosThetaO * cosThetaI ) );
		color *= wi.z;

		return color;

	}

	// bsdf
	void getLobeWeights(
		vec3 wo, vec3 wi, vec3 wh, vec3 clearcoatWo, const in SurfaceRecord surf,
		float heroWavelength,
		inout float diffuseWeight, inout float specularWeight, inout float transmissionWeight, inout float clearcoatWeight
	) {

		float metalness = surf.metalness;
		float transmission = surf.transmission;
		BsdfLayeredInterfaceResponse interfaceResponse =
			surfaceLayeredInterfaceResponse(
				abs( wo.z ), wo, surf, heroWavelength
			);
		float fEstimate = clamp(
			luminance( interfaceResponse.reflectance ), 0.0, 1.0
		);

		float transSpecularProb = mix( max( 0.25, fEstimate ), 1.0, metalness );
		float diffSpecularProb = 0.5 + 0.5 * metalness;

		diffuseWeight = ( 1.0 - transmission ) * ( 1.0 - diffSpecularProb );
		specularWeight = transmission * transSpecularProb + ( 1.0 - transmission ) * diffSpecularProb;
		transmissionWeight = transmission * ( 1.0 - transSpecularProb );
		clearcoatWeight = surf.clearcoat * schlickFresnel( clearcoatWo.z, 0.04 );

		float totalWeight = diffuseWeight + specularWeight + transmissionWeight + clearcoatWeight;
		if (
			! ( totalWeight > 0.0 ) ||
			isnan( totalWeight ) || isinf( totalWeight )
		) {
			diffuseWeight = 1.0;
			specularWeight = 0.0;
			transmissionWeight = 0.0;
			clearcoatWeight = 0.0;
			return;
		}
		diffuseWeight /= totalWeight;
		specularWeight /= totalWeight;
		transmissionWeight /= totalWeight;
		clearcoatWeight /= totalWeight;
		representedCategoricalProbabilities4(
			diffuseWeight,
			specularWeight,
			transmissionWeight,
			clearcoatWeight
		);
	}

        void getSamplingLobeWeights(
                vec3 wo, vec3 clearcoatWo, const in SurfaceRecord surf, float heroWavelength,
                inout float diffuseWeight, inout float specularWeight, inout float transmissionWeight, inout float clearcoatWeight
        ) {

		// The path sampler chooses a lobe before a candidate wi exists. Use the
		// same incident-direction-independent policy anywhere a mixed BSDF PDF is
		// reported, including NEE/BDPT bsdfResult() calls for arbitrary light
		// directions. The individual lobe BRDF values remain evaluated at the real
		// wi; only the mixture-selection probabilities are frozen to the sampling
		// policy so MIS sees the PDF of the distribution that could generate wi.
		getLobeWeights( wo, wo, vec3( 0, 0, 1 ), clearcoatWo, surf, heroWavelength, diffuseWeight, specularWeight, transmissionWeight, clearcoatWeight );

        }

        bool bsdfBaseLobesAreDelta( const in SurfaceRecord surf ) {

            // Anisotropy only shapes a lobe with nonzero width. At exact-zero
            // roughness both axes collapse to the same discrete direction.
            return surf.filteredRoughness <= 0.0;

        }

        bool bsdfClearcoatLobeIsDelta( const in SurfaceRecord surf ) {

                return surf.filteredClearcoatRoughness <= 0.0;

        }

        bool bsdfDeltaDirectionMatches( vec3 sampledDirection, vec3 candidateDirection ) {

                return dot(
                        normalize( sampledDirection ),
                        normalize( candidateDirection )
                ) >= 1.0 - 2e-6;

        }

        vec3 bsdfDeltaTransmissionDirection(
                vec3 wo,
                const in SurfaceRecord surf,
                float heroWavelength
        ) {

		float eta = transmissionEtaAtHero( surf, heroWavelength, wo );
		if ( ! ( eta > 0.0 ) || isnan( eta ) || isinf( eta ) ) {
			return vec3( 0.0 );
		}
		float orientation = wo.z < 0.0 ? -1.0 : 1.0;
		vec3 canonicalDirection = refract(
			normalize( - wo * orientation ),
			vec3( 0.0, 0.0, 1.0 ),
			eta
		);
		if ( ! ( dot( canonicalDirection, canonicalDirection ) > 1e-16 ) ) {
			return vec3( 0.0 );
		}
		return normalize( canonicalDirection ) * orientation;

        }

	struct ThinSheetTransmissionSample {
		vec3 worldDirection;
		vec3 throughput;
		float pdfFwd;
		float pdfRev;
		float sampledRoughness;
		bool sampledDelta;
		bool valid;
	};

	bool sampleThinSheetInterface(
		vec3 wo,
		const in SurfaceRecord surf,
		float heroWavelength,
		vec2 uv,
		out vec3 wi,
		out vec3 valueCos,
		out float pdfFwd,
		out float pdfRev,
		out float sampledRoughness
	) {
		valueCos = vec3( 0.0 );
		pdfFwd = 0.0;
		pdfRev = 0.0;
		sampledRoughness = 0.0;
		bool delta = bsdfBaseLobesAreDelta( surf );
		if ( delta ) {
			wi = bsdfDeltaTransmissionDirection( wo, surf, heroWavelength );
			if ( ! ( dot( wi, wi ) > 1e-16 ) ) return false;
			float eta = transmissionEtaAtHero( surf, heroWavelength, wo );
			if ( ! ( eta > 0.0 ) || isnan( eta ) || isinf( eta ) ) {
				return false;
			}
			BsdfLayeredInterfaceResponse interfaceResponse =
				surfaceLayeredInterfaceResponse(
					abs( wo.z ), wo, surf, heroWavelength
				);
			valueCos = interfaceResponse.baseTransmittance * eta * eta;
			pdfFwd = 1.0;
			pdfRev = 1.0;
			return
				all( greaterThanEqual( valueCos, vec3( 0.0 ) ) ) &&
				! any( isnan( valueCos ) ) && ! any( isinf( valueCos ) );
		}

		wi = sampleTransmissionInterfaceDirection(
			wo, surf, heroWavelength, uv
		);
		if ( ! ( dot( wi, wi ) > 1e-16 ) ) return false;
		pdfFwd = transmissionInterfaceEval(
			wo, wi, surf, heroWavelength, valueCos
		);
		vec3 reverseValueCos;
		pdfRev = transmissionInterfaceEval(
			wi, wo, surf, heroWavelength, reverseValueCos
		);
		float orientation = wo.z < 0.0 ? -1.0 : 1.0;
		float eta = transmissionEtaAtHero( surf, heroWavelength, wo );
		vec3 wh = getHalfVector( wi * orientation, wo * orientation, eta );
		sampledRoughness = sqrt( max( 1.0 - wh.z * wh.z, 0.0 ) );
		return
			pdfFwd > 0.0 && pdfRev > 0.0 &&
			all( greaterThanEqual( valueCos, vec3( 0.0 ) ) ) &&
			! isnan( sampledRoughness ) && ! isinf( sampledRoughness );
	}

	ThinSheetTransmissionSample sampleThinSheetTransmission(
		vec3 worldWo,
		const in SurfaceRecord surf,
		float heroWavelength,
		float transmissionWeight,
		vec2 entryUv,
		vec2 exitUv
	) {
		ThinSheetTransmissionSample result;
		result.worldDirection = normalize( surf.normal );
		result.throughput = vec3( 0.0 );
		result.pdfFwd = 0.0;
		result.pdfRev = 0.0;
		result.sampledRoughness = 0.0;
		result.sampledDelta = false;
		result.valid = false;
		if ( ! ( transmissionWeight > 0.0 ) ) return result;

		vec3 entryWo = normalize( transpose( surf.normalBasis ) * worldWo );
		vec3 entryWi;
		vec3 entryValue;
		float entryPdfFwd;
		float entryPdfRev;
		float entryRoughness;
		if ( ! sampleThinSheetInterface(
			entryWo, surf, heroWavelength, entryUv, entryWi,
			entryValue, entryPdfFwd, entryPdfRev, entryRoughness
		) ) return result;

		vec3 internalWorldDirection = normalize( surf.normalBasis * entryWi );
		SurfaceRecord exitSurf = oppositeFacingSurface( surf, false );
		// The authored coherent ThinFilmStack is one physical coating. The entry
		// interface evaluates it once; the opposite substrate face retains its
		// bare dielectric response without applying the same stack a second time.
		// exitWo lies below the reversed basis, so transmissionEtaAtHero already
		// takes the reciprocal of the entry ratio. Pre-reciprocating exitSurf.eta
		// here would invert it twice and apply air-to-glass at both interfaces.
		exitSurf.thinFilmEnabled = 0.0;
		exitSurf.thinFilmLayerCount = 0.0;
		vec3 exitWoWorld = - internalWorldDirection;
		vec3 exitWo = normalize(
			transpose( exitSurf.normalBasis ) * exitWoWorld
		);
		vec3 exitWi;
		vec3 exitValue;
		float exitPdfFwd;
		float exitPdfRev;
		float exitRoughness;
		if ( ! sampleThinSheetInterface(
			exitWo, exitSurf, heroWavelength, exitUv, exitWi,
			exitValue, exitPdfFwd, exitPdfRev, exitRoughness
		) ) {
			// The bounded single-pass sheet estimator owns no internal-reflection
			// chain. Exit TIR is therefore a null transmission draw, not a sample
			// redirected into the entry reflection strategy.
			return result;
		}

		result.worldDirection = normalize( exitSurf.normalBasis * exitWi );
		float reverseDiffuseWeight;
		float reverseSpecularWeight;
		float reverseTransmissionWeight;
		float reverseClearcoatWeight;
		vec3 reverseClearcoatWo = normalize(
			transpose( exitSurf.clearcoatBasis ) * result.worldDirection
		);
		getSamplingLobeWeights(
			exitWi, reverseClearcoatWo, exitSurf, heroWavelength,
			reverseDiffuseWeight, reverseSpecularWeight,
			reverseTransmissionWeight, reverseClearcoatWeight
		);
		result.pdfFwd = transmissionWeight * entryPdfFwd * exitPdfFwd;
		result.pdfRev =
			reverseTransmissionWeight * exitPdfRev * entryPdfRev;
		result.throughput =
			surf.transmission * surf.color * entryValue * exitValue *
			activeLayerThroughput( surf, heroWavelength ) *
			oppositeLayerThroughput( surf, heroWavelength );
		result.sampledDelta =
			bsdfBaseLobesAreDelta( surf ) &&
			bsdfBaseLobesAreDelta( exitSurf );
		result.sampledRoughness = entryRoughness + exitRoughness;
		result.valid =
			result.pdfFwd > 0.0 && result.pdfRev > 0.0 &&
			all( greaterThanEqual( result.throughput, vec3( 0.0 ) ) ) &&
			! any( isnan( result.throughput ) ) &&
			! any( isinf( result.throughput ) ) &&
			! isnan( result.pdfFwd ) && ! isinf( result.pdfFwd ) &&
			! isnan( result.pdfRev ) && ! isinf( result.pdfRev );
		return result;
	}

	ThinSheetTransmissionSample sampleThinSheetTransmission(
		vec3 worldWo,
		const in SurfaceRecord surf,
		float heroWavelength,
		float transmissionWeight
	) {
		return sampleThinSheetTransmission(
			worldWo, surf, heroWavelength, transmissionWeight,
			rand2( 13 ), rand2( 17 )
		);
	}

	bool thinSheetExactVisibilityTransmission(
		vec3 worldWo,
		vec3 connectionDirection,
		const in SurfaceRecord surf,
		float heroWavelength,
		out vec3 attenuation
	) {
		attenuation = vec3( 0.0 );
		if (
			! surf.thinFilm || ! ( surf.transmission > 0.0 ) ||
			surf.roughness != 0.0 || surf.oppositeRoughness != 0.0
		) return false;
		ThinSheetTransmissionSample sheetSample = sampleThinSheetTransmission(
			worldWo, surf, heroWavelength, 1.0,
			vec2( 0.5 ), vec2( 0.5 )
		);
		if (
			! sheetSample.valid || ! sheetSample.sampledDelta ||
			! bsdfDeltaDirectionMatches(
				sheetSample.worldDirection, connectionDirection
			)
		) return false;
		attenuation = pathThroughputFromRgb(
			sheetSample.throughput, heroWavelength
		);
		return
			all( greaterThanEqual( attenuation, vec3( 0.0 ) ) ) &&
			! any( isnan( attenuation ) ) &&
			! any( isinf( attenuation ) );
	}

        float bsdfDeltaPdfLocal(
                vec3 wo,
                vec3 clearcoatWo,
                vec3 wi,
                vec3 clearcoatWi,
                const in SurfaceRecord surf,
                float heroWavelength,
                float specularWeight,
                float transmissionWeight,
                float clearcoatWeight,
                out bool deltaMeasure
        ) {

                deltaMeasure = false;
                float pdf = 0.0;
                if ( bsdfBaseLobesAreDelta( surf ) ) {

                        vec3 reflected = - reflect( wo, vec3( 0.0, 0.0, 1.0 ) );
                        if (
                                specularWeight > 0.0 &&
                                wi.z > 0.0 &&
                                bsdfDeltaDirectionMatches( reflected, wi )
                        ) {

                                pdf += specularWeight;
                                deltaMeasure = true;

                        }

                        vec3 transmitted =
                                bsdfDeltaTransmissionDirection( wo, surf, heroWavelength );
                        if (
                                transmissionWeight > 0.0 &&
								! surf.thinFilm &&
                                wi.z < 0.0 &&
                                length( transmitted ) > 1e-8 &&
                                bsdfDeltaDirectionMatches( transmitted, wi )
                        ) {

                                pdf += transmissionWeight;
                                deltaMeasure = true;

                        }

                }

                if ( bsdfClearcoatLobeIsDelta( surf ) && clearcoatWeight > 0.0 ) {

                        vec3 reflectedClearcoat =
                                - reflect( clearcoatWo, vec3( 0.0, 0.0, 1.0 ) );
                        if (
                                clearcoatWi.z >= 0.0 &&
                                bsdfDeltaDirectionMatches( reflectedClearcoat, clearcoatWi )
                        ) {

                                pdf += clearcoatWeight;
                                deltaMeasure = true;

                        }

                }

                return pdf;

        }

        float bsdfPdfLocal(
                vec3 wo,
                vec3 clearcoatWo,
                vec3 wi,
                vec3 clearcoatWi,
                const in SurfaceRecord surf,
                float heroWavelength,
                float diffuseWeight,
                float specularWeight,
                float transmissionWeight,
                float clearcoatWeight,
                out bool deltaMeasure
        ) {

                float deltaPdf = bsdfDeltaPdfLocal(
                        wo,
                        clearcoatWo,
                        wi,
                        clearcoatWi,
                        surf,
                        heroWavelength,
                        specularWeight,
                        transmissionWeight,
                        clearcoatWeight,
                        deltaMeasure
                );
                if ( deltaMeasure ) return deltaPdf;

                float dpdf = 0.0;
                float spdf = 0.0;
                float tpdf = 0.0;
                float cpdf = 0.0;
                if ( diffuseWeight > 0.0 && wi.z > 0.0 ) {

                        dpdf = wi.z / PI;

                }
                if (
                        specularWeight > 0.0 &&
                        wi.z > 0.0 &&
                        ! bsdfBaseLobesAreDelta( surf )
                ) {

                        vec3 wh = getHalfVector( wi, wo );
                        spdf = ggxPdfForSurface( wo, wh, surf ) /
                                max( 4.0 * dot( wo, wh ), 1e-12 );

                }
                if (
                        transmissionWeight > 0.0 &&
						! surf.thinFilm &&
                        wi.z < 0.0 &&
                        ! bsdfBaseLobesAreDelta( surf )
                ) {
			vec3 ignoredInterfaceCos;
			tpdf = transmissionInterfaceEval(
				wo, wi, surf, heroWavelength, ignoredInterfaceCos
			);

                }
                if (
                        clearcoatWeight > 0.0 &&
                        clearcoatWi.z >= 0.0 &&
                        ! bsdfClearcoatLobeIsDelta( surf )
                ) {

                        vec3 wh = getHalfVector( clearcoatWi, clearcoatWo );
                        cpdf = ggxPDF(
                                clearcoatWo,
                                wh,
                                surf.filteredClearcoatRoughness
                        ) / max( 4.0 * dot( clearcoatWi, wh ), 1e-12 );

                }

                return
                        dpdf * diffuseWeight +
                        spdf * specularWeight +
                        tpdf * transmissionWeight +
                        cpdf * clearcoatWeight;

        }

        float bsdfPdfResult(
                vec3 worldWo,
                vec3 worldWi,
                const in SurfaceRecord surf,
                float heroWavelength,
                out bool deltaMeasure
        ) {

                if ( surf.volumeParticle ) {

                        deltaMeasure = false;
                        return mediumPhasePdf(
                                worldWo, worldWi, surf.sssAnisotropyG
                        );

                }

		SurfaceRecord orientedSurf = surf;
		mat3 normalInvBasis = transpose( orientedSurf.normalBasis );
		vec3 wo = normalize( normalInvBasis * worldWo );
		vec3 wi = normalize( normalInvBasis * worldWi );
		if ( wo.z < 0.0 ) {
			orientedSurf = oppositeFacingSurface( surf, true );
			normalInvBasis = transpose( orientedSurf.normalBasis );
			wo = normalize( normalInvBasis * worldWo );
			wi = normalize( normalInvBasis * worldWi );
		}
		mat3 clearcoatInvBasis = transpose( orientedSurf.clearcoatBasis );
                vec3 clearcoatWo = normalize( clearcoatInvBasis * worldWo );
                vec3 clearcoatWi = normalize( clearcoatInvBasis * worldWi );

                float diffuseWeight;
                float specularWeight;
                float transmissionWeight;
                float clearcoatWeight;
                getSamplingLobeWeights(
                        wo,
                        clearcoatWo,
			orientedSurf,
                        heroWavelength,
                        diffuseWeight,
                        specularWeight,
                        transmissionWeight,
                        clearcoatWeight
                );
                return bsdfPdfLocal(
                        wo,
                        clearcoatWo,
                        wi,
                        clearcoatWi,
			orientedSurf,
                        heroWavelength,
                        diffuseWeight,
                        specularWeight,
                        transmissionWeight,
                        clearcoatWeight,
                        deltaMeasure
                );

        }

        float bsdfDeltaEvalLocal(
                vec3 wo,
                vec3 clearcoatWo,
                vec3 wi,
                vec3 clearcoatWi,
                const in SurfaceRecord surf,
                float heroWavelength,
                float specularWeight,
                float transmissionWeight,
                float clearcoatWeight,
                inout vec3 color
        ) {

                bool deltaMeasure;
                float pdf = bsdfDeltaPdfLocal(
                        wo,
                        clearcoatWo,
                        wi,
                        clearcoatWi,
                        surf,
                        heroWavelength,
                        specularWeight,
                        transmissionWeight,
                        clearcoatWeight,
                        deltaMeasure
                );
                color = vec3( 0.0 );
                if ( ! deltaMeasure ) return 0.0;

                vec3 baseDelta = vec3( 0.0 );
                if ( bsdfBaseLobesAreDelta( surf ) ) {

                        vec3 reflected = - reflect( wo, vec3( 0.0, 0.0, 1.0 ) );
                        if (
                                specularWeight > 0.0 &&
                                wi.z > 0.0 &&
                                bsdfDeltaDirectionMatches( reflected, wi )
                        ) {

						baseDelta += surfaceLayeredInterfaceResponse(
							abs( wo.z ), wo, surf, heroWavelength
						).reflectance;

                        }

                        vec3 transmitted =
                                bsdfDeltaTransmissionDirection( wo, surf, heroWavelength );
                        if (
                                transmissionWeight > 0.0 &&
								! surf.thinFilm &&
                                wi.z < 0.0 &&
                                length( transmitted ) > 1e-8 &&
                                bsdfDeltaDirectionMatches( transmitted, wi )
                        ) {

	                                float eta = transmissionEtaAtHero( surf, heroWavelength, wo );
						BsdfLayeredInterfaceResponse interfaceResponse =
							surfaceLayeredInterfaceResponse(
								abs( wo.z ), wo, surf, heroWavelength
							);
						vec3 transmissionColor =
							surf.transmission *
							surf.color *
							interfaceResponse.baseTransmittance *
							eta * eta;
						baseDelta += transmissionColor;

                        }

                }

                vec3 clearcoatDelta = vec3( 0.0 );
                float clearcoatF = 0.0;
                if ( ( surf.lobeMask & 8u ) != 0u ) {

                        clearcoatF = schlickFresnel( abs( clearcoatWo.z ), 0.04 );
                        if (
                                bsdfClearcoatLobeIsDelta( surf ) &&
                                clearcoatWeight > 0.0
                        ) {

                                vec3 reflectedClearcoat =
                                        - reflect( clearcoatWo, vec3( 0.0, 0.0, 1.0 ) );
                                if (
                                        clearcoatWi.z >= 0.0 &&
                                        bsdfDeltaDirectionMatches(
                                                reflectedClearcoat,
                                                clearcoatWi
                                        )
                                ) {

                                        clearcoatDelta =
                                                vec3( surf.clearcoat * clearcoatF );

                                }

                        }
                        if ( clearcoatWi.z >= 0.0 ) {

                                baseDelta *= 1.0 - surf.clearcoat * clearcoatF;

                        }

                }

                if ( ( surf.lobeMask & 4u ) != 0u ) {

                        baseDelta *= mix(
                                1.0,
                                sheenAlbedoScaling( wo, wi, surf ),
                                surf.sheen
                        );

                }
                color =
                        ( baseDelta + clearcoatDelta ) *
                        activeLayerThroughput( surf, heroWavelength );
                return pdf;

        }

        float bsdfEval(
                vec3 wo, vec3 clearcoatWo, vec3 wi, vec3 clearcoatWi, const in SurfaceRecord surf,
                float heroWavelength,
                float diffuseWeight, float specularWeight, float transmissionWeight, float clearcoatWeight, inout float specularPdf, inout vec3 color
        ) {

		float spdf = 0.0;
		float dpdf = 0.0;
		float tpdf = 0.0;
		float cpdf = 0.0;
		color = vec3( 0.0 );

		vec3 halfVector = getHalfVector( wi, wo, surf.eta );

		if ( diffuseWeight > 0.0 && wi.z > 0.0 ) {

			dpdf = diffuseEval(
				wo, wi, halfVector, surf, heroWavelength, color
			);

		}

                if (
                        specularWeight > 0.0 &&
                        wi.z > 0.0 &&
                        ! bsdfBaseLobesAreDelta( surf )
                ) {

			vec3 outColor;
			spdf = specularEval( wo, wi, getHalfVector( wi, wo ), surf, heroWavelength, outColor );
			color += outColor;

		}

                if (
                        transmissionWeight > 0.0 &&
						! surf.thinFilm &&
                        wi.z < 0.0 &&
                        ! bsdfBaseLobesAreDelta( surf )
                ) {

		vec3 transmissionHalfVector = getHalfVector(
			wi, wo, transmissionEtaAtHero( surf, heroWavelength, wo )
		);
			vec3 transmissionColor;
			tpdf = transmissionEval( wo, wi, transmissionHalfVector, surf, heroWavelength, transmissionColor );
			color += transmissionColor;

		}

		if ( ( surf.lobeMask & 4u ) != 0u ) {
			color *= mix( 1.0, sheenAlbedoScaling( wo, wi, surf ), surf.sheen );
			color += sheenColor( wo, wi, halfVector, surf ) * surf.sheen;
		}

                if (
                        ( surf.lobeMask & 8u ) != 0u &&
                        clearcoatWi.z >= 0.0 &&
                        clearcoatWeight > 0.0 &&
                        ! bsdfClearcoatLobeIsDelta( surf )
                ) {

			vec3 clearcoatHalfVector = getHalfVector( clearcoatWo, clearcoatWi );
                        cpdf = clearcoatEval( clearcoatWo, clearcoatWi, clearcoatHalfVector, surf, color );

                } else if (
                        ( surf.lobeMask & 8u ) != 0u &&
                        clearcoatWi.z >= 0.0 &&
                        clearcoatWeight > 0.0
                ) {

                        float clearcoatF =
                                schlickFresnel( abs( clearcoatWo.z ), 0.04 );
                        color *= 1.0 - surf.clearcoat * clearcoatF;

                }

		// RFE-03 / Sprint 14: apply selected front/back layer absorption exactly once
		// in the BSDF evaluation flow, after all lobes have been summed.
		// activeLayerThroughput() returns vec3(1.0) when surf.hasActiveLayer is
		// false, so non-layered materials are unaffected.
		color *= activeLayerThroughput( surf, heroWavelength );

		float pdf =
			dpdf * diffuseWeight
			+ spdf * specularWeight
			+ tpdf * transmissionWeight
			+ cpdf * clearcoatWeight;

		specularPdf = spdf * specularWeight + cpdf * clearcoatWeight;

		return pdf;

	}

	float bsdfResult( vec3 worldWo, vec3 worldWi, const in SurfaceRecord surf, float heroWavelength, inout vec3 color ) {

           if ( surf.volumeParticle ) {

                   float phasePdf = mediumPhasePdf(
                           worldWo, worldWi, surf.sssAnisotropyG
                   );
                   color = surf.color * phasePdf;
                   return phasePdf;

		}

		SurfaceRecord orientedSurf = surf;
		mat3 normalInvBasis = transpose( orientedSurf.normalBasis );
		vec3 wo = normalize( normalInvBasis * worldWo );
		vec3 wi = normalize( normalInvBasis * worldWi );
		if ( wo.z < 0.0 ) {
			orientedSurf = oppositeFacingSurface( surf, true );
			normalInvBasis = transpose( orientedSurf.normalBasis );
			wo = normalize( normalInvBasis * worldWo );
			wi = normalize( normalInvBasis * worldWi );
		}

		mat3 clearcoatInvBasis = transpose( orientedSurf.clearcoatBasis );
		vec3 clearcoatWo = normalize( clearcoatInvBasis * worldWo );
		vec3 clearcoatWi = normalize( clearcoatInvBasis * worldWi );

		float diffuseWeight;
		float specularWeight;
		float transmissionWeight;
                float clearcoatWeight;
                getSamplingLobeWeights(
			wo, clearcoatWo, orientedSurf, heroWavelength,
			diffuseWeight, specularWeight, transmissionWeight,
			clearcoatWeight
		);

                float deltaPdf = bsdfDeltaEvalLocal(
                        wo,
                        clearcoatWo,
                        wi,
                        clearcoatWi,
				orientedSurf,
                        heroWavelength,
                        specularWeight,
                        transmissionWeight,
                        clearcoatWeight,
                        color
                );
                if ( deltaPdf > 0.0 ) return deltaPdf;

                float specularPdf;
		return bsdfEval(
			wo, clearcoatWo, wi, clearcoatWi, orientedSurf,
			heroWavelength, diffuseWeight, specularWeight,
			transmissionWeight, clearcoatWeight, specularPdf, color
		);

	}

	// Explicit participating-medium continuation. Surface BSDFs never invoke a
	// second SSS shortcut; free flight and HG vertices own scattering transport.
	ScatterRecord bsdfSample( vec3 worldWo, const in SurfaceRecord surf, float heroWavelength ) {

           if ( surf.volumeParticle ) {

                        ScatterRecord sampleRec;
                        sampleRec.specularPdf = 0.0;
                        sampleRec.direction = sampleMediumPhase(
                                worldWo, surf.sssAnisotropyG, rand2( 16 )
                        );
			 sampleRec.pdf = mediumPhasePdf(
				 worldWo, sampleRec.direction, surf.sssAnisotropyG
			 );
			 sampleRec.pdfRev = sampleRec.pdf;
			 sampleRec.sampledDelta = false;
			 sampleRec.sampledNonConnectable = false;
			 sampleRec.sampledRoughness = 0.0;
			 sampleRec.throughput = pathThroughputFromRgb(
				 surf.color * sampleRec.pdf, heroWavelength
			 );
			return sampleRec;

		}

		mat3 normalBasis = surf.normalBasis;
		mat3 invBasis = transpose( normalBasis );
		mat3 clearcoatNormalBasis = surf.clearcoatBasis;
		mat3 clearcoatInvBasis = transpose( clearcoatNormalBasis );
		vec3 wo = normalize( invBasis * worldWo );
		vec3 clearcoatWo = normalize( clearcoatInvBasis * worldWo );

		float diffuseWeight;
		float specularWeight;
		float transmissionWeight;
		float clearcoatWeight;
		getSamplingLobeWeights( wo, clearcoatWo, surf, heroWavelength, diffuseWeight, specularWeight, transmissionWeight, clearcoatWeight );

		float pdf[4];
		pdf[0] = diffuseWeight;
		pdf[1] = specularWeight;
		pdf[2] = transmissionWeight;
		pdf[3] = clearcoatWeight;

		float cdf[4];
		cdf[0] = pdf[0];
		cdf[1] = pdf[1] + cdf[0];
		cdf[2] = pdf[2] + cdf[1];
		cdf[3] = pdf[3] + cdf[2];

		if ( cdf[3] != 0.0 ) {

			float invMaxCdf = 1.0 / cdf[3];
			cdf[0] *= invMaxCdf;
			cdf[1] *= invMaxCdf;
			cdf[2] *= invMaxCdf;
			cdf[3] *= invMaxCdf;

		} else {

			cdf[0] = 1.0;
			cdf[1] = 0.0;
			cdf[2] = 0.0;
			cdf[3] = 0.0;

		}

                vec3 wi;
		vec3 clearcoatWi;
		bool sampledDelta = false;
		bool rejectedTransmission = false;
		bool sampledThinSheet = false;
		ThinSheetTransmissionSample thinSheetSample;

                float r = rand( 15 );
                if ( r < cdf[0] ) {

			wi = diffuseDirection( wo, surf );
			clearcoatWi = normalize( clearcoatInvBasis * normalize( normalBasis * wi ) );

                } else if ( r < cdf[1] ) {

                        wi = specularDirection( wo, surf );
                        clearcoatWi = normalize( clearcoatInvBasis * normalize( normalBasis * wi ) );
                        sampledDelta = bsdfBaseLobesAreDelta( surf );

		} else if ( r < cdf[2] ) {

			if ( surf.thinFilm ) {
				thinSheetSample = sampleThinSheetTransmission(
					worldWo, surf, heroWavelength, transmissionWeight
				);
				if ( thinSheetSample.valid ) {
					wi = normalize(
						invBasis * thinSheetSample.worldDirection
					);
					clearcoatWi = normalize(
						clearcoatInvBasis * thinSheetSample.worldDirection
					);
					sampledDelta = thinSheetSample.sampledDelta;
					sampledThinSheet = true;
				} else {
					clearcoatWi = vec3( 0.0, 0.0, 1.0 );
					rejectedTransmission = true;
				}
			} else {
				wi = cauchyDispersionEnabled( surf )
					? dispersionTransmissionDirection( wo, surf, heroWavelength )
					: transmissionDirection( wo, surf );
				if ( dot( wi, wi ) > 1e-16 ) {
					clearcoatWi = normalize(
						clearcoatInvBasis * normalize( normalBasis * wi )
					);
				} else {
					// TIR rejects the selected transmission sample.
					// Keep every returned field finite; zero density
					// and throughput make the placeholder direction inert.
					clearcoatWi = vec3( 0.0, 0.0, 1.0 );
					rejectedTransmission = true;
				}
				sampledDelta = bsdfBaseLobesAreDelta( surf );
			}

                } else {

                        clearcoatWi = clearcoatDirection( clearcoatWo, surf );
                        wi = normalize( invBasis * normalize( clearcoatNormalBasis * clearcoatWi ) );
                        sampledDelta = surf.filteredClearcoatRoughness <= 0.0;

		}

                ScatterRecord result;
                if ( rejectedTransmission ) {

			result.pdf = 0.0;
			result.pdfRev = 0.0;
			result.specularPdf = 0.0;
			result.throughput = vec3( 0.0 );
                        result.direction = normalize(
                                surf.normalBasis * vec3( 0.0, 0.0, 1.0 )
                        );
			result.sampledDelta = sampledDelta;
			result.sampledNonConnectable = false;
			result.sampledRoughness = 0.0;
			return result;

		}
		if ( sampledThinSheet ) {

			// This augmented event is evaluated only in the sampled latent-interface
			// measure. Arbitrary-direction BSDF queries deliberately report zero for
			// through-sheet directions, so preserve both exact joint densities here.
			result.pdf = thinSheetSample.pdfFwd;
			result.pdfRev = thinSheetSample.pdfRev;
			result.specularPdf = thinSheetSample.pdfFwd;
			result.throughput = pathThroughputFromRgb(
				thinSheetSample.throughput, heroWavelength
			);
			result.direction = thinSheetSample.worldDirection;
			result.sampledDelta = thinSheetSample.sampledDelta;
			result.sampledNonConnectable = ! thinSheetSample.sampledDelta;
			result.sampledRoughness = thinSheetSample.sampledRoughness;
			return result;

		}
		vec3 resultColor;
                if ( sampledDelta ) {

                        result.pdf = bsdfDeltaEvalLocal(
                                wo,
                                clearcoatWo,
                                wi,
                                clearcoatWi,
                                surf,
                                heroWavelength,
                                specularWeight,
                                transmissionWeight,
                                clearcoatWeight,
                                resultColor
                        );
                        result.specularPdf = result.pdf;

                } else {

                        result.pdf = bsdfEval(
                                wo,
                                clearcoatWo,
                                wi,
                                clearcoatWi,
                                surf,
                                heroWavelength,
                                diffuseWeight,
                                specularWeight,
                                transmissionWeight,
                                clearcoatWeight,
                                result.specularPdf,
                                resultColor
                        );

                }
		result.throughput = pathThroughputFromRgb( resultColor, heroWavelength );
		result.direction = normalize( surf.normalBasis * wi );
		result.sampledDelta = sampledDelta;
		bool reverseDeltaMeasure;
		result.pdfRev = bsdfPdfResult(
			result.direction, worldWo, surf, heroWavelength,
			reverseDeltaMeasure
		);
		if (
			! ( result.pdfRev >= 0.0 ) ||
			isnan( result.pdfRev ) || isinf( result.pdfRev )
		) result.pdfRev = 0.0;
		result.sampledNonConnectable = false;
		result.sampledRoughness = 0.0;
		return result;

	}

`;

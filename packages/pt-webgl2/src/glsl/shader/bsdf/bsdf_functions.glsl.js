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

	// Sprint 7: TRANSLUCENT material flag bit for SSS single-scatter path.
	// This is packed in MaterialsTexture sample 14 and unpacked into material.flags
	// by material_struct.glsl.js::readMaterialInfo().
	const uint TRANSLUCENT_BIT = 0x10u;  // bit 4

	// RFE-03 / Sprint 14 — defined before bsdfEval since GLSL parses top-down and
	// some drivers don't accept forward-declarations whose params reference user structs.
	vec3 pathThroughputFromRgb( vec3 rgb, float heroWavelength ) {
		if ( uSpectralRendering == 0 ) return max( rgb, vec3( 0.0 ) );
		return vec3( heroScalarFromRgb( rgb, heroWavelength ) );
	}

	vec3 activeLayerThroughput( const in SurfaceRecord surf, float heroWavelength ) {
		if ( ! surf.hasActiveLayer ) return vec3( 1.0 );
		return pathThroughputFromRgb( surf.activeLayerTransmission, heroWavelength );
	}

	float sampleExponentialDistance( float xi, float sigmaT, float maxDistance ) {
		if ( sigmaT <= 0.0 ) return maxDistance;
		float u = max( 1.0 - clamp( xi, 0.0, 0.999999 ), 1e-6 );
		return min( - log( u ) / sigmaT, maxDistance );
	}

	float hg_phase( float cosTheta, float g ) {
		float gg = clamp( g, -0.9999, 0.9999 );
		float g2 = gg * gg;
		float denom = 1.0 + g2 - 2.0 * gg * clamp( cosTheta, -1.0, 1.0 );
		return ( 1.0 - g2 ) / ( 4.0 * PI * denom * sqrt( denom ) );
	}

        vec3 sampleHG_glsl( float u1, float u2, float g, vec3 forward ) {
                float gg = clamp( g, -0.9999, 0.9999 );
                float cosTheta;
                float a = 1.0 - 2.0 * u2;
                if ( gg == 0.0 ) {
                        cosTheta = a;
                } else if ( abs( gg ) < 1e-3 ) {
                        // Stable expansion of the exact inversion through g².
                        // Unlike the old isotropic cutoff, every nonzero
                        // authored asymmetry still changes the sample.
                        float a2 = a * a;
                        cosTheta = a + 1.5 * gg * ( 1.0 - a2 )
                                + 2.0 * gg * gg * ( a * a2 - a );
                } else {
                        // Using 1-u preserves the exact-zero branch's sample
                        // orientation while leaving the HG distribution
                        // unchanged because the variate remains uniform.
                        float xi = 1.0 - u2;
                        float sqrtTerm = ( 1.0 - gg * gg ) / ( 1.0 - gg + 2.0 * gg * xi );
                        cosTheta = ( 1.0 + gg * gg - sqrtTerm * sqrtTerm ) / ( 2.0 * gg );
		}
		cosTheta = clamp( cosTheta, -1.0, 1.0 );
		float sinTheta = sqrt( max( 0.0, 1.0 - cosTheta * cosTheta ) );
		float phi = 2.0 * PI * u1;
		vec3 localDir = vec3( sinTheta * cos( phi ), sinTheta * sin( phi ), cosTheta );
		return normalize( getBasisFromNormal( normalize( forward ) ) * localDir );
	}

	// diffuse
	float diffuseEval( vec3 wo, vec3 wi, vec3 wh, const in SurfaceRecord surf, inout vec3 color ) {

		// https://schuttejoe.github.io/post/disneybsdf/
		float fl = schlickFresnel( wi.z, 0.0 );
		float fv = schlickFresnel( wo.z, 0.0 );

		float metalFactor = ( 1.0 - surf.metalness );
		float transFactor = ( 1.0 - surf.transmission );
		float rr = 0.5 + 2.0 * surf.roughness * fl * fl;
		float retro = rr * ( fl + fv + fl * fv * ( rr - 1.0f ) );
		float lambert = ( 1.0f - 0.5f * fl ) * ( 1.0f - 0.5f * fv );

		// Subsurface scattering is handled by the dedicated sssSample() path (Sprint 7),
		// gated by surf.sssSigmaT > 0 and the TRANSLUCENT_BIT. A diffuse sub-surface
		// approximation inside diffuseEval would double-count with that path and is not
		// needed; this note is resolved.
		float F = disneyFresnel( wo, wi, wh, surf.f0, surf.eta, surf.metalness );
		color = ( 1.0 - F ) * transFactor * metalFactor * wi.z * surf.color * ( retro + lambert ) / PI;

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

	// ── Sprint 12: Cauchy IOR at arbitrary wavelength ────────────────────────────
	//
	// cauchyIORatLambda: evaluate IOR at a given wavelength using the three-term Cauchy formula.
	// GLSL mirror of @vitrum/shared-samplers/src/cauchyIor.ts::cauchyIOR.
	//
	// Parameters: lambdaNm in nm; A, B, C in µm units (Sprint 12 coefficient form).
	//   n(λ) = A + B/λ² + C/λ⁴    (λ in µm)
	//
	// This function is the Sprint 12 replacement for Sprint 8's per-channel Cauchy approach.
	// It is called at the hero wavelength sampled from sampleHeroWavelengthMIS in the main loop.
	//
	// New uniforms: iorCauchyA, iorCauchyB, iorCauchyC (see PhysicalPathTracingMaterial.js).
	// Legacy Sprint 8 scalar dispersion uniforms were removed; per-material
	// dispersion now flows through surf.dispersionStrength.
	//
	float cauchyIORatLambda( float lambdaNm, float A, float B, float C ) {
		float lambdaUm = lambdaNm * 0.001;  // nm → µm
		float lam2 = lambdaUm * lambdaUm;
		float lam4 = lam2 * lam2;
		// Exact-zero fast path: positive finite Cauchy support must not disappear.
		if ( C == 0.0 ) return A + B / lam2;
		return A + B / lam2 + C / lam4;
	}

	bool cauchyDispersionEnabled( const in SurfaceRecord surf ) {

		return surf.dispersionStrength > 0.0 && ( abs( iorCauchyB ) > 0.0 || abs( iorCauchyC ) > 0.0 );

	}

	float surfaceIorAtHero( const in SurfaceRecord surf, float heroWavelength ) {

		if ( ! cauchyDispersionEnabled( surf ) ) {

			return surf.ior;

		}

		float iorAtHero = cauchyIORatLambda( heroWavelength, iorCauchyA, iorCauchyB, iorCauchyC );
		float iorDelta = iorAtHero - iorCauchyA;
		float dispersionBasis = max( abs( iorCauchyB ), abs( iorCauchyC ) );
		if ( ! ( dispersionBasis > 0.0 ) || isnan( dispersionBasis ) || isinf( dispersionBasis ) ) return surf.ior;
		float dispersionScale = surf.dispersionStrength / dispersionBasis;
		dispersionScale = clamp( dispersionScale, 0.0, 4.0 );
		return max( 1.0, surf.ior + iorDelta * dispersionScale );

	}

	float transmissionEtaAtHero( const in SurfaceRecord surf, float heroWavelength ) {

		float ior = surfaceIorAtHero( surf, heroWavelength );
		return surf.frontFace ? 1.0 / ior : ior;

	}

	// specular
	float specularEval( vec3 wo, vec3 wi, vec3 wh, const in SurfaceRecord surf, float heroWavelength, inout vec3 color ) {

		// if roughness is set to 0 then D === NaN which results in black pixels
		float metalness = surf.metalness;
		float roughness = surf.filteredRoughness;

		float eta = surf.eta;
		float f0 = surf.f0;

		vec3 f0Color = mix( f0 * surf.specularColor * surf.specularIntensity, surf.color, surf.metalness );
		vec3 f90Color = vec3( mix( surf.specularIntensity, 1.0, surf.metalness ) );
		vec3 F = evaluateFresnel( dot( wo, wh ), eta, f0Color, f90Color );

		// Skip iridescence Fresnel computation when lobeMask bit 4 is clear
		// (iridescence == 0) or a future shader-internal lite policy opts out.
		if ( ( surf.lobeMask & 16u ) != 0u && ! surf.liteMode ) {
			vec3 iridescenceF = evalIridescence( 1.0, surf.iridescenceIor, dot( wi, wh ), surf.iridescenceThickness, f0Color );
			F = mix( F, iridescenceF, surf.iridescence );
		}
		if ( surf.thinFilmEnabled > 0.5 && surf.thinFilmLayerCount > 0.5 ) {
			float viewCos = surf.thinFilmAngleDependent ? abs( wo.z ) : 1.0;
			vec2 thinFilmRt = thinFilmTMM(
				surf.materialIndex,
				int( surf.thinFilmLayerCount + 0.5 ),
				heroWavelength,
				max( surf.ior, 1.0 ),
				surf.thinFilmIncidentIor,
				viewCos
			);
			F = clamp( F + ( vec3( 1.0 ) - F ) * thinFilmRt.x, vec3( 0.0 ), vec3( 1.0 ) );
		}

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
		// Skipped only when a future shader-internal lite policy opts out.
		if ( ! surf.liteMode ) {
			vec3 Favg = f0Color + ( vec3( 1.0 ) - f0Color ) * ( 1.0 / 21.0 );
			color += ggxMultiscatter( roughness, abs( wo.z ), abs( wi.z ), Favg );
		}

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

		float eta = transmissionEtaAtHero( surf, heroWavelength );
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

		float F = dielectricFresnel( abs( woDotH ), eta );
		float D = ggxDistributionForSurface( wh, surf );
		float G = ggxShadowMaskG2ForSurface( wi, wo, surf );
		// Walter et al. rough-dielectric BTDF in radiance-transport mode,
		// multiplied by |n·wi| because ScatterRecord.throughput stores f*cos.
		// eta^2 is 1/(eta_t/eta_i)^2 for this shader's eta convention.
		float btdfCos =
			( 1.0 - F ) * D * G * abs( wiDotH * woDotH ) * eta * eta /
			max( abs( wo.z ) * denom, 1e-12 );
		color = surf.transmission * surf.color * btdfCos;
		if ( surf.thinFilmEnabled > 0.5 && surf.thinFilmLayerCount > 0.5 ) {
			float viewCos = surf.thinFilmAngleDependent ? abs( wo.z ) : 1.0;
			vec2 thinFilmRt = thinFilmTMM(
				surf.materialIndex,
				int( surf.thinFilmLayerCount + 0.5 ),
				heroWavelength,
				max( surf.ior, 1.0 ),
				surf.thinFilmIncidentIor,
				viewCos
			);
			color *= thinFilmRt.y;
		}

		return pdfWi;

	}

        vec3 transmissionDirection( vec3 wo, const in SurfaceRecord surf ) {

		float eta = surf.eta;
                vec3 halfVector = ggxDirectionForSurface( wo, surf, rand2( 13 ) );
                vec3 lightDirection = refract( normalize( - wo ), halfVector, eta );

                // GLSL refract returns exactly zero for total internal reflection.
                // Never feed that sentinel to normalize, including the virtual
                // second boundary of a thin sheet.
                if ( ! ( dot( lightDirection, lightDirection ) > 1e-16 ) ) {
                        return vec3( 0.0 );
                }

                if ( surf.thinFilm ) {

                        vec3 exitDirection = refract(
                                normalize( - lightDirection ),
                                - vec3( 0.0, 0.0, 1.0 ),
                                1.0 / eta
                        );
                        if ( ! ( dot( exitDirection, exitDirection ) > 1e-16 ) ) {
                                return vec3( 0.0 );
                        }
                        lightDirection = - exitDirection;

		}

		return normalize( lightDirection );

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

	vec3 mediumAlbedoThroughput( vec3 rgb, float heroWavelength ) {
		if ( uSpectralRendering == 0 ) return max( rgb, vec3( 0.0 ) );
		// RGB scattering is an explicitly approximate contract row. Project the
		// actual per-material σ_s/σ_t albedo to the sampled hero wavelength;
		// a single scene-global sigmoid spectrum cannot represent more than one
		// participating material and previously remained at an unreachable flat
		// default. Surface reflectance continues to use genuine per-material
		// Jakob-Hanika coefficients through evalSpectrum().
		return vec3( heroScalarFromRgb( rgb, heroWavelength ) );
	}

        vec3 attenuationSigmaA( vec3 attColor, float attDist ) {
                if ( attDist <= 0.0 || isinf( attDist ) ) return vec3( 0.0 );
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
                if ( dist <= 0.0 ) return vec3( 1.0 );
                return exp(
                        - fogTrueExtinction(
                                materialsTex, fog, heroWavelength
                        ) * dist
                );
        }

        // The ray-distance proposal is sampled from fog.opacity, the packed
        // scalar majorant.  This ratio converts that proposal's survival into
        // the authored RGB / hero-wavelength Beer law without changing the
        // proposal PDFs used by BDPT MIS.
        vec3 fogFreeFlightRatioWeight(
                sampler2D materialsTex,
                const in FogMaterial fog,
                float dist,
                float heroWavelength
        ) {
                if ( dist <= 0.0 || fog.opacity <= 0.0 ) return vec3( 1.0 );
                vec3 sigmaT = fogTrueExtinction(
                        materialsTex, fog, heroWavelength
                );
                return exp( ( vec3( fog.opacity ) - sigmaT ) * dist );
        }

        float mediumPhasePdf( vec3 worldWo, vec3 worldWi, float g ) {
                float cosTheta = clamp(
                        dot( - normalize( worldWo ), normalize( worldWi ) ),
                        -1.0,
                        1.0
                );
                float g2 = g * g;
                float denominator = pow( 1.0 + g2 - 2.0 * g * cosTheta, 1.5 );
                return ( 1.0 - g2 ) / ( 4.0 * PI * denominator );
        }

        vec3 sampleMediumPhase( vec3 worldWo, float g, vec2 uv ) {
                float cosTheta;
                float a = 1.0 - 2.0 * uv.x;
                if ( g == 0.0 ) {
                        cosTheta = a;
                } else if ( abs( g ) < 1e-3 ) {
                        // Stable expansion of the exact inversion through g².
                        // Every nonzero authored asymmetry remains observable;
                        // there is no isotropic epsilon cutoff.
                        float a2 = a * a;
                        cosTheta = a + 1.5 * g * ( 1.0 - a2 )
                                + 2.0 * g * g * ( a * a2 - a );
                } else {
                        // Complementing the uniform variate keeps the finite-g
                        // inversion continuous with the exact-zero orientation.
                        float xi = 1.0 - uv.x;
                        float ratio = ( 1.0 - g * g ) / ( 1.0 - g + 2.0 * g * xi );
                        cosTheta = ( 1.0 + g * g - ratio * ratio ) / ( 2.0 * g );
                }
                cosTheta = clamp( cosTheta, -1.0, 1.0 );
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

	// Sprint 12: dielectric transmission with hero-wavelength Cauchy IOR.
	// Uses global Cauchy coefficients (iorCauchyA/B/C) with per-material base IOR
	// preserved by applying only the spectral delta from iorCauchyA.
        vec3 dispersionTransmissionDirection( vec3 wo, const in SurfaceRecord surf, float heroWavelength ) {

		float eta = transmissionEtaAtHero( surf, heroWavelength );

                vec3 halfVector = ggxDirectionForSurface( wo, surf, rand2( 13 ) );
                vec3 lightDirection = refract( normalize( - wo ), halfVector, eta );

                if ( ! ( dot( lightDirection, lightDirection ) > 1e-16 ) ) {
                        return vec3( 0.0 );
                }

                if ( surf.thinFilm ) {
                        vec3 exitDirection = refract(
                                normalize( - lightDirection ),
                                - vec3( 0.0, 0.0, 1.0 ),
                                1.0 / eta
                        );
                        if ( ! ( dot( exitDirection, exitDirection ) > 1e-16 ) ) {
                                return vec3( 0.0 );
                        }
                        lightDirection = - exitDirection;
                }

		return normalize( lightDirection );

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
		float eta = transmissionEtaAtHero( surf, heroWavelength );
		float f0 = iorRatioToF0( eta );
		float fEstimate = disneyFresnel( wo, wi, wh, f0, eta, surf.metalness );

		float transSpecularProb = mix( max( 0.25, fEstimate ), 1.0, metalness );
		float diffSpecularProb = 0.5 + 0.5 * metalness;

		diffuseWeight = ( 1.0 - transmission ) * ( 1.0 - diffSpecularProb );
		specularWeight = transmission * transSpecularProb + ( 1.0 - transmission ) * diffSpecularProb;
		transmissionWeight = transmission * ( 1.0 - transSpecularProb );
		clearcoatWeight = surf.clearcoat * schlickFresnel( clearcoatWo.z, 0.04 );

		float totalWeight = diffuseWeight + specularWeight + transmissionWeight + clearcoatWeight;
		diffuseWeight /= totalWeight;
		specularWeight /= totalWeight;
		transmissionWeight /= totalWeight;
		clearcoatWeight /= totalWeight;
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

                float eta = transmissionEtaAtHero( surf, heroWavelength );
                vec3 direction = refract(
                        normalize( - wo ),
                        vec3( 0.0, 0.0, 1.0 ),
                        eta
                );
                if ( ! ( dot( direction, direction ) > 1e-16 ) ) {
                        return vec3( 0.0 );
                }
                if ( surf.thinFilm ) {

                        vec3 exitDirection = refract(
                                normalize( - direction ),
                                - vec3( 0.0, 0.0, 1.0 ),
                                1.0 / eta
                        );
                        if ( ! ( dot( exitDirection, exitDirection ) > 1e-16 ) ) {
                                return vec3( 0.0 );
                        }
                        direction = - exitDirection;

                }
                return normalize( direction );

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
                        wi.z < 0.0 &&
                        ! bsdfBaseLobesAreDelta( surf )
                ) {

                        float eta = transmissionEtaAtHero( surf, heroWavelength );
                        vec3 wh = getHalfVector( wi, wo, eta );
                        float wiDotH = dot( wi, wh );
                        float woDotH = dot( wo, wh );
                        float sqrtDenom = wiDotH + eta * woDotH;
                        float denom = sqrtDenom * sqrtDenom;
                        if ( wiDotH * woDotH < 0.0 && denom > 1e-12 ) {

                                tpdf =
                                        ggxPdfForSurface( wo, wh, surf ) *
                                        abs( wiDotH ) /
                                        denom;

                        }

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

                mat3 normalInvBasis = transpose( surf.normalBasis );
                vec3 wo = normalize( normalInvBasis * worldWo );
                vec3 wi = normalize( normalInvBasis * worldWi );
                mat3 clearcoatInvBasis = transpose( surf.clearcoatBasis );
                vec3 clearcoatWo = normalize( clearcoatInvBasis * worldWo );
                vec3 clearcoatWi = normalize( clearcoatInvBasis * worldWi );

                float diffuseWeight;
                float specularWeight;
                float transmissionWeight;
                float clearcoatWeight;
                getSamplingLobeWeights(
                        wo,
                        clearcoatWo,
                        surf,
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
                        surf,
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

                                vec3 f0Color = mix(
                                        surf.f0 * surf.specularColor * surf.specularIntensity,
                                        surf.color,
                                        surf.metalness
                                );
                                vec3 f90Color = vec3(
                                        mix( surf.specularIntensity, 1.0, surf.metalness )
                                );
                                vec3 F = evaluateFresnel(
                                        abs( wo.z ),
                                        surf.eta,
                                        f0Color,
                                        f90Color
                                );
                                if ( ( surf.lobeMask & 16u ) != 0u && ! surf.liteMode ) {

                                        vec3 iridescenceF = evalIridescence(
                                                1.0,
                                                surf.iridescenceIor,
                                                abs( wi.z ),
                                                surf.iridescenceThickness,
                                                f0Color
                                        );
                                        F = mix( F, iridescenceF, surf.iridescence );

                                }
                                if (
                                        surf.thinFilmEnabled > 0.5 &&
                                        surf.thinFilmLayerCount > 0.5
                                ) {

                                        float viewCos =
                                                surf.thinFilmAngleDependent
                                                        ? abs( wo.z )
                                                        : 1.0;
                                        vec2 thinFilmRt = thinFilmTMM(
                                                surf.materialIndex,
                                                int( surf.thinFilmLayerCount + 0.5 ),
                                                heroWavelength,
                                                max( surf.ior, 1.0 ),
                                                surf.thinFilmIncidentIor,
                                                viewCos
                                        );
                                        F = clamp(
                                                F + ( vec3( 1.0 ) - F ) * thinFilmRt.x,
                                                vec3( 0.0 ),
                                                vec3( 1.0 )
                                        );

                                }
                                baseDelta += F;

                        }

                        vec3 transmitted =
                                bsdfDeltaTransmissionDirection( wo, surf, heroWavelength );
                        if (
                                transmissionWeight > 0.0 &&
                                wi.z < 0.0 &&
                                length( transmitted ) > 1e-8 &&
                                bsdfDeltaDirectionMatches( transmitted, wi )
                        ) {

                                float eta = transmissionEtaAtHero( surf, heroWavelength );
                                float F = dielectricFresnel( abs( wo.z ), eta );
                                vec3 transmissionColor =
                                        surf.transmission *
                                        surf.color *
                                        ( 1.0 - F ) *
                                        eta * eta;
                                if (
                                        surf.thinFilmEnabled > 0.5 &&
                                        surf.thinFilmLayerCount > 0.5
                                ) {

                                        float viewCos =
                                                surf.thinFilmAngleDependent
                                                        ? abs( wo.z )
                                                        : 1.0;
                                        vec2 thinFilmRt = thinFilmTMM(
                                                surf.materialIndex,
                                                int( surf.thinFilmLayerCount + 0.5 ),
                                                heroWavelength,
                                                max( surf.ior, 1.0 ),
                                                surf.thinFilmIncidentIor,
                                                viewCos
                                        );
                                        transmissionColor *= thinFilmRt.y;

                                }
                                baseDelta += transmissionColor;

                        }

                }

                vec3 clearcoatDelta = vec3( 0.0 );
                float clearcoatF = 0.0;
                if ( ( surf.lobeMask & 8u ) != 0u && ! surf.liteMode ) {

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

                if ( ( surf.lobeMask & 4u ) != 0u && ! surf.liteMode ) {

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

			dpdf = diffuseEval( wo, wi, halfVector, surf, color );
			color *= 1.0 - surf.transmission;

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
                        wi.z < 0.0 &&
                        ! bsdfBaseLobesAreDelta( surf )
                ) {

			vec3 transmissionHalfVector = getHalfVector( wi, wo, transmissionEtaAtHero( surf, heroWavelength ) );
			vec3 transmissionColor;
			tpdf = transmissionEval( wo, wi, transmissionHalfVector, surf, heroWavelength, transmissionColor );
			color += transmissionColor;

		}

		if ( ( surf.lobeMask & 4u ) != 0u && ! surf.liteMode ) {
			color *= mix( 1.0, sheenAlbedoScaling( wo, wi, surf ), surf.sheen );
			color += sheenColor( wo, wi, halfVector, surf ) * surf.sheen;
		}

                if (
                        ( surf.lobeMask & 8u ) != 0u &&
                        ! surf.liteMode &&
                        clearcoatWi.z >= 0.0 &&
                        clearcoatWeight > 0.0 &&
                        ! bsdfClearcoatLobeIsDelta( surf )
                ) {

			vec3 clearcoatHalfVector = getHalfVector( clearcoatWo, clearcoatWi );
                        cpdf = clearcoatEval( clearcoatWo, clearcoatWi, clearcoatHalfVector, surf, color );

                } else if (
                        ( surf.lobeMask & 8u ) != 0u &&
                        ! surf.liteMode &&
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

		mat3 normalInvBasis = transpose( surf.normalBasis );
		vec3 wo = normalize( normalInvBasis * worldWo );
		vec3 wi = normalize( normalInvBasis * worldWi );

		mat3 clearcoatInvBasis = transpose( surf.clearcoatBasis );
		vec3 clearcoatWo = normalize( clearcoatInvBasis * worldWo );
		vec3 clearcoatWi = normalize( clearcoatInvBasis * worldWi );

		float diffuseWeight;
		float specularWeight;
		float transmissionWeight;
                float clearcoatWeight;
                getSamplingLobeWeights( wo, clearcoatWo, surf, heroWavelength, diffuseWeight, specularWeight, transmissionWeight, clearcoatWeight );

                float deltaPdf = bsdfDeltaEvalLocal(
                        wo,
                        clearcoatWo,
                        wi,
                        clearcoatWi,
                        surf,
                        heroWavelength,
                        specularWeight,
                        transmissionWeight,
                        clearcoatWeight,
                        color
                );
                if ( deltaPdf > 0.0 ) return deltaPdf;

                float specularPdf;
                return bsdfEval( wo, clearcoatWo, wi, clearcoatWi, surf, heroWavelength, diffuseWeight, specularWeight, transmissionWeight, clearcoatWeight, specularPdf, color );

	}

	// Sprint 7: SSS single scatter via HG phase function.
	// Called when a ray exits the back face of a TRANSLUCENT material
	// (gated by surf.sssSigmaT > 0 — see PhysicalPathTracingMaterial.js).
	// The scatter position is sampled from an exponential distribution along
	// the refracted direction; the scattered direction is sampled from HG.
	// Mirrors @vitrum/shared-samplers/src/hgPhase.ts::sampleHG.
	ScatterRecord sssSample( vec3 worldWo, const in SurfaceRecord surf, float heroWavelength ) {

		// Per-material SSS parameters come from the SurfaceRecord (packed from the
		// MaterialsTexture: material.sssSigmaT/sssAnisotropyG/sssSigmaS via
		// get_surface_record_function, surface_record_struct). The legacy global
		// legacy global SSS uniforms were NEVER assigned per-material — only their
		// constructor defaults (sigmaT=0, g=0, albedo=0.9)
		// — so reading them collapsed SSS to a degenerate Beer-Lambert=1 term with a
		// fixed albedo (the per-material magnitudes were packed + gated on but never
		// consumed). surf.* IS the per-material data and is already in scope here.
		vec3 sigmaS = max( surf.sssSigmaS, vec3( 0.0 ) );
		vec3 sigmaA = attenuationSigmaA( surf.attenuationColor, surf.attenuationDistance );
		vec3 sigmaT = max( sigmaA + sigmaS, vec3( 0.0 ) );
		float sigmaTMajorant = max( surf.sssSigmaT, max( sigmaT.x, max( sigmaT.y, sigmaT.z ) ) );
		float tScatter = sampleExponentialDistance( rand( 17 ), sigmaTMajorant, 1e6 );
		float beerLambert = exp( - sigmaTMajorant * tScatter );
		vec3 mediumAlbedo = vec3(
			sigmaT.x > 0.0 ? sigmaS.x / sigmaT.x : 0.0,
			sigmaT.y > 0.0 ? sigmaS.y / sigmaT.y : 0.0,
			sigmaT.z > 0.0 ? sigmaS.z / sigmaT.z : 0.0
		);

		vec3 rd = normalize( - worldWo ); // refracted direction approximation
		vec3 scatterDir = sampleHG_glsl( rand( 18 ), rand( 19 ), surf.sssAnisotropyG, rd );

                ScatterRecord sssRec;
                sssRec.pdf = hg_phase( dot( rd, scatterDir ), surf.sssAnisotropyG );
                sssRec.specularPdf = 0.0;
                sssRec.direction = scatterDir;
                sssRec.sampledDelta = false;
		// Medium single-scatter albedo is projected from this material's authored
		// RGB σ_s/σ_t to the hero wavelength. Beer-Lambert attenuation remains an
		// explicit scalar so units (reflectance × transmittance) are preserved.
		sssRec.throughput = mediumAlbedoThroughput( mediumAlbedo, heroWavelength ) * beerLambert;
		return sssRec;

	}

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
                        sampleRec.sampledDelta = false;
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

                float r = rand( 15 );
                if ( r <= cdf[0] ) {

			wi = diffuseDirection( wo, surf );
			clearcoatWi = normalize( clearcoatInvBasis * normalize( normalBasis * wi ) );

                } else if ( r <= cdf[1] ) {

                        wi = specularDirection( wo, surf );
                        clearcoatWi = normalize( clearcoatInvBasis * normalize( normalBasis * wi ) );
                        sampledDelta = bsdfBaseLobesAreDelta( surf );

                } else if ( r <= cdf[2] ) {

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

                } else {

                        clearcoatWi = clearcoatDirection( clearcoatWo, surf );
                        wi = normalize( invBasis * normalize( clearcoatNormalBasis * clearcoatWi ) );
                        sampledDelta = surf.filteredClearcoatRoughness <= 0.0;

		}

                ScatterRecord result;
                if ( rejectedTransmission ) {

                        result.pdf = 0.0;
                        result.specularPdf = 0.0;
                        result.throughput = vec3( 0.0 );
                        result.direction = normalize(
                                surf.normalBasis * vec3( 0.0, 0.0, 1.0 )
                        );
                        result.sampledDelta = sampledDelta;
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
                return result;

	}

`;

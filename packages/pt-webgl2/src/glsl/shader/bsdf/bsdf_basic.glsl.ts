/** Compact opaque Disney-base/GGX mixture for scene-proven basic materials. */
export const BSDF_BASIC_GLSL = /* glsl */ `
  vec3 pathThroughputFromRgb( vec3 rgb, float heroWavelength ) {
    if ( uSpectralRendering == 0 ) return max( rgb, vec3( 0.0 ) );
    return vec3( heroScalarFromRgb( rgb, heroWavelength ) );
  }

  vec3 activeLayerThroughput( SurfaceRecord surf, float heroWavelength ) {
    return vec3( 1.0 );
  }

  vec3 transmissionAttenuationThroughput(
    sampler2D materialsTex, float dist, vec3 attColor, float attDist,
    bool hasSpectral, uint materialIndex, float heroWavelength
  ) {
    return vec3( 1.0 );
  }

  vec3 attenuationSigmaA( vec3 attColor, float attDist ) {
    if ( attDist <= 0.0 || isinf( attDist ) ) return vec3( 0.0 );
    vec3 transmittance = min( attColor, vec3( 1.0 ) );
    return max( - log( transmittance ) / attDist, vec3( 0.0 ) );
  }

  vec3 fogTrueExtinction(
    sampler2D materialsTex, const in FogMaterial fog,
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
    float sigmaSHero = heroScalarFromRgb( fog.sigmaS, heroWavelength );
    return vec3( sigmaAHero + sigmaSHero );
  }

  vec3 fogSegmentTransmittance(
    sampler2D materialsTex, const in FogMaterial fog,
    float dist, float heroWavelength
  ) {
    return extinctionTransmittance(
      fogTrueExtinction( materialsTex, fog, heroWavelength ),
      dist
    );
  }

  float fogFreeFlightSampleDistance(
    sampler2D materialsTex, const in FogMaterial fog,
    float heroWavelength, vec2 u
  ) {
    vec3 sigmaT = fogTrueExtinction( materialsTex, fog, heroWavelength );
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
    if (
      isnan( sampledExtinction ) || sampledExtinction < 0.0 ||
      isnan( u.y ) || isinf( u.y ) || u.y < 0.0 || u.y > 1.0 ||
      u.y == 0.0
    ) return INFINITY;
    if ( isinf( sampledExtinction ) ) return 0.0;
    if ( sampledExtinction == 0.0 ) return INFINITY;
    if ( u.y == 1.0 ) return 0.0;
    return - log( u.y ) / sampledExtinction;
  }

  float fogExtinctionCollisionDensity( float extinction, float dist ) {
    if (
      isnan( extinction ) || extinction < 0.0 ||
      isnan( dist ) || dist < 0.0
    ) return 0.0;
    if ( dist == 0.0 && isinf( extinction ) ) return INFINITY;
    if ( isinf( extinction ) || isinf( dist ) ) return 0.0;
    return extinction * extinctionTransmittance( extinction, dist );
  }

  float fogProposalSurvival(
    sampler2D materialsTex, const in FogMaterial fog,
    float dist, float heroWavelength
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
    sampler2D materialsTex, const in FogMaterial fog,
    float dist, float heroWavelength
  ) {
    vec3 sigmaT = fogTrueExtinction( materialsTex, fog, heroWavelength );
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

  vec3 fogFreeFlightSurvivalWeight(
    sampler2D materialsTex, const in FogMaterial fog,
    float dist, float heroWavelength
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
    sampler2D materialsTex, const in FogMaterial fog,
    float dist, float heroWavelength
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

  float hg_phase( float cosTheta, float g ) {
    float gg = clamp( g, -0.999999, 0.999999 );
    float a = abs( gg );
    float clampedCos = clamp( cosTheta, -1.0, 1.0 );
    float alignedCos = gg >= 0.0 ? clampedCos : - clampedCos;
    float oneMinusA = 1.0 - a;
    float denominator =
      oneMinusA * oneMinusA +
      2.0 * a * ( 1.0 - alignedCos );
    return
      ( oneMinusA * ( 1.0 + a ) ) /
      ( 4.0 * PI * denominator * sqrt( denominator ) );
  }

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
      float ratio = ( 1.0 - gg * gg ) / ( 1.0 + gg * q );
      cosTheta = ( 1.0 + gg * gg - ratio * ratio ) / ( 2.0 * gg );
    }
    return clamp( cosTheta, -1.0, 1.0 );
  }

  float mediumPhasePdf( vec3 worldWo, vec3 worldWi, float g ) {
    float cosTheta = clamp(
      dot( - normalize( worldWo ), normalize( worldWi ) ), -1.0, 1.0
    );
    return hg_phase( cosTheta, g );
  }

  vec3 sampleMediumPhase( vec3 worldWo, float g, vec2 uv ) {
    float cosTheta = sampleHgCosTheta( uv.x, g );
    float sinTheta = sqrt( max( 1.0 - cosTheta * cosTheta, 0.0 ) );
    float phi = 2.0 * PI * uv.y;
    vec3 localDirection = vec3(
      sinTheta * cos( phi ), sinTheta * sin( phi ), cosTheta
    );
    return normalize(
      getBasisFromNormal( - normalize( worldWo ) ) * localDirection
    );
  }

  void basicLobeWeights(
    SurfaceRecord surf, out float diffuseWeight, out float specularWeight
  ) {
    specularWeight = 0.5 + 0.5 * surf.metalness;
    diffuseWeight = 1.0 - specularWeight;
    diffuseWeight = representedBernoulliProbabilityF32( diffuseWeight );
    specularWeight = 1.0 - diffuseWeight;
  }

  bool basicBaseLobeIsDelta( SurfaceRecord surf ) {
    return surf.filteredRoughness <= 0.0;
  }

  bool basicDeltaDirectionMatches( vec3 sampled, vec3 candidate ) {
    return dot( normalize( sampled ), normalize( candidate ) ) >= 1.0 - 2e-6;
  }

  float basicDeltaPdfLocal(
    vec3 wo, vec3 wi, SurfaceRecord surf, float specularWeight,
    out bool deltaMeasure
  ) {
    deltaMeasure = false;
    if ( ! basicBaseLobeIsDelta( surf ) || ! ( specularWeight > 0.0 ) ) {
      return 0.0;
    }
    vec3 reflected = - reflect( wo, vec3( 0.0, 0.0, 1.0 ) );
    if ( wi.z > 0.0 && basicDeltaDirectionMatches( reflected, wi ) ) {
      deltaMeasure = true;
      return specularWeight;
    }
    return 0.0;
  }

  float basicDeltaEvalLocal(
    vec3 wo, vec3 wi, SurfaceRecord surf, float specularWeight,
    out vec3 color
  ) {
    bool deltaMeasure;
    float pdf = basicDeltaPdfLocal(
      wo, wi, surf, specularWeight, deltaMeasure
    );
    color = vec3( 0.0 );
    if ( ! deltaMeasure ) return 0.0;
    vec3 f0Color = mix( vec3( surf.f0 ), surf.color, surf.metalness );
    color = evaluateFresnel(
      abs( wo.z ), surf.eta, f0Color, vec3( 1.0 )
    );
    return pdf;
  }

  float basicBsdfEval(
    vec3 wo, vec3 wi, SurfaceRecord surf,
    float diffuseWeight, float specularWeight,
    out float specularPdf, out vec3 color
  ) {
    color = vec3( 0.0 );
    specularPdf = 0.0;
    if ( wo.z <= 0.0 || wi.z <= 0.0 ) return 0.0;

    float roughness = clamp( surf.filteredRoughness, 0.0, 1.0 );
    vec3 halfVector = normalize( wo + wi );
    float diffusePdf = wi.z / PI;
    float fl = schlickFresnel( wi.z, 0.0 );
    float fv = schlickFresnel( wo.z, 0.0 );
    float rr = 0.5 + 2.0 * surf.roughness * fl * fl;
    float retro = rr * ( fl + fv + fl * fv * ( rr - 1.0 ) );
    float lambert = ( 1.0 - 0.5 * fl ) * ( 1.0 - 0.5 * fv );
    float diffuseFresnel = disneyFresnel(
      wo, wi, halfVector, surf.f0, surf.eta, surf.metalness
    );
    vec3 diffuse =
      ( 1.0 - diffuseFresnel ) * ( 1.0 - surf.metalness ) *
      wi.z * surf.color * ( retro + lambert ) / PI;

    vec3 f0Color = mix( vec3( surf.f0 ), surf.color, surf.metalness );
    vec3 specular = vec3( 0.0 );
    float reflectionPdf = 0.0;
    if ( roughness > 0.0 ) {
      vec3 fresnel = evaluateFresnel(
        dot( wo, halfVector ), surf.eta, f0Color, vec3( 1.0 )
      );
      float distribution = ggxDistribution( halfVector, roughness );
      float geometry = ggxShadowMaskG2( wi, wo, roughness );
      specular = fresnel * distribution * geometry / ( 4.0 * abs( wo.z ) );
      vec3 averageFresnel =
        f0Color + ( vec3( 1.0 ) - f0Color ) * ( 1.0 / 21.0 );
      specular += ggxMultiscatter(
        roughness, abs( wo.z ), abs( wi.z ), averageFresnel
      );
      float halfPdf = ggxPDF( wo, halfVector, roughness );
      float reflectionJacobian = 4.0 * abs( dot( wo, halfVector ) );
      if ( reflectionJacobian > 0.0 ) {
        reflectionPdf = halfPdf / reflectionJacobian;
      }
    }

    color = diffuse + specular;
    specularPdf = reflectionPdf * specularWeight;
    return diffusePdf * diffuseWeight + reflectionPdf * specularWeight;
  }

  float bsdfResult(
    vec3 worldWo, vec3 worldWi, SurfaceRecord surf,
    float heroWavelength, inout vec3 color
  ) {
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
    float diffuseWeight;
    float specularWeight;
    basicLobeWeights( surf, diffuseWeight, specularWeight );
    float deltaPdf = basicDeltaEvalLocal(
      wo, wi, surf, specularWeight, color
    );
    if ( deltaPdf > 0.0 ) return deltaPdf;
    float specularPdf;
    return basicBsdfEval(
      wo, wi, surf, diffuseWeight, specularWeight, specularPdf, color
    );
  }

  float bsdfPdfResult(
    vec3 worldWo, vec3 worldWi, SurfaceRecord surf,
    float heroWavelength, out bool deltaMeasure
  ) {
    if ( surf.volumeParticle ) {
      deltaMeasure = false;
      return mediumPhasePdf( worldWo, worldWi, surf.sssAnisotropyG );
    }
    vec3 wo = normalize( transpose( surf.normalBasis ) * worldWo );
    vec3 wi = normalize( transpose( surf.normalBasis ) * worldWi );
    float diffuseWeight;
    float specularWeight;
    basicLobeWeights( surf, diffuseWeight, specularWeight );
    float deltaPdf = basicDeltaPdfLocal(
      wo, wi, surf, specularWeight, deltaMeasure
    );
    if ( deltaMeasure ) return deltaPdf;
    vec3 ignoredColor;
    float ignoredSpecularPdf;
    return basicBsdfEval(
      wo, wi, surf, diffuseWeight, specularWeight,
      ignoredSpecularPdf, ignoredColor
    );
  }

  ScatterRecord bsdfSample(
    vec3 worldWo, SurfaceRecord surf, float heroWavelength
  ) {
    if ( surf.volumeParticle ) {
      ScatterRecord volumeResult;
      volumeResult.specularPdf = 0.0;
      volumeResult.direction = sampleMediumPhase(
        worldWo, surf.sssAnisotropyG, rand2( 16 )
      );
      volumeResult.pdf = mediumPhasePdf(
        worldWo, volumeResult.direction, surf.sssAnisotropyG
      );
      volumeResult.pdfRev = volumeResult.pdf;
      volumeResult.sampledDelta = false;
      volumeResult.sampledNonConnectable = false;
      volumeResult.sampledRoughness = 0.0;
      volumeResult.throughput = pathThroughputFromRgb(
        surf.color * volumeResult.pdf, heroWavelength
      );
      return volumeResult;
    }

    vec3 wo = normalize( transpose( surf.normalBasis ) * worldWo );
    float diffuseWeight;
    float specularWeight;
    basicLobeWeights( surf, diffuseWeight, specularWeight );
    vec3 wi;
    bool sampledDelta = false;
    if ( rand( 15 ) < diffuseWeight ) {
      vec3 sampled = sampleSphere( rand2( 11 ) );
      sampled.z += 1.0;
      wi = normalize( sampled );
    } else {
      if ( basicBaseLobeIsDelta( surf ) ) {
        wi = - reflect( wo, vec3( 0.0, 0.0, 1.0 ) );
        sampledDelta = true;
      } else {
        float roughness = clamp( surf.filteredRoughness, 0.0, 1.0 );
        vec3 halfVector = ggxDirection( wo, vec2( roughness ), rand2( 12 ) );
        wi = - reflect( wo, halfVector );
      }
    }

    ScatterRecord result;
    vec3 resultColor;
    if ( sampledDelta ) {
      result.pdf = basicDeltaEvalLocal(
        wo, wi, surf, specularWeight, resultColor
      );
      result.specularPdf = result.pdf;
    } else {
      result.pdf = basicBsdfEval(
        wo, wi, surf, diffuseWeight, specularWeight,
        result.specularPdf, resultColor
      );
    }
    result.direction = normalize( surf.normalBasis * wi );
    result.throughput = pathThroughputFromRgb( resultColor, heroWavelength );
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

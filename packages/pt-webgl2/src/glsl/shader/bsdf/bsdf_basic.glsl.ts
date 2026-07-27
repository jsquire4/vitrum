/** Compact opaque Disney-base/GGX mixture for scene-proven basic materials. */
export const BSDF_BASIC_GLSL = /* glsl */ `
  const uint TRANSLUCENT_BIT = 0x10u;

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
    if ( dist <= 0.0 ) return vec3( 1.0 );
    return exp(
      - fogTrueExtinction( materialsTex, fog, heroWavelength ) * dist
    );
  }

  vec3 fogFreeFlightRatioWeight(
    sampler2D materialsTex, const in FogMaterial fog,
    float dist, float heroWavelength
  ) {
    if ( dist <= 0.0 || fog.opacity <= 0.0 ) return vec3( 1.0 );
    vec3 sigmaT = fogTrueExtinction(
      materialsTex, fog, heroWavelength
    );
    return exp( ( vec3( fog.opacity ) - sigmaT ) * dist );
  }

  float mediumPhasePdf( vec3 worldWo, vec3 worldWi, float g ) {
    float cosTheta = clamp(
      dot( - normalize( worldWo ), normalize( worldWi ) ), -1.0, 1.0
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
      float a2 = a * a;
      cosTheta = a + 1.5 * g * ( 1.0 - a2 )
        + 2.0 * g * g * ( a * a2 - a );
    } else {
      float xi = 1.0 - uv.x;
      float ratio = ( 1.0 - g * g ) / ( 1.0 - g + 2.0 * g * xi );
      cosTheta = ( 1.0 + g * g - ratio * ratio ) / ( 2.0 * g );
    }
    cosTheta = clamp( cosTheta, -1.0, 1.0 );
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
      volumeResult.sampledDelta = false;
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
    return result;
  }

  ScatterRecord sssSample(
    vec3 worldWo, SurfaceRecord surf, float heroWavelength
  ) {
    return bsdfSample( worldWo, surf, heroWavelength );
  }
`;

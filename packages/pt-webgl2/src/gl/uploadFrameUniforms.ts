// uploadFrameUniforms — the per-draw uniform-upload body extracted from
// GlResources.drawAccumStep (T3-D / D11-6). Sets every individual GLSL uniform the
// copied fork trace shader reads (no FrameParams UBO — a UBO bind alone renders
// black on a real driver; verified). BEHAVIOR-PRESERVING: the exact same setter
// sequence, gates, and CMF-upload constants as the pre-extraction inline body.

import type { GlProgram } from './glProgram.js';
import type { FrameUniforms } from './glResources.js';
import {
  CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE,
  X_CMF_CDF, Y_CMF_CDF, Z_CMF_CDF,
  X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL,
} from '@vitrum/shared-samplers';

// The CIE 1931 CMF tables / CDFs / integrals the spectral path needs. The wavelength
// sampler evaluates the hero wavelength against these CIE 1931 tables/CDFs and
// reconstructs RGB via the integrals; without them every uniform is 0 →
// wavelengthPdf=0 → black. The CDFs are Float64Array in @vitrum/shared-samplers
// (length 82); GL needs Float32.
const CMF_X_F32 = Float32Array.from(CIE_X_TABLE);
const CMF_Y_F32 = Float32Array.from(CIE_Y_TABLE);
const CMF_Z_F32 = Float32Array.from(CIE_Z_TABLE);
const CMF_XCDF_F32 = Float32Array.from(X_CMF_CDF);
const CMF_YCDF_F32 = Float32Array.from(Y_CMF_CDF);
const CMF_ZCDF_F32 = Float32Array.from(Z_CMF_CDF);

/**
 * Upload the per-frame individual uniforms for one accumulation draw. `samples` is
 * the count already accumulated since the last clear (drives opacity 1/(N+1)).
 * The caller must have called `prog.use()`-equivalent binding beforehand — this
 * fn calls `prog.use()` itself (byte-identical to the inline body which did so).
 */
export function uploadFrameUniforms(
  prog: GlProgram,
  samples: number,
  seed: number,
  frame: FrameUniforms,
): void {
  prog.use();
  // The copied fork GLSL reads INDIVIDUAL uniforms (no FrameParams UBO).
  prog.setInt('seed', seed);
  prog.setFloat('opacity', 1 / (samples + 1));
  prog.setVec2('resolution', frame.resolution[0], frame.resolution[1]);
  prog.setInt('bounces', frame.bounces);
  prog.setInt('transmissiveBounces', frame.transmissiveBounces);
  prog.setFloat('filterGlossyFactor', frame.filterGlossyFactor);
  prog.setInt('materialLodDepth', frame.materialLodDepth);
  prog.setFloat('uRadianceClamp', frame.radianceClamp);
  prog.setMat4('cameraWorldMatrix', frame.cameraWorldMatrix);
  prog.setMat4('invProjectionMatrix', frame.invProjectionMatrix);
  prog.setFloat('environmentIntensity', frame.environmentIntensity);
  prog.setFloat('backgroundBlur', frame.backgroundBlur);
  // H3 FIX (2026-06-09): upload backgroundAlpha. Directly-visible background
  // (NO_HIT first ray) sets `pc_fragColor.a = backgroundAlpha`, then the running
  // average multiplies by `opacity`; in the 'normal' regime the SRC_ALPHA blend
  // weights the fragment by that alpha. Never uploaded → defaulted to 0 → the
  // background contributed `src*0 + dst*1` every frame and NEVER accumulated
  // (directly-visible sky/HDRI rendered black). 1 = opaque (accumulates like
  // geometry); <1 routes to the alpha-composite regime (see #regime).
  prog.setFloat('backgroundAlpha', frame.backgroundAlpha);
  prog.setMat4('environmentRotation', frame.environmentRotation);
  prog.setInt('uSpectralRendering', frame.spectralEnabled ? 1 : 0);
  if (frame.spectralEnabled) {
    // H2 FIX (2026-06-09): upload the CIE CMF tables + CDFs + integrals. Before
    // this, GlProgram.setFloatArray had ZERO callers, so uCmfX/Y/Z, the three
    // CDFs, and the integrals all defaulted to 0 → wavelengthPdf=0 →
    // wavelengthToRGB() returned vec3(0) → `spectral: true` rendered BLACK.
    // Constant data, cheap re-upload; gated so non-spectral frames skip it.
    // (u_jakobCoeffs / iorCauchy stay at their flat-spectrum / no-dispersion
    // defaults — those refine spectral reflectance colour, not black-vs-lit.)
    prog.setFloatArray('uCmfX', CMF_X_F32);
    prog.setFloatArray('uCmfY', CMF_Y_F32);
    prog.setFloatArray('uCmfZ', CMF_Z_F32);
    prog.setFloatArray('uXCmfCdf', CMF_XCDF_F32);
    prog.setFloatArray('uYCmfCdf', CMF_YCDF_F32);
    prog.setFloatArray('uZCmfCdf', CMF_ZCDF_F32);
    prog.setFloat('uXCmfIntegral', X_CMF_INTEGRAL);
    prog.setFloat('uYCmfIntegral', Y_CMF_INTEGRAL);
    prog.setFloat('uZCmfIntegral', Z_CMF_INTEGRAL);
  }
  prog.setInt('uCausticStrategy', frame.causticStrategy);
  prog.setFloat('uMneeMaxIterations', frame.mneeMaxIterations);
  prog.setFloat('uMneeMaxChainLength', frame.mneeMaxChainLength);
  // H2 follow-on: scene-global spectral dispersion + reflectance coefficients.
  // Default (0,0,0)/(0,0,0) keep the no-dispersion / flat-S≡½ no-op path, so a
  // non-dispersive spectral frame is unchanged. Set unconditionally (cheap scalar
  // uploads; gated to nothing-but-defaults when the host supplies no dispersion).
  prog.setVec3('u_jakobCoeffs', frame.jakobCoeffs[0], frame.jakobCoeffs[1], frame.jakobCoeffs[2]);
  prog.setFloat('iorCauchyA', frame.iorCauchy[0]);
  prog.setFloat('iorCauchyB', frame.iorCauchy[1]);
  prog.setFloat('iorCauchyC', frame.iorCauchy[2]);
  // Flag-plumbing audit (2026-06-10): upload the PhysicalCamera DoF uniforms when
  // dof is enabled. The FEATURE_DOF GLSL gate is compiled in only when opts.dof was
  // set (see #traceFeatures), so these setters are inactive no-ops otherwise — but
  // we still gate the upload to skip the work for the common pinhole path.
  if (frame.dof != null) {
    prog.setFloat('physicalCamera.focusDistance', frame.dof.focusDistance);
    prog.setFloat('physicalCamera.bokehSize', frame.dof.bokehSize);
    prog.setInt('physicalCamera.apertureBlades', frame.dof.apertureBlades);
    prog.setFloat('physicalCamera.apertureRotation', frame.dof.apertureRotation);
    prog.setFloat('physicalCamera.anamorphicRatio', frame.dof.anamorphicRatio);
  }
  // A5 — eye pass: light-subpath pass OFF, bind the built light-path texture, and
  // upload the bounce count the connection sweep iterates. `#bindSceneTextures`
  // binds a dummy for `uBdptLightPathTex` when bdpt is off (or the build failed);
  // here we override it with the real result so the connection pass reads vertices.
  if (frame.bdpt) {
    prog.setInt('uBdptLightSubpathPass', 0);
    prog.setInt('uBdptMaxLightBounces', frame.bdptMaxLightBounces);
  }
}

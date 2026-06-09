// The three accumulation blend regimes (plan/three-removal/02-gl-framework.md §3) —
// verbatim from the fork's PathTracingRenderer.js:18-53 renderTask, re-expressed as raw
// WebGL2 blend state (we have no THREE material `.blending`/`.opacity`).
//
// A WebGL2 path tracer accumulates by blending each new sample into a float render target.
// The regime is chosen by the host from caps + backgroundAlpha (WebGLPathTracer.js:470-473):
//   needsAlphaComposite = backgroundAlpha !== 1 || !EXT_float_blend.
//
//   'additive'        (Regime 1, FEATURE_ADDITIVE_ACCUM): buffer = SUM(rgb)/COUNT(alpha).
//                     Needs EXT_float_blend. Frag writes premultiplied sample, alpha=1/sample
//                     → alpha channel counts. Host clears to 0 first. (fork :31-41)
//   'alpha-composite' (Regime 2, !EXT_float_blend OR bgAlpha≠1): PT pass renders with NO blend
//                     into the primary target, then a BlendMaterial fullscreen quad composites
//                     into a ping-pong pair with opacity = 1/(samples+1). (fork :42-47, :156-165)
//   'normal'          (Regime 3, default w/ float-blend): SRC_ALPHA/ONE_MINUS_SRC_ALPHA running
//                     average; material.opacity = 1/(samples+1), frag alpha = 1. (fork :48-52)

import type { AccumRegime } from '../featureTypes.js';

/**
 * Set the GL blend state for one accumulation draw.
 *
 * `samples` is the count of samples ALREADY accumulated (used by the running-average regimes
 * to derive the opacity 1/(samples+1) — the caller still uploads that opacity as a uniform;
 * this function only configures the fixed-function blend stage).
 *
 * Regime 2's PT pass itself uses NO blend here (the running-average composite is a SEPARATE
 * BlendMaterial quad pass the resource owner drives afterwards into the ping-pong pair).
 */
export function setBlendForRegime(
  gl: WebGL2RenderingContext,
  regime: AccumRegime,
  samples: number,
): void {
  void samples; // opacity is uploaded as a uniform; the blend equation is sample-independent.
  switch (regime) {
    case 'additive':
      // SUM(rgb)/COUNT(alpha): straight ONE/ONE additive (needs EXT_float_blend).
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
      return;
    case 'alpha-composite':
      // PT pass renders unblended; the composite happens in the BlendMaterial quad pass.
      gl.disable(gl.BLEND);
      return;
    case 'normal':
      // Running average: lerp(dst, src, opacity) via SRC_ALPHA/ONE_MINUS_SRC_ALPHA.
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      return;
    default: {
      const _exhaustive: never = regime;
      throw new Error(`pt-webgl2: unknown accumulation regime '${String(_exhaustive)}'`);
    }
  }
}

// GL capability probe (plan/three-removal/02-gl-framework.md §1, §3, §4).
//
// The two extensions decide which accumulation regime the renderer runs:
//   - EXT_color_buffer_float — makes RGBA32F render targets color-renderable (the
//     fork's FloatType accumulation FBOs). Without it we cannot accumulate in float.
//   - EXT_float_blend        — allows GL blending into a 32-bit-float draw buffer
//     (the running-average Regime 3 + the additive Regime 1). Absent → Regime 2
//     (BlendMaterial ping-pong) is selected automatically (02 §3, WebGLPathTracer.js:470-473).
// The MAX_* limits feed sampler-unit / MRT / data-texture sizing decisions in
// GlResources + the scene packers.

export interface GlCaps {
  /** EXT_color_buffer_float present — RGBA32F targets are color-renderable. */
  readonly floatColorRenderable: boolean;
  /** EXT_float_blend present — GL blend works on float draw buffers (enables Regime 1/3). */
  readonly floatBlend: boolean;
  /** gl.MAX_DRAW_BUFFERS — MRT attachment budget (need ≥3 for gNormalDepth/gAlbedo). */
  readonly maxDrawBuffers: number;
  /** gl.MAX_TEXTURE_IMAGE_UNITS — sampler-unit budget for GlProgram link-time assignment. */
  readonly maxTexUnits: number;
  /** gl.MAX_TEXTURE_SIZE — square data-texture dimension ceiling. */
  readonly maxTexSize: number;
  /** gl.MAX_ARRAY_TEXTURE_LAYERS — layer-count ceiling for sampler2DArray textures. */
  readonly maxArrayLayers: number;
}

/**
 * Probe the live WebGL2 context for the capabilities the render framework needs.
 * Requesting an extension via `getExtension` both activates it and reports presence,
 * so we call it once here (the activation persists for the context lifetime).
 */
export function probeGlCaps(gl: WebGL2RenderingContext): GlCaps {
  return {
    floatColorRenderable: gl.getExtension('EXT_color_buffer_float') != null,
    floatBlend: gl.getExtension('EXT_float_blend') != null,
    maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS) as number,
    maxTexUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number,
    maxTexSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
    maxArrayLayers: gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number,
  };
}

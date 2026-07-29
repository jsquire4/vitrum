// GL capability probe (plan/three-removal/02-gl-framework.md §1, §3, §4).
//
// EXT_color_buffer_float makes RGBA32F render targets color-renderable (the
// fork's FloatType accumulation FBOs). Without it we cannot accumulate in float.
//
// Accumulation itself is deliberately shader-composited on every device. The
// trace pass writes an MRT whose auxiliary attachments must retain last-sample
// semantics, then the separately resolved NEE term is added to radiance before
// one running-mean update. WebGL2 fixed-function blending cannot express that
// per-attachment ownership, so EXT_float_blend is neither requested nor probed.
// The retained MAX_* limits feed MRT and data-texture sizing decisions in
// GlResources. Other resource-specific limits are queried where they are
// consumed so this object cannot imply unused enforcement.

export interface GlCaps {
  /** gl.MAX_DRAW_BUFFERS — MRT attachment budget (need ≥3 for gNormalDepth/gAlbedo). */
  readonly maxDrawBuffers: number;
  /** gl.MAX_TEXTURE_SIZE — square data-texture dimension ceiling. */
  readonly maxTexSize: number;
}

/**
 * Probe the live WebGL2 context for the capabilities the render framework needs.
 * Requesting an extension via `getExtension` both activates it and reports presence,
 * so we call it once here (the activation persists for the context lifetime).
 */
export function probeGlCaps(gl: WebGL2RenderingContext): GlCaps {
  // Activates float color attachments for the context. Individual framebuffer
  // creation checks completeness, so retaining an unconsumed boolean here would
  // duplicate no usable decision.
  gl.getExtension('EXT_color_buffer_float');
  return {
    maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS) as number,
    maxTexSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
  };
}

// Render-target + framebuffer helpers (plan/three-removal/02-gl-framework.md §4).
//
// Formats are VERBATIM from the fork's PathTracingRenderer.js:271-290 — every accumulation
// RT is RGBA32F (`format RGBAFormat, type FloatType`) with NEAREST min/mag filtering.
// The optional MRT extras (gNormalDepth @ COLOR_ATTACHMENT1, gAlbedo @ COLOR_ATTACHMENT2)
// mirror the fork's PhysicalPathTracingMaterial.js:244-245 MRT outputs; on a non-MRT device
// we attach only attachment 0 and drawBuffers([COLOR_ATTACHMENT0]) — locations 1/2 are
// "harmlessly ignored" by the shader then (fork comment, plan 02 §4).

/** Allocate one RGBA32F, NEAREST, CLAMP_TO_EDGE color texture sized w×h (no data upload). */
export function createColorTexture(gl: WebGL2RenderingContext, w: number, h: number): WebGLTexture {
  return createTexture(gl, w, h, gl.RGBA32F, gl.RGBA, gl.FLOAT);
}

/**
 * Allocate the present (tonemapped) target — DELIBERATELY RGBA32F, not RGBA8.
 *
 * The present texture is what `resultTexture()` returns and therefore what the
 * public `FrameRendered.primaryRadiance` hands to hosts. Hosts (and the
 * wsl-gpu GPU validation harnesses) read it with `readPixels(RGBA, FLOAT)`;
 * a UNORM8 attachment makes that read an INVALID_OPERATION that fails
 * SILENTLY (buffer stays zeroed, framebuffer still "complete") — an all-black
 * readback with no error. Display-referred data does not *need* float
 * precision, but the float READBACK CONTRACT does. Sweep note (2026-06-11):
 * D10.11 briefly switched this to RGBA8 and broke every external float
 * readback of primaryRadiance; reverted. Do not change this format without
 * migrating every primaryRadiance consumer to UNSIGNED_BYTE reads.
 */
export function createPresentTexture(gl: WebGL2RenderingContext, w: number, h: number): WebGLTexture {
  return createTexture(gl, w, h, gl.RGBA32F, gl.RGBA, gl.FLOAT);
}

/** Internal helper: allocate a 2D texture with the given internalFormat/format/type. */
function createTexture(
  gl: WebGL2RenderingContext, w: number, h: number,
  internalFormat: number, format: number, type: number,
): WebGLTexture {
  const tex = gl.createTexture();
  if (tex == null) throw new Error('pt-webgl2: failed to create render-target texture');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

/** A render target: an FBO with a primary color texture + optional MRT aux textures. */
export interface RenderTarget {
  readonly fbo: WebGLFramebuffer;
  /** COLOR_ATTACHMENT0 — the PT accumulation texture. */
  readonly color: WebGLTexture;
  /** COLOR_ATTACHMENT1 — packed normal+depth g-buffer (null when MRT disabled). */
  readonly normalDepth: WebGLTexture | null;
  /** COLOR_ATTACHMENT2 — albedo g-buffer (null when MRT disabled). */
  readonly albedo: WebGLTexture | null;
  /** The drawBuffers list this target was built with (1 entry, or 3 with MRT aux). */
  readonly drawBuffers: GLenum[];
  readonly width: number;
  readonly height: number;
  destroy(): void;
}

/**
 * Create an RGBA32F render target. When `withAux` is true and the device supports ≥3
 * draw buffers, also attach gNormalDepth/gAlbedo at COLOR_ATTACHMENT1/2 and set up a
 * 3-wide drawBuffers list; otherwise attach only attachment 0.
 */
export function createRenderTarget(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  withAux: boolean,
): RenderTarget {
  const fbo = gl.createFramebuffer();
  if (fbo == null) throw new Error('pt-webgl2: failed to create framebuffer');

  const color = createColorTexture(gl, w, h);
  let normalDepth: WebGLTexture | null = null;
  let albedo: WebGLTexture | null = null;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);

  const drawBuffers: GLenum[] = [gl.COLOR_ATTACHMENT0];
  if (withAux) {
    normalDepth = createColorTexture(gl, w, h);
    albedo = createColorTexture(gl, w, h);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, normalDepth, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, albedo, 0);
    drawBuffers.push(gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2);
  }
  gl.drawBuffers(drawBuffers);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  return {
    fbo,
    color,
    normalDepth,
    albedo,
    drawBuffers,
    width: w,
    height: h,
    destroy() {
      gl.deleteFramebuffer(fbo);
      gl.deleteTexture(color);
      if (normalDepth != null) gl.deleteTexture(normalDepth);
      if (albedo != null) gl.deleteTexture(albedo);
    },
  };
}

/** Bind a render target's FBO and re-declare its drawBuffers list (MRT-aware). */
export function bindRenderTarget(gl: WebGL2RenderingContext, target: RenderTarget): void {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
  gl.drawBuffers(target.drawBuffers);
}

/**
 * Clear a render target to (0,0,0,0) — verbatim PathTracingRenderer.js:400-433 reset().
 * Binds the FBO, sets a transparent-black clear color, and clears the color buffer.
 */
export function clearRenderTarget(gl: WebGL2RenderingContext, target: RenderTarget): void {
  bindRenderTarget(gl, target);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

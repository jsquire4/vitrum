// Render-target + framebuffer helpers (plan/three-removal/02-gl-framework.md §4).
//
// Progressive radiance and linear depth retain RGBA32F. Bounded albedo and the
// display-referred present target use RGBA16F. All targets use NEAREST sampling.
// The optional MRT extras (gNormalDepth @ COLOR_ATTACHMENT1, gAlbedo @
// COLOR_ATTACHMENT2) mirror the fork's PhysicalPathTracingMaterial.js outputs.

import { allocGlTexture } from './texAlloc.js';

export type RenderTargetFormat = 'rgba32f' | 'rgba16f';

/** Allocate one floating-point color texture sized w×h (no data upload). */
function createColorTexture(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  targetFormat: RenderTargetFormat = 'rgba32f',
  resourceName = 'render-target',
): WebGLTexture {
  return createTexture(
    gl,
    w,
    h,
    targetFormat === 'rgba16f' ? gl.RGBA16F : gl.RGBA32F,
    gl.RGBA,
    targetFormat === 'rgba16f' ? gl.HALF_FLOAT : gl.FLOAT,
    resourceName,
  );
}

/** Internal helper: allocate a 2D texture with the given internalFormat/format/type. */
function createTexture(
  gl: WebGL2RenderingContext, w: number, h: number,
  internalFormat: number, format: number, type: number,
  resourceName: string,
): WebGLTexture {
  return allocGlTexture(gl, {
    kind: 'rect',
    width: w,
    height: h,
    internalFormat,
    format,
    type,
    data: null,
    resourceName,
  });
}

function framebufferStatusName(gl: WebGL2RenderingContext, status: number): string {
  const entries: readonly (readonly [number, string])[] = [
    [gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT, 'FRAMEBUFFER_INCOMPLETE_ATTACHMENT'],
    [gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT, 'FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT'],
    [gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS, 'FRAMEBUFFER_INCOMPLETE_DIMENSIONS'],
    [gl.FRAMEBUFFER_UNSUPPORTED, 'FRAMEBUFFER_UNSUPPORTED'],
    [gl.FRAMEBUFFER_INCOMPLETE_MULTISAMPLE, 'FRAMEBUFFER_INCOMPLETE_MULTISAMPLE'],
  ];
  return entries.find(([value]) => value === status)?.[1] ?? `0x${status.toString(16)}`;
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

/** Four RGBA32F attachments used by the portable one-vertex NEE handoff. */
export interface NeeCandidateTarget {
  readonly fbo: WebGLFramebuffer;
  readonly textures: readonly [WebGLTexture, WebGLTexture, WebGLTexture, WebGLTexture];
  readonly drawBuffers: [GLenum, GLenum, GLenum, GLenum];
  readonly width: number;
  readonly height: number;
  destroy(): void;
}

/**
 * Two RGBA32F progressive-history colors sharing one full-tier auxiliary pair.
 *
 * Each framebuffer owns one history color, while both framebuffers attach the
 * same normal/depth and albedo textures. The renderer alternates the color
 * framebuffers so the trace shader can read the previous running mean and write
 * the next one without retaining a third full-resolution raw-sample texture.
 */
export interface ProgressiveTarget {
  readonly fbos: readonly [WebGLFramebuffer, WebGLFramebuffer];
  readonly colors: readonly [WebGLTexture, WebGLTexture];
  /** RGBA32F normal/depth: full range is retained for linear scene depth. */
  readonly normalDepth: WebGLTexture | null;
  /** RGBA16F albedo: bounded material reflectance does not require 32-bit storage. */
  readonly albedo: WebGLTexture | null;
  readonly drawBuffers: GLenum[];
  readonly width: number;
  readonly height: number;
  destroy(): void;
}

/**
 * Allocate the dedicated four-attachment NEE candidate target.
 *
 * WebGL2 requires at least four draw buffers/color attachments, but the check is
 * still explicit: a broken/virtualized context must fail before any partial
 * target is published. Every allocation and framebuffer failure retires all
 * resources created by this attempt.
 */
export function createNeeCandidateTarget(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
): NeeCandidateTarget {
  const maxDrawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS) as number;
  const maxColorAttachments = gl.getParameter(gl.MAX_COLOR_ATTACHMENTS) as number;
  if (maxDrawBuffers < 4 || maxColorAttachments < 4) {
    throw new Error(
      'pt-webgl2: the NEE resolve pipeline requires four RGBA32F draw buffers ' +
        `(device reports ${maxDrawBuffers} draw buffers and ${maxColorAttachments} color attachments)`,
    );
  }

  const fbo = gl.createFramebuffer();
  if (fbo == null) throw new Error('pt-webgl2: failed to create NEE candidate framebuffer');
  const textures: WebGLTexture[] = [];
  const drawBuffers: [GLenum, GLenum, GLenum, GLenum] = [
    gl.COLOR_ATTACHMENT0,
    gl.COLOR_ATTACHMENT1,
    gl.COLOR_ATTACHMENT2,
    gl.COLOR_ATTACHMENT3,
  ];
  try {
    for (let attachment = 0; attachment < 4; attachment += 1) {
      const texture = createColorTexture(gl, w, h);
      textures.push(texture);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    for (let attachment = 0; attachment < 4; attachment += 1) {
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        drawBuffers[attachment]!,
        gl.TEXTURE_2D,
        textures[attachment]!,
        0,
      );
    }
    gl.drawBuffers(drawBuffers);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(
        `pt-webgl2: NEE candidate framebuffer ${w}×${h} is incomplete ` +
          `(${framebufferStatusName(gl, status)})`,
      );
    }
  } catch (error) {
    for (const texture of textures) gl.deleteTexture(texture);
    gl.deleteFramebuffer(fbo);
    throw error;
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  if (textures.length !== 4) {
    for (const texture of textures) gl.deleteTexture(texture);
    gl.deleteFramebuffer(fbo);
    throw new Error('pt-webgl2: NEE candidate target allocation did not complete');
  }
  const completeTextures = textures as unknown as [
    WebGLTexture,
    WebGLTexture,
    WebGLTexture,
    WebGLTexture,
  ];
  let destroyed = false;
  return {
    fbo,
    textures: completeTextures,
    drawBuffers,
    width: w,
    height: h,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      gl.deleteFramebuffer(fbo);
      for (const texture of completeTextures) gl.deleteTexture(texture);
    },
  };
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
  targetFormat: RenderTargetFormat = 'rgba32f',
): RenderTarget {
  const fbo = gl.createFramebuffer();
  if (fbo == null) throw new Error('pt-webgl2: failed to create framebuffer');

  let color: WebGLTexture | null = null;
  let normalDepth: WebGLTexture | null = null;
  let albedo: WebGLTexture | null = null;
  const drawBuffers: GLenum[] = [gl.COLOR_ATTACHMENT0];
  try {
    color = createColorTexture(gl, w, h, targetFormat);
    if (withAux) {
      normalDepth = createColorTexture(gl, w, h, 'rgba32f', 'normal/depth render-target');
      albedo = createColorTexture(gl, w, h, 'rgba16f', 'albedo render-target');
      drawBuffers.push(gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
    if (normalDepth != null && albedo != null) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, normalDepth, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, albedo, 0);
    }
    gl.drawBuffers(drawBuffers);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(
        `pt-webgl2: framebuffer ${w}×${h} is incomplete (${framebufferStatusName(gl, status)})`,
      );
    }
  } catch (error) {
    if (color != null) gl.deleteTexture(color);
    if (normalDepth != null) gl.deleteTexture(normalDepth);
    if (albedo != null) gl.deleteTexture(albedo);
    gl.deleteFramebuffer(fbo);
    throw error;
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  const completeColor = color;
  if (completeColor == null) throw new Error('pt-webgl2: render-target color allocation did not complete');
  let destroyed = false;

  return {
    fbo,
    color: completeColor,
    normalDepth,
    albedo,
    drawBuffers,
    width: w,
    height: h,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      gl.deleteFramebuffer(fbo);
      gl.deleteTexture(completeColor);
      if (normalDepth != null) gl.deleteTexture(normalDepth);
      if (albedo != null) gl.deleteTexture(albedo);
    },
  };
}

/**
 * Allocate the compact progressive target transactionally.
 *
 * The two RGBA32F colors are the running-mean ping-pong pair. Both FBOs attach
 * the same auxiliary textures because auxiliaries are last-sample products and
 * are overwritten on every trace draw.
 */
export function createProgressiveTarget(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  withAux: boolean,
): ProgressiveTarget {
  const fbos: WebGLFramebuffer[] = [];
  const colors: WebGLTexture[] = [];
  let normalDepth: WebGLTexture | null = null;
  let albedo: WebGLTexture | null = null;
  const drawBuffers: GLenum[] = [gl.COLOR_ATTACHMENT0];
  try {
    for (let i = 0; i < 2; i += 1) {
      const fbo = gl.createFramebuffer();
      if (fbo == null) {
        throw new Error('pt-webgl2: failed to create progressive framebuffer');
      }
      fbos.push(fbo);
      colors.push(
        createColorTexture(gl, w, h, 'rgba32f', `progressive history ${i}`),
      );
    }
    if (withAux) {
      normalDepth = createColorTexture(
        gl,
        w,
        h,
        'rgba32f',
        'progressive normal/depth',
      );
      albedo = createColorTexture(gl, w, h, 'rgba16f', 'progressive albedo');
      drawBuffers.push(gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2);
    }
    for (let i = 0; i < 2; i += 1) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbos[i]!);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        colors[i]!,
        0,
      );
      if (normalDepth != null && albedo != null) {
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT1,
          gl.TEXTURE_2D,
          normalDepth,
          0,
        );
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT2,
          gl.TEXTURE_2D,
          albedo,
          0,
        );
      }
      gl.drawBuffers(drawBuffers);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(
          `pt-webgl2: progressive framebuffer ${w}×${h} is incomplete ` +
            `(${framebufferStatusName(gl, status)})`,
        );
      }
    }
  } catch (error) {
    for (const texture of colors) gl.deleteTexture(texture);
    if (normalDepth != null) gl.deleteTexture(normalDepth);
    if (albedo != null) gl.deleteTexture(albedo);
    for (const fbo of fbos) gl.deleteFramebuffer(fbo);
    throw error;
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  if (fbos.length !== 2 || colors.length !== 2) {
    throw new Error('pt-webgl2: progressive target allocation did not complete');
  }
  const completeFbos = fbos as unknown as [WebGLFramebuffer, WebGLFramebuffer];
  const completeColors = colors as unknown as [WebGLTexture, WebGLTexture];
  let destroyed = false;
  return {
    fbos: completeFbos,
    colors: completeColors,
    normalDepth,
    albedo,
    drawBuffers,
    width: w,
    height: h,
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      for (const fbo of completeFbos) gl.deleteFramebuffer(fbo);
      for (const color of completeColors) gl.deleteTexture(color);
      if (normalDepth != null) gl.deleteTexture(normalDepth);
      if (albedo != null) gl.deleteTexture(albedo);
    },
  };
}

export function bindProgressiveTarget(
  gl: WebGL2RenderingContext,
  target: ProgressiveTarget,
  colorIndex: 0 | 1,
): void {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbos[colorIndex]);
  gl.drawBuffers(target.drawBuffers);
}

export function clearProgressiveTarget(
  gl: WebGL2RenderingContext,
  target: ProgressiveTarget,
): void {
  for (const fbo of target.fbos) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.drawBuffers(target.drawBuffers);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }
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

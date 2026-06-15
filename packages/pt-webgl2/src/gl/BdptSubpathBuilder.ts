// BdptSubpathBuilder — extracted BDPT light-subpath machinery (D10.1).
// Owns the per-frame light-subpath ping-pong pair and the per-column draw loop that
// builds the light-vertex texture for the eye pass's connection sweep (A5, 2026-06-10).
//
// Previously private fields + methods of GlResources (#bdptLightPath, #bdptCopyFbo,
// #buildBdptLightSubpath, #blitBdpt, #ensureBdptLightPath). Extracted here so
// GlResources's drawAccumStep remains the coordinator without encoding the full
// BDPT protocol inline.
//
// Protocol contract (unchanged):
//   • build(prog, scene, seed, frame) is called BEFORE the eye-pass draw.
//   • Returns the WebGLTexture holding all light-path vertex columns, or null when
//     there is nothing to connect to (no analytic lights).
//   • Allocates the ping-pong pair lazily on the first call; disposes on destroy().

import type { FrameUniforms } from './glResources.js';
import type { UploadedSceneTextures } from '../scene/sceneTextures.js';
import { GlProgram } from './glProgram.js';
import { createRenderTarget, clearRenderTarget, type RenderTarget } from './framebuffer.js';

/** A5 — BDPT light-path ping-pong dimensions (matches the GLSL kernel layout). */
const BDPT_LIGHT_PATH_COLS = 3;
const BDPT_LIGHT_PATH_ROWS = 4;

/**
 * Manages the BDPT light-subpath ping-pong textures and issues the per-column
 * draw passes that populate them. Owned by GlResources; created once per engine
 * lifetime and destroyed with it.
 */
export class BdptSubpathBuilder {
  readonly #gl: WebGL2RenderingContext;
  /** Ping-pong pair — allocated lazily when bdpt:true is first encountered. */
  #lightPath: [RenderTarget, RenderTarget] | null = null;
  /** Scratch FBO for the per-column blit (copy built columns forward). */
  #copyFbo: WebGLFramebuffer | null = null;

  constructor(gl: WebGL2RenderingContext) {
    this.#gl = gl;
  }

  /**
   * A5 — build the BDPT light subpath for this sample and return the texture holding
   * all light-path vertex columns (to be bound as `uBdptLightPathTex` for the eye
   * pass's connection sweep). Returns null when there is nothing to connect to (no
   * analytic lights) — the caller then leaves the dummy bound and the frame renders
   * unidirectionally.
   *
   * Per-column protocol (one fullscreen draw over a 3×4 viewport per bounce):
   *   read  = the texture holding columns < col already built this frame
   *   write = the other ping-pong slot
   *   1. blit read → write (copy already-built columns forward; the kernel `discard`s
   *      every column != uBdptVertexCol, so without this they'd be lost on the swap)
   *   2. set uBdptLightSubpathPass=1, uBdptVertexCol=col, uBdptMaxLightBounces
   *   3. bind read as uBdptLightPathTex (bounce k reads column k-1 from it)
   *   4. draw → write column `col` is overwritten with the new vertex
   *   5. swap read/write
   * After the loop, `read` holds all columns. Reading and writing the SAME texture in
   * one draw is a WebGL2 feedback loop (undefined), which the read≠write ping-pong +
   * pre-blit avoids.
   */
  build(
    prog: GlProgram,
    scene: UploadedSceneTextures,
    seed: number,
    frame: FrameUniforms,
    bindSceneTextures: (prog: GlProgram, scene: UploadedSceneTextures, bdptTex: WebGLTexture | null) => void,
    drawFullscreen: () => void,
  ): WebGLTexture | null {
    if (scene.lightCount === 0) return null; // nothing to sample → unidirectional fallback
    const gl = this.#gl;
    this.#ensurePair();
    const pair = this.#lightPath;
    if (pair == null) return null;
    const copyFbo = this.#copyFbo;
    if (copyFbo == null) return null;

    const cols = Math.max(1, Math.min(frame.bdptMaxLightBounces, BDPT_LIGHT_PATH_COLS));

    // Clear both slots so unbuilt columns read as (0,0,0,0); column 0 row 0 .w==0 is
    // BDPT_KIND_LIGHT, so an all-zero column is NOT auto-invalid — but the kernel only
    // ever connects to columns it actually wrote, and the connection sweep iterates
    // [0, uBdptMaxLightBounces); a failed sample writes BDPT_KIND_INVALID (.w==3)
    // explicitly. Clearing keeps stale prior-frame columns out.
    clearRenderTarget(gl, pair[0]);
    clearRenderTarget(gl, pair[1]);

    prog.use();
    // The light-subpath pass shares the eye program; flip the pass flag + upload the
    // per-pass scalars. The scene textures (BVH/materials/lights) are bound below.
    prog.setInt('seed', seed);
    prog.setInt('uBdptLightSubpathPass', 1);
    prog.setInt('uBdptMaxLightBounces', cols);
    // The light subpath traces scene rays → needs the same per-frame transforms the
    // eye pass reads (lightsDenom uses environmentIntensity; traceScene reads none of
    // the camera matrices but initRenderState / fog do touch a few). Upload the load-
    // bearing ones; the kernel ignores the rest in the subpath branch.
    prog.setVec2('resolution', BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS);
    prog.setInt('bounces', frame.bounces);
    prog.setInt('transmissiveBounces', frame.transmissiveBounces);
    prog.setFloat('environmentIntensity', frame.environmentIntensity);
    prog.setMat4('environmentRotation', frame.environmentRotation);
    prog.setMat4('cameraWorldMatrix', frame.cameraWorldMatrix);
    prog.setMat4('invProjectionMatrix', frame.invProjectionMatrix);

    gl.disable(gl.BLEND); // vertex writes overwrite; no accumulation in the subpath.

    let readIdx = 0;
    for (let col = 0; col < cols; col += 1) {
      const read = pair[readIdx]!;
      const write = pair[1 - readIdx]!;

      // 1. Copy already-built columns (< col) read → write so they survive the swap.
      if (col > 0) this.#blit(read, write, copyFbo);

      // 2/3. Per-column scalars + the read texture as uBdptLightPathTex.
      prog.setInt('uBdptVertexCol', col);
      // Bind scene textures with the read slot as the light-path source. (For col 0
      // the kernel ignores the texture; binding the read slot is harmless.)
      bindSceneTextures(prog, scene, read.color);

      // 4. Draw the 3×4 viewport into the write slot.
      const { fbo } = write;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.viewport(0, 0, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS);
      drawFullscreen();

      // 5. Swap — `write` now holds columns ≤ col.
      readIdx = 1 - readIdx;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // After the loop, pair[readIdx] is the most-recently-written slot (all columns).
    return pair[readIdx]!.color;
  }

  dispose(): void {
    if (this.#lightPath != null) {
      this.#lightPath[0].destroy();
      this.#lightPath[1].destroy();
      this.#lightPath = null;
    }
    if (this.#copyFbo != null) {
      this.#gl.deleteFramebuffer(this.#copyFbo);
      this.#copyFbo = null;
    }
  }

  // ── private ────────────────────────────────────────────────────────────────

  /** Lazily allocate the ping-pong pair + copy FBO. */
  #ensurePair(): void {
    if (this.#lightPath != null) return;
    const gl = this.#gl;
    this.#lightPath = [
      createRenderTarget(gl, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS, false),
      createRenderTarget(gl, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS, false),
    ];
    this.#copyFbo = gl.createFramebuffer();
    if (this.#copyFbo == null) throw new Error('pt-webgl2: failed to create BDPT copy FBO');
  }

  /** Copy `src` color into `dst` color via a framebuffer blit (preserve built columns). */
  #blit(src: RenderTarget, dst: RenderTarget, copyFbo: WebGLFramebuffer): void {
    const gl = this.#gl;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, copyFbo);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, src.color, 0);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst.fbo);
    gl.blitFramebuffer(
      0, 0, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS,
      0, 0, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS,
      gl.COLOR_BUFFER_BIT, gl.NEAREST,
    );
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
  }
}

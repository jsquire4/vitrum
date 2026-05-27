/**
 * BdptLightPathBuffer — host-side ping-pong texture for the fork's BDPT
 * light-subpath cache.
 *
 * Sprint 10c (2026-05-12) shipped the fork's BDPT integrator + light-subpath
 * ping-pong (fork commit `98f4446`) and the vitrum bridge `ForkBridgeBdptOptions`
 * (vitrum commit `398dfce`). The remaining gap was host-side ownership of the
 * RGBA32F `lightPathTex` (width = maxLightBounces, height = 3) that the fork's
 * connection pass reads each iteration: hosts had to allocate + dispose it
 * themselves, and there was no example wiring.
 *
 * This helper owns the texture lifecycle so a host can:
 *
 * ```ts
 * import { BdptLightPathBuffer, createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
 *
 * const engine = await createPTEngine_WebGL2(canvas, {
 *   extensions: {
 *     'vitrum.ptWebgl.bdpt': true,
 *     'vitrum.ptWebgl.bdptMaxLightBounces': 3,
 *   },
 * });
 *
 * const bdpt = new BdptLightPathBuffer({ maxLightBounces: 3 });
 *
 * function frame() {
 *   // 1. Run the host's light-subpath draw pass into bdpt.renderTarget
 *   //    (the host owns this draw call; it uses the fork's light-subpath
 *   //    GLSL kernel via whatever mechanism the host already has in place).
 *   //    See plan/sprint-10c-pt-fork-patch.md for the kernel signature.
 *   //
 *   // 2. Hand the texture to the engine so the next renderFrame's
 *   //    connection pass reads it.
 *   engine.bdptAdvanceFrame(bdpt.texture);
 *
 *   // 3. Standard renderFrame — the fork's connect pass now reads
 *   //    bdpt.texture each sample.
 *   engine.renderFrame(input);
 * }
 *
 * // On teardown:
 * bdpt.dispose();
 * engine.dispose();
 * ```
 *
 * Lifecycle: the WebGLRenderTarget is constructed eagerly (three.js
 * defers the actual GL texture allocation until first use, but the JS
 * wrapper is live immediately). `dispose()` releases the underlying GL
 * resources. `BdptLightPathBuffer` is intentionally a single-texture
 * wrapper rather than a true ping-pong because the fork's connect pass
 * reads the same texture every iteration (Veach 1997 §10.3) — the
 * "advance" is the host re-writing it once per accumulation tick.
 *
 * References:
 * - Veach 1997, "Robust Monte Carlo Methods for Light Transport Simulation,"
 *   PhD thesis, §10.3 (BDPT MIS connection formulae).
 * - `plan/sprint-10c-pt-fork-patch.md` (vitrum repo).
 */

import type { Scene } from '@vitrum/core';
import * as THREE from 'three';
import type { WebGLRenderer } from 'three';
import { fillBdptLightPathWebGL } from './bdpt/fillBdptLightPathWebGL.js';

export interface BdptLightPathBufferOptions {
  /**
   * Number of light-subpath bounces to cache. Must match the
   * `extensions['vitrum.ptWebgl.bdptMaxLightBounces']` value passed at
   * engine construction (`PTEngineWebGL2` clamps to 1..3).
   *
   * @default 3
   */
  readonly maxLightBounces?: number;
}

const LIGHT_PATH_BUFFER_HEIGHT = 3 as const;

export class BdptLightPathBuffer {
  /** Cached width = maxLightBounces. Texture dimensions are `width × 3`. */
  readonly maxLightBounces: number;

  /** RGBA32F WebGLRenderTarget. The host writes into `.texture` via its
   *  own light-subpath draw pass; the fork's connect pass reads from it. */
  readonly renderTarget: THREE.WebGLRenderTarget;

  /** Direct handle to the underlying THREE.Texture for use as
   *  `ForkBridgeBdptOptions.lightPathTex`. */
  readonly texture: THREE.Texture;

  private _disposed = false;

  constructor(options: BdptLightPathBufferOptions = {}) {
    const max = options.maxLightBounces ?? 3;
    if (!Number.isFinite(max) || max < 1 || max > 3) {
      throw new RangeError(
        `[BdptLightPathBuffer] maxLightBounces must be in 1..3 (got ${max}). ` +
        `The fork hard-caps at BDPT_MAX_LIGHT_BOUNCES = 3.`,
      );
    }
    this.maxLightBounces = max;
    this.renderTarget = new THREE.WebGLRenderTarget(max, LIGHT_PATH_BUFFER_HEIGHT, {
      type: THREE.FloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    this.renderTarget.texture.name = 'vitrum.bdpt.lightPath';
    this.texture = this.renderTarget.texture;
  }

  /** Release the underlying GL resources. Idempotent. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.renderTarget.dispose();
  }

  /** True when `dispose()` has been called. */
  get disposed(): boolean {
    return this._disposed;
  }

  /**
   * CPU bounce-0 fill from a @vitrum/core Scene (rect/point/spot/directional emitters).
   * Call once per frame before `engine.bdptAdvanceFrame(this.texture)`.
   */
  fillFromScene(renderer: WebGLRenderer, scene: Scene, frameSeed: number): void {
    if (this._disposed) {
      throw new Error('[BdptLightPathBuffer] fillFromScene after dispose');
    }
    // Upload only — do not bind/clear this.renderTarget on the shared renderer; that
    // leaked GL state and blacked the path-tracer canvas on some drivers.
    fillBdptLightPathWebGL(renderer, this.texture, this.maxLightBounces, scene, frameSeed);
  }
}

/**
 * forkAccess.ts — single-point encapsulation of `three-gpu-pathtracer` fork
 * internals consumed by `@vitrum/pt-webgl`.
 *
 * WHY: two production files (`forkUniformBridge.ts` and `ptEngineWebGL2.ts`)
 * historically reached through `tracer._pathTracer.material.uniforms` and
 * `tracer.target.texture` independently. When the fork upstreams an official
 * accessor — or renames `_pathTracer` — both call sites break in lockstep and
 * each must be patched in isolation.
 *
 * After this module:
 *   - All fork-private navigation lives here. Production callers go through
 *     `ForkAccess.getMaterial(tracer)` / `ForkAccess.getRenderTexture(tracer)`.
 *   - When the fork lands an official accessor (e.g. `getMaterial()` /
 *     `getRenderTexture()` methods on `WebGLPathTracer`), exactly one file
 *     changes — this one.
 *
 * The `_pathTracer` field is an underscore-prefixed private of the upstream
 * `WebGLPathTracer` class. We keep `_pathTracer.material` and `target.texture`
 * typed loosely (with optional chaining) so we never throw on shape drift —
 * callers receive `null` and degrade gracefully.
 */

/** Minimal three-side material shape we depend on. The fork's path-tracer
 *  material is a `ShaderMaterial`-like with `.uniforms` keyed by uniform name.
 *  Each uniform is a `{ value: T }` cell that the renderer reads each frame. */
export interface MaterialLike {
  uniforms?: Record<string, { value: unknown }>;
}

/** Opaque render-texture handle exposed via `WebGLPathTracer.target.texture`.
 *  Returned to the host as `FrameOutput.primaryRadiance` (typed as
 *  `BackendTexture` in `@vitrum/core`) — we don't narrow the type here so
 *  consumers cast at their own boundaries (`three.Texture`, `WebGLTexture`,
 *  etc.). */
export type RenderTextureHandle = unknown;

/** Loose structural shape of the fork's `WebGLPathTracer` that exposes the
 *  fork-private accessors we currently reach for. Every member is optional
 *  so callers can pass arbitrary stubs (production tracer, test fakes) and
 *  `ForkAccess` will return `null` rather than throw on missing structure. */
interface ForkPathTracerLike {
  _pathTracer?: {
    material?: MaterialLike;
  };
  target?: {
    texture?: RenderTextureHandle;
  };
}

/**
 * Test-only helper: build a stub object that mimics the fork's path-tracer
 * shape so `ForkAccess.getMaterial` and `ForkAccess.getRenderTexture` can
 * navigate it without re-implementing the fork structure in every test file.
 *
 * The production fork-private navigation path lives entirely inside
 * `ForkAccess` (this file). Test fixtures construct shapes via this helper
 * so individual test files never write the underscore-prefixed key directly
 * — adding a single point of change when the fork lands an official accessor.
 *
 * Pass `material` (typically `{ uniforms: { ... } }`) and optionally a
 * `renderTexture` value; everything else is filled with safe defaults.
 */
export function makeForkPathTracerStubForTests(opts?: {
  material?: MaterialLike;
  renderTexture?: RenderTextureHandle;
}): ForkPathTracerLike {
  // Build sub-objects conditionally so we never assign `undefined` to fields
  // declared as plain optional (TypeScript's exactOptionalPropertyTypes
  // distinguishes "missing key" from "key explicitly set to undefined").
  const inner: ForkPathTracerLike = {
    _pathTracer: {},
    target: {},
  };
  if (opts?.material !== undefined) inner._pathTracer!.material = opts.material;
  if (opts?.renderTexture !== undefined) inner.target!.texture = opts.renderTexture;
  return inner;
}

/**
 * Static helper class encapsulating all fork-private access. Stateless;
 * methods are class-static so call sites read like a namespace
 * (`ForkAccess.getMaterial(...)`) and the encapsulation surface is grep-able.
 *
 * Adding a new fork-private reach? Add a method here. Removing a fork-private
 * dependency? Delete the method here. Never inline a `tracer._pathTracer.*`
 * read at a call site; that defeats the encapsulation.
 */
export class ForkAccess {
  /**
   * Resolve the fork's path-tracer shader material. Returns `null` when the
   * tracer has not yet built its internal pipeline (pre-`setScene()`) or when
   * the fork shape has drifted under us.
   *
   * Currently navigates `tracer._pathTracer.material`. The leading underscore
   * marks this as a fork private — when an official accessor lands (e.g.
   * `WebGLPathTracer.getMaterial()`), update this method only.
   */
  static getMaterial(tracer: unknown): MaterialLike | null {
    const t = tracer as ForkPathTracerLike | null | undefined;
    return t?._pathTracer?.material ?? null;
  }

  /**
   * Resolve the fork's HDR accumulation render-texture handle. Returns `null`
   * when the tracer has no allocated target yet.
   *
   * The fork exposes `target` publicly today (no underscore), so this is
   * arguably less brittle than `getMaterial` — but routing it through the
   * same façade keeps the fork-coupling surface in one file. Future fork
   * renames or wrapper changes touch exactly this method.
   */
  static getRenderTexture(tracer: unknown): RenderTextureHandle | null {
    const t = tracer as ForkPathTracerLike | null | undefined;
    return t?.target?.texture ?? null;
  }
}

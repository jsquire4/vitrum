/**
 * DDGILight — the light-source shape consumed by ProbeUpdatePass.
 *
 * This type is intentionally NOT @vitrum/core's SceneEmitter. The DDGI
 * probe-update pass consumes a host-domain projection of lights that is
 * specifically tagged with the runtime kinds the walkaround pipeline
 * recognises ('sun' | 'fixture' | 'teaLight'). SceneEmitter uses an
 * abstract emitter taxonomy ('directional' | 'point' | 'spot' | …) that
 * the core contract defines independently of any specific renderer's
 * conventions. Keeping DDGILight separate lets the host bridge map its
 * domain objects without polluting the core contract with walkaround-
 * specific kind strings.
 *
 * Fields verified against probeUpdatePass.ts _uploadLights (the only
 * consumer in this package):
 *   - kind      — switched on: 'sun' | 'fixture' | 'teaLight'
 *   - intensity — multiplied by _sunIntensityMul for sun lights
 *   - position  — accessed via unsafe cast on fixture/teaLight lights
 *   - on        — filter: only lights where on===true are uploaded
 */
export interface DDGILight {
  /** Runtime kind tag. Only 'sun', 'fixture', and 'teaLight' are handled;
   *  any other kind is silently skipped in _uploadLights. */
  readonly kind: 'sun' | 'fixture' | 'teaLight' | string;

  /** Photometric intensity value (arbitrary units; host normalises to
   *  whatever scale the renderer expects). */
  readonly intensity: number;

  /** Whether this light is active. ProbeUpdatePass filters to only
   *  lights where on === true before uploading to the GPU UBO. */
  readonly on: boolean;

  /** World-space position. Only used for 'fixture' and 'teaLight' kinds.
   *  Accessed via an unsafe cast in the original source because LightSource
   *  did not expose position on the base type — DDGILight makes it explicit
   *  and optional (sun lights have no meaningful position). */
  readonly position?: { readonly x: number; readonly y: number; readonly z: number };
}

// `DDGIDeviceHandle` interface removed 2026-05-18 — was defined here but
// never imported anywhere in the workspace (only referenced in JSDoc
// comments inside HybridEngine.ts by spelling, not by type). DDGI's
// input shape currently uses an inline object literal via `DDGIFrameInputs`
// which exposes the `device` + `renderer` fields directly.

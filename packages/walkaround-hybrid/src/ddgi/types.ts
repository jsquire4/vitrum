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
 *   - direction — sun travel-direction; packed for 'sun' lights (see below)
 *   - on        — filter: only lights where on===true are uploaded
 */
export interface DDGILight {
  /** Runtime kind tag. Only 'sun', 'fixture', and 'teaLight' are handled;
   *  any other kind is warn-skipped by the DDGI probe-light packer. */
  // eslint-disable-next-line @typescript-eslint/no-redundant-type-constituents -- intentional: literal union widens to string for extensibility while preserving IDE autocomplete for the known kinds
  readonly kind: 'sun' | 'fixture' | 'teaLight' | string;

  /** Optional source-emitter id, preserved when a DDGILight is mapped
   *  directly from a `@vitrum/core` `SceneEmitter` (see
   *  `coreEmittersToDDGILights`). Ignored by `_uploadLights` / the GPU
   *  packer (the DDGI light UBO has no id slot); carried purely so host-
   *  side code can correlate a DDGI light back to its core emitter for
   *  incremental updates / introspection. Lights derived from the raw-
   *  THREE escape-hatch path leave this undefined. */
  readonly id?: string;

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

  /** Sun TRAVEL direction (the direction light propagates — points away from
   *  the sun, downward for an overhead sun). Only consumed by the GPU packer
   *  for `kind: 'sun'` lights; the WGSL probe shader negates it (`lightDir =
   *  normalize(-light.direction)`) to recover the toward-light direction for
   *  `dot(N, L)`. Carried directly from a `@vitrum/core` `directional` emitter
   *  via `coreEmitterToDDGILight` (whose `direction` points AT the light, so
   *  the mapper negates it to a travel direction). When absent, the packer
   *  falls back to the legacy hardcoded `(0,-1,0)` straight-down sun so any
   *  host-supplied sun light without an explicit direction is unchanged. */
  readonly direction?: { readonly x: number; readonly y: number; readonly z: number };

  /** RGB radiance multiplier for point-like lights (`fixture` / `teaLight`)
   *  AND for `sun` lights (the latter previously hardcoded `(1,0.95,0.85)` in
   *  the packer; now read from this field when present, falling back to that
   *  warm-white default otherwise). */
  readonly color?: { readonly r: number; readonly g: number; readonly b: number };

  /** SPOT fixtures only — the cone axis as a TOWARD-LIGHT unit vector (= the core
   *  `SpotEmitter.direction`, `normalize(position - target)`, UNnegated, unlike
   *  the sun's travel `direction`). When present (length ≈ 1), the probe shader
   *  applies the cone falloff `smoothstep(spotCosOuter, spotCosInner,
   *  dot(toLightDir, spotAxis))`; absent (point fixture) ⇒ omnidirectional. */
  readonly spotAxis?: { readonly x: number; readonly y: number; readonly z: number };
  /** SPOT fixtures only — cos of the INNER (full-intensity) half-angle =
   *  cos(angle·(1−penumbra)). Falloff = 1 at/above this. */
  readonly spotCosInner?: number;
  /** SPOT fixtures only — cos of the OUTER half-angle = cos(angle). Falloff = 0
   *  at/below this (outside the cone). */
  readonly spotCosOuter?: number;
}

// `DDGIDeviceHandle` interface removed 2026-05-18 — was defined here but
// never imported anywhere in the workspace (only referenced in JSDoc
// comments inside HybridEngine.ts by spelling, not by type). DDGI's
// input shape currently uses an inline object literal via `DDGIFrameInputs`
// which exposes the `device` + `renderer` fields directly.

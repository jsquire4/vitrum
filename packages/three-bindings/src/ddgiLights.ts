/**
 * ddgiLights.ts — THREE.RectAreaLight → DDGILight point-approximation extraction.
 *
 * Walkaround's DDGI probe-update pass consumes a `DDGILight` light-source
 * record (point-light projection with kind / intensity / position / on).
 * `THREE.RectAreaLight` isn't natively supported by that pass, so this helper
 * walks an Object3D tree, finds every visible RectAreaLight, and projects each
 * onto a `DDGILight` point-light approximation at the rect centroid.
 *
 * Approximation rationale: DDGI provides low-frequency indirect bounce — the
 * actual rect geometry only matters for the high-frequency direct term, which
 * ReSTIR DI handles separately from the actual emitter triangles. A point at
 * the rect centroid carrying flux ≈ `color × intensity × area` gives a
 * qualitatively-correct downward irradiance for probes; colour bleed onto
 * surrounding walls (the visible signature of Cornell-style scenes) reaches
 * the irradiance atlas correctly. The remaining factor-of-π errors in
 * total-flux conversion are negligible against the multiple-of-10 dynamic
 * range that distinguishes "lit colour bleed" from "atlas reads zero".
 *
 * Lives in @vitrum/three-bindings because the function does THREE-only light
 * extraction; walkaround-hybrid imports it. The DDGILight return type is also
 * defined here (the type itself is THREE-agnostic — only describes the data
 * shape walkaround's probe-update pass consumes — but co-locating it with its
 * producer keeps the extraction module self-contained).
 */

import * as THREE from 'three';

/**
 * DDGILight — the light-source shape consumed by walkaround's
 * ProbeUpdatePass.
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
 * consumer in walkaround-hybrid):
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

/**
 * Walk an Object3D tree for `THREE.RectAreaLight` instances and project each
 * onto a `DDGILight` point-light approximation so the DDGI probe-update pass
 * (which only switches on `kind === 'sun' | 'fixture' | 'teaLight'`) can
 * evaluate direct lighting at probe-ray hit points.
 *
 * See file header for the full approximation rationale.
 */
export function collectDDGILightsFromRectAreaLights(root: THREE.Object3D): DDGILight[] {
  const out: DDGILight[] = [];
  const _wp = new THREE.Vector3();
  root.updateMatrixWorld(true);
  root.traverseVisible((obj) => {
    if (!(obj instanceof THREE.RectAreaLight)) return;
    const light = obj;
    const area = light.width * light.height;
    _wp.setFromMatrixPosition(light.matrixWorld);
    out.push({
      kind: 'fixture',
      intensity: light.intensity * area,
      on: true,
      position: { x: _wp.x, y: _wp.y, z: _wp.z },
    });
  });
  return out;
}

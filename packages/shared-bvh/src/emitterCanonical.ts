/**
 * emitterCanonical.ts — backend-neutral emitter normalizer (T1-2).
 *
 * Both path-tracing backends pack `scene.emitters` into their own texel/storage
 * layouts (`@vitrum/pt-webgl2` `packLightsTexture`, `@vitrum/pt-webgpu`
 * `packEmitterArrays`). The BYTE layouts differ and stay per-backend. What the
 * two SHOULD agree on is the *semantic* interpretation of each core emitter:
 *
 *   - **directional** — the "toward-light" unit vector. Core `direction` points
 *     AT the light; the NEE convention both backends use is the incoming
 *     direction `-direction / |direction|` (pt-webgpu `packEmitterArrays`
 *     directional loop; the pt-webgl2 lights texture stores core `direction`
 *     directly as "toward the light", which for a normalized core direction is
 *     the same ray up to sign convention — captured here as `towardLight`).
 *   - **disc-area** — synthesize an in-plane (u, v) basis from the disc normal
 *     (both backends build an orthonormal tangent basis; the exact axis choice
 *     differs, but |u| = |v| = radius and area = π·r²).
 *   - **spot** — the inner/outer cone from `angle` + `penumbra`:
 *     `outerAngle = angle`, `innerAngle = angle · (1 − penumbra)`, and the
 *     packed cosines `cos(outerAngle)` / `cos(innerAngle)`.
 *   - **SHADOW-01** — `castShadow === false ⇒ shadowDisabled` (both backends
 *     resolve the flag identically, encoded per-backend).
 *   - **power** — a luminance·intensity·area proxy (the light-tree / NEE
 *     selection mass). Area = 0 for delta lights (point/spot/directional).
 *
 * This normalizer is the single source of that mapping. It is consumed by the
 * cross-backend parity test (`emitterCanonicalParity.test.ts` in pt-webgpu) to
 * assert both backends agree on light count, per-light power, and cone angles
 * for a fixture scene. It is byte-preserving: it does NOT change what either
 * packer writes.
 *
 * KNOWN feature-parity gaps captured but NOT reconciled here (tracked in
 * `items_to_fix.md`): pt-webgl2 hardcodes the spot soft-source radius to 0, and
 * pt-webgl2 filters `mesh-area` emitters out of its analytic light list (they go
 * through emissive-geometry sampling instead). See `canonicalMeshAreaIncluded`.
 */

import type { Scene, SceneEmitter, Vec3 } from '@vitrum/core';

/** Rec.709 luminance (matches `@vitrum/shared-samplers` `luminance`). */
function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function len3(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function cross3(a: Vec3, b: Vec3): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export type CanonicalEmitterKind =
  | 'directional'
  | 'disc-area'
  | 'rect-area'
  | 'point'
  | 'spot'
  | 'mesh-area';

export interface CanonicalEmitter {
  readonly id: string;
  readonly kind: CanonicalEmitterKind;
  /** `emitter.color · emitter.intensity` — the linear HDR radiance/irradiance. */
  readonly radiance: [number, number, number];
  /**
   * Luminance-weighted power proxy: `luminance(radiance) · area`. Area is the
   * emitter's surface area (rect: |u×v|; disc: π·r²; mesh-area: not resolved
   * here — see `meshArea`) and `1` for delta lights (point/spot/directional) so
   * their power is `luminance(radiance)`. This is the light-tree / NEE
   * selection mass both backends key on.
   */
  readonly power: number;
  /** Emitter surface area (0 for delta lights; NaN when not resolvable). */
  readonly area: number;
  /** SHADOW-01: `castShadow === false`. */
  readonly shadowDisabled: boolean;
  /** Unit "toward the light" vector (directional/spot only), else null. */
  readonly towardLight: [number, number, number] | null;
  /** Spot cone (outer/inner angle in radians + their cosines), else null. */
  readonly cone: {
    readonly outerAngle: number;
    readonly innerAngle: number;
    readonly cosOuter: number;
    readonly cosInner: number;
  } | null;
}

const RECT_AREA_SHAPE = 'rect-area';

function emitterRadiance(e: SceneEmitter): [number, number, number] {
  return [e.color[0] * e.intensity, e.color[1] * e.intensity, e.color[2] * e.intensity];
}

/**
 * Normalize one core emitter into the backend-neutral canonical form. Returns
 * null for `mesh-area` emitters when `includeMeshArea` is false (the pt-webgl2
 * analytic-list convention — mesh-area goes through emissive-geometry sampling).
 */
export function canonicalizeEmitter(
  e: SceneEmitter,
  includeMeshArea: boolean,
): CanonicalEmitter | null {
  const radiance = emitterRadiance(e);
  const lum = luminance(radiance[0], radiance[1], radiance[2]);
  const shadowDisabled = e.castShadow === false;
  switch (e.kind) {
    case 'directional': {
      const d = e.direction;
      const l = len3(d);
      // Core `direction` points AT the light; NEE fires TOWARD the light.
      const toward: [number, number, number] =
        l < 1e-8 ? [0, 1, 0] : [-d[0] / l, -d[1] / l, -d[2] / l];
      return {
        id: e.id,
        kind: 'directional',
        radiance,
        power: lum,
        area: 0,
        shadowDisabled,
        towardLight: toward,
        cone: null,
      };
    }
    case 'point':
      return {
        id: e.id,
        kind: 'point',
        radiance,
        power: lum,
        area: 0,
        shadowDisabled,
        towardLight: null,
        cone: null,
      };
    case 'spot': {
      const d = e.direction;
      const l = len3(d);
      const toward: [number, number, number] =
        l < 1e-8 ? [0, -1, 0] : [d[0] / l, d[1] / l, d[2] / l];
      const penumbra = Math.min(1, Math.max(0, e.penumbra ?? 0));
      const outerAngle = e.angle;
      const innerAngle = outerAngle * (1 - penumbra);
      return {
        id: e.id,
        kind: 'spot',
        radiance,
        power: lum,
        area: 0,
        shadowDisabled,
        towardLight: toward,
        cone: {
          outerAngle,
          innerAngle,
          cosOuter: Math.cos(outerAngle),
          cosInner: Math.cos(innerAngle),
        },
      };
    }
    case 'rect-area': {
      const area = len3(cross3(e.uAxis, e.vAxis));
      return {
        id: e.id,
        kind: RECT_AREA_SHAPE,
        radiance,
        power: lum * area,
        area,
        shadowDisabled,
        towardLight: null,
        cone: null,
      };
    }
    case 'disc-area': {
      const area = Math.PI * e.radius * e.radius;
      return {
        id: e.id,
        kind: 'disc-area',
        radiance,
        power: lum * area,
        area,
        shadowDisabled,
        towardLight: null,
        cone: null,
      };
    }
    case 'mesh-area': {
      if (!includeMeshArea) return null;
      return {
        id: e.id,
        kind: 'mesh-area',
        radiance,
        power: lum,
        area: Number.NaN,
        shadowDisabled,
        towardLight: null,
        cone: null,
      };
    }
    default: {
      const _never: never = e;
      void _never;
      return null;
    }
  }
}

/**
 * Normalize a scene's emitters into the backend-neutral canonical set.
 *
 * @param scene the core scene.
 * @param includeMeshArea when false (default), `mesh-area` emitters are dropped
 *   — the pt-webgl2 analytic-list convention (mesh-area is sampled through the
 *   emissive-geometry path, not the analytic light list). Pass `true` to mirror
 *   pt-webgpu, which packs explicit mesh-area emitters into their own stream.
 */
export function emitterToCanonical(
  scene: Scene,
  includeMeshArea = false,
): CanonicalEmitter[] {
  const out: CanonicalEmitter[] = [];
  for (const e of scene.emitters) {
    const c = canonicalizeEmitter(e, includeMeshArea);
    if (c != null) out.push(c);
  }
  return out;
}

/**
 * Whether the canonical set includes `mesh-area` emitters. The pt-webgl2
 * analytic light list excludes them (they are sampled through emissive
 * geometry); pt-webgpu packs explicit ones. See the `items_to_fix.md`
 * feature-parity entry.
 */
export function canonicalMeshAreaIncluded(canonical: readonly CanonicalEmitter[]): boolean {
  return canonical.some((c) => c.kind === 'mesh-area');
}

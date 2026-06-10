/**
 * B15 — scene-scale-aware radiometric clamp defaults (road-to-100 campaign 2).
 *
 * Background
 * ----------
 * Several walkaround tunables are radiometric CLAMPS whose Cornell-baseline
 * defaults were tuned against the canonical ±1 Cornell box (world AABB ≈
 * [-1,-1,-1]..[1,1,1], diagonal ≈ 2·√3 ≈ 3.4641). They bite at large scene
 * scale: with emitter intensity held fixed, blowing the geometry up by ×s makes
 * characteristic surface-to-emitter distances grow ×s, so the irradiance /
 * radiance arriving at surfaces falls as ×1/s². A clamp tuned to clip only the
 * firefly tail at Cornell scale then clips the LEGITIMATE signal at large scale
 * (the size-200 "bimodal clipping" instability, A-7 lead #5). The squared-
 * distance FLOOR has the opposite problem — a fixed 0.01 m² floor that was a
 * sensible near-contact epsilon at Cornell scale becomes negligibly small (no
 * longer guarding the near-emitter singularity) at large scale.
 *
 * The law
 * -------
 * Pure GEOMETRIC inverse-square scaling, normalized so that a Cornell-diagonal
 * scene reproduces today's exact defaults (factor == 1 ⇒ byte-identical):
 *
 *   s = D_scene / D_cornell                         (scene diagonal ratio)
 *
 *   • IRRADIANCE / RADIANCE clamps  (W/m², W/sr/m²) scale ×(1/s²):
 *       restirGiIrrClamp, directFireflyClamp, indirectFireflyClamp[c]
 *     Rationale: peak irradiance/radiance at a surface from a fixed-intensity
 *     emitter ∝ intensity·area / d², and d ∝ D_scene, so the radiometric
 *     magnitude (hence the clip threshold that preserves the same tail) ∝ 1/s².
 *
 *   • The squared-DISTANCE floor  (m²) scales ×s²:
 *       emitterDist2Floor
 *     Rationale: it is a length², so to remain the same FRACTION of the scene's
 *     characteristic distance it must scale with s².
 *
 *   • restirGiWCap is NOT scaled. It is a UNITLESS RIS unbiased-weight cap
 *     (variance-bounding), not a radiometric magnitude — scene scale does not
 *     change the statistics it bounds. (Documented per-knob in
 *     HybridEngineTuning's TUNABLE_DEFINITIONS table notes.)
 *
 * Honesty caveat (documented at the call site + per knob): the law assumes the
 * host holds emitter INTENSITY fixed across scene scales (the size-200 regime).
 * A host that scales emitter power physically with the scene should set the
 * clamp explicitly; HOST OVERRIDES ARE NEVER SCALED — an explicitly-set tuning
 * value passes through verbatim (see {@link ScaleAwareInputs.hostExplicit}).
 *
 * The estimate uses only the scene's mesh/analytic primitive bounds (cheap,
 * synchronous, transform/instance-aware) — it does NOT wait for the async BVH
 * build. Degenerate scenes (no finite extent, single point) fall back to the
 * Cornell diagonal ⇒ factor 1 ⇒ defaults unchanged.
 */
import type { Scene, ScenePrimitive, Mat4 } from '@vitrum/core';
import type { Tunables } from './HybridEngineTuning.js';

/** Canonical Cornell-box world-space diagonal (±1 box ⇒ 2·√3). The reference
 *  scale at which every clamp default reproduces its tuned value exactly. */
export const CORNELL_DIAGONAL = 2 * Math.sqrt(3);

/** The radiometric knobs scaled by B15, with their dimensionality exponent on
 *  the scale ratio `s = D_scene / D_cornell`. Irradiance/radiance clamps use
 *  −2 (×1/s²); the squared-distance floor uses +2 (×s²). restirGiWCap is
 *  deliberately ABSENT (unitless variance cap — not scaled). */
const SCALE_EXPONENTS: Partial<Record<keyof Tunables, number>> = {
  restirGiIrrClamp: -2,
  directFireflyClamp: -2,
  emitterDist2Floor: +2,
};

/** Host-explicit-override flags for the B15-scaled knobs. A `true` entry means
 *  the host set that knob explicitly (via `opts.tuning` or a subsystem
 *  sub-object), so it must pass through UN-scaled. */
export interface ScaleAwareHostExplicit {
  readonly restirGiIrrClamp: boolean;
  readonly directFireflyClamp: boolean;
  readonly emitterDist2Floor: boolean;
  readonly indirectFireflyClamp: boolean;
}

export interface ScaleAwareInputs {
  /** The Cornell-baseline tunables resolved by `readTunables` (host overrides
   *  already applied — see {@link ScaleAwareInputs.hostExplicit} for which
   *  knobs were host-set). */
  readonly baseTunables: Tunables;
  /** The Cornell-baseline `indirectFireflyClamp` tuple (lives outside the
   *  number-only Tunables table). */
  readonly baseIndirectFireflyClamp: readonly [number, number, number];
  /** Per-knob host-explicit flags. */
  readonly hostExplicit: ScaleAwareHostExplicit;
}

export interface ScaleAwareResult {
  readonly tunables: Tunables;
  readonly indirectFireflyClamp: readonly [number, number, number];
  /** The scene diagonal used (for telemetry / debug). */
  readonly sceneDiagonal: number;
  /** The scale ratio `D_scene / D_cornell` (1 ⇒ Cornell-scale ⇒ no change). */
  readonly scaleRatio: number;
}

// ── World-space AABB from scene primitives ─────────────────────────────────

/** Apply a column-major Mat4 to a point. Identity-safe. */
function transformPoint(
  m: Mat4 | undefined,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  if (m == null) return [x, y, z];
  // Column-major: m[col*4 + row]. Mat4 is a fixed 16-element layout, so the
  // indexed reads are always defined (the non-null assertions satisfy
  // noUncheckedIndexedAccess without a per-element guard).
  const tx = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
  const ty = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
  const tz = m[2]! * x + m[6]! * y + m[10]! * z + m[14]!;
  return [tx, ty, tz];
}

interface Bounds {
  minX: number; minY: number; minZ: number;
  maxX: number; maxY: number; maxZ: number;
  any: boolean;
}

function expand(b: Bounds, x: number, y: number, z: number): void {
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
  if (x < b.minX) b.minX = x;
  if (y < b.minY) b.minY = y;
  if (z < b.minZ) b.minZ = z;
  if (x > b.maxX) b.maxX = x;
  if (y > b.maxY) b.maxY = y;
  if (z > b.maxZ) b.maxZ = z;
  b.any = true;
}

/** Accumulate a primitive's positions (transform/instance-aware) into bounds. */
function accumulatePrimitive(b: Bounds, prim: ScenePrimitive): void {
  // analytic primitives that survived to the render-scene view are already
  // mesh fallbacks; the authored `_lastScene` may still carry `analytic`, which
  // has no `positions` — skip it here (its mesh fallback is the bounds source).
  const positions = (prim as { positions?: Float32Array }).positions;
  if (positions == null) return;
  const transform = (prim as { transform?: Mat4 }).transform;
  const instances = (prim as { instances?: ReadonlyArray<Mat4> }).instances;
  const n = positions.length;
  if (instances != null && instances.length > 0) {
    for (const inst of instances) {
      for (let i = 0; i + 2 < n; i += 3) {
        const [x, y, z] = transformPoint(inst, positions[i]!, positions[i + 1]!, positions[i + 2]!);
        expand(b, x, y, z);
      }
    }
    return;
  }
  for (let i = 0; i + 2 < n; i += 3) {
    const [x, y, z] = transformPoint(transform, positions[i]!, positions[i + 1]!, positions[i + 2]!);
    expand(b, x, y, z);
  }
}

/**
 * Compute the world-space AABB diagonal of a scene from its primitive
 * positions (transform- and instance-aware). Returns the {@link CORNELL_DIAGONAL}
 * for a scene with no finite extent (empty / degenerate / point), so the scale
 * ratio is 1 and the defaults are unchanged.
 */
export function sceneWorldDiagonal(scene: Scene | null): number {
  if (scene == null || scene.primitives.length === 0) return CORNELL_DIAGONAL;
  const b: Bounds = {
    minX: Infinity, minY: Infinity, minZ: Infinity,
    maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity,
    any: false,
  };
  for (const prim of scene.primitives) accumulatePrimitive(b, prim);
  if (!b.any) return CORNELL_DIAGONAL;
  const dx = b.maxX - b.minX;
  const dy = b.maxY - b.minY;
  const dz = b.maxZ - b.minZ;
  const diag = Math.sqrt(dx * dx + dy * dy + dz * dz);
  return diag > 1e-6 ? diag : CORNELL_DIAGONAL;
}

// ── The scaling law ─────────────────────────────────────────────────────────

/**
 * Derive scene-scale-aware clamp defaults from a scene's world diagonal.
 *
 * INVARIANT: at `D_scene ≈ CORNELL_DIAGONAL` the returned values are
 * byte-identical to the input baselines (scaleRatio == 1 ⇒ s^k == 1). Host
 * overrides (`hostExplicit[knob] === true`) ALWAYS pass through un-scaled.
 */
export function deriveScaleAwareClamps(
  scene: Scene | null,
  inputs: ScaleAwareInputs,
): ScaleAwareResult {
  const diag = sceneWorldDiagonal(scene);
  const s = diag / CORNELL_DIAGONAL;

  // Exact Cornell-scale short-circuit guarantees byte-identity (no float drift
  // from pow(1, k)). |s - 1| within a tight epsilon ⇒ leave the baselines
  // untouched.
  const isCornellScale = Math.abs(s - 1) < 1e-6;

  const scaled: Record<string, number> = { ...inputs.baseTunables };

  if (!isCornellScale && s > 0 && Number.isFinite(s)) {
    for (const key of Object.keys(SCALE_EXPONENTS) as Array<keyof Tunables>) {
      const k = SCALE_EXPONENTS[key]!;
      const knob = key as keyof ScaleAwareHostExplicit;
      // Only the three scalar knobs are in both maps; the membership check keeps
      // TS + the host-explicit gate honest.
      if (
        knob === 'restirGiIrrClamp' ||
        knob === 'directFireflyClamp' ||
        knob === 'emitterDist2Floor'
      ) {
        if (inputs.hostExplicit[knob]) continue; // host override → never scaled
        const base = inputs.baseTunables[key];
        scaled[key as string] = base * Math.pow(s, k);
      }
    }
  }

  // indirectFireflyClamp (tuple) scales ×1/s² unless host-set.
  let indirect = inputs.baseIndirectFireflyClamp;
  if (!isCornellScale && s > 0 && Number.isFinite(s) && !inputs.hostExplicit.indirectFireflyClamp) {
    const f = Math.pow(s, -2);
    indirect = [
      inputs.baseIndirectFireflyClamp[0] * f,
      inputs.baseIndirectFireflyClamp[1] * f,
      inputs.baseIndirectFireflyClamp[2] * f,
    ];
  }

  return {
    tunables: Object.freeze(scaled) as unknown as Tunables,
    indirectFireflyClamp: indirect,
    sceneDiagonal: diag,
    scaleRatio: s,
  };
}

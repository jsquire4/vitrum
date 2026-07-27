/**
 * Single-sourced flat-emitter stride-walk for pt-webgpu's CPU NEE / BDPT oracles.
 *
 * pt-webgpu's selectable-light enumeration order is fixed:
 *
 *   directional? · point[stride12] · spot[stride16] · rect-area[stride16]
 *     · mesh-triangle-area[stride28] · env?
 *
 * H51-D bumped the strides: point 8→12 (added distance/decay vec4),
 * spot 12→16 (added penumbra inner-cone cosine in slot 2.w + distance/decay vec4).
 *
 * Disc-area emitters are lowered by emitterPacking.ts into the mesh-triangle
 * section as an equal-area fan, so the stride walk still mirrors the GPU kernel
 * without a dedicated disc storage layout.
 *
 * That walk was open-coded FOUR times — `bdptEmitterPower`, `sampleBdptBounce0Cpu`
 * (both in `./bdptEmitterPickCpu.ts`) and `buildLightTreeInputForScene`
 * (`../scene/emitterPacking.ts`). The per-emitter-kind stride/offset arithmetic
 * MUST be byte-identical across all consumers because the flat emitter index feeds
 * the RNG-correlated power-weighted pick — any reorder or value drift would
 * silently de-correlate the CPU oracle from the GPU kernel.
 *
 * This module owns the POSITIONAL middle of the walk (point → spot → rect → mesh)
 * as a single generator yielding one record per packed light, in exact walk order.
 * The non-positional directional/env ENDS stay with each consumer because their
 * per-consumer semantics differ (power term vs. sampled vertex vs. light-tree leaf
 * with a union-AABB), but they bracket the SAME positional sequence this generator
 * produces, so the flat index alignment is single-sourced here.
 *
 * The packed arrays consumed here are exactly the `packEmitterArrays` output the
 * GPU uploads, so the strides below are the load-bearing layout
 * contract — they are NOT free to change without the matching WGSL change.
 */

export type Vec3 = [number, number, number];

/**
 * Float strides of the four positional packed-light arrays (floats per light).
 * H51-D: point 8→12 (position, radiance, [distance, decay, 0, 0]);
 *        spot  12→16 (position, dir+cosOuter, radiance+cosInner, [distance, decay, 0, 0]).
 */
export const POINT_LIGHT_STRIDE = 12;
export const SPOT_LIGHT_STRIDE = 16;
export const RECT_AREA_LIGHT_STRIDE = 16;
export const MESH_AREA_LIGHT_STRIDE = 28;

/** A single positional light, decoded from its packed stride layout. */
export type PositionalEmitter =
  | {
      readonly kind: 'point';
      readonly index: number;
      /** Light position. */
      readonly position: Vec3;
      readonly radiance: Vec3;
    }
  | {
      readonly kind: 'spot';
      readonly index: number;
      readonly position: Vec3;
      /** Spot axis as packed (NOT renormalized — consumers normalize as needed). */
      readonly axis: Vec3;
      readonly cosOuter: number;
      readonly radiance: Vec3;
      readonly cosInner: number;
    }
  | {
      readonly kind: 'rect';
      readonly index: number;
      readonly position: Vec3;
      readonly uAxis: Vec3;
      readonly vAxis: Vec3;
      readonly radiance: Vec3;
      /** Shape discriminator: 0.0 = rect (default), 1.0 = analytic disc.
       *  Packed in emission.w of the rect-area record. */
      readonly shapeTag: number;
    }
  | {
      readonly kind: 'mesh';
      readonly index: number;
      /** Triangle vertices A, B, C (world space). */
      readonly triA: Vec3;
      readonly triB: Vec3;
      readonly triC: Vec3;
      readonly radiance: Vec3;
    };

/** The packed positional-light arrays + their counts (subset of UploadedSceneBuffers). */
export interface PositionalEmitterArrays {
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly pointLightsData: ArrayLike<number>;
  readonly spotLightsData: ArrayLike<number>;
  readonly rectAreaLightsData: ArrayLike<number>;
  readonly meshAreaLightsData: ArrayLike<number>;
}

const v3 = (a: ArrayLike<number>, o: number): Vec3 => [a[o]!, a[o + 1]!, a[o + 2]!];

/**
 * Yield every positional selectable light in the EXACT pt-webgpu walk order:
 * point[stride12] → spot[stride16] → rect[stride16] → mesh[stride16].
 *
 * The directional slot precedes this sequence and the env slot follows it; those
 * are appended by callers because their per-consumer handling differs. The flat
 * index of the k-th yielded record is `directionalPresent ? k + 1 : k`.
 */
export function* walkPositionalEmitters(
  sb: PositionalEmitterArrays,
): Generator<PositionalEmitter> {
  for (let i = 0; i < sb.pointLightCount; i += 1) {
    // H51-D stride 12: position(0..2), _(3), radiance(4..6), _(7), [distance(8), decay(9), 0, 0]
    const o = i * POINT_LIGHT_STRIDE;
    yield {
      kind: 'point',
      index: i,
      position: v3(sb.pointLightsData, o),
      radiance: v3(sb.pointLightsData, o + 4),
    };
  }
  for (let i = 0; i < sb.spotLightCount; i += 1) {
    // H51-D stride 16: position(0..2), _(3), axis(4..6), cosOuter(7), radiance(8..10), cosInner(11), [dist(12), decay(13), 0, 0]
    const o = i * SPOT_LIGHT_STRIDE;
    yield {
      kind: 'spot',
      index: i,
      position: v3(sb.spotLightsData, o),
      axis: v3(sb.spotLightsData, o + 4),
      cosOuter: sb.spotLightsData[o + 7] ?? 0,
      radiance: v3(sb.spotLightsData, o + 8),
      cosInner: sb.spotLightsData[o + 11] ?? 0,
    };
  }
  for (let i = 0; i < sb.rectAreaLightCount; i += 1) {
    const o = i * RECT_AREA_LIGHT_STRIDE;
    yield {
      kind: 'rect',
      index: i,
      position: v3(sb.rectAreaLightsData, o),
      uAxis: v3(sb.rectAreaLightsData, o + 4),
      vAxis: v3(sb.rectAreaLightsData, o + 8),
      radiance: v3(sb.rectAreaLightsData, o + 12),
      // Shape discriminator packed in emission.w: 0.0 = rect, 1.0 = disc.
      shapeTag: sb.rectAreaLightsData[o + 15] ?? 0,
    };
  }
  for (let i = 0; i < sb.meshAreaLightCount; i += 1) {
    const o = i * MESH_AREA_LIGHT_STRIDE;
    yield {
      kind: 'mesh',
      index: i,
      triA: v3(sb.meshAreaLightsData, o),
      triB: v3(sb.meshAreaLightsData, o + 4),
      triC: v3(sb.meshAreaLightsData, o + 8),
      radiance: v3(sb.meshAreaLightsData, o + 12),
    };
  }
}

/** Quad area = 4·|u×v| (matches the WGSL rect-area NEE term for rect lights). */
export function rectQuadArea(uAxis: Vec3, vAxis: Vec3): number {
  const cross: Vec3 = [
    uAxis[1] * vAxis[2] - uAxis[2] * vAxis[1],
    uAxis[2] * vAxis[0] - uAxis[0] * vAxis[2],
    uAxis[0] * vAxis[1] - uAxis[1] * vAxis[0],
  ];
  return 4 * Math.hypot(cross[0], cross[1], cross[2]);
}

/**
 * Disc area = π·|uAxis|² (matches the WGSL disc NEE term).
 * uAxis carries tangent × radius so |uAxis| = radius.
 */
export function discArea(uAxis: Vec3): number {
  const r = Math.hypot(uAxis[0], uAxis[1], uAxis[2]);
  return Math.PI * r * r;
}

/** Triangle area = 0.5·|(B−A)×(C−A)| (matches the WGSL mesh-area NEE term). */
export function meshTriangleArea(a: Vec3, b: Vec3, c: Vec3): number {
  const ab: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const cross: Vec3 = [
    ab[1] * ac[2] - ab[2] * ac[1],
    ab[2] * ac[0] - ab[0] * ac[2],
    ab[0] * ac[1] - ab[1] * ac[0],
  ];
  return 0.5 * Math.hypot(cross[0], cross[1], cross[2]);
}

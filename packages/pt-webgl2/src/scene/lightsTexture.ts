// Lights importance-sampling packer — THREE-free port of the fork's
// `LightsInfoUniformStruct.js` (three-gpu-pathtracer) onto the @vitrum/core
// `SceneEmitter` contract. Reproduces the verified 6-texel-per-light RGBA32F
// layout that the kept GLSL decoder (`structs/lights_struct.glsl.js`,
// `readLightInfo`) reads via `texelFetch1D`.
//
// Provenance: gkjohnson/three-gpu-pathtracer LightsInfoUniformStruct.js (MIT);
// see CREDITS.md. Layout spec: plan/three-removal/03-scene-bvh-packers.md §5.
//
// Per-light texel layout (24 floats):
//   s0 = (worldPos.xyz, type)
//   s1 = (color.rgb, intensity)
//   s2 = (u-vector.xyz, power)
//   s3 = (v-vector.xyz, area)
//   s4 = (reserved 0, decay, distance, coneCos)    — spot/point only
//   s5 = (penumbraCos, castShadowDisabled, angularDiameter, 0)
//        — s5.r spot only; s5.g ALL kinds; s5.b directional only
//        (SHADOW-01: s5.g — the former IES padding slot — carries 1.0 when the
//        emitter sets castShadow:false, 0.0 default; byte-identical for default
//        scenes. Consumed by readLightInfo → LightRecord.castShadowDisabled →
//        directLightContribution's analytic-light shadow-test skip.)
//        Directional angularDiameter is stored in s5.b; default/no-spread stays 0
//        so legacy hard directional scenes remain byte-identical.
//
// Light types (must match the GLSL `#define`s):
//   RECT_AREA = 0, CIRC_AREA = 1, SPOT = 2, DIR = 3, POINT = 4

import type { SceneEmitter, Vec3 } from '@vitrum/core';
import {
  luminance,
  vecNormalize as normalize,
  tangentBasis,
} from '@vitrum/shared-samplers';
import {
  classifyAreaVectorF32,
  normalizeDirectionF32,
} from './areaEmitterGeometry.js';
import {
  WEBGL2_F32_MIN_NORMAL,
  multiplyNonNegativeFloat32,
  requireFiniteFloat32,
  requireNonNegativeFloat32,
} from './float32Policy.js';
import type { LightsTextureData } from './sceneTextures.js';

// ── D10.10: dev-only slot-cursor guard ────────────────────────────────────────
// assertSlotCursor(k, expected, kind) throws at runtime (dev / test builds only)
// when the write cursor k does not match the expected channel count after packing
// a light of the given kind. This catches packing bugs (off-by-one, missed fields,
// wrong stride) that would otherwise silently produce wrong GPU data.
//
// Gated to non-production builds (import.meta.env.DEV || process.env.NODE_ENV !== 'production').
// Always active in the test harness (NODE_ENV=test, not production).
const _DEV_ASSERT = /* @__PURE__ */ (() => {
  const g = globalThis as Record<string, unknown>;
  try {
    if (g['__vitest_environment__'] != null) return true;
  } catch { /* noop */ }
  try {
    const proc = g['process'] as { env?: Record<string, string | undefined> } | undefined;
    return proc?.env?.['NODE_ENV'] !== 'production';
  } catch { return true; }
})();

function assertSlotCursor(k: number, expected: number, kind: string): void {
  if (_DEV_ASSERT && k !== expected) {
    throw new Error(
      `[pt-webgl2 lightsTexture] packing bug for '${kind}': expected slot cursor ${expected}, got ${k}. ` +
      'Check the RGBA32F layout vs §5 (6 texels × 4 channels = 24 channels per light).',
    );
  }
}

/** Texels per light in the packed RGBA32F grid (24 floats). Matches the GLSL
 *  decoder's `index * 6u` stride (`lights_struct.glsl.js`). */
export const LIGHT_PIXELS = 6;

const RECT_AREA_LIGHT = 0;
const CIRC_AREA_LIGHT = 1;
const SPOT_LIGHT = 2;
const DIR_LIGHT = 3;
const POINT_LIGHT = 4;

function selectionPowerF32(value: number, context: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${context} selection power must be finite and non-negative.`);
  }
  if (value === 0) return 0;
  const stored = Math.fround(value);
  if (!(stored > 0) || !Number.isFinite(stored)) {
    throw new RangeError(
      `${context} selection power must survive finite positive RGBA32F storage.`,
    );
  }
  return stored;
}

function addSelectionPowerF32(sum: number, value: number): number {
  const next = Math.fround(sum + value);
  if (!Number.isFinite(next)) {
    throw new RangeError(
      '@vitrum/pt-webgl2: analytic-light selection-power sum overflows float32.',
    );
  }
  if (value > 0 && next === sum) {
    throw new RangeError(
      '@vitrum/pt-webgl2: a positive analytic-light selection power loses ' +
        'proposal support in the cumulative float32 distribution.',
    );
  }
  return next;
}

function finiteVec3F32(value: Vec3, context: string): Vec3 {
  return [
    requireFiniteFloat32(value[0], `${context}[0]`),
    requireFiniteFloat32(value[1], `${context}[1]`),
    requireFiniteFloat32(value[2], `${context}[2]`),
  ];
}

function nonNegativeVec3F32(value: Vec3, context: string): Vec3 {
  return [
    requireNonNegativeFloat32(value[0], `${context}[0]`),
    requireNonNegativeFloat32(value[1], `${context}[1]`),
    requireNonNegativeFloat32(value[2], `${context}[2]`),
  ];
}

interface PackedLightRadiometry {
  readonly color: Vec3;
  readonly intensity: number;
  readonly radiance: Vec3;
}

interface PackedSpotCone {
  readonly coneCos: number;
  readonly penumbraCos: number;
}

/**
 * Mirror the GLSL source-radiance operand order:
 * `light.color * light.intensity`. Both operands first cross separate RGBA32F
 * lanes, then each component multiplication is evaluated in shader f32.
 */
function packLightRadiometry(
  colorSource: Vec3,
  intensitySource: number,
  context: string,
): PackedLightRadiometry {
  const color = nonNegativeVec3F32(colorSource, `${context} color`);
  const intensity = requireNonNegativeFloat32(
    intensitySource,
    `${context} intensity`,
  );
  const radiance: Vec3 = [
    multiplyNonNegativeFloat32(
      color[0],
      intensity,
      `${context} color[0] * intensity`,
    ),
    multiplyNonNegativeFloat32(
      color[1],
      intensity,
      `${context} color[1] * intensity`,
    ),
    multiplyNonNegativeFloat32(
      color[2],
      intensity,
      `${context} color[2] * intensity`,
    ),
  ];
  return { color, intensity, radiance };
}

/**
 * Pack the exact cosine lanes consumed by GLSL spot attenuation. A zero
 * penumbra intentionally produces equal edges and selects the shader's defined
 * hard-cone branch. Positive authored cone/penumbra support must not collapse
 * to that branch merely because the derived cosines share one f32 value.
 */
function packSpotCone(
  angle: number,
  penumbra: number,
  context: string,
): PackedSpotCone {
  if (!Number.isFinite(angle) || !(angle > 0) || angle > Math.PI) {
    throw new RangeError(
      `${context} angle must be finite, positive, and <= PI.`,
    );
  }
  if (!Number.isFinite(penumbra) || penumbra < 0 || penumbra > 1) {
    throw new RangeError(
      `${context} penumbra must be finite and in [0, 1].`,
    );
  }
  const coneCos = requireFiniteFloat32(
    Math.cos(angle),
    `${context} cone cosine`,
  );
  const penumbraCos = requireFiniteFloat32(
    Math.cos(angle * (1 - penumbra)),
    `${context} penumbra cosine`,
  );
  if (coneCos === 1) {
    throw new RangeError(
      `${context} positive angle collapses to a zero-width cone after ` +
      'WebGL float32 cosine storage.',
    );
  }
  if (penumbra > 0 && penumbraCos === coneCos) {
    throw new RangeError(
      `${context} positive penumbra collapses to the hard-cone edge after ` +
      'WebGL float32 cosine storage.',
    );
  }
  return { coneCos, penumbraCos };
}

const WEBGL2_F32_EPSILON_AT_ONE = 2 ** -23;
const DIRECTIONAL_BASIS_ADD_GUARD_ULPS = 8;
const DIRECTIONAL_MIN_RIM_COMPONENT =
  DIRECTIONAL_BASIS_ADD_GUARD_ULPS * WEBGL2_F32_EPSILON_AT_ONE;
const DIRECTIONAL_MIN_ANGULAR_DIAMETER = 2 * Math.asin(
  Math.sqrt(3) * DIRECTIONAL_MIN_RIM_COMPONENT,
);

/**
 * A positive directional diameter is a finite-cone contract, not a hint that
 * may silently collapse to the hard/delta branch. Validate the exact f32
 * intermediates used by `sampleDirectionalCone`: its stable
 * `1-cos(d/2) = 2*sin²(d/4)` term must remain at least normal, and the
 * reciprocal solid-angle PDF must remain finite.
 */
function packDirectionalAngularDiameter(
  value: number | undefined,
  context: string,
): number {
  if (value == null || value === 0) return 0;
  if (!Number.isFinite(value) || value < 0 || value > Math.PI) {
    throw new RangeError(
      `${context} angularDiameter must be finite and in [0, PI].`,
    );
  }
  const stored = requireNonNegativeFloat32(
    value,
    `${context} angularDiameter`,
  );
  const quarterAngle = Math.fround(stored * 0.25);
  const sinQuarter = Math.fround(
    quarterAngle < 1e-3 ? quarterAngle : Math.sin(quarterAngle),
  );
  const sinQuarterSquared = Math.fround(sinQuarter * sinQuarter);
  const oneMinusCosHalf = Math.fround(2 * sinQuarterSquared);
  const solidAngle = Math.fround(
    Math.fround(2 * Math.PI) * oneMinusCosHalf,
  );
  const pdf = Math.fround(1 / solidAngle);
  if (
    !(oneMinusCosHalf >= WEBGL2_F32_MIN_NORMAL) ||
    !(solidAngle > 0) ||
    !Number.isFinite(solidAngle) ||
    !(pdf > 0) ||
    !Number.isFinite(pdf)
  ) {
    throw new RangeError(
      `${context} angularDiameter is too small to retain a finite-cone ` +
      'solid angle and PDF in WebGL float32; use 0 for a delta directional light.',
    );
  }
  // For every unit transverse rim vector, at least one world component has
  // magnitude >= sin(d/2)/sqrt(3). Require that guaranteed component to clear
  // eight binary32 ULPs at unit magnitude, covering basis construction,
  // matrix addition, and final normalization. Otherwise a valid local cone can
  // quantize back to its axis after the world-basis transform.
  const rimSin = Math.fround(Math.sin(Math.fround(stored * 0.5)));
  const guaranteedWorldComponent = Math.fround(rimSin / Math.sqrt(3));
  if (!(guaranteedWorldComponent >= DIRECTIONAL_MIN_RIM_COMPONENT)) {
    throw new RangeError(
      `${context} angularDiameter is too small to survive world-basis ` +
      `float32 addition; use 0 or at least ${DIRECTIONAL_MIN_ANGULAR_DIAMETER} radians.`,
    );
  }
  return stored;
}

// Rec.709 luminance + normalize/tangentBasis are single-sourced in
// `@vitrum/shared-samplers`. The shared `vecNormalize` preserves this packer's
// historical `<1e-12 → [0,0,0]` degeneracy contract, and the shared `luminance`
// uses the same Rec.709 coefficients.

interface PackedAreaGeometry {
  readonly u: Vec3;
  readonly v: Vec3;
  readonly area: number;
}

function discAreaTangentBasis(normal: Vec3, id: string): {
  readonly t: Vec3;
  readonly b: Vec3;
} {
  const n = normalizeDirectionF32(normal);
  if (n == null) {
    throw new RangeError(
      `@vitrum/pt-webgl2: disc-area emitter "${id}" has a degenerate normal.`,
    );
  }
  const ref: Vec3 = Math.abs(n[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const t = normalizeDirectionF32([
    ref[1] * n[2] - ref[2] * n[1],
    ref[2] * n[0] - ref[0] * n[2],
    ref[0] * n[1] - ref[1] * n[0],
  ]);
  if (t == null) {
    throw new RangeError(
      `@vitrum/pt-webgl2: disc-area emitter "${id}" has a degenerate tangent basis.`,
    );
  }
  const b = normalizeDirectionF32([
    n[1] * t[2] - n[2] * t[1],
    n[2] * t[0] - n[0] * t[2],
    n[0] * t[1] - n[1] * t[0],
  ]);
  if (b == null) {
    throw new RangeError(
      `@vitrum/pt-webgl2: disc-area emitter "${id}" has a degenerate bitangent basis.`,
    );
  }
  return { t, b };
}

/**
 * Quantize the exact full-span axes consumed by GLSL before deriving any
 * geometry-dependent fields. Core validation operates on authored JS numbers,
 * while the light texture is RGBA32F; an otherwise-valid pair can therefore
 * overflow or become collinear only at this backend boundary.
 */
function packAreaGeometry(
  kind: 'rect-area' | 'disc-area',
  id: string,
  sourceU: Vec3,
  sourceV: Vec3,
  areaScale: number,
): PackedAreaGeometry {
  const u: Vec3 = [
    Math.fround(sourceU[0]),
    Math.fround(sourceU[1]),
    Math.fround(sourceU[2]),
  ];
  const v: Vec3 = [
    Math.fround(sourceV[0]),
    Math.fround(sourceV[1]),
    Math.fround(sourceV[2]),
  ];
  const areaMeasure = classifyAreaVectorF32(u, v, areaScale);
  if (!areaMeasure.valid) {
    throw new RangeError(
      `@vitrum/pt-webgl2: ${kind} emitter "${id}" full-span axes do not retain ` +
      `strictly positive finite area in float32 shader arithmetic (${areaMeasure.reason}).`,
    );
  }
  return { u, v, area: areaMeasure.area };
}

// ── D11-12: shared s0/s1 header helpers ───────────────────────────────────────
// Every light kind writes the same two leading texels:
//   s0 = (worldPos.xyz, type)   s1 = (color.rgb, intensity)
// Previously each of the 5 arms re-spelled these 8 `data[base + k++] = …` writes
// with a fragile hand-advanced `k` cursor. These helpers own the s0/s1 writes and
// advance an explicit cursor object so the header can never drift per-kind. The
// `assertSlotCursor` guards downstream still pin the final per-kind channel count.

interface SlotCursor {
  k: number;
}

/** Write s0 = (position.xyz, type) at `base + cursor.k`, advancing the cursor by 4. */
function writePositionType(
  data: Float32Array,
  base: number,
  cursor: SlotCursor,
  position: Vec3,
  type: number,
): void {
  data[base + cursor.k++] = position[0];
  data[base + cursor.k++] = position[1];
  data[base + cursor.k++] = position[2];
  data[base + cursor.k++] = type;
}

/** Write s1 = (color.rgb, intensity) at `base + cursor.k`, advancing the cursor by 4. */
function writeColorIntensity(
  data: Float32Array,
  base: number,
  cursor: SlotCursor,
  color: Vec3,
  intensity: number,
): void {
  data[base + cursor.k++] = color[0];
  data[base + cursor.k++] = color[1];
  data[base + cursor.k++] = color[2];
  data[base + cursor.k++] = intensity;
}

/**
 * Pack the scene's emitters into the 6-texel-per-light RGBA32F square grid the
 * WebGL2 path tracer reads. Mirrors the fork's `LightsInfoUniformStruct.updateFrom`
 * byte layout exactly (§5), driven from @vitrum/core `SceneEmitter`s rather than
 * THREE light objects.
 *
 * Unmappable / approximated fields vs the THREE source (documented for the caller):
 *  - IES profiles: removed (not in @vitrum/core contract). s5.g is reserved padding.
 *  - Spot emitters are delta-position sources. The inherited radius/area lanes
 *    remain reserved zeroes; finite source extent is represented by rect/disc.
 *  - `disc-area` (CIRC_AREA): core gives centre+normal+radius; the in-plane
 *    (u, v) axes are synthesized from a deterministic tangent basis, each
 *    scaled by `2*radius` (full diameter, matching the fork's full-extent
 *    width/height convention) before the π/4 circular correction.
 *  - `mesh-area`: emissive geometry is sampled via the BVH/material path, not
 *    the analytic light list → skipped here (not representable in this 6px slot).
 */
export function packLightsTexture(
  emitters: readonly SceneEmitter[],
): LightsTextureData {
  // mesh-area emitters are not analytic lights in this layout — they are handled
  // by emissive-geometry sampling. Filter them out of the analytic light list.
  const lights = emitters.filter((e) => e.kind !== 'mesh-area');

  const pixelCount = Math.max(lights.length * LIGHT_PIXELS, 1);
  const dim = Math.ceil(Math.sqrt(pixelCount));
  const data = new Float32Array(dim * dim * 4);
  let selectionPowerSum = 0;
  const selectionPowers: number[] = [];

  for (let i = 0; i < lights.length; i += 1) {
    const l = lights[i]!;
    const base = i * LIGHT_PIXELS * 4;
    const cursor: SlotCursor = { k: 0 };

    const lightContext = `${l.kind} emitter "${l.id}"`;
    const { color, intensity, radiance } = packLightRadiometry(
      l.color,
      l.intensity,
      lightContext,
    );
    const lum = luminance(radiance[0], radiance[1], radiance[2]);
    let packedSelectionPower = 0;

    switch (l.kind) {
      case 'rect-area': {
        // s0: position / type   s1: color / intensity
        writePositionType(
          data,
          base,
          cursor,
          finiteVec3F32(l.position, `${lightContext} position`),
          RECT_AREA_LIGHT,
        );
        writeColorIntensity(data, base, cursor, color, intensity);
        // Core uAxis/vAxis are HALF-extents, while the retained WebGL2 light
        // sampler/intersector uses the fork's FULL-span convention with
        // coordinates in [-0.5, 0.5]. Double both axes at this backend boundary
        // so the sampled surface is p ± uAxis ± vAxis and its area is
        // 4·|uAxis×vAxis|, matching the core contract and pt-webgpu.
        const { u, v, area } = packAreaGeometry(
          'rect-area',
          l.id,
          [2 * l.uAxis[0], 2 * l.uAxis[1], 2 * l.uAxis[2]],
          [2 * l.vAxis[0], 2 * l.vAxis[1], 2 * l.vAxis[2]],
          1,
        );
        // s2: u / power
        data[base + cursor.k++] = u[0];
        data[base + cursor.k++] = u[1];
        data[base + cursor.k++] = u[2];
        packedSelectionPower = selectionPowerF32(
          lum * area,
          `rect-area emitter "${l.id}"`,
        );
        data[base + cursor.k++] = packedSelectionPower;
        // s3: v / area  (full-span area = |u × v|)
        data[base + cursor.k++] = v[0];
        data[base + cursor.k++] = v[1];
        data[base + cursor.k++] = v[2];
        data[base + cursor.k++] = area;
        // rect-area packs s0..s3 (16 channels); s4..s5 stay zero (no cone/radius).
        assertSlotCursor(cursor.k, 16, 'rect-area');
        break;
      }
      case 'disc-area': {
        // s0: position / type (CIRC_AREA)   s1: color / intensity
        writePositionType(
          data,
          base,
          cursor,
          finiteVec3F32(l.position, `${lightContext} position`),
          CIRC_AREA_LIGHT,
        );
        writeColorIntensity(data, base, cursor, color, intensity);
        // Synthesize an in-plane (u, v) basis from the disc normal; each axis
        // spans the full diameter (2*radius) to match the fork's full-extent
        // width/height convention, then π/4 corrects rectangle→disc.
        const { t, b } = discAreaTangentBasis(l.normal, l.id);
        const d = 2 * l.radius;
        const areaScale = Math.PI / 4.0;
        const { u, v, area } = packAreaGeometry(
          'disc-area',
          l.id,
          [t[0] * d, t[1] * d, t[2] * d],
          [b[0] * d, b[1] * d, b[2] * d],
          areaScale,
        );
        // s2: u / power
        data[base + cursor.k++] = u[0];
        data[base + cursor.k++] = u[1];
        data[base + cursor.k++] = u[2];
        packedSelectionPower = selectionPowerF32(
          lum * area,
          `disc-area emitter "${l.id}"`,
        );
        data[base + cursor.k++] = packedSelectionPower;
        // s3: v / area
        data[base + cursor.k++] = v[0];
        data[base + cursor.k++] = v[1];
        data[base + cursor.k++] = v[2];
        data[base + cursor.k++] = area;
        // disc-area packs s0..s3 (16 channels); s4..s5 stay zero (no cone/radius).
        assertSlotCursor(cursor.k, 16, 'disc-area');
        break;
      }
      case 'spot': {
        // s0: position / type   s1: color / intensity
        writePositionType(
          data,
          base,
          cursor,
          finiteVec3F32(l.position, `${lightContext} position`),
          SPOT_LIGHT,
        );
        writeColorIntensity(data, base, cursor, color, intensity);
        const penumbra = l.penumbra ?? 0;
        const { coneCos, penumbraCos } = packSpotCone(
          l.angle,
          penumbra,
          lightContext,
        );
        // The fork builds a lookAt basis: u = (1,0,0), v = (0,1,0) rotated into
        // the light's orientation. Core gives the forward `direction`; we build
        // a basis whose -w is the spot direction and read its u/v axes.
        const forward = finiteVec3F32(
          normalize(l.direction),
          `${lightContext} normalized direction`,
        );
        // The GLSL/fork convention recovers the spot's BACK axis as cross(u,v)
        // and negates it to obtain the forward emission direction. Build the
        // plane around -forward so both NEE cone attenuation and BDPT emission
        // point along the authored core direction.
        const basis = tangentBasis([-forward[0], -forward[1], -forward[2]]);
        const t = finiteVec3F32(basis.t, `${lightContext} tangent`);
        const b = finiteVec3F32(basis.b, `${lightContext} bitangent`);
        // s2: u / power
        data[base + cursor.k++] = t[0];
        data[base + cursor.k++] = t[1];
        data[base + cursor.k++] = t[2];
        packedSelectionPower = selectionPowerF32(
          lum,
          `spot emitter "${l.id}"`,
        );
        data[base + cursor.k++] = packedSelectionPower;
        // s3: v / reserved area (zero for the delta-position spot contract).
        data[base + cursor.k++] = b[0];
        data[base + cursor.k++] = b[1];
        data[base + cursor.k++] = b[2];
        cursor.k += 1;
        // s4: reserved radius / decay / distance / coneCos
        cursor.k += 1;
        data[base + cursor.k++] = requireNonNegativeFloat32(
          l.decay ?? 2,
          `${lightContext} decay`,
        );
        data[base + cursor.k++] = requireNonNegativeFloat32(
          l.distance ?? 0,
          `${lightContext} distance`,
        );
        data[base + cursor.k++] = coneCos;
        // s5: penumbraCos / castShadowDisabled / 0 / 0. The shared shadow
        // flag is written after the kind-specific switch below.
        data[base + cursor.k++] = penumbraCos;
        cursor.k += 1;
        // spot packs through s5.g, including reserved zero area/radius lanes.
        assertSlotCursor(cursor.k, 22, 'spot');
        break;
      }
      case 'point': {
        // s0: position / type   s1: color / intensity
        const position = finiteVec3F32(
          l.position,
          `${lightContext} position`,
        );
        writePositionType(data, base, cursor, position, POINT_LIGHT);
        writeColorIntensity(data, base, cursor, color, intensity);
        // s2: u = world position again / power (fork stores worldPosition here)
        data[base + cursor.k++] = position[0];
        data[base + cursor.k++] = position[1];
        data[base + cursor.k++] = position[2];
        packedSelectionPower = selectionPowerF32(
          lum,
          `point emitter "${l.id}"`,
        );
        data[base + cursor.k++] = packedSelectionPower;
        // s3 unused (zero); cursor advances over it
        cursor.k += 4;
        // s4: radius=0 / decay / distance / coneCos=0
        cursor.k += 1; // radius slot (0)
        data[base + cursor.k++] = requireNonNegativeFloat32(
          l.decay ?? 2,
          `${lightContext} decay`,
        );
        data[base + cursor.k++] = requireNonNegativeFloat32(
          l.distance ?? 0,
          `${lightContext} distance`,
        );
        // s4.a coneCos + s5 stay 0 — point lights are isotropic.
        // point packs: s0(4) + s1(4) + s2(4) + s3(skip4) + s4.r(skip1) + s4.g/s4.b(2) = 19 channels.
        assertSlotCursor(cursor.k, 19, 'point');
        break;
      }
      case 'directional': {
        // s0: position / type. Directional lights have no position; the fork
        // packs the world position (used only as an origin offset). Core gives
        // a direction only → store origin (0,0,0).
        writePositionType(data, base, cursor, [0, 0, 0], DIR_LIGHT);
        // s1: color / intensity
        writeColorIntensity(data, base, cursor, color, intensity);
        // s2: u = direction TOWARD the light / power. Core `direction` is the
        // unit vector pointing AT the light (matches the fork's
        // normalize(worldPos - target)).
        const dir = finiteVec3F32(
          normalize(l.direction),
          `${lightContext} normalized direction`,
        );
        data[base + cursor.k++] = dir[0];
        data[base + cursor.k++] = dir[1];
        data[base + cursor.k++] = dir[2];
        packedSelectionPower = selectionPowerF32(
          lum,
          `directional emitter "${l.id}"`,
        );
        data[base + cursor.k++] = packedSelectionPower;
        const angularDiameter = packDirectionalAngularDiameter(
          l.angularDiameter,
          lightContext,
        );
        data[base + 22] = angularDiameter;
        // directional packs s0(4) + s1(4) + s2(4) + s5.b(angularDiameter).
        assertSlotCursor(cursor.k, 12, 'directional');
        break;
      }
      default: {
        // Exhaustiveness guard — `mesh-area` already filtered out above.
        const _never: never = l;
        void _never;
      }
    }

    selectionPowerSum = addSelectionPowerF32(
      selectionPowerSum,
      packedSelectionPower,
    );
    selectionPowers.push(packedSelectionPower);

    // SHADOW-01 — s5.g (channel 21, the former IES padding slot) carries the
    // emitter castShadowDisabled flag for EVERY light kind. Default (castShadow
    // true/undefined) writes 0.0 — byte-identical to the pre-SHADOW-01 grid.
    data[base + 21] = l.castShadow === false ? 1 : 0;
  }

  for (const power of selectionPowers) {
    if (
      power > 0 &&
      !(Math.fround(power / selectionPowerSum) > 0)
    ) {
      throw new RangeError(
        '@vitrum/pt-webgl2: a positive analytic-light selection probability ' +
          'collapses to zero in float32.',
      );
    }
  }

  return { data, dim, kind: 'rgba32f', lightCount: lights.length };
}

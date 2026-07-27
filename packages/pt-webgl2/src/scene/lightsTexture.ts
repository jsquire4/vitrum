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
  vecCross as cross,
  vecLength as lengthOf,
  vecNormalize as normalize,
  tangentBasis,
} from '@vitrum/shared-samplers';
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

// Rec.709 luminance + the cross/normalize/tangentBasis/length Vec3 helpers are
// single-sourced in `@vitrum/shared-samplers` (imported above under the local
// aliases `luminance`/`cross`/`lengthOf`/`normalize`/`tangentBasis`). The
// shared `vecNormalize` preserves this packer's historical `<1e-12 → [0,0,0]`
// degeneracy contract, and the shared `luminance` uses the same Rec.709
// coefficients, so the packed `power` field stays byte-for-byte identical.

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

  for (let i = 0; i < lights.length; i += 1) {
    const l = lights[i]!;
    const base = i * LIGHT_PIXELS * 4;
    const cursor: SlotCursor = { k: 0 };

    const [cr, cg, cb] = l.color;
    const color: Vec3 = [cr, cg, cb];
    const lum = luminance(cr, cg, cb);

    switch (l.kind) {
      case 'rect-area': {
        // s0: position / type   s1: color / intensity
        writePositionType(data, base, cursor, l.position, RECT_AREA_LIGHT);
        writeColorIntensity(data, base, cursor, color, l.intensity);
        // u-vector + power. Core uAxis/vAxis are the in-plane span vectors; the
        // fork's width/height are their lengths.
        const u = l.uAxis;
        const v = l.vAxis;
        const width = lengthOf(u);
        const height = lengthOf(v);
        // s2: u / power
        data[base + cursor.k++] = u[0];
        data[base + cursor.k++] = u[1];
        data[base + cursor.k++] = u[2];
        data[base + cursor.k++] = lum * l.intensity * (width * height);
        // s3: v / area  (area = |u × v|)
        data[base + cursor.k++] = v[0];
        data[base + cursor.k++] = v[1];
        data[base + cursor.k++] = v[2];
        data[base + cursor.k++] = lengthOf(cross(u, v));
        // rect-area packs s0..s3 (16 channels); s4..s5 stay zero (no cone/radius).
        assertSlotCursor(cursor.k, 16, 'rect-area');
        break;
      }
      case 'disc-area': {
        // s0: position / type (CIRC_AREA)   s1: color / intensity
        writePositionType(data, base, cursor, l.position, CIRC_AREA_LIGHT);
        writeColorIntensity(data, base, cursor, color, l.intensity);
        // Synthesize an in-plane (u, v) basis from the disc normal; each axis
        // spans the full diameter (2*radius) to match the fork's full-extent
        // width/height convention, then π/4 corrects rectangle→disc.
        const { t, b } = tangentBasis(l.normal);
        const d = 2 * l.radius;
        const u: Vec3 = [t[0] * d, t[1] * d, t[2] * d];
        const v: Vec3 = [b[0] * d, b[1] * d, b[2] * d];
        const areaScale = Math.PI / 4.0;
        // s2: u / power
        data[base + cursor.k++] = u[0];
        data[base + cursor.k++] = u[1];
        data[base + cursor.k++] = u[2];
        data[base + cursor.k++] = lum * l.intensity * (d * d) * areaScale;
        // s3: v / area
        data[base + cursor.k++] = v[0];
        data[base + cursor.k++] = v[1];
        data[base + cursor.k++] = v[2];
        data[base + cursor.k++] = lengthOf(cross(u, v)) * areaScale;
        // disc-area packs s0..s3 (16 channels); s4..s5 stay zero (no cone/radius).
        assertSlotCursor(cursor.k, 16, 'disc-area');
        break;
      }
      case 'spot': {
        // s0: position / type   s1: color / intensity
        writePositionType(data, base, cursor, l.position, SPOT_LIGHT);
        writeColorIntensity(data, base, cursor, color, l.intensity);
        // The fork builds a lookAt basis: u = (1,0,0), v = (0,1,0) rotated into
        // the light's orientation. Core gives the forward `direction`; we build
        // a basis whose -w is the spot direction and read its u/v axes.
        const forward = normalize(l.direction);
        // The GLSL/fork convention recovers the spot's BACK axis as cross(u,v)
        // and negates it to obtain the forward emission direction. Build the
        // plane around -forward so both NEE cone attenuation and BDPT emission
        // point along the authored core direction.
        const { t, b } = tangentBasis([-forward[0], -forward[1], -forward[2]]);
        // s2: u / power
        data[base + cursor.k++] = t[0];
        data[base + cursor.k++] = t[1];
        data[base + cursor.k++] = t[2];
        data[base + cursor.k++] = lum * l.intensity;
        // s3: v / reserved area (zero for the delta-position spot contract).
        data[base + cursor.k++] = b[0];
        data[base + cursor.k++] = b[1];
        data[base + cursor.k++] = b[2];
        cursor.k += 1;
        // s4: reserved radius / decay / distance / coneCos
        cursor.k += 1;
        data[base + cursor.k++] = l.decay ?? 2;
        data[base + cursor.k++] = l.distance ?? 0;
        data[base + cursor.k++] = Math.cos(l.angle);
        // s5: penumbraCos / castShadowDisabled / 0 / 0. The shared shadow
        // flag is written after the kind-specific switch below.
        data[base + cursor.k++] = Math.cos(l.angle * (1 - (l.penumbra ?? 0)));
        cursor.k += 1;
        // spot packs through s5.g, including reserved zero area/radius lanes.
        assertSlotCursor(cursor.k, 22, 'spot');
        break;
      }
      case 'point': {
        // s0: position / type   s1: color / intensity
        writePositionType(data, base, cursor, l.position, POINT_LIGHT);
        writeColorIntensity(data, base, cursor, color, l.intensity);
        // s2: u = world position again / power (fork stores worldPosition here)
        data[base + cursor.k++] = l.position[0];
        data[base + cursor.k++] = l.position[1];
        data[base + cursor.k++] = l.position[2];
        data[base + cursor.k++] = lum * l.intensity;
        // s3 unused (zero); cursor advances over it
        cursor.k += 4;
        // s4: radius=0 / decay / distance / coneCos=0
        cursor.k += 1; // radius slot (0)
        data[base + cursor.k++] = l.decay ?? 2;
        data[base + cursor.k++] = l.distance ?? 0;
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
        writeColorIntensity(data, base, cursor, color, l.intensity);
        // s2: u = direction TOWARD the light / power. Core `direction` is the
        // unit vector pointing AT the light (matches the fork's
        // normalize(worldPos - target)).
        const dir = normalize(l.direction);
        data[base + cursor.k++] = dir[0];
        data[base + cursor.k++] = dir[1];
        data[base + cursor.k++] = dir[2];
        data[base + cursor.k++] = lum * l.intensity;
        const angularDiameter =
          l.angularDiameter != null && Number.isFinite(l.angularDiameter) && l.angularDiameter > 0
            ? l.angularDiameter
            : 0;
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

    // SHADOW-01 — s5.g (channel 21, the former IES padding slot) carries the
    // emitter castShadowDisabled flag for EVERY light kind. Default (castShadow
    // true/undefined) writes 0.0 — byte-identical to the pre-SHADOW-01 grid.
    data[base + 21] = l.castShadow === false ? 1 : 0;
  }

  return { data, dim, kind: 'rgba32f', lightCount: lights.length };
}

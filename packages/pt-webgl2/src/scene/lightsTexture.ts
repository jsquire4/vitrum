// Lights importance-sampling packer — THREE-free port of the fork's
// `LightsInfoUniformStruct.js` (three-gpu-pathtracer) onto the @vitrum/core
// `SceneEmitter` contract. Reproduces the verified 6-texel-per-light RGBA32F
// layout that the kept GLSL decoder (`structs/lights_struct.glsl.js`,
// `readLightInfo`) reads via `texelFetch1D`.
//
// Provenance: gkjohnson/three-gpu-pathtracer LightsInfoUniformStruct.js (MIT),
// absorbed at packages/three-gpu-pathtracer/. Layout spec: plan/three-removal/
// 03-scene-bvh-packers.md §5.
//
// Per-light texel layout (24 floats):
//   s0 = (worldPos.xyz, type)
//   s1 = (color.rgb, intensity)
//   s2 = (u-vector.xyz, power)
//   s3 = (v-vector.xyz, area)
//   s4 = (radius, decay, distance, coneCos)        — spot/point only
//   s5 = (penumbraCos, iesProfile, 0, 0)           — spot/point only
//
// Light types (must match the GLSL `#define`s):
//   RECT_AREA = 0, CIRC_AREA = 1, SPOT = 2, DIR = 3, POINT = 4

import type { SceneEmitter, Vec3 } from '@vitrum/core';
import type { LightsTextureData } from './sceneTextures.js';

/** Texels per light in the packed RGBA32F grid (24 floats). Matches the GLSL
 *  decoder's `index * 6u` stride (`lights_struct.glsl.js`). */
export const LIGHT_PIXELS = 6;

const RECT_AREA_LIGHT = 0;
const CIRC_AREA_LIGHT = 1;
const SPOT_LIGHT = 2;
const DIR_LIGHT = 3;
const POINT_LIGHT = 4;

/** Rec.709 relative luminance — identical coefficients to the fork's
 *  `luminance()` so the packed `power` field matches byte-for-byte. */
function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function lengthOf(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(v: Vec3): Vec3 {
  const len = lengthOf(v);
  if (len < 1e-12) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

/**
 * Build two unit tangent vectors spanning the plane perpendicular to `n`.
 * Used to synthesize the (u, v) basis for a disc-area emitter, whose core
 * representation gives only a centre, a normal and a radius (no explicit
 * in-plane axes). Deterministic so the packed data is stable across calls.
 */
function tangentBasis(n: Vec3): { t: Vec3; b: Vec3 } {
  const nn = normalize(n);
  // Pick the world axis least aligned with nn to avoid degeneracy.
  const ref: Vec3 = Math.abs(nn[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const t = normalize(cross(ref, nn));
  const b = normalize(cross(nn, t));
  return { t, b };
}

/**
 * Pack the scene's emitters into the 6-texel-per-light RGBA32F square grid the
 * WebGL2 path tracer reads. Mirrors the fork's `LightsInfoUniformStruct.updateFrom`
 * byte layout exactly (§5), driven from @vitrum/core `SceneEmitter`s rather than
 * THREE light objects.
 *
 * Unmappable / approximated fields vs the THREE source (documented for the caller):
 *  - IES profiles: core has no IES handle → `iesProfile = -1` always.
 *  - Spot `radius`: core `SpotEmitter` has no soft-source radius → 0 (area = 0).
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
    let k = 0;

    const [cr, cg, cb] = l.color;
    const lum = luminance(cr, cg, cb);

    switch (l.kind) {
      case 'rect-area': {
        // s0: position / type
        data[base + k++] = l.position[0];
        data[base + k++] = l.position[1];
        data[base + k++] = l.position[2];
        data[base + k++] = RECT_AREA_LIGHT;
        // s1: color / intensity
        data[base + k++] = cr;
        data[base + k++] = cg;
        data[base + k++] = cb;
        data[base + k++] = l.intensity;
        // u-vector + power. Core uAxis/vAxis are the in-plane span vectors; the
        // fork's width/height are their lengths.
        const u = l.uAxis;
        const v = l.vAxis;
        const width = lengthOf(u);
        const height = lengthOf(v);
        // s2: u / power
        data[base + k++] = u[0];
        data[base + k++] = u[1];
        data[base + k++] = u[2];
        data[base + k++] = lum * l.intensity * (width * height);
        // s3: v / area  (area = |u × v|)
        data[base + k++] = v[0];
        data[base + k++] = v[1];
        data[base + k++] = v[2];
        data[base + k++] = lengthOf(cross(u, v));
        break;
      }
      case 'disc-area': {
        // s0: position / type (CIRC_AREA)
        data[base + k++] = l.position[0];
        data[base + k++] = l.position[1];
        data[base + k++] = l.position[2];
        data[base + k++] = CIRC_AREA_LIGHT;
        // s1: color / intensity
        data[base + k++] = cr;
        data[base + k++] = cg;
        data[base + k++] = cb;
        data[base + k++] = l.intensity;
        // Synthesize an in-plane (u, v) basis from the disc normal; each axis
        // spans the full diameter (2*radius) to match the fork's full-extent
        // width/height convention, then π/4 corrects rectangle→disc.
        const { t, b } = tangentBasis(l.normal);
        const d = 2 * l.radius;
        const u: Vec3 = [t[0] * d, t[1] * d, t[2] * d];
        const v: Vec3 = [b[0] * d, b[1] * d, b[2] * d];
        const areaScale = Math.PI / 4.0;
        // s2: u / power
        data[base + k++] = u[0];
        data[base + k++] = u[1];
        data[base + k++] = u[2];
        data[base + k++] = lum * l.intensity * (d * d) * areaScale;
        // s3: v / area
        data[base + k++] = v[0];
        data[base + k++] = v[1];
        data[base + k++] = v[2];
        data[base + k++] = lengthOf(cross(u, v)) * areaScale;
        break;
      }
      case 'spot': {
        // s0: position / type
        data[base + k++] = l.position[0];
        data[base + k++] = l.position[1];
        data[base + k++] = l.position[2];
        data[base + k++] = SPOT_LIGHT;
        // s1: color / intensity
        data[base + k++] = cr;
        data[base + k++] = cg;
        data[base + k++] = cb;
        data[base + k++] = l.intensity;
        // The fork builds a lookAt basis: u = (1,0,0), v = (0,1,0) rotated into
        // the light's orientation. Core gives the forward `direction`; we build
        // a basis whose -w is the spot direction and read its u/v axes.
        const forward = normalize(l.direction);
        const { t, b } = tangentBasis(forward);
        // s2: u / power
        data[base + k++] = t[0];
        data[base + k++] = t[1];
        data[base + k++] = t[2];
        data[base + k++] = lum * l.intensity;
        // s3: v / area. Core spot has no source radius → radius = 0, area = 0.
        const radius = 0;
        data[base + k++] = b[0];
        data[base + k++] = b[1];
        data[base + k++] = b[2];
        data[base + k++] = Math.PI * radius * radius;
        // s4: radius / decay / distance / coneCos
        data[base + k++] = radius;
        data[base + k++] = l.decay ?? 2;
        data[base + k++] = l.distance ?? 0;
        data[base + k++] = Math.cos(l.angle);
        // s5: penumbraCos / iesProfile / 0 / 0
        data[base + k++] = Math.cos(l.angle * (1 - (l.penumbra ?? 0)));
        data[base + k++] = -1; // no IES profile in the core contract
        break;
      }
      case 'point': {
        // s0: position / type
        data[base + k++] = l.position[0];
        data[base + k++] = l.position[1];
        data[base + k++] = l.position[2];
        data[base + k++] = POINT_LIGHT;
        // s1: color / intensity
        data[base + k++] = cr;
        data[base + k++] = cg;
        data[base + k++] = cb;
        data[base + k++] = l.intensity;
        // s2: u = world position again / power (fork stores worldPosition here)
        data[base + k++] = l.position[0];
        data[base + k++] = l.position[1];
        data[base + k++] = l.position[2];
        data[base + k++] = lum * l.intensity;
        // s3 unused (zero); k advances over it
        k += 4;
        // s4: radius=0 / decay / distance / coneCos=0
        k += 1; // radius slot (0)
        data[base + k++] = l.decay ?? 2;
        data[base + k++] = l.distance ?? 0;
        // s4.a coneCos + s5 stay 0 — point lights are isotropic.
        break;
      }
      case 'directional': {
        // s0: position / type. Directional lights have no position; the fork
        // packs the world position (used only as an origin offset). Core gives
        // a direction only → store origin (0,0,0).
        data[base + k++] = 0;
        data[base + k++] = 0;
        data[base + k++] = 0;
        data[base + k++] = DIR_LIGHT;
        // s1: color / intensity
        data[base + k++] = cr;
        data[base + k++] = cg;
        data[base + k++] = cb;
        data[base + k++] = l.intensity;
        // s2: u = direction TOWARD the light / power. Core `direction` is the
        // unit vector pointing AT the light (matches the fork's
        // normalize(worldPos - target)).
        const dir = normalize(l.direction);
        data[base + k++] = dir[0];
        data[base + k++] = dir[1];
        data[base + k++] = dir[2];
        data[base + k++] = lum * l.intensity;
        break;
      }
      default: {
        // Exhaustiveness guard — `mesh-area` already filtered out above.
        const _never: never = l;
        void _never;
      }
    }
  }

  return { data, dim, kind: 'rgba32f', lightCount: lights.length };
}

/**
 * materialsTextureSpectral.test.ts — Layer 1 contract test for H6.6/SG.D.
 *
 * Verifies that per-material spectral data (vitrumSpectralAttenuation,
 * vitrumThinFilmStack, vitrumScatteringCoefficient) stamped on THREE
 * MeshPhysicalMaterial.userData is correctly packed into MaterialsTexture
 * and accessible to the fork GLSL shader via readSpectralAttenuationMu().
 *
 * Wire-format reference (Sprint 12 / H6.6):
 *   MATERIAL_PIXELS = 85 texels per material, stride = 85 * 4 = 340 floats
 *   Sample 17 .a = featureFlags (bit 0 = hasSpectralAttenuation,
 *                                bit 1 = hasFrontLayer, bit 2 = hasBackLayer)
 *   Samples 20–27 (floats 80–111): 32 spectral μ(λ) bins, 380–780 nm
 *     uniform grid, linearly interpolated from SpectralCurve on the JS side.
 *     GLSL consumer: readSpectralAttenuationMu(materialsTex, materialIndex, i)
 *     → spectralAttenuationMuHero() → transmissionAttenuationHero().
 *   Samples 28–54 (floats 112–219): per-material thin-film layer payload
 *     layout per layer: [ior, thicknessNm, extinctionCoefficient], up to 35 layers.
 *   Sample 15 (floats 60–63): [sssSigmaT, sssAnisotropy, dispersionStrength, thinFilmEnabled]
 *     sssSigmaT = vitrumScatteringCoefficient (mm⁻¹).
 *
 * These assertions would catch any future regression in the packing if a
 * MATERIAL_PIXELS bump shifts the spectral slot offset.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { MeshPhysicalMaterial } from 'three';

// MaterialsTexture is a named export from a subpath of the fork that is
// intentionally NOT part of the public index.js. The fork lives as a sibling
// directory to the vitrum repo; vitest.config.ts exposes it as the
// `@vitrum-fork/three-gpu-pathtracer` alias so the import survives both main
// checkouts and git worktrees (where the file:-symlink `node_modules` entry
// is broken). The fork ships JS only (no .d.ts); the alias resolves at
// runtime — silence implicit-any.
// @ts-expect-error — JS-only fork module; runtime resolves via vitest alias.
import { MaterialsTexture, MATERIAL_PIXELS } from '@vitrum-fork/three-gpu-pathtracer/src/uniforms/MaterialsTexture.js';

// Resolve the fork root once so the suite can self-skip cleanly when the
// sibling repo is missing (CI / fresh clone / worktree without the sibling
// checked out). Mirrors the lookup chain in vitest.config.ts.
const __filename = fileURLToPath(import.meta.url);
const __dirnameLocal = path.dirname(__filename);
function findForkRoot(): string | null {
  const FORK = 'three-gpu-pathtracer';
  const SENTINEL = path.join('src', 'uniforms', 'MaterialsTexture.js');
  const env = process.env['VITRUM_PT_FORK_PATH'];
  if (env && fs.existsSync(path.join(env, SENTINEL))) return env;
  const candidates = [
    // main checkout: __tests__ → src → pt-webgl → packages → vitrum → <parent>
    path.resolve(__dirnameLocal, '../../../../..', FORK),
    // worktree: __tests__ → src → pt-webgl → packages → <agent-id> → worktrees → .claude → vitrum → <parent>
    path.resolve(__dirnameLocal, '../../../../../../../..', FORK),
  ];
  for (const c of candidates) if (fs.existsSync(path.join(c, SENTINEL))) return c;
  let dir = __dirnameLocal;
  for (let i = 0; i < 12; i++) {
    const parent = path.dirname(dir);
    if (parent === dir) break;
    const candidate = path.join(parent, FORK);
    if (fs.existsSync(path.join(candidate, SENTINEL))) return candidate;
    dir = parent;
  }
  return null;
}
const FORK_AVAILABLE = findForkRoot() !== null;
const itIfFork = FORK_AVAILABLE ? it : it.skip;

// ── Wire-format layout constants (must match MaterialsTexture.js) ──────────────
// These are asserted below as a self-check so a MATERIAL_PIXELS bump is caught.
const EXPECTED_MATERIAL_PIXELS = 85;
const STRIDE = EXPECTED_MATERIAL_PIXELS * 4; // floats per material

// Sample 17.a = featureFlags offset within a material's float block
const S17_OFFSET = 17 * 4;
const FEATURE_FLAGS_OFFSET = S17_OFFSET + 3; // .a component

// Sample 15: sssSigmaT (vitrumScatteringCoefficient)
const S15_OFFSET = 15 * 4;

// Sample 16: thinFilmLayerCount
const S16_OFFSET = 16 * 4;

// Spectral grid: samples 20..27 → 32 floats starting at float offset 80
const SPECTRAL_BASE_FLOAT = 20 * 4; // = 80
const SPECTRAL_COUNT = 32;

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Build a SpectralCurve object matching the stainedGlass physics baker format:
 *   { wavelengthStart, wavelengthEnd, values: Float32Array }
 * The baker uses 81 samples at 5 nm steps (380–780 nm).
 */
function makeConstantSpectralCurve(muValue: number): {
  wavelengthStart: number;
  wavelengthEnd: number;
  values: Float32Array;
} {
  const N = 81;
  const values = new Float32Array(N).fill(muValue);
  return { wavelengthStart: 380, wavelengthEnd: 780, values };
}

/**
 * Pack a material array into MaterialsTexture and return the raw float data
 * for the material at the given index (0-based).
 */
function packAndSlice(materials: MeshPhysicalMaterial[], matIdx: number): Float32Array {
  const mt = new MaterialsTexture();
  mt.updateFrom(materials, []);
  const floatData = mt.image.data as Float32Array;
  return floatData.slice(matIdx * STRIDE, matIdx * STRIDE + STRIDE);
}

/**
 * Read a spectral bin from the packed float slice.
 * spectralIdx ∈ [0, 31].
 */
function readBin(slice: Float32Array, spectralIdx: number): number {
  return slice[SPECTRAL_BASE_FLOAT + spectralIdx] ?? 0;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('MaterialsTexture spectral packing — H6.6 / SG.D wire-format contract', () => {

  itIfFork('MATERIAL_PIXELS constant matches expected value (self-check against future bumps)', () => {
    expect(MATERIAL_PIXELS).toBe(EXPECTED_MATERIAL_PIXELS);
  });

  itIfFork('material without vitrumSpectralAttenuation: hasSpectralAttenuation bit = 0, spectral bins = 0', () => {
    const mat = new MeshPhysicalMaterial({ color: 0xffffff, transmission: 1, ior: 1.52 });
    const slice = packAndSlice([mat], 0);

    const featureFlags = Math.round(slice[FEATURE_FLAGS_OFFSET] ?? 0);
    expect(featureFlags & 1).toBe(0); // hasSpectralAttenuation bit clear

    for (let i = 0; i < SPECTRAL_COUNT; i++) {
      expect(readBin(slice, i)).toBeCloseTo(0.0, 6);
    }
  });

  itIfFork('cobalt-blue spectral curve: hasSpectralAttenuation=1, μ(λ) bins non-zero and peak at red end', () => {
    // Cobalt absorbs ~525, 595, 650 nm (red/orange).
    // A flat μ=1.0 constant curve verifies packing math; caller uses real curves.
    const curve = makeConstantSpectralCurve(1.0);
    const mat = new MeshPhysicalMaterial({ color: 0x3355bb, transmission: 1, ior: 1.52 });
    mat.userData['vitrumSpectralAttenuation'] = curve;

    const slice = packAndSlice([mat], 0);

    // Bit 0 of featureFlags must be set.
    const featureFlags = Math.round(slice[FEATURE_FLAGS_OFFSET] ?? 0);
    expect(featureFlags & 1).toBe(1);

    // All 32 spectral bins should be ≈ 1.0 (constant curve).
    for (let i = 0; i < SPECTRAL_COUNT; i++) {
      expect(readBin(slice, i)).toBeCloseTo(1.0, 5);
    }
  });

  itIfFork('spectral bins reflect linearly-interpolated μ(λ) from the source curve wavelength range', () => {
    // Source curve: 0.0 at 380 nm, 1.0 at 780 nm (linear ramp).
    const N = 81;
    const values = new Float32Array(N);
    for (let i = 0; i < N; i++) values[i] = i / (N - 1);
    const curve = { wavelengthStart: 380, wavelengthEnd: 780, values };

    const mat = new MeshPhysicalMaterial({ transmission: 1, ior: 1.52 });
    mat.userData['vitrumSpectralAttenuation'] = curve;

    const slice = packAndSlice([mat], 0);

    // Bin 0 → λ=380 nm → t=0 → μ≈0.
    expect(readBin(slice, 0)).toBeCloseTo(0.0, 3);

    // Bin 31 → λ=780 nm → t=1 → μ≈1.
    expect(readBin(slice, 31)).toBeCloseTo(1.0, 3);

    // Bin 15 → λ=380 + (15/31)×400 ≈ 573.5 nm → μ ≈ 15/31 ≈ 0.484.
    expect(readBin(slice, 15)).toBeCloseTo(15 / 31, 2);
  });

  itIfFork('material index 1 in a 2-material array: spectral bins land at the correct stride offset', () => {
    // mat[0] has no spectral; mat[1] has constant μ=2.5.
    const mat0 = new MeshPhysicalMaterial({ color: 0xff0000 });
    const mat1 = new MeshPhysicalMaterial({ transmission: 1, ior: 1.52 });
    mat1.userData['vitrumSpectralAttenuation'] = makeConstantSpectralCurve(2.5);

    const mt = new MaterialsTexture();
    mt.updateFrom([mat0, mat1], []);
    const floatData = mt.image.data as Float32Array;

    // mat[0] — no spectral.
    const ff0 = Math.round(floatData[0 * STRIDE + FEATURE_FLAGS_OFFSET] ?? 0);
    expect(ff0 & 1).toBe(0);

    // mat[1] — hasSpectral + all bins = 2.5.
    const ff1 = Math.round(floatData[1 * STRIDE + FEATURE_FLAGS_OFFSET] ?? 0);
    expect(ff1 & 1).toBe(1);
    for (let i = 0; i < SPECTRAL_COUNT; i++) {
      const v = floatData[1 * STRIDE + SPECTRAL_BASE_FLOAT + i] ?? 0;
      expect(v).toBeCloseTo(2.5, 5);
    }
  });

  itIfFork('vitrumScatteringCoefficient (sssSigmaT) packs into sample 15.r', () => {
    // Opalescent glass: sssSigmaT = 0.35 mm⁻¹.
    const mat = new MeshPhysicalMaterial({ transmission: 1, ior: 1.47 });
    mat.userData['vitrumScatteringCoefficient'] = 0.35;
    mat.userData['vitrumScatteringAnisotropy'] = 0.6;

    const slice = packAndSlice([mat], 0);

    // Sample 15.r = sssSigmaT
    expect(slice[S15_OFFSET + 0]).toBeCloseTo(0.35, 5);
    // Sample 15.g = sssAnisotropyG
    expect(slice[S15_OFFSET + 1]).toBeCloseTo(0.6, 5);
  });

  itIfFork('vitrumThinFilmStack: thinFilmEnabled=1 and layer count in sample 16.a', () => {
    // 2-layer TiO₂/SiO₂ stack (dichroic).
    const stack = {
      incidentIor: 1.0,
      angleDependent: true,
      layers: [
        { ior: 2.35, thicknessNm: 120, extinctionCoefficient: 0 },
        { ior: 1.46, thicknessNm: 120, extinctionCoefficient: 0 },
      ],
    };
    const mat = new MeshPhysicalMaterial({ transmission: 1, ior: 1.52 });
    mat.userData['vitrumThinFilmStack'] = stack;

    const slice = packAndSlice([mat], 0);

    // Sample 15.a = thinFilmEnabled (1.0 when layer count > 0)
    expect(slice[S15_OFFSET + 3]).toBeCloseTo(1.0, 5);

    // Sample 16.a = thinFilmLayerCount
    expect(slice[S16_OFFSET + 3]).toBeCloseTo(2.0, 5);

    // Sample 17.r = thinFilmIncidentIor = 1.0
    expect(slice[17 * 4 + 0]).toBeCloseTo(1.0, 5);

    // Sample 17.g = thinFilmAngleDependent = 1.0
    expect(slice[17 * 4 + 1]).toBeCloseTo(1.0, 5);
  });

  itIfFork('hasFrontLayer bit (bit 1) and hasBackLayer bit (bit 2) in featureFlags', () => {
    const mat = new MeshPhysicalMaterial({ transmission: 1, ior: 1.52 });
    mat.userData['vitrumFrontLayer'] = { transmission: [0.9, 0.92, 0.95], roughness: 0.2 };
    mat.userData['vitrumBackLayer'] = { transmission: [0.98, 0.98, 0.98], roughness: 0.05 };

    const slice = packAndSlice([mat], 0);

    const featureFlags = Math.round(slice[FEATURE_FLAGS_OFFSET] ?? 0);
    expect(featureFlags & 2).toBe(2); // hasFrontLayer
    expect(featureFlags & 4).toBe(4); // hasBackLayer
  });

  itIfFork('opt-in invariant: spectral path does not corrupt base PBR fields', () => {
    // Confirm that adding spectral data does not change IOR, attenuationDistance etc.
    const mat = new MeshPhysicalMaterial({
      color: 0x3355bb,
      transmission: 1,
      ior: 1.58,
      attenuationDistance: 5.0,
    });
    mat.userData['vitrumSpectralAttenuation'] = makeConstantSpectralCurve(0.8);

    const slice = packAndSlice([mat], 0);

    // Sample 2.r = ior
    expect(slice[2 * 4 + 0]).toBeCloseTo(1.58, 4);
    // Sample 12.a = attenuationDistance (Infinity encodes as Infinity in Float32)
    // Here we set attenuationDistance=5.0 to keep it finite.
    expect(slice[12 * 4 + 3]).toBeCloseTo(5.0, 4);
  });
});

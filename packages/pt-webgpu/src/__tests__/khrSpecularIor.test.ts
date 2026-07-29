import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  effectiveMaterialIor,
  KHR_MATERIALS_IOR_INFINITY_APPROX,
  materialToPackedVec4s,
} from '../scene/materialPacking.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL } from '../wgsl/pathTrace/material.wgsl.js';

function khrDielectricF0(
  ior: number,
  specularColor: number,
  specularStrength: number,
): number {
  const ratio = (ior - 1) / (ior + 1);
  return Math.min(ratio * ratio * specularColor, 1) * specularStrength;
}

describe('KHR_materials_ior + KHR_materials_specular semantics', () => {
  it('preserves authored zero IOR and specularColor factors above one in packing', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.8, 0.7, 0.6],
      roughness: 0.4,
      metallic: 0,
      ior: 0,
      specularColor: [1.25, 4, 0],
      specularIntensity: 0.5,
    });
    expect(packed[9]).toBe(0);
    expect(packed.slice(27 * 4, 27 * 4 + 4)).toEqual([1.25, 4, 0, 0.5]);
  });

  it('maps IOR zero to a finite refraction-safe transport value', () => {
    const ior = effectiveMaterialIor(0);
    expect(ior).toBe(KHR_MATERIALS_IOR_INFINITY_APPROX);
    expect(Number.isFinite(ior)).toBe(true);
    const etaIncidentOverTransmitted = 1 / ior;
    const cosIncident = 0.3;
    const cosTransmitted = Math.sqrt(Math.max(
      0,
      1 - etaIncidentOverTransmitted ** 2 * (1 - cosIncident ** 2),
    ));
    expect(Number.isFinite(cosTransmitted)).toBe(true);
    expect(cosTransmitted).toBeCloseTo(1, 12);
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain(
      `    ${KHR_MATERIALS_IOR_INFINITY_APPROX},`,
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain('m2.y == 0.0,');
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).not.toContain(
      'mat.ior = clamp(m2.y, 1.0, 2.5);',
    );
  });

  it('clamps IOR-derived colored F0 before specular strength', () => {
    // At IOR 1.5, base F0 is 0.04. A color factor of 30 saturates it to one,
    // then strength 0.5 yields 0.5. Clamping color first would incorrectly
    // produce 0.02.
    expect(khrDielectricF0(1.5, 30, 0.5)).toBeCloseTo(0.5, 15);
    const helper = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf('fn materialSpecularF0('),
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn materialSpecularFresnelSchlick(',
      ),
    );
    const colorClamp = helper.indexOf('let coloredDielectricF0 = min(');
    const strength = helper.indexOf(
      'coloredDielectricF0 * clamp(specularIntensity, 0.0, 1.0);',
    );
    expect(colorClamp).toBeGreaterThanOrEqual(0);
    expect(strength).toBeGreaterThan(colorClamp);
    expect(helper).not.toContain('clamp(specularColor');
  });

  it('keeps unbounded color factors through every mapped material payload', () => {
    const sources = [
      '../wgsl/bdpt/bdptLightSubpath.wgsl.ts',
      '../wgsl/pathTrace/caustic.wgsl.ts',
      '../wgsl/pathTrace/reservoirPtHero.wgsl.ts',
      '../wgsl/pathTrace/restirPtProducer.wgsl.ts',
      '../wgsl/pathTrace/shadePrologue.wgsl.ts',
      '../wgsl/pathTrace/sppmBindings.wgsl.ts',
    ].map((path) => readFileSync(new URL(path, import.meta.url), 'utf8')).join('\n');
    expect(sources).not.toMatch(/specularColor\w*\s*=\s*clamp\(/);
    expect(sources).toContain(
      'mat.specularColor * sampleSpecularColorTexture(',
    );
  });

  it('uses maximum iridescence thickness when no texture sample overrides it', () => {
    const helper = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf('fn iridescenceModifiedF0('),
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf('fn materialSpecularF0('),
    );
    expect(helper).toContain('let thicknessNm = max(thicknessMax, 0.0);');
    expect(helper).not.toContain('mix(thicknessMin, thicknessMax');
  });
});

import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';

function wgslFunction(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  expect(start, `missing WGSL function ${name}`).toBeGreaterThanOrEqual(0);
  const bodyStart = source.indexOf('{', start);
  expect(bodyStart, `missing body for WGSL function ${name}`).toBeGreaterThan(start);

  let depth = 0;
  for (let cursor = bodyStart; cursor < source.length; cursor += 1) {
    const char = source[cursor];
    if (char === '{') depth += 1;
    if (char === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(start, cursor + 1);
    }
  }
  throw new Error(`unterminated WGSL function ${name}`);
}

function compact(source: string): string {
  return source.replace(/\s+/g, ' ').trim();
}

describe('opaque zero-roughness PDF coherence', () => {
  it('classifies only smooth transmissive dielectrics as delta reflection', () => {
    const predicate = compact(
      wgslFunction(PT_WEBGPU_PATH_TRACE_BSDF_WGSL, 'bsdfBaseReflectionIsDelta'),
    );
    expect(predicate).toContain(
      'return metallic == 0.0 && transmission > 0.0 && bsdfDielectricIsSmooth(roughness);',
    );

    const sampler = compact(
      wgslFunction(
        PT_WEBGPU_PATH_TRACE_BSDF_WGSL,
        'sampleNextBounceDirectionWithClearcoatNormal',
      ),
    );
    expect(sampler).toContain(
      'if (transmission > 0.0 && metallic == 0.0)',
    );
    expect(sampler).toContain('glossyReflectionSample(');
  });

  it('uses that same classification in every base-reflection PDF path', () => {
    for (const functionName of [
      'brdfDirectionalPdfThinFilm',
      'brdfDirectionalPdfFullWithClearcoatNormal',
      'brdfDirectionalPdfWithIridescence',
    ]) {
      const pdf = wgslFunction(PT_WEBGPU_PATH_TRACE_BSDF_WGSL, functionName);
      expect(pdf, functionName).toContain(
        'bsdfBaseReflectionIsDelta(roughness, metallic, transmission)',
      );
    }

    const defaultPdf = wgslFunction(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL,
      'brdfDirectionalPdfWithIridescence',
    );
    expect(defaultPdf).not.toContain(
      '0.0, bsdfDielectricIsSmooth(roughness)',
    );
  });
});

import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import { SPPM_GROUP3_BINDINGS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';

describe('caustic receiver finite-BSDF parity', () => {
  it('evaluates MNEE receivers with transmission and interface IOR', () => {
    const receiver = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf('let receiverDistance'),
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf('let emitterFactor'),
    );
    expect(receiver).toContain('evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(receiver).toContain(
      'baseColor, roughness, metallic, transmission, etaTOverI,',
    );
    expect(receiver).toContain(
      'anisotropy, anisotropyRotation, thinFilm, true,',
    );
    expect(receiver).not.toContain('evaluateBrdfFullWithClearcoatNormal(');
  });

  it('threads transmission and eta through the progressive SPPM surface gather', () => {
    const gather = SPPM_GROUP3_BINDINGS_WGSL.slice(
      SPPM_GROUP3_BINDINGS_WGSL.indexOf('fn sppmUpdateProgressiveKind('),
      SPPM_GROUP3_BINDINGS_WGSL.indexOf('fn sppmProgressiveEstimateKind('),
    );
    expect(gather).toContain('transmission : f32,');
    expect(gather).toContain('etaTOverI : f32,');
    expect(gather).toContain('evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(gather).toContain(
      'baseColor, roughness, metallic, transmission, etaTOverI,',
    );
    expect(gather).not.toContain('evaluateBrdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'baseColor, roughness, metallic, transmission, etaTOverI,',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'transmission,\n          surfaceEtaTOverI,\n          mat.clearcoat,',
    );
  });
});

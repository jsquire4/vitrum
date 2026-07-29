import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL as code } from '../wgsl/pathTrace/caustic.wgsl.js';
import { PT_WEBGPU_TRACE_WGSL, composePtWebgpuTraceWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('unified MNEE refraction closure', () => {
  it('replays exact delta transmission and extinction at every solved vertex', () => {
    expect(code).toContain('if (!mneeFacetHasDeltaTransmission(optics) ||');
    expect(code).toContain('eventFactor = mneeFacetTransmissionFactorWithEta(');
    expect(code).toContain('solvedStack = mneeChainCommitTransmission(solvedStack, optics);');
    expect(code).toContain('volumeTransmittance = volumeTransmittance * segment.transmittance;');
  });

  it('tracks nested media by object identity', () => {
    expect(code).toContain('objectIds: array<u32, 8>');
    expect(code).toContain('stack.objectIds[index] = optics.objectId;');
    expect(code).toContain('stack.objectIds[scan] == optics.objectId');
    expect(code).toContain('stack.objectIds[shift] = stack.objectIds[shift + 1u];');
  });

  it('keeps spectral dispersion, coherent films, and every Jacobian domain', () => {
    expect(code).toContain('ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);');
    expect(code).toContain('materialDielectricLayeredInterface(');
    expect(code).toContain('interfaceResponse.baseTransmittance');
    expect(code).toContain('mneeBoundedChainAreaPdfDet(');
    expect(code).toContain('mneeBoundedChainDirectionalFocusingDet(');
    expect(code).toContain('mneeBoundedChainFocusingDet(');
  });

  it('has no retired one-interface/cone sibling in either composition', () => {
    for (const source of [PT_WEBGPU_TRACE_WGSL, composePtWebgpuTraceWgsl(true)]) {
      expect(source).toContain('fn boundedManifoldCaustic(');
      expect(source).not.toContain('fn pointLightRefractionCaustic(');
      expect(source).not.toContain('traceSpecularTransmissiveChain');
    }
  });
});

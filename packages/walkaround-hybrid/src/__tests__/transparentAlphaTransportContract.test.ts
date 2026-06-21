import { describe, expect, it } from 'vitest';

import { RESTIR_CAST_PRIMARY_WGSL } from '../shaders/restirCastPrimary.wgsl.js';
import { RIS_WGSL } from '../shaders/ris.wgsl.js';
import { RIS_GI_WGSL } from '../shaders/risGi.wgsl.js';
import { buildRisGiNrcModule } from '../shaders/risGiNrc.wgsl.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shaders/shadingTerms.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';

const nrcGiModule = buildRisGiNrcModule({
  levels: 1,
  featuresPerEntry: 4,
  oneBlobBins: 4,
  width: 16,
  outWidth: 3,
  hidden: 1,
});

describe('transparent alpha transport contract', () => {
  it('keeps fractional alpha-blend surfaces out of ReSTIR primary and reconnection vertices', () => {
    for (const [name, shader] of [
      ['restir-cast-primary', RESTIR_CAST_PRIMARY_WGSL],
      ['ris-di', RIS_WGSL],
      ['ris-gi', RIS_GI_WGSL],
      ['ris-gi-nrc', nrcGiModule.source],
      ['shade-primary', SHADE_WGSL],
    ] as const) {
      expect(shader, name).toContain('traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(');
      expect(shader, name).not.toMatch(/\btraceSceneFirstHitAlphaMaskTextured\(/);
    }
  });

  it('keeps shade-side direct-light visibility on alpha-aware transmittance predicates', () => {
    expect(SHADING_TERMS_WGSL).toContain('traceSceneAnyAlphaMaskTextured(');
    expect(SHADING_TERMS_WGSL).toContain('traceSceneAlphaTintTransmittanceTextured(');
    expect(SHADING_TERMS_WGSL).toContain('var shadowT = vec3f(1.0);');
  });

  it('keeps transparent OIT lighting local instead of coupling it to ReSTIR reservoirs', () => {
    expect(TRANSPARENT_OIT_WGSL).toContain('fn oitLayerAreaEmitterNEE(');
    expect(TRANSPARENT_OIT_WGSL).toContain('traceSceneAlphaTintTransmittanceTextured(');
    expect(TRANSPARENT_OIT_WGSL).toContain('sampleEmitterLeAtXi(e, xi)');
    expect(TRANSPARENT_OIT_WGSL).not.toMatch(/var<storage,\s*(read|read_write)>[^;]*reservoir/i);
    expect(TRANSPARENT_OIT_WGSL).not.toMatch(/\b(load|store|update|resolve)\w*Reservoir\b/);
    expect(TRANSPARENT_OIT_WGSL).not.toContain('selectedEmitter');
    expect(TRANSPARENT_OIT_WGSL).not.toContain('risFinal');
  });
});

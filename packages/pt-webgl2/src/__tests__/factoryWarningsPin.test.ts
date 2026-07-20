// T3-D characterization pin: the createPTEngine_WebGL2 factory emits a fixed set
// of construction-time warnings (and throws) driven by option validation. The
// D11-1 extraction of options.validate.ts must preserve every [code, message]
// exactly. This test drives the factory with option sets that trip each
// validation path and pins the emitted warning codes + messages, plus the
// throw messages. Captured on the pre-refactor source; green before the split.

import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGL2 } from '../index.js';
import type { PTEngineWebGL2Options } from '../index.js';
import { createMockGl } from './mockGl.js';

function baseOpts(over: Partial<PTEngineWebGL2Options>): PTEngineWebGL2Options {
  return { device: createMockGl(), ...over };
}

async function collectWarnings(
  over: Partial<PTEngineWebGL2Options>,
): Promise<Array<{ code: string; message: string }>> {
  const warnings: Array<{ code: string; message: string }> = [];
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    const e = await createPTEngine_WebGL2(
      baseOpts({ ...over, onWarning: (w) => warnings.push({ code: w.code, message: w.message }) }),
    );
    e.dispose();
  } finally {
    vi.restoreAllMocks();
  }
  return warnings;
}

describe('createPTEngine_WebGL2 factory warning pins (T3-D)', () => {
  it('denoiser:auto with no model URL resolves to none with a warning', async () => {
    const w = await collectWarnings({ denoiser: 'auto' });
    expect(w.map((x) => x.code)).toEqual(['pt-webgl2.denoiser-auto-resolved']);
    expect(w[0]!.message).toContain("denoiser:'auto' resolved to 'none'");
    expect(w[0]!.message).toContain('no-host-model-assets');
  });

  it('manifold-nee caustic strategy warns with the approximation message', async () => {
    const w = await collectWarnings({ causticStrategy: 'manifold-nee' });
    expect(w.map((x) => x.code)).toContain('pt-webgl2.caustic-strategy-approximation');
    const cw = w.find((x) => x.code === 'pt-webgl2.caustic-strategy-approximation')!;
    expect(cw.message).toContain('deterministic refraction-walk heuristic');
    expect(cw.message).toContain('causticStrategy="manifold-nee"');
  });

  it('photon-map caustic strategy warns with the cone-traced message', async () => {
    const w = await collectWarnings({ causticStrategy: 'photon-map' });
    const cw = w.find((x) => x.code === 'pt-webgl2.caustic-strategy-approximation')!;
    expect(cw.message).toContain('deterministic cone-traced photon estimate');
  });

  it('bdptOptions.maxLightBounces > limit clamps with a warning', async () => {
    const w = await collectWarnings({ bdptOptions: { maxLightBounces: 99 } });
    const cw = w.find((x) => x.code === 'pt-webgl2.bdpt-max-light-bounces-clamped')!;
    expect(cw).toBeDefined();
    expect(cw.message).toContain('clamping to supported WebGL2 BDPT light-subpath limit 3');
  });

  it('non-integer bdptOptions.maxLightBounces rounds with a warning', async () => {
    const w = await collectWarnings({ bdptOptions: { maxLightBounces: 2.7 } });
    const cw = w.find((x) => x.code === 'pt-webgl2.bdpt-max-light-bounces-rounded')!;
    expect(cw).toBeDefined();
    expect(cw.message).toContain('rounding down to integer 2');
  });

  it('unsupported denoiser warns and degrades', async () => {
    const w = await collectWarnings({ denoiser: 'svgf' as never });
    const cw = w.find((x) => x.code === 'pt-webgl2.unsupported-denoiser')!;
    expect(cw).toBeDefined();
    expect(cw.message).toContain('Degrading to no-denoise');
  });

  it('equirectangular + dof warns dof is ignored', async () => {
    const w = await collectWarnings({
      cameraType: 'equirectangular',
      dof: { focusDistance: 1, bokehSize: 0.1 },
    });
    const cw = w.find((x) => x.code === 'pt-webgl2.equirectangular-dof-ignored')!;
    expect(cw).toBeDefined();
    expect(cw.message).toContain('dof is ignored when cameraType is "equirectangular"');
  });

  // ── Throwing validation paths ──────────────────────────────────────────────
  it('non-WebGL2 device throws', async () => {
    await expect(createPTEngine_WebGL2({ device: {} as never })).rejects.toThrow(
      /device must be a WebGL2RenderingContext/,
    );
  });

  it('maxBounces < 1 throws RangeError', async () => {
    await expect(createPTEngine_WebGL2(baseOpts({ maxBounces: 0 }))).rejects.toThrow(
      /maxBounces must be >= 1/,
    );
  });

  it('maxSamplesPerPixel < 1 throws RangeError', async () => {
    await expect(
      createPTEngine_WebGL2(baseOpts({ maxSamplesPerPixel: 0 })),
    ).rejects.toThrow(/maxSamplesPerPixel must be >= 1/);
  });

  it('negative materialLodDepth throws RangeError', async () => {
    await expect(
      createPTEngine_WebGL2(baseOpts({ materialLodDepth: -1 })),
    ).rejects.toThrow(/materialLodDepth must be a finite number >= 0/);
  });

  it('bdptOptions.maxLightBounces < 1 throws RangeError', async () => {
    await expect(
      createPTEngine_WebGL2(baseOpts({ bdptOptions: { maxLightBounces: 0 } })),
    ).rejects.toThrow(/bdptOptions.maxLightBounces must be a finite number >= 1/);
  });

  it('bdpt:true with multi-vertex depth but no opt-in throws', async () => {
    await expect(
      createPTEngine_WebGL2(
        baseOpts({ bdpt: true, bdptOptions: { maxLightBounces: 3 } }),
      ),
    ).rejects.toThrow(/multi-vertex BDPT research path/);
  });

  it('bdpt:true + multiVertex opt-in warns (research mode)', async () => {
    const w = await collectWarnings({
      bdpt: true,
      bdptOptions: { maxLightBounces: 3, experimentalMultiVertex: true },
    });
    const cw = w.find((x) => x.code === 'pt-webgl2.bdpt-multivertex-research-mode')!;
    expect(cw).toBeDefined();
    expect(cw.message).toContain('multi-vertex BDPT research path');
  });

  it('negative backgroundBlur throws RangeError', async () => {
    await expect(
      createPTEngine_WebGL2(baseOpts({ backgroundBlur: -1 })),
    ).rejects.toThrow(/backgroundBlur must be a finite number >= 0/);
  });
});

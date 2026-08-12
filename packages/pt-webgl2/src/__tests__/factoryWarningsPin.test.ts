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
  it('denoiser:auto with no model URL resolves to oidn-final via the default URL', async () => {
    const w = await collectWarnings({ denoiser: 'auto' });
    expect(w.map((x) => x.code)).toEqual(['pt-webgl2.denoiser-auto-resolved']);
    expect(w[0]!.message).toContain("denoiser:'auto' resolved to 'oidn-final'");
    expect(w[0]!.message).toContain('default-oidn-model-url');
  });

  it('accepts named BDPT and rejects unsupported caustic strategies/options', async () => {
    expect(await collectWarnings({ causticStrategy: 'bdpt' })).toEqual([]);

    for (const legacy of [
      { causticStrategy: 'none' },
      { causticStrategy: 'manifold-nee' },
      { causticStrategy: 'photon-map' },
    ]) {
      await expect(
        createPTEngine_WebGL2(baseOpts(legacy as unknown as Partial<PTEngineWebGL2Options>)),
      ).rejects.toThrow(/causticStrategy must be one of/);
    }

    await expect(
      createPTEngine_WebGL2(baseOpts({ causticOptions: { mneeMaxIterations: 8 } } as never)),
    ).rejects.toThrow(/causticOptions are not accepted/);
  });

  it('treats absent and explicitly undefined legacy caustic fields identically', async () => {
    const warnings = await collectWarnings({
      causticStrategy: undefined,
      causticOptions: undefined,
    } as unknown as Partial<PTEngineWebGL2Options>);
    expect(warnings).toEqual([]);
  });

  it('rejects unsupported or non-integral BDPT light depths', async () => {
    for (const maxLightBounces of [2.7, 9, 99, Number.POSITIVE_INFINITY]) {
      await expect(
        createPTEngine_WebGL2(
          baseOpts({
            bdptOptions: { maxLightBounces },
          }),
        ),
      ).rejects.toThrow(/integer in the supported range 1\.\.8/);
    }
  });

  it('unsupported denoiser rejects without degrading', async () => {
    await expect(
      createPTEngine_WebGL2(baseOpts({ denoiser: 'svgf' as never })),
    ).rejects.toThrow(/denoiser must be one of/);
  });

  it('equirectangular + dof rejects instead of being ignored', async () => {
    await expect(createPTEngine_WebGL2(baseOpts({
      cameraType: 'equirectangular',
      dof: { focusDistance: 1, bokehSize: 0.1 },
    }))).rejects.toThrow(/dof is unsupported.*equirectangular/);
  });

  // ── Throwing validation paths ──────────────────────────────────────────────
  it('non-WebGL2 device throws', async () => {
    await expect(createPTEngine_WebGL2({ device: {} as never })).rejects.toThrow(
      /device must be a WebGL2RenderingContext/,
    );
  });

  it('rejects maxBounces outside the statically supported integer range', async () => {
    await expect(createPTEngine_WebGL2(baseOpts({ maxBounces: 0 }))).rejects.toThrow(
      /maxBounces must be an integer in the supported range 1\.\.32/,
    );
    await expect(createPTEngine_WebGL2(baseOpts({ maxBounces: 2.5 }))).rejects.toThrow(
      /maxBounces must be an integer in the supported range 1\.\.32/,
    );
    await expect(createPTEngine_WebGL2(baseOpts({ maxBounces: 33 }))).rejects.toThrow(
      /maxBounces must be an integer in the supported range 1\.\.32/,
    );
  });

  it('non-finite maxBounces and maxSamplesPerPixel throw RangeError', async () => {
    await expect(createPTEngine_WebGL2(baseOpts({ maxBounces: Number.NaN }))).rejects.toThrow(
      /maxBounces must be an integer in the supported range 1\.\.32/,
    );
    await expect(
      createPTEngine_WebGL2(baseOpts({ maxSamplesPerPixel: Number.POSITIVE_INFINITY })),
    ).rejects.toThrow(/maxSamplesPerPixel must be a positive safe integer/);
  });

  it('maxSamplesPerPixel < 1 throws RangeError', async () => {
    await expect(createPTEngine_WebGL2(baseOpts({ maxSamplesPerPixel: 0 }))).rejects.toThrow(
      /maxSamplesPerPixel must be a positive safe integer/,
    );
  });

  it('negative materialLodDepth throws RangeError', async () => {
    await expect(createPTEngine_WebGL2(baseOpts({ materialLodDepth: -1 }))).rejects.toThrow(
      /materialLodDepth must be an integer in 0/,
    );
  });

  it('bdptOptions.maxLightBounces < 1 throws RangeError', async () => {
    await expect(
      createPTEngine_WebGL2(baseOpts({ bdptOptions: { maxLightBounces: 0 } })),
    ).rejects.toThrow(/integer in the supported range 1\.\.8/);
  });

  it('rejects nonempty BDPT tuning when no BDPT selector is enabled', async () => {
    await expect(
      createPTEngine_WebGL2(baseOpts({ bdptOptions: { maxLightBounces: 4 } })),
    ).rejects.toThrow(/bdptOptions requires bdpt:true or causticStrategy:"bdpt"/);

    await expect(
      createPTEngine_WebGL2(baseOpts({
        causticStrategy: 'bdpt',
        bdptOptions: { maxLightBounces: 4 },
      })),
    ).resolves.toBeDefined();
  });

  it('bdpt:true rejects a light subpath above the eight-vertex bound', async () => {
    await expect(
      createPTEngine_WebGL2(baseOpts({ bdpt: true, bdptOptions: { maxLightBounces: 9 } })),
    ).rejects.toThrow(/integer in the supported range 1\.\.8/);
  });

  it('negative backgroundBlur throws RangeError', async () => {
    await expect(createPTEngine_WebGL2(baseOpts({ backgroundBlur: -1 }))).rejects.toThrow(
      /backgroundBlur must be >= 0/,
    );
  });
});

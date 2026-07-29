import { createHash } from 'node:crypto';
import { describe, expect, it, vi } from 'vitest';
import {
  LIGHT_TREE_TRAVERSAL_WGSL,
  SOBOL_DIMENSION_COUNT,
  initOwenScrambledSobolStream,
  nextOwenScrambledSobolU32,
  owenScrambledSobolU32,
  sobolFrameKey,
} from '@vitrum/shared-samplers';
import { createPTEngine_WebGPU } from '../index.js';
import { GpuResources } from '../gpuResources.js';
import {
  composePtWebgpuCompositeTraceWgsl,
  composeSppmPhotonPassWgsl,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import {
  PT_WEBGPU_TRACE_LITE_WGSL,
  composePtWebgpuTraceLiteWgsl,
} from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import {
  composeRestirPtResolveWgsl,
  composeRestirPtSpatialWgsl,
  composeRestirPtProducerWgsl,
  composeRestirPtTemporalWgsl,
} from '../wgsl/pathTrace/restirPtCompose.wgsl.js';
import { PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE } from '../webgpuLimits.js';
import { PT_WEBGPU_SOBOL_RNG_WGSL } from '../wgsl/common.wgsl.js';
import { composePtWebgpuAdjointPassWgsl } from '../wgsl/pathTrace/adjointPass.wgsl.js';
import { RESTIR_PT_TEMPORAL_WGSL } from '../wgsl/pathTrace/restirPtTemporal.wgsl.js';
import { RESTIR_PT_SPATIAL_WGSL } from '../wgsl/pathTrace/restirPtSpatial.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { installGpuConstStubs } from './gpuStub.js';

const SOBOL_DIMENSION_AUDIT_2026_07_21 = true;

function expectOrderedNeedles(
  source: string,
  needles: Array<readonly [label: string, needle: string]>,
): void {
  let cursor = 0;
  for (const [label, needle] of needles) {
    const at = source.indexOf(needle, cursor);
    if (at < cursor) {
      throw new Error(`Missing or out-of-order Sobol dimension audit needle "${label}": ${needle}`);
    }
    cursor = at + needle.length;
  }
}

function makeDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(() => ({ finish: vi.fn(() => ({})) })),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makePipelineDevice() {
  installGpuConstStubs();
  const shaderModules: Array<{ label?: string; code: string }> = [];
  const device = {
    createBuffer: vi.fn((desc?: { label?: string }) => ({
      label: desc?.label ?? '',
      destroy: vi.fn(),
    })),
    createShaderModule: vi.fn((desc: { label?: string; code: string }) => {
      shaderModules.push(desc);
      return {};
    }),
    createBindGroupLayout: vi.fn((desc: { label?: string }) => ({ label: desc.label })),
    createPipelineLayout: vi.fn((desc: { label?: string }) => ({ label: desc.label })),
    createComputePipeline: vi.fn((desc: { label?: string; compute: { entryPoint: string } }) => ({
      label: desc.label,
      entryPoint: desc.compute.entryPoint,
    })),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
  return { device, shaderModules };
}

describe('pt-webgpu sampling options', () => {
  it('keeps PCG as the default full and lite shader composition', () => {
    expect(composePtWebgpuTraceWgsl(false)).toContain(
      'fn pcgNext(state: ptr<function, u32>) -> u32',
    );
    expect(composePtWebgpuTraceWgsl(false)).not.toContain('ptSobolNextU32');
    expect(composePtWebgpuTraceLiteWgsl()).toBe(PT_WEBGPU_TRACE_LITE_WGSL);
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('ptSobolNextU32');
  });

  it('composes the binding-free Sobol RNG for full, lite, composite, SPPM, and ReSTIR-PT paths when requested', () => {
    const full = composePtWebgpuTraceWgsl(false, { sampling: 'sobol' });
    const lite = composePtWebgpuTraceLiteWgsl({ sampling: 'sobol' });
    const compositeSss = composePtWebgpuCompositeTraceWgsl(false, { sampling: 'sobol' });
    const compositeBdpt = composePtWebgpuCompositeTraceWgsl(true, { sampling: 'sobol' });
    const sppmPhoton = composeSppmPhotonPassWgsl({ sampling: 'sobol' });
    const restirProducer = composeRestirPtProducerWgsl({ sampling: 'sobol' });
    const restirTemporal = composeRestirPtTemporalWgsl({ sampling: 'sobol' });
    const restirSpatial = composeRestirPtSpatialWgsl({ sampling: 'sobol' });
    const restirResolve = composeRestirPtResolveWgsl({ sampling: 'sobol' });
    const adjoint = composePtWebgpuAdjointPassWgsl('sobol');

    for (const wgsl of [
      full,
      lite,
      compositeSss,
      compositeBdpt,
      sppmPhoton,
      restirProducer,
      restirTemporal,
      restirSpatial,
      restirResolve,
      adjoint,
    ]) {
      expect(wgsl).toContain('fn ptSobolNextU32(state: ptr<function, PtRngState>) -> u32');
      expect(wgsl).toContain('fn ptSobolNestedUniformScrambleBase2(x: u32, seed: u32) -> u32');
      expect(wgsl).toContain('fn pcgInit(px: u32, py: u32, frameKey: u32) -> PtRngState');
      expect(wgsl).toContain('fn ptRngFrameKey(frameSeed: u32, frameIndex: u32) -> u32');
      expect(wgsl).toContain('fn rand_f32(state: ptr<function, PtRngState>) -> f32');
      expect(wgsl).toContain(
        '(*state).fallbackState = (*state).fallbackState * 747796405u + 2891336453u;',
      );
    }
    expect(adjoint).toContain(
      'var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(frameSeed, 0u));',
    );
  });

  it('pins every Sobol production composition to one assignment digest', () => {
    const compositions: Array<readonly [string, string]> = [
      ['trace.full', composePtWebgpuTraceWgsl(false, { sampling: 'sobol' })],
      ['trace.lite', composePtWebgpuTraceLiteWgsl({ sampling: 'sobol' })],
      ['composite.sss', composePtWebgpuCompositeTraceWgsl(false, { sampling: 'sobol' })],
      ['composite.bdpt', composePtWebgpuCompositeTraceWgsl(true, { sampling: 'sobol' })],
      ['sppm.photon', composeSppmPhotonPassWgsl({ sampling: 'sobol' })],
      ['restir.producer', composeRestirPtProducerWgsl({ sampling: 'sobol' })],
      ['restir.temporal', composeRestirPtTemporalWgsl({ sampling: 'sobol' })],
      ['restir.spatial', composeRestirPtSpatialWgsl({ sampling: 'sobol' })],
      ['restir.resolve', composeRestirPtResolveWgsl({ sampling: 'sobol' })],
    ];
    const digest = createHash('sha256');
    const assignmentNeedles = [
      'PtRngState',
      'ptRngFrameKey',
      'ptSobol',
      'PT_SOBOL_DIMENSION_COUNT',
      'rand_f32(',
      'pcgInit(',
      'sampleIndex',
      'sequenceKey',
      'fallbackState',
      'pixelX',
      'pixelY',
      '.dimension',
    ];
    for (const [name, wgsl] of compositions) {
      digest.update(name);
      digest.update(String.fromCharCode(0));
      digest.update(
        wgsl
          .split(String.fromCharCode(10))
          .filter((line) => assignmentNeedles.some((needle) => line.includes(needle)))
          .map((line) => line.trim())
          .join(String.fromCharCode(10)),
      );
      digest.update(String.fromCharCode(0));
    }
    // A5 threads the active RNG into visibility helpers so stochastic alpha
    // blends remain unbiased for shadow and reconnection rays.
    // C65 expands the shared Joe-Kuo direction table to 512 dimensions; the
    // C28-C40/KHR pass also changes the composed production call graph.
    // C35 composes its native t=1 strategy helper only for BDPT-on modules, so
    // Sobol's default and composite-off assignment surface stays legacy-exact.
    expect(digest.digest('hex')).toBe(
      '8bf839f5011d981e70cb99dff48b429ca92e34f457968c3995e703c578920084',
    );
  });
  it('builds full and lite path-trace modules from the selected Sobol RNG', () => {
    const fullStub = makePipelineDevice();
    const full = new GpuResources(fullStub.device, 'full', false, false, undefined, 'sobol');
    full.ensurePipeline();
    expect(
      fullStub.shaderModules.find((m) => m.label === 'vitrum.pt-webgpu.pathTrace.full')?.code,
    ).toContain('ptSobolNextU32');

    const liteStub = makePipelineDevice();
    const lite = new GpuResources(liteStub.device, 'lite', false, false, undefined, 'sobol');
    lite.ensurePipeline();
    expect(
      liteStub.shaderModules.find((m) => m.label === 'vitrum.pt-webgpu.pathTrace.lite')?.code,
    ).toContain('ptSobolNextU32');
  });

  it('reports selected Sobol as an active native capability', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeDevice(),
      sampling: 'sobol',
    });

    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-sobol-sampling')).toBe(true);
    expect(engine.capabilities.supportDetails?.samplingSequences).toEqual({
      default: 'pcg',
      modes: { pcg: 'native', sobol: 'native' },
      sobol: {
        lowDiscrepancyDimensions: 512,
        continuation: 'independent-pcg',
        sampleBlockSize: 65536,
        frameIndexPeriod: 4294967296,
      },
    });
    engine.dispose();
  });

  it('rejects an unknown sampling mode before construction', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeDevice(),
      sampling: 'halton' as never,
    })).rejects.toThrow(/sampling is unsupported.*halton/);
  });

  it('pins the Sobol dimension stream and assignment anchors across pt-webgpu pipelines', () => {
    expect(SOBOL_DIMENSION_AUDIT_2026_07_21).toBe(true);
    expectOrderedNeedles(PT_WEBGPU_SOBOL_RNG_WGSL, [
      ['explicit state', 'struct PtRngState {'],
      ['sample slot', 'sampleIndex: u32,'],
      ['monotonic dimension', 'dimension: u32,'],
      ['full pixel x', 'pixelX: u32,'],
      ['full pixel y', 'pixelY: u32,'],
      ['frame-block key', 'sequenceKey: u32,'],
      ['independent overflow state', 'fallbackState: u32,'],
      ['shared direction domain', 'const SOBOL_DIRECTION_DIMENSION_COUNT = 512u;'],
      [
        'Sobol prefix length',
        'const PT_SOBOL_DIMENSION_COUNT = SOBOL_DIRECTION_DIMENSION_COUNT;',
      ],
      ['sample extraction', 'let sampleIndex = frameKey & 0x0000ffffu;'],
      ['block extraction', 'let sequenceKey = frameKey >> 16u;'],
      ['full x retained', 'state.pixelX = px;'],
      ['full y retained', 'state.pixelY = py;'],
      ['overflow crossing seed', 'PT_SOBOL_DIMENSION_COUNT,'],
      ['32-bit draw position', 'let dim = (*state).dimension;'],
      ['overflow branch', 'if (dim >= PT_SOBOL_DIMENSION_COUNT) {'],
      ['independent continuation', 'return ptSobolFallbackNext(state);'],
      [
        'fixed stream seed',
        'let streamSeed = ptSobolHash(ptSobolHashCombine(pixelSeed, (*state).sequenceKey));',
      ],
      ['dimension component seed', 'let seed = ptSobolHash(ptSobolHashCombine(streamSeed, dim));'],
      ['common index permutation', 'let shuffleSeed = ptSobolHashCombine(streamSeed, 0u);'],
      ['dimension-indexed component', 'var result = ptSobolTextureComponent(shuffledIndex, dim);'],
      ['monotonic increment', '(*state).dimension = dim + 1u;'],
    ]);
    expect(PT_WEBGPU_SOBOL_RNG_WGSL).not.toContain('ptSobolHashCombine(pathIndex, dim)');
    expect(PT_WEBGPU_SOBOL_RNG_WGSL).not.toContain('let dim = (*state) & 0xffu;');
    expect(PT_WEBGPU_SOBOL_RNG_WGSL).not.toContain('(dim + 1u) & 0xffu');
    expect(PT_WEBGPU_SOBOL_RNG_WGSL).not.toContain('dim % 256u');
    expect(PT_WEBGPU_SOBOL_RNG_WGSL).not.toContain('fn rand3(');

    const composedSobol = composePtWebgpuTraceWgsl(false, { sampling: 'sobol' });
    expectOrderedNeedles(composedSobol, [
      ['frame block', 'let sampleBlock = frameIndex >> 16u;'],
      ['seed key', 'let seedKey = ptSobolHash(frameSeed) & 0x0000ffffu;'],
      [
        'bijective block step',
        'let blockKey = (seedKey + sampleBlock * 0x00009e37u) & 0x0000ffffu;',
      ],
      ['block plus sample', 'return (blockKey << 16u) | (frameIndex & 0x0000ffffu);'],
    ]);

    const parityState = initOwenScrambledSobolStream(9, 10, sobolFrameKey(123, 0));
    parityState.dimension = SOBOL_DIMENSION_COUNT - 2;
    expect(Array.from({ length: 7 }, () => nextOwenScrambledSobolU32(parityState))).toEqual([
      0x08785900, 0xe8b6ec00, 0x9d8c7ac8, 0xed2474d0,
      0xb1a14c1e, 0x96716f84, 0x74ede162,
    ]);
    const lowDiscrepancyDimensions = Array.from(
      { length: SOBOL_DIMENSION_COUNT },
      (_unused, dim) =>
      owenScrambledSobolU32(12345, dim, 7),
    );
    expect(new Set(lowDiscrepancyDimensions).size)
      .toBe(lowDiscrepancyDimensions.length);

    expectOrderedNeedles(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL, [
      [
        'main stream seed',
        'var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(params.frameSeed, params.frameIndex));',
      ],
      ['camera jitter dims 0-1', 'let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));'],
      ['primary ray consumes jitter', 'var ray = generatePrimaryRay(gid.x, gid.y, jitter);'],
      [
        'spectral hero dims 2-3 when enabled',
        'sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng)),',
      ],
      [
        'alpha visibility uses same stream',
        'hitMaterialId(hit), hit.triIndex, hit.baryVW, hit.instanceIndex, &rng,',
      ],
      [
        'canonical light selection consumes the next dimension block',
        'lightSelection = sampleCanonicalDirectLight(',
      ],
      [
        'area-light surface pair consumes adjacent dimensions',
        'let xi1 = rand_f32(&rng);\n          let xi2 = rand_f32(&rng);',
      ],
      [
        'environment importance uses the shared stream',
        'let envSample = sampleEnvironmentImportance(&rng);',
      ],
      [
        'next-bounce source lobe uses the remaining stream',
        'let bs = sampleNextBounceDirectionWithClearcoatNormal(',
      ],
      [
        'russian roulette consumes after bounce sampling',
        'let rr = russianRoulette(&rng, throughput);',
      ],
    ]);

    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'floor(rand_f32(rng) * f32(lightCount))',
    );

    expectOrderedNeedles(PT_WEBGPU_PATH_TRACE_BSDF_WGSL, [
      ['transmission lobe mixture', 'let xiLobe = rand_f32(rng) * lobeWeightSum;'],
      [
        'transmissive microfacet normal samples the shared stream',
        'let wmLocal = sampleGgxVndfTangent(',
      ],
      [
        'transmissive sheen branch samples the shared stream',
        'let bs = charlieSheenSample(rng, -incomingDir, normal, tanT, tanB, sheenRoughness);',
      ],
      ['opaque lobe mixture', 'let xiLobe = rand_f32(rng) * lobeWeightSum;'],
      [
        'opaque diffuse branch samples the shared stream',
        'let bs = cosineHemisphereSample(rng, normal);',
      ],
    ]);

    expectOrderedNeedles(SPPM_PHOTON_PASS_WGSL, [
      [
        'photon stream seed',
        'var rng = pcgInit(photonIdx, params.frameSeed, params.frameIndex ^ 0xdeadbeefu);',
      ],
      ['photon light pick', 'floor(rand_f32(&rng) * f32(availableLightCount))'],
      [
        'directional source disk pair',
        'let r2d  = sqrt(rand_f32(&rng)) * extent;\n      let phi2 = 2.0 * PI * rand_f32(&rng);',
      ],
      [
        'point source sphere pair',
        'photonDir    = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));',
      ],
      ['rect/disc emitter pair', 'let xi1 = rand_f32(&rng);\n        let xi2 = rand_f32(&rng);'],
      [
        'environment launch disk pair',
        'let r2d = sqrt(rand_f32(&rng)) * extent;\n      let phi = 2.0 * PI * rand_f32(&rng);',
      ],
      [
        'photon hash insertion consumes no tie-breaker dimension',
        'SPPM_PHOTON_KIND_SURFACE,',
      ],
    ]);

    for (const [label, needle] of [
      ['source lobe selection uses stream', 'let xiSource = rand_f32(rng) * lobeWeightSum;'],
      ['source base split uses stream', 'if (rand_f32(rng) < specProb)'],
      [
        'suffix direct area pair uses stream',
        'let xi1r = rand_f32(rng);\n    let xi2r = rand_f32(rng);',
      ],
    ] as Array<readonly [string, string]>) {
      expect(RESTIR_PT_PRODUCER_WGSL, label).toContain(needle);
    }
    expectOrderedNeedles(RESTIR_PT_PRODUCER_WGSL, [
      [
        'producer stream seed',
        'var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(params.frameSeed, params.frameIndex));',
      ],
      ['producer camera jitter dims 0-1', 'let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));'],
      [
        'producer spectral hero dims 2-3',
        'let hero = sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng));',
      ],
      [
        'producer calls alpha-aware primary trace',
        'let vTrace = rptTraceClosestAfterAlpha(primaryRay, &rng);',
      ],
      ['producer calls source-lobe sampler', 'let wiRecon = rptSampleSourceReconnectionDirection('],
      [
        'reservoir update tie-breaker uses stream',
        '&r, xs, ns, Lo, heroLambda, pdfSrc, logCandidateWeight, &rng,',
      ],
    ]);
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'fn rptTraceClosestAfterAlpha(rayIn: Ray, rng: ptr<function, PtRngState>)',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'hitMaterialId(hit), hit.triIndex, hit.baryVW, hit.instanceIndex, rng,',
    );
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'ptRngFrameKey(params.frameSeed ^ 0x9B7Fu, params.frameIndex)',
    );
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      'ptRngFrameKey(params.frameSeed ^ 0xBCD3u, params.frameIndex)',
    );
  });

  it('keeps the maximum-bounce alpha and high-depth light-tree path in the Sobol prefix', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'let bounceLimit = max(1u, min(params.maxBounces, 8u));',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'let alphaSurfaceHitLimit = sceneSurfaceHitLimit();',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'if (alphaSurfaceHitCount >= alphaSurfaceHitLimit) {',
    );
    expect(LIGHT_TREE_TRAVERSAL_WGSL).toContain(
      'for (var guard: u32 = 0u; guard < nodeCount + 1u; guard = guard + 1u)',
    );
    expect(LIGHT_TREE_TRAVERSAL_WGSL).toContain('if (rand_f32(rng) < pL)');

    const cameraAndSpectralDraws = 4;
    const maximumBounceAlphaDraws = 8 * 8;
    const highDepthTreeNodes = 513;
    const highDepthTreeDraws = (highDepthTreeNodes - 1) / 2;
    const auditedDrawBudget = cameraAndSpectralDraws + maximumBounceAlphaDraws + highDepthTreeDraws;
    expect(auditedDrawBudget).toBeLessThanOrEqual(SOBOL_DIMENSION_COUNT);

    const stream = initOwenScrambledSobolStream(9, 10, sobolFrameKey(123, 0));
    const fallbackBeforeDraws = stream.fallbackState;
    const draws = Array.from({ length: auditedDrawBudget }, () =>
      nextOwenScrambledSobolU32(stream),
    );
    expect(draws.slice(-4)).toEqual([
      0xc4b83500, 0x41b58100, 0xf4567900, 0x0904d300,
    ]);
    expect(stream.dimension).toBe(auditedDrawBudget);
    expect(stream.fallbackState).toBe(fallbackBeforeDraws);
  });
});

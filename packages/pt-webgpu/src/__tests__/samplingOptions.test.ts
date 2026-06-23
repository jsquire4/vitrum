import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import { initOwenScrambledSobolState, owenScrambledSobolU32 } from '@vitrum/shared-samplers';
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
  composeRestirPtProducerWgsl,
  composePtWebgpuReuseWgsl,
} from '../wgsl/pathTrace/restirPtCompose.wgsl.js';
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from '../webgpuLimits.js';
import { PT_WEBGPU_SOBOL_RNG_WGSL } from '../wgsl/common.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { installGpuConstStubs } from './gpuStub.js';

const SOBOL_DIMENSION_AUDIT_2026_06_21 = true;

function expectOrderedNeedles(source: string, needles: Array<readonly [label: string, needle: string]>): void {
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
    createBuffer: vi.fn((desc?: { label?: string }) => ({ label: desc?.label ?? '', destroy: vi.fn() })),
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
    expect(composePtWebgpuTraceWgsl(false)).toContain('fn pcgNext(state: ptr<function, u32>) -> u32');
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
    const restirCombined = composePtWebgpuReuseWgsl({ sampling: 'sobol' });

    for (const wgsl of [full, lite, compositeSss, compositeBdpt, sppmPhoton, restirProducer, restirCombined]) {
      expect(wgsl).toContain('fn ptSobolNextU32(state: ptr<function, u32>) -> u32');
      expect(wgsl).toContain('fn ptSobolNestedUniformScrambleBase2(x: u32, seed: u32) -> u32');
      expect(wgsl).toContain('fn pcgInit(px: u32, py: u32, frameSeed: u32) -> u32');
      expect(wgsl).toContain('fn ptRngFrameKey(frameSeed: u32, frameIndex: u32) -> u32');
      expect(wgsl).toContain('fn rand_f32(state: ptr<function, u32>) -> f32');
      expect(wgsl).not.toContain('(*state) = (*state) * 747796405u + 2891336453u;');
    }
  });

  it('builds full and lite path-trace modules from the selected Sobol RNG', () => {
    const fullStub = makePipelineDevice();
    const full = new GpuResources(fullStub.device, 'full', false, false, undefined, 'sobol');
    full.ensurePipeline();
    expect(fullStub.shaderModules.find((m) => m.label === 'vitrum.pt-webgpu.pathTrace.full')?.code)
      .toContain('ptSobolNextU32');

    const liteStub = makePipelineDevice();
    const lite = new GpuResources(liteStub.device, 'lite', false, false, undefined, 'sobol');
    lite.ensurePipeline();
    expect(liteStub.shaderModules.find((m) => m.label === 'vitrum.pt-webgpu.pathTrace.lite')?.code)
      .toContain('ptSobolNextU32');
  });

  it('surfaces opt-in Sobol as an experimental capability with structured warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeDevice(),
      sampling: 'sobol',
      onWarning: (w) => structured.push(w),
    });

    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-sobol-sampling')).toBe(true);
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.sobol-sampling-experimental' &&
      w.details?.sampling === 'sobol' &&
      w.details?.rotation === 'ranked-8x8' &&
      Array.isArray(w.details?.promotionTails) &&
      !w.details.promotionTails.includes('owen-scrambling') &&
      !w.details.promotionTails.includes('blue-noise-rotation') &&
      !w.details.promotionTails.includes('broader-dimension-audit') &&
      w.details.promotionTails.includes('equal-time-rmse-ab'),
    )).toBe(true);
    const warningText = warn.mock.calls.map((c) => c.join(' ')).join('\n');
    expect(warningText).toContain("sampling:'sobol'");
    expect(warningText).toContain('Owen-scrambled Sobol RNG');
    expect(warningText).toContain('tiled ranked rotation');
    expect(warningText).toContain('dimension-assignment audit is pinned');
    expect(warningText).not.toContain('Owen scrambling, blue-noise');
    warn.mockRestore();
  });

  it('pins the Sobol dimension stream and assignment anchors across pt-webgpu pipelines', () => {
    expect(SOBOL_DIMENSION_AUDIT_2026_06_21).toBe(true);
    expectOrderedNeedles(PT_WEBGPU_SOBOL_RNG_WGSL, [
      ['monotonic frame sample key', 'let sampleIndex = frameSeed & 0x0000ffffu;'],
      ['per-pixel scramble slot', 'let rotationTile = ptSobolHash(ptSobolHashCombine(pixelSeed, frameSeed >> 16u)) & 0xffu;'],
      ['sample index high bits', 'let pathIndex = ((*state) >> 16u) & 0x0000ffffu;'],
      ['tile rank middle bits', 'let rotationTile = ((*state) >> 8u) & 0xffu;'],
      ['dimension low bits', 'let dim = (*state) & 0xffu;'],
      ['dimension-seeded scramble', 'let seed = ptSobolHash(ptSobolHashCombine(pathIndex, dim));'],
      ['dimension-indexed component', 'var result = ptSobolTextureComponent(shuffledIndex, dim);'],
      ['dimension-indexed rotation', 'ptSobolBlueNoiseRotation(rotationTile, dim)'],
      ['monotonic dimension increment', '((dim + 1u) & 0xffu)'],
    ]);
    expect(initOwenScrambledSobolState(9, 10, 123)).toBe(0x007b2400);
    const first32Dimensions = Array.from(
      { length: 32 },
      (_unused, dim) => owenScrambledSobolU32(12_345, dim, 7),
    );
    expect(new Set(first32Dimensions).size).toBe(first32Dimensions.length);

    expectOrderedNeedles(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL, [
      ['main stream seed', 'var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(params.frameSeed, params.frameIndex));'],
      ['camera jitter dims 0-1', 'let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));'],
      ['primary ray consumes jitter', 'var ray = generatePrimaryRay(gid.x, gid.y, jitter);'],
      ['spectral hero dims 2-3 when enabled', 'let hero = sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng));'],
      ['alpha visibility uses same stream', 'alphaTestPassThrough(hitMaterialId(hit), hit.triIndex, hit.baryVW, &rng)'],
      ['light tree selection consumes next dimensions', 'sampleLightTree(hitPos, LT_DIST2_FLOOR, params.lightTreeNodeCount, &rng)'],
      ['uniform light fallback consumes one dimension', 'floor(rand_f32(&rng) * f32(lightCount))'],
      ['area-light surface pair consumes adjacent dimensions', 'let xi1 = rand_f32(&rng);\n          let xi2 = rand_f32(&rng);'],
      ['environment importance uses the shared stream', 'let envSample = sampleEnvironmentImportance(&rng);'],
      ['next-bounce source lobe uses the remaining stream', 'let bs = sampleNextBounceDirectionWithClearcoatNormal('],
      ['russian roulette consumes after bounce sampling', 'let rr = russianRoulette(&rng, throughput);'],
    ]);

    expectOrderedNeedles(PT_WEBGPU_PATH_TRACE_BSDF_WGSL, [
      ['transmission lobe mixture', 'let xiLobe = rand_f32(rng) * lobeWeightSum;'],
      ['transmissive glossy branch samples the shared stream', 'bs = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);'],
      ['transmissive sheen branch samples the shared stream', 'let bs = charlieSheenSample(rng, -incomingDir, normal, tanT, tanB, sheenRoughness);'],
      ['opaque lobe mixture', 'let xiLobe = rand_f32(rng) * lobeWeightSum;'],
      ['opaque diffuse branch samples the shared stream', 'let bs = cosineHemisphereSample(rng, normal);'],
    ]);

    expectOrderedNeedles(SPPM_PHOTON_PASS_WGSL, [
      ['photon stream seed', 'var rng = pcgInit(photonIdx, params.frameSeed, params.frameIndex ^ 0xdeadbeefu);'],
      ['photon light pick', 'floor(rand_f32(&rng) * f32(availableLightCount))'],
      ['directional source disk pair', 'let r2d  = sqrt(rand_f32(&rng)) * extent;\n      let phi2 = 2.0 * PI * rand_f32(&rng);'],
      ['point source sphere pair', 'photonDir    = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));'],
      ['rect/disc emitter pair', 'let xi1 = rand_f32(&rng);\n        let xi2 = rand_f32(&rng);'],
      ['environment launch disk pair', 'let r2d = sqrt(rand_f32(&rng)) * extent;\n      let phi = 2.0 * PI * rand_f32(&rng);'],
      ['photon hash reservoir tie-breaker', 'sppmInsertPhoton(hp, flux, ray.direction, sppmStats.currentRadius, rand_f32(&rng));'],
    ]);

    for (const [label, needle] of [
      ['source lobe selection uses stream', 'let xiSource = rand_f32(rng) * lobeWeightSum;'],
      ['source base split uses stream', 'if (rand_f32(rng) < specProb)'],
      ['suffix direct area pair uses stream', 'let xi1r = rand_f32(rng);\n    let xi2r = rand_f32(rng);'],
    ] as Array<readonly [string, string]>) {
      expect(RESTIR_PT_PRODUCER_WGSL, label).toContain(needle);
    }
    expectOrderedNeedles(RESTIR_PT_PRODUCER_WGSL, [
      ['producer stream seed', 'var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(params.frameSeed, params.frameIndex));'],
      ['producer camera jitter dims 0-1', 'let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));'],
      ['producer spectral hero dims 2-3', 'let hero = sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng));'],
      ['producer alpha visibility uses stream', 'alphaTestPassThrough(hitMaterialId(vHit), vHit.triIndex, vHit.baryVW, &rng)'],
      ['producer calls source-lobe sampler', 'let wiRecon = rptSampleSourceReconnectionDirection('],
      ['reservoir update tie-breaker uses stream', 'updateReservoirPT(&r, xs, ns, Lo, pdfSrc, wCandidate, &rng);'],
    ]);
  });
});

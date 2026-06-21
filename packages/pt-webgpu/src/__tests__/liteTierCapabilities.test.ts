/**
 * H12 — lite-tier capabilities truth: the lite kernel uses a reduced binding
 * layout; analytic shapes, BDPT, and mesh-area emitters are absent from the
 * lite shader path. Mesh/skinned/instanced primitives are statically
 * supported by baking them into one world-space BLAS at setScene time; transform
 * and topology mutations stay unsupported because the fast paths are TLAS-based.
 *
 * B12 (2026-06-10) — point/spot/rect/disc-area emitters and HDRI environments are now
 * supported on the lite tier via texture packing (liteLightTex, liteEnvTex,
 * liteEnvCdfTex). Tests updated to reflect genuine support.
 *
 * Also covers the contract-honesty fix: lite-tier supportDetails must reflect
 * the actual lite binding budget (not the full-tier ledger).
 */
import { describe, expect, it, vi } from 'vitest';
import { asMat4, type EngineWarning, type Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE } from '../webgpuLimits.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

function makeLiteDevice(): GPUDevice {
  return {
    // Limits below the full-tier threshold → traceTier resolves to 'lite'.
    limits: {
      maxStorageBuffersPerShaderStage: 8,
      maxStorageTexturesPerShaderStage: 4,
    },
    createCommandEncoder: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeFullDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

/** A lite device that also has the GPU stubs needed to call setScene(). */
function makeLiteDeviceForSetScene(): GPUDevice {
  installGpuConstStubs();
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 8,
      maxStorageTexturesPerShaderStage: 4,
    },
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createCommandEncoder: vi.fn(),
    ...textureStubMethods(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

/** A full-tier device with the GPU stubs needed to call setScene(). */
function makeFullDeviceForSetScene(): GPUDevice {
  installGpuConstStubs();
  return {
    limits: {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 8,
    },
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createCommandEncoder: vi.fn(),
    ...textureStubMethods(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

describe('H12: lite-tier capabilities truth', () => {
  it('lite tier: supportedAnalyticShapes is empty', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    expect(engine.capabilities.supportedAnalyticShapes.size).toBe(0);
    engine.dispose();
    warn.mockRestore();
  });

  // B12 — point/spot/rect/disc-area now supported via texture packing.
  it('lite tier: supportedEmitterKinds contains directional + point + spot + rect-area + disc-area', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const kinds = engine.capabilities.supportedEmitterKinds;
    expect(kinds.has('directional')).toBe(true);
    expect(kinds.has('point')).toBe(true);
    expect(kinds.has('spot')).toBe(true);
    expect(kinds.has('rect-area')).toBe(true);
    expect(kinds.has('disc-area')).toBe(true);
    // Explicit mesh-area emitters remain unsupported (no triangle-emitter NEE path in lite kernel).
    expect(kinds.has('mesh-area')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  // B12 — HDRI environment now supported via texture packing.
  it('lite tier: supportedEnvironmentKinds includes hdri (B12)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const envs = engine.capabilities.supportedEnvironmentKinds;
    expect(envs?.has('none')).toBe(true);
    expect(envs?.has('procedural-sky')).toBe(true);
    expect(envs?.has('hdri')).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: reports transform/topology mutation fallback rebuilds and primitive limits', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    expect(engine.capabilities.incrementalPatchSupport).toEqual({
      transform: false,
      positions: false,
      material: true,
      emitter: true,
      topology: false,
    });
    const primitiveKinds = engine.capabilities.supportedPrimitiveKinds!;
    expect(primitiveKinds.has('mesh')).toBe(true);
    expect(primitiveKinds.has('skinned-mesh')).toBe(true);
    expect(primitiveKinds.has('instanced-mesh')).toBe(true);
    const sd = engine.capabilities.supportDetails!;
    expect(sd.primitives['instanced-mesh']).toBe('native');
    expect(sd.mutations.transform).toBe('fallback-rebuild');
    expect(sd.mutations.positions).toBe('fallback-rebuild');
    expect(sd.mutations.material).toBe('native');
    expect(sd.mutations.topology).toBe('fallback-rebuild');
    expect(sd.materials.baseColor).toBe('native');
    expect(sd.materials.clearcoat).toBe('native');
    expect(sd.materials.baseColorMap).toBe('unsupported');
    expect(sd.materials.normalMap).toBe('unsupported');
    expect(sd.materials.normalScale).toBe('unsupported');
    expect(sd.materials.alphaMode).toBe('unsupported');
    expect(sd.materials.opacity).toBe('unsupported');
    expect(sd.materials.thickness).toBe('approximate');
    expect(sd.materials.thicknessMap).toBe('unsupported');
    expect(sd.materials.envMapIntensity).toBe('unsupported');
    expect(sd.materials.anisotropy).toBe('unsupported');
    expect(sd.materials.anisotropyRotation).toBe('unsupported');
    expect(sd.materials.frontLayer).toBe('approximate');
    expect(sd.materials.backLayer).toBe('approximate');
    expect(sd.materials.displacementMap).toBe('approximate');
    expect(sd.materials.displacementScale).toBe('approximate');
    expect(sd.materials.displacementBias).toBe('approximate');
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: pt-webgpu-bdpt is absent from experimentalFeatures even with bdpt:true', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDevice(),
      bdpt: true,
      bdptOptions: { maxLightBounces: 2 },
      onWarning: (w) => structured.push(w),
    });
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-bdpt')).toBe(false);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-lite-tier')).toBe(true);
    expect(structured.map((w) => w.code)).toContain('pt-webgpu.lite-tier');
    expect(structured.map((w) => w.code)).not.toContain('pt-webgpu.bdpt-multivertex-research-mode');
    expect(warn.mock.calls.some((c) => String(c[0]).includes('multi-vertex BDPT research path'))).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('full tier: analytic shapes, all emitter kinds, hdri, and bdpt are available', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeFullDevice(), bdpt: true });
    expect(engine.capabilities.supportedAnalyticShapes.size).toBeGreaterThan(0);
    expect(engine.capabilities.supportedEmitterKinds.has('rect-area')).toBe(true);
    expect(engine.capabilities.supportedEnvironmentKinds?.has('hdri')).toBe(true);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-bdpt')).toBe(true);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-lite-tier')).toBe(false);
    engine.dispose();
  });

  // B12 — point/spot/rect-area upgraded to 'native'.
  it('lite tier: supportDetails emitters shows point/spot/rect-area as native (B12)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const sd = engine.capabilities.supportDetails!;
    expect(sd.emitters.directional).toBe('native');
    expect(sd.emitters.point).toBe('native');
    expect(sd.emitters.spot).toBe('native');
    expect(sd.emitters['rect-area']).toBe('native');
    expect(sd.emitters['disc-area']).toBe('native');
    // Explicit mesh-area emitters remain unsupported.
    expect(sd.emitters['mesh-area']).toBe('unsupported');
    engine.dispose();
    warn.mockRestore();
  });

  // B12 — HDRI env upgraded to 'native'.
  it('lite tier: supportDetails environments shows hdri as native (B12)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const sd = engine.capabilities.supportDetails!;
    expect(sd.environments.hdri).toBe('native');
    expect(sd.environments.none).toBe('native');
    // procedural-sky uses the shared finite-resolution Preetham equirect bake.
    expect(sd.environments['procedural-sky']).toBe('approximate');
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: supportDetails analyticShapes all unsupported', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const sd = engine.capabilities.supportDetails!;
    for (const grade of Object.values(sd.analyticShapes)) {
      expect(grade).toBe('unsupported');
    }
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: supportDetails analytic primitive is unsupported', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const sd = engine.capabilities.supportDetails!;
    expect(sd.primitives.analytic).toBe('unsupported');
    engine.dispose();
    warn.mockRestore();
  });

  it('full tier: supportDetails emitters all native', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeFullDevice() });
    const sd = engine.capabilities.supportDetails!;
    expect(sd.emitters.directional).toBe('native');
    expect(sd.emitters.point).toBe('native');
    expect(sd.emitters['rect-area']).toBe('native');
    engine.dispose();
  });

  it('lite tier: setScene warns when scene contains analytic primitives', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    // The warn must fire BEFORE the GPU upload — use try/catch for GPU stub noise.
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'a',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),  // cx,cy,cz,radius
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('analytic') && c.includes('Lite tier'))).toBe(true);
    expect(structured).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'pt-webgpu.lite-analytic-primitive',
        details: expect.objectContaining({
          count: 1,
          primitiveIds: ['a'],
          primitiveKinds: ['analytic'],
          requiredTier: 'full',
          fallback: 'ignore-unsupported-lite-primitive',
        }),
      }),
    ]));
    expect(calls.some((c) => c.includes('silently ignored'))).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: setScene does NOT warn for static instanced meshes', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'instanced-mesh',
          id: 'im',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          instances: [asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1])],
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('instanced-mesh') && c.includes('Lite tier'))).toBe(false);
    expect(structured.some((w) => w.code === 'pt-webgpu.lite-instanced-mesh')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: setScene does NOT warn for static mesh transforms', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          transform: asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('non-identity transforms') && c.includes('Lite tier'))).toBe(false);
    expect(structured.some((w) => w.code === 'pt-webgpu.lite-primitive-transform')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('setScene warns when displacement map handles are not CPU-readable', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.8, 0.2, 0.1],
            roughness: 0.3,
            metallic: 0,
            displacementMap: { handle: { id: 'height' } },
            displacementScale: 0.2,
            displacementBias: -0.1,
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.scene-pack-warning' &&
      w.message.includes('displacementMap') &&
      w.message.includes('vertex displacement skipped') &&
      typeof w.details?.warning === 'string' &&
      w.details.warning.includes('Primitive "m" displacementMap')
    )).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it('setScene warns when receiveShadow:false is supplied', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.8, 0.2, 0.1],
            roughness: 0.3,
            metallic: 0,
          },
          receiveShadow: false,
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) =>
      c.includes('receiveShadow:false') &&
      c.includes('m'),
    )).toBe(true);
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.reserved-receive-shadow' &&
      w.phase === 'setScene' &&
      w.method === 'setScene' &&
      Array.isArray(w.details?.primitiveIds) &&
      w.details.primitiveIds.includes('m'),
    )).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it('updatePrimitive warns when receiveShadow:false is supplied', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    try {
      engine.setScene({
        primitives: [
          {
            kind: 'mesh',
            id: 'm',
            positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
            material: {
              baseColor: [0.8, 0.2, 0.1],
              roughness: 0.3,
              metallic: 0,
            },
          },
        ],
        emitters: [],
        environment: { kind: 'none' },
      });
      structured.length = 0;
      warn.mockClear();

      expect(engine.updatePrimitive).toBeTypeOf('function');
      engine.updatePrimitive!('m', { receiveShadow: false } as never);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.reserved-receive-shadow' &&
      w.phase === 'mutation' &&
      w.method === 'updatePrimitive' &&
      Array.isArray(w.details?.primitiveIds) &&
      w.details.primitiveIds.includes('m'),
    )).toBe(true);
    expect(warn.mock.calls.flat().map(String).some((m) =>
      m.includes('updatePrimitive("m")') && m.includes('receiveShadow:false'),
    )).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it('full tier updatePrimitive warns with primitive-scoped details for scalar displacement fields', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeFullDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    try {
      engine.setScene({
        primitives: [
          {
            kind: 'mesh',
            id: 'm',
            positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
            material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
          },
        ],
        emitters: [],
        environment: { kind: 'none' },
      });

      expect(engine.updatePrimitive).toBeTypeOf('function');
      engine.updatePrimitive!('m', {
        material: {
          roughness: 0.42,
          displacementScale: 0.2,
          displacementBias: -0.1,
        },
      } as never);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }

    const repackWarnings = structured.filter((w) =>
      w.code === 'pt-webgpu.primitive-material-repack'
    );
    expect(repackWarnings).toHaveLength(1);
    expect(repackWarnings[0]).toMatchObject({
      backend: 'pt-webgpu',
      phase: 'mutation',
      method: 'updatePrimitive',
      details: {
        id: 'm',
        fallbackReason: 'material-texture-descriptor-repack',
        geometryFields: ['displacementBias', 'displacementScale'],
      },
    });
    engine.dispose();
    warn.mockRestore();
  });

  it('full tier photon-map honors castShadow:false emitter source treatment without approximation warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeFullDeviceForSetScene(),
      causticStrategy: 'photon-map',
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'receiver',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [
        {
          kind: 'point',
          id: 'ghost-point',
          color: [1, 1, 1],
          intensity: 4,
          position: [0, 2, 1],
          castShadow: false,
        },
      ],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const approx = structured.find((w) =>
      w.code === 'pt-webgpu.sppm-emitter-cast-shadow-approximation'
    );
    expect(approx).toBeUndefined();
    expect(warn.mock.calls.some((c) =>
      c.join(' ').includes('SPPM photon-map source treatment remains approximate'),
    )).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: setScene warns for thicknessMap but not scalar thickness (CAP-01)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.8, 0.2, 0.1],
            roughness: 0.3,
            metallic: 0,
            // Scalar thickness is shared-buffer approximate; thicknessMap is
            // unsupported only on lite because it needs full-tier group-3 maps.
            thickness: 0.2,
            thicknessMap: { handle: { id: 'thickness' } },
            displacementMap: { handle: { id: 'displacement' } },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('thicknessMap') && !c.includes('displacementMap'))).toBe(true);
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.unsupported-material-fields' &&
      Array.isArray(w.details?.fields) &&
      w.details.fields.includes('thicknessMap') &&
      !w.details.fields.includes('thickness') &&
      !w.details.fields.includes('displacementMap') &&
      Array.isArray(w.details?.primitiveIds) &&
      w.details.primitiveIds.includes('m') &&
      Array.isArray(w.details?.primitiveFields) &&
      w.details.primitiveFields.some((entry) =>
        entry.primitiveId === 'm' &&
        entry.fields.includes('thicknessMap') &&
        !entry.fields.includes('thickness') &&
        !entry.fields.includes('displacementMap')
      ),
    )).toBe(true);
    // Consumed fields must NOT appear in the warning.
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.unsupported-material-fields' &&
      Array.isArray(w.details?.fields) &&
      (w.details.fields.includes('baseColor') || w.details.fields.includes('roughness')),
    )).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: setScene warns when full-tier material texture and alpha fields are supplied', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.8, 0.2, 0.1],
            roughness: 0.3,
            metallic: 0,
            baseColorMap: { handle: { id: 'albedo' } },
            normalMap: { handle: { id: 'normal' } },
            normalScale: 0.5,
            alphaMode: 'blend',
            opacity: 0.5,
            envMapIntensity: 0.25,
            anisotropy: 0.5,
            frontLayer: {
              transmission: [1, 1, 1],
              normalMap: { handle: { id: 'front-normal' } },
              normalScale: 0.75,
            },
            backLayer: {
              transmission: [1, 1, 1],
              normalMap: { handle: { id: 'back-normal' } },
              normalScale: 0.5,
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw — expected */
    }
    expect(structured).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'pt-webgpu.unsupported-material-fields',
        details: expect.objectContaining({
          fields: expect.arrayContaining([
            'baseColorMap',
            'normalMap',
            'normalScale',
            'alphaMode',
            'opacity',
            'envMapIntensity',
            'anisotropy',
            'frontLayer.normalMap',
            'frontLayer.normalScale',
            'backLayer.normalMap',
            'backLayer.normalScale',
          ]),
          primitiveIds: ['m'],
          primitiveFields: expect.arrayContaining([
            expect.objectContaining({
              primitiveId: 'm',
              fields: expect.arrayContaining([
                'baseColorMap',
                'normalMap',
                'normalScale',
                'alphaMode',
                'opacity',
                'envMapIntensity',
                'anisotropy',
                'frontLayer.normalMap',
                'frontLayer.normalScale',
                'backLayer.normalMap',
                'backLayer.normalScale',
              ]),
            }),
          ]),
        }),
      }),
    ]));
    engine.dispose();
    warn.mockRestore();
  });

  it('full tier: setScene does not warn unsupported-material-fields for per-face layer normal fields', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeFullDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'layered',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.8, 0.2, 0.1],
            roughness: 0.3,
            metallic: 0,
            frontLayer: {
              transmission: [0.9, 0.8, 0.7],
              roughness: 0.2,
              normalMap: { handle: { id: 'front-normal' } },
              normalScale: 0.75,
            },
            backLayer: {
              transmission: [0.7, 0.8, 0.9],
              roughness: 0.6,
              normalMap: { handle: { id: 'back-normal' } },
              normalScale: 0.5,
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw — expected */
    }
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.unsupported-material-fields' &&
      Array.isArray(w.details?.fields) &&
      (
        w.details.fields.includes('frontLayer.normalMap') ||
        w.details.fields.includes('frontLayer.normalScale') ||
        w.details.fields.includes('backLayer.normalMap') ||
        w.details.fields.includes('backLayer.normalScale') ||
        w.details.fields.includes('frontLayer') ||
        w.details.fields.includes('backLayer')
      ),
    )).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: setScene warns when non-constant COLOR_0 vertex colors would be dropped', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'colored',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw — expected */
    }
    expect(structured).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'pt-webgpu.lite-unsupported-vertex-colors',
        details: expect.objectContaining({
          primitiveIds: ['colored'],
          bakedWhen: 'constant-rgb-alpha-one',
          requiredTier: 'full',
        }),
      }),
    ]));
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: setScene does not warn for constant RGB COLOR_0 that can bake into baseColor', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'constant-colored',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          colors: new Float32Array([
            0.5, 0.25, 1,
            0.5, 0.25, 1,
            0.5, 0.25, 1,
          ]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw — expected */
    }
    expect(structured.some((w) => w.code === 'pt-webgpu.lite-unsupported-vertex-colors')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it('setScene does NOT emit unsupported-material-fields for a fully supported material (CAP-01)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.8, 0.2, 0.1],
            roughness: 0.3,
            metallic: 0,
            clearcoat: 0.25,
            specularIntensity: 0.5,
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw — expected */
    }
    expect(structured.some((w) => w.code === 'pt-webgpu.unsupported-material-fields')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  // B12 — point emitters no longer warn (they are now supported via texture packing).
  it('lite tier: setScene does NOT warn for point/spot/rect-area emitters (B12 supported)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDeviceForSetScene() });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [{ kind: 'point', id: 'p', position: [0, 0, 0], color: [1, 1, 1], intensity: 1 }],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw — expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    // Must NOT warn that point lights are unsupported.
    expect(calls.some((c) => c.includes('point') && c.toLowerCase().includes('unsupported'))).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  // B12 follow-up — disc-area is native in lite via the rect/disc texture record;
  // explicit mesh-area remains unsupported.
  it('lite tier: setScene does not warn for disc-area but still warns for mesh-area emitters', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [
        { kind: 'disc-area', id: 'd', position: [0, 1, 0], normal: [0, -1, 0], radius: 0.5, color: [1, 1, 1], intensity: 1 },
        { kind: 'mesh-area', id: 'ma', meshId: 'm', color: [1, 1, 1], intensity: 1 },
      ],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('disc-area') && c.includes('Lite tier'))).toBe(false);
    expect(calls.some((c) => c.includes('mesh-area') && c.includes('Lite tier'))).toBe(true);
    expect(structured).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'pt-webgpu.lite-unsupported-emitters',
        details: expect.objectContaining({
          kinds: 'mesh-area',
          count: 1,
          emitterIds: ['ma'],
          emitterKinds: ['mesh-area'],
          requiredTier: 'full',
          fallback: 'ignore-unsupported-lite-emitter',
        }),
      }),
    ]));
    expect(calls.some((c) => c.includes('silently ignored'))).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  // Item 19 closure — lite tier now packs every directional emitter into liteLightTex.
  it('lite tier: setScene does NOT warn when scene contains multiple directional emitters', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [
        { kind: 'directional', id: 'd1', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1 },
        { kind: 'directional', id: 'd2', direction: [1, -1, 0], color: [0.8, 0.9, 1], intensity: 0.5 },
      ],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after setScene — expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('first directional') || (c.includes('directional') && c.includes('only')))).toBe(false);
    expect(structured.some((w) => w.code === 'pt-webgpu.lite-multiple-directional-emitters')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  // Item 19 — exactly 1 directional does NOT trigger the multi-directional warn.
  it('lite tier: setScene does NOT warn for exactly 1 directional emitter', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDeviceForSetScene() });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [
        { kind: 'directional', id: 'd1', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1 },
      ],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw — expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    // Must NOT warn about first-only directional rendering for a single-directional scene.
    expect(calls.some((c) => c.includes('first directional') || c.includes('directional') && c.includes('only'))).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  // B12 — HDRI environments are supported via texture packing; unreadable opaque
  // handles are a payload-readability fallback, not an unsupported-environment warning.
  it('lite tier: setScene emits structured unreadable-HDRI fallback without unsupported warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeLiteDeviceForSetScene(),
      onWarning: (w) => structured.push(w),
    });
    warn.mockClear();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'hdri', hdri: undefined as unknown as NonNullable<unknown> },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw — expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    // Must NOT warn that hdri is unsupported.
    expect(calls.some((c) => c.includes('hdri') && c.toLowerCase().includes('unsupported'))).toBe(false);
    expect(structured).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'pt-webgpu.hdri-unreadable',
        backend: 'pt-webgpu',
        phase: 'setScene',
        method: 'setScene',
        details: expect.objectContaining({
          fallback: 'no-environment',
          warning: expect.stringContaining('lacks CPU pixel data'),
        }),
      }),
    ]));
    expect(
      structured.some(
        (w) =>
          w.code === 'pt-webgpu.scene-pack-warning' &&
          typeof w.details?.warning === 'string' &&
          w.details.warning.includes('HDRI environment'),
      ),
    ).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });
});

/**
 * H12 — lite-tier capabilities truth: the lite kernel uses a reduced binding
 * layout; analytic shapes, BDPT, disc-area, and mesh-area emitters are absent
 * from the lite shader path. Mesh/skinned/instanced primitives are statically
 * supported by baking them into one world-space BLAS at setScene time; transform
 * and topology mutations stay unsupported because the fast paths are TLAS-based.
 *
 * B12 (2026-06-10) — point/spot/rect-area emitters and HDRI environments are now
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

describe('H12: lite-tier capabilities truth', () => {
  it('lite tier: supportedAnalyticShapes is empty', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    expect(engine.capabilities.supportedAnalyticShapes.size).toBe(0);
    engine.dispose();
    warn.mockRestore();
  });

  // B12 — point/spot/rect-area now supported via texture packing.
  it('lite tier: supportedEmitterKinds contains directional + point + spot + rect-area', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    const kinds = engine.capabilities.supportedEmitterKinds;
    expect(kinds.has('directional')).toBe(true);
    expect(kinds.has('point')).toBe(true);
    expect(kinds.has('spot')).toBe(true);
    expect(kinds.has('rect-area')).toBe(true);
    // disc-area and mesh-area remain unsupported (no NEE path in lite kernel).
    expect(kinds.has('disc-area')).toBe(false);
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

  it('lite tier: reports transform/topology mutation gaps and primitive limits', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice() });
    expect(engine.capabilities.incrementalPatchSupport).toEqual({
      transform: false,
      positions: true,
      material: false,
      emitter: true,
      topology: false,
    });
    const primitiveKinds = engine.capabilities.supportedPrimitiveKinds!;
    expect(primitiveKinds.has('mesh')).toBe(true);
    expect(primitiveKinds.has('skinned-mesh')).toBe(true);
    expect(primitiveKinds.has('instanced-mesh')).toBe(true);
    const sd = engine.capabilities.supportDetails!;
    expect(sd.primitives['instanced-mesh']).toBe('native');
    expect(sd.mutations.transform).toBe('unsupported');
    expect(sd.mutations.material).toBe('fallback-rebuild');
    expect(sd.mutations.topology).toBe('unsupported');
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
    expect(sd.materials.displacementMap).toBe('unsupported');
    expect(sd.materials.displacementScale).toBe('unsupported');
    expect(sd.materials.displacementBias).toBe('unsupported');
    engine.dispose();
    warn.mockRestore();
  });

  it('lite tier: pt-webgpu-bdpt is absent from experimentalFeatures even with bdpt:true', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({ device: makeLiteDevice(), bdpt: true });
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-bdpt')).toBe(false);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-lite-tier')).toBe(true);
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
    // disc-area and mesh-area remain unsupported.
    expect(sd.emitters['disc-area']).toBe('unsupported');
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
    const engine = await createPTEngine_WebGPU({ device: makeLiteDeviceForSetScene() });
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

  it('setScene warns when displacement material fields are supplied', async () => {
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
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) =>
      c.includes('displacementMap') &&
      c.includes('displacementScale') &&
      c.includes('displacementBias'),
    )).toBe(true);
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.unsupported-displacement-material' &&
      Array.isArray(w.details?.fields) &&
      w.details.fields.includes('displacementMap') &&
      w.details.fields.includes('displacementScale') &&
      w.details.fields.includes('displacementBias'),
    )).toBe(true);
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
      !w.details.fields.includes('displacementMap'),
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
          ]),
        }),
      }),
    ]));
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

  // B12 — disc-area and mesh-area still warn (no NEE path in lite kernel).
  it('lite tier: setScene warns when scene contains disc-area or mesh-area emitters', async () => {
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
      emitters: [{ kind: 'disc-area', id: 'd', position: [0, 1, 0], normal: [0, -1, 0], radius: 0.5, color: [1, 1, 1], intensity: 1 }],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — that's expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('disc-area') && c.includes('Lite tier'))).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  // Item 19 — lite tier warns when scene has ≥2 directional emitters (first-only rendering).
  it('lite tier: setScene warns when scene contains ≥2 directional emitters (item 19)', async () => {
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
        { kind: 'directional', id: 'd2', direction: [1, -1, 0], color: [0.8, 0.9, 1], intensity: 0.5 },
      ],
      environment: { kind: 'none' },
    };
    try {
      engine.setScene(scene);
    } catch {
      /* GPU stubs may throw after the warn — expected */
    }
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('directional') && c.includes('2'))).toBe(true);
    expect(calls.some((c) => c.includes('lite') || c.includes('Lite'))).toBe(true);
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

  // B12 — HDRI environments no longer warn (supported via texture packing).
  it('lite tier: setScene does NOT warn for hdri environment (B12 supported)', async () => {
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
    engine.dispose();
    warn.mockRestore();
  });
});

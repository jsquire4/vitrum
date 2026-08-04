import { describe, expect, it, vi } from 'vitest';
import { SceneBvh } from '@vitrum/shared-bvh';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { ProbeGrid } from '../probeGrid.js';
import { ProbeUpdateAtlasTextureCache } from '../probeUpdateAtlasCache.js';
import type { ProbeUpdateGpuState } from '../probeUpdateGpuState.js';
import { ProbeUpdatePass } from '../probeUpdatePass.js';
import { readPackedProbeStateFromIrradianceAtlas } from '../probeState.js';

installWebGPUPolyfills();

interface TrackedBuffer {
  readonly gpu: GPUBuffer;
  readonly destroy: ReturnType<typeof vi.fn>;
}

interface TrackedTexture {
  readonly gpu: GPUTexture;
  readonly destroy: ReturnType<typeof vi.fn>;
}

interface PassInternals {
  _gpu: ProbeUpdateGpuState | null;
  _grid: ProbeGrid;
  _atlasCache: ProbeUpdateAtlasTextureCache;
  _maxProbes: number;
  _emitterTrisData: Float32Array;
  _emitterTrisCount: number;
  _hasEnv: boolean;
  _envRotationY: number;
  _envIntensity: number;
  _envMapView: GPUTextureView | null;
  _lastTangentSource: ArrayBuffer | null;
  _lastVertexColorSource: ArrayBuffer | null;
  _ensureRayResultsCapacity(device: GPUDevice, maxProbes: number): void;
  _uploadActiveProbeIndices(device: GPUDevice, active: Uint32Array<ArrayBuffer>): void;
  _uploadEmitterTris(device: GPUDevice): void;
  _syncTangentTexture(device: GPUDevice, data: ArrayBuffer | null): void;
  _syncVertexColorTexture(device: GPUDevice, data: ArrayBuffer | null): void;
  _readbackRgba16f(
    device: GPUDevice,
    texture: GPUTexture,
    width: number,
    height: number,
  ): Promise<Uint16Array>;
}

function makePass(): { readonly pass: ProbeUpdatePass; readonly internal: PassInternals } {
  const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
  return { pass, internal: pass as unknown as PassInternals };
}

function trackedBuffer(size: number): TrackedBuffer {
  const destroy = vi.fn();
  return {
    gpu: { size, destroy } as unknown as GPUBuffer,
    destroy,
  };
}

function trackedTexture(
  width = 1,
  height = 1,
  createView: () => GPUTextureView = vi.fn(() => ({} as GPUTextureView)),
): TrackedTexture {
  const destroy = vi.fn();
  return {
    gpu: {
      width,
      height,
      createView,
      destroy,
    } as unknown as GPUTexture,
    destroy,
  };
}

describe('ProbeUpdatePass candidate-first resource replacement', () => {
  it('preserves ray-results identity and max-probe capacity after allocation failure', () => {
    const { internal } = makePass();
    const previous = trackedBuffer(64);
    internal._gpu = { rayResultsBuf: previous.gpu } as ProbeUpdateGpuState;
    internal._maxProbes = 1;
    let fail = true;
    const candidates: TrackedBuffer[] = [];
    const device = {
      createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
        if (fail) {
          fail = false;
          throw new Error('ray allocation failed');
        }
        const candidate = trackedBuffer(descriptor.size);
        candidates.push(candidate);
        return candidate.gpu;
      }),
    } as unknown as GPUDevice;

    expect(() => internal._ensureRayResultsCapacity(device, 2)).toThrow('ray allocation failed');
    expect(internal._gpu.rayResultsBuf).toBe(previous.gpu);
    expect(internal._maxProbes).toBe(1);
    expect(previous.destroy).not.toHaveBeenCalled();

    internal._ensureRayResultsCapacity(device, 2);
    expect(internal._gpu.rayResultsBuf).toBe(candidates[0]!.gpu);
    expect(internal._maxProbes).toBe(2);
    expect(previous.destroy).toHaveBeenCalledTimes(1);
  });

  it('writes an active-probe growth candidate before publication and retries a failed write', () => {
    const { internal } = makePass();
    const previous = trackedBuffer(4);
    internal._gpu = { activeProbesBuf: previous.gpu } as ProbeUpdateGpuState;
    const candidates: TrackedBuffer[] = [];
    let failWrite = true;
    const device = {
      createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
        const candidate = trackedBuffer(descriptor.size);
        candidates.push(candidate);
        return candidate.gpu;
      }),
      queue: {
        writeBuffer: vi.fn(() => {
          if (failWrite) {
            failWrite = false;
            throw new Error('active write failed');
          }
        }),
      },
    } as unknown as GPUDevice;
    const active = new Uint32Array([0, 1]);

    expect(() => internal._uploadActiveProbeIndices(device, active)).toThrow('active write failed');
    expect(internal._gpu.activeProbesBuf).toBe(previous.gpu);
    expect(previous.destroy).not.toHaveBeenCalled();
    expect(candidates[0]!.destroy).toHaveBeenCalledTimes(1);

    internal._uploadActiveProbeIndices(device, active);
    expect(internal._gpu.activeProbesBuf).toBe(candidates[1]!.gpu);
    expect(previous.destroy).toHaveBeenCalledTimes(1);
  });

  it('writes an emitter-triangle growth candidate before publication and retries a failed write', () => {
    const { internal } = makePass();
    const previous = trackedBuffer(80);
    internal._gpu = {
      emitterTrisBuf: previous.gpu,
      emitterTrisCount: 0,
    } as ProbeUpdateGpuState;
    // Two 20-float records plus two 4-float represented-alias entries.
    internal._emitterTrisData = new Float32Array(48);
    internal._emitterTrisCount = 2;
    const candidates: TrackedBuffer[] = [];
    let failWrite = true;
    const device = {
      createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
        const candidate = trackedBuffer(descriptor.size);
        candidates.push(candidate);
        return candidate.gpu;
      }),
      queue: {
        writeBuffer: vi.fn(() => {
          if (failWrite) {
            failWrite = false;
            throw new Error('emitter write failed');
          }
        }),
      },
    } as unknown as GPUDevice;

    expect(() => internal._uploadEmitterTris(device)).toThrow('emitter write failed');
    expect(internal._gpu.emitterTrisBuf).toBe(previous.gpu);
    expect(internal._gpu.emitterTrisCount).toBe(0);
    expect(previous.destroy).not.toHaveBeenCalled();
    expect(candidates[0]!.destroy).toHaveBeenCalledTimes(1);

    internal._uploadEmitterTris(device);
    expect(internal._gpu.emitterTrisBuf).toBe(candidates[1]!.gpu);
    expect(internal._gpu.emitterTrisCount).toBe(2);
    expect(previous.destroy).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['tangent', '_syncTangentTexture', 'bvhTangentTexture', '_lastTangentSource'],
    ['vertex-color', '_syncVertexColorTexture', 'bvhVertexColorTexture', '_lastVertexColorSource'],
  ] as const)(
    'preserves the prior %s texture when candidate view creation fails',
    (_name, method, textureField, sourceField) => {
      const { internal } = makePass();
      const previous = trackedTexture();
      internal._gpu = { [textureField]: previous.gpu } as unknown as ProbeUpdateGpuState;
      internal[sourceField] = null;
      const candidates: TrackedTexture[] = [];
      let failView = true;
      const device = {
        createTexture: vi.fn((descriptor: GPUTextureDescriptor) => {
          const size = descriptor.size as GPUExtent3DDict;
          const candidate = trackedTexture(
            Number(size.width),
            Number(size.height),
            vi.fn(() => {
              if (failView) {
                failView = false;
                throw new Error('candidate view failed');
              }
              return {} as GPUTextureView;
            }),
          );
          candidates.push(candidate);
          return candidate.gpu;
        }),
        queue: { writeTexture: vi.fn() },
        limits: { maxTextureDimension2D: 8192 },
      } as unknown as GPUDevice;
      const source = new ArrayBuffer(16);

      expect(() => internal[method](device, source)).toThrow('candidate view failed');
      expect(internal._gpu[textureField]).toBe(previous.gpu);
      expect(internal[sourceField]).toBeNull();
      expect(previous.destroy).not.toHaveBeenCalled();
      expect(candidates[0]!.destroy).toHaveBeenCalledTimes(1);

      internal[method](device, source);
      expect(internal._gpu[textureField]).toBe(candidates[1]!.gpu);
      expect(internal[sourceField]).toBe(source);
      expect(previous.destroy).toHaveBeenCalledTimes(1);
    },
  );

  it(
    'preserves the prior visibility scratch texture and cache size after allocation failure',
    () => {
      const cache = new ProbeUpdateAtlasTextureCache();
      const gpu = { visScratchTex: null } as ProbeUpdateGpuState;
      let fail = false;
      const created: TrackedTexture[] = [];
      const device = {
        createTexture: vi.fn((descriptor: GPUTextureDescriptor) => {
          if (fail) {
            fail = false;
            throw new Error('scratch allocation failed');
          }
          const size = descriptor.size as readonly number[];
          const candidate = trackedTexture(Number(size[0]), Number(size[1]));
          created.push(candidate);
          return candidate.gpu;
        }),
      } as unknown as GPUDevice;
      const firstAtlas = { width: 2, height: 2 } as GPUTexture;
      const secondAtlas = { width: 4, height: 4 } as GPUTexture;
      const previous = cache.getOrCreateScratchTexture(device, gpu, firstAtlas);

      fail = true;
      expect(() => cache.getOrCreateScratchTexture(device, gpu, secondAtlas))
        .toThrow('scratch allocation failed');
      expect(gpu.visScratchTex).toBe(previous);
      expect(created[0]!.destroy).not.toHaveBeenCalled();

      const replacement = cache.getOrCreateScratchTexture(device, gpu, secondAtlas);
      expect(replacement).not.toBe(previous);
      expect(created[0]!.destroy).toHaveBeenCalledTimes(1);
    },
  );

  it('preserves an external environment binding when placeholder view creation fails', () => {
    const { pass, internal } = makePass();
    const externalView = {} as GPUTextureView;
    const sampler = {} as GPUSampler;
    const candidates: TrackedTexture[] = [];
    let failView = true;
    const device = {
      createTexture: vi.fn(() => {
        const candidate = trackedTexture(1, 1, vi.fn(() => {
          if (failView) {
            failView = false;
            throw new Error('env view failed');
          }
          return {} as GPUTextureView;
        }));
        candidates.push(candidate);
        return candidate.gpu;
      }),
    } as unknown as GPUDevice;
    internal._gpu = {
      device,
      envMapView: externalView,
      envMapOwnedByPass: false,
      envMapPlaceholderTex: null,
    } as ProbeUpdateGpuState;
    internal._hasEnv = true;
    internal._envRotationY = 0.75;
    internal._envIntensity = 3;
    internal._envMapView = externalView;

    expect(() => pass.setEnvironment(null, null, 0, 0, false)).toThrow('env view failed');
    expect(internal._gpu.envMapView).toBe(externalView);
    expect(internal._gpu.envMapOwnedByPass).toBe(false);
    expect(internal._gpu.envMapPlaceholderTex).toBeNull();
    expect(candidates[0]!.destroy).toHaveBeenCalledTimes(1);
    expect(internal._hasEnv).toBe(true);
    expect(internal._envRotationY).toBe(0.75);
    expect(internal._envIntensity).toBe(3);
    expect(internal._envMapView).toBe(externalView);

    pass.setEnvironment(null, null, 0, 0, false);
    expect(internal._gpu.envMapView).not.toBe(externalView);
    expect(internal._gpu.envMapOwnedByPass).toBe(true);
    expect(internal._gpu.envMapPlaceholderTex).toBe(candidates[1]!.gpu);
  });

  it.each(['encoder', 'finish', 'submit'] as const)(
    'destroys readback staging after %s failure',
    async (stage) => {
      const { internal } = makePass();
      const staging = trackedBuffer(256);
      const encoder = {
        copyTextureToBuffer: vi.fn(),
        finish: vi.fn(() => {
          if (stage === 'finish') throw new Error('finish failed');
          return {} as GPUCommandBuffer;
        }),
      } as unknown as GPUCommandEncoder;
      const device = {
        createBuffer: vi.fn(() => staging.gpu),
        createCommandEncoder: vi.fn(() => {
          if (stage === 'encoder') throw new Error('encoder failed');
          return encoder;
        }),
        queue: {
          submit: vi.fn(() => {
            if (stage === 'submit') throw new Error('submit failed');
          }),
        },
      } as unknown as GPUDevice;

      await expect(internal._readbackRgba16f(device, {} as GPUTexture, 1, 1))
        .rejects.toThrow(`${stage} failed`);
      expect(staging.destroy).toHaveBeenCalledTimes(1);
    },
  );

  it.each([1, 2])(
    'keeps all live DDGI texture identities after import upload #%i fails, then retries',
    (failUploadAt) => {
      const { pass, internal } = makePass();
      internal._grid.computeFromBounds({ min: [0, 0, 0], max: [1, 1, 1] }, 100, 3);
      internal._grid.allocateAtlases();
      const irrSlot = internal._grid.irradianceReadTex;
      const visSlot = internal._grid.visibilityReadTex;
      if (!irrSlot || !visSlot) throw new Error('expected allocated DDGI atlas slots');
      const created: TrackedTexture[] = [];
      let uploadCount = 0;
      let failureAvailable = true;
      const device = {
        createTexture: vi.fn((descriptor: GPUTextureDescriptor) => {
          const size = descriptor.size as readonly number[];
          const texture = trackedTexture(Number(size[0]), Number(size[1]));
          created.push(texture);
          return texture.gpu;
        }),
        queue: {
          writeTexture: vi.fn(() => {
            uploadCount += 1;
            if (failureAvailable && uploadCount === failUploadAt) {
              failureAvailable = false;
              throw new Error(`atlas upload ${failUploadAt}`);
            }
          }),
        },
      } as unknown as GPUDevice;
      const liveIrr = internal._atlasCache.getOrCreateAtlasTexture(
        device,
        irrSlot,
        'rgba16float',
      );
      const liveVis = internal._atlasCache.getOrCreateAtlasTexture(
        device,
        visSlot,
        'rgba16float',
      );
      const probeStateData = new Float32Array(internal._grid.probeCount * 4);
      probeStateData.set([
        internal._grid.worldSpacing * 0.25,
        0,
        0,
        1,
      ]);
      const snap = {
        irrW: irrSlot.width,
        irrH: irrSlot.height,
        visW: visSlot.width,
        visH: visSlot.height,
        probeStateW: internal._grid.dims.x,
        probeStateH: internal._grid.dims.y * internal._grid.dims.z,
        irrData: new Uint16Array(irrSlot.width * irrSlot.height * 4),
        visData: new Uint16Array(visSlot.width * visSlot.height * 4),
        probeStateData,
      };

      expect(() => pass.importAtlasData(device, snap)).toThrow(`atlas upload ${failUploadAt}`);
      expect(pass.getReadAtlasGPUTextures()).toEqual({
        irradiance: liveIrr,
        visibility: liveVis,
      });
      expect(created[0]!.destroy).not.toHaveBeenCalled();
      expect(created[1]!.destroy).not.toHaveBeenCalled();
      for (const candidate of created.slice(2)) {
        expect(candidate.destroy).toHaveBeenCalledTimes(1);
      }

      expect(pass.importAtlasData(device, snap)).toBe(true);
      expect(pass.getReadAtlasGPUTextures()).toEqual({
        irradiance: created.at(-2)!.gpu,
        visibility: created.at(-1)!.gpu,
      });
      expect(created[0]!.destroy).toHaveBeenCalledTimes(1);
      expect(created[1]!.destroy).toHaveBeenCalledTimes(1);

      const latestUploads = vi.mocked(device.queue.writeTexture).mock.calls.slice(-2);
      const uploadedIrradiance = latestUploads[0]![1] as Uint16Array;
      const decoded = readPackedProbeStateFromIrradianceAtlas(
        uploadedIrradiance,
        {
          dimsX: internal._grid.dims.x,
          dimsY: internal._grid.dims.y,
          dimsZ: internal._grid.dims.z,
          irradianceWidth: irrSlot.width,
          irradianceHeight: irrSlot.height,
          spacing: internal._grid.worldSpacing,
        },
      );
      expect(decoded[0]).toBeCloseTo(probeStateData[0]!, 3);
      expect(decoded[3]).toBe(1);
    },
  );

  it('rejects malformed atlas snapshots before allocating or uploading', () => {
    const { pass, internal } = makePass();
    internal._grid.computeFromBounds({ min: [0, 0, 0], max: [1, 1, 1] }, 100, 3);
    internal._grid.allocateAtlases();
    const irrSlot = internal._grid.irradianceReadTex;
    const visSlot = internal._grid.visibilityReadTex;
    if (!irrSlot || !visSlot) throw new Error('expected allocated DDGI atlas slots');
    const device = {
      createTexture: vi.fn((descriptor: GPUTextureDescriptor) => {
        const size = descriptor.size as readonly number[];
        return trackedTexture(Number(size[0]), Number(size[1])).gpu;
      }),
      queue: { writeTexture: vi.fn() },
    } as unknown as GPUDevice;
    internal._atlasCache.getOrCreateAtlasTexture(device, irrSlot, 'rgba16float');
    internal._atlasCache.getOrCreateAtlasTexture(device, visSlot, 'rgba16float');
    vi.mocked(device.createTexture).mockClear();

    expect(pass.importAtlasData(device, {
      irrW: irrSlot.width,
      irrH: irrSlot.height,
      visW: visSlot.width,
      visH: visSlot.height,
      probeStateW: internal._grid.dims.x,
      probeStateH: internal._grid.dims.y * internal._grid.dims.z,
      irrData: new Uint16Array(0),
      visData: new Uint16Array(visSlot.width * visSlot.height * 4),
      probeStateData: new Float32Array(internal._grid.probeCount * 4),
    })).toBe(false);
    expect(device.createTexture).not.toHaveBeenCalled();
    expect(device.queue.writeTexture).not.toHaveBeenCalled();
  });

  it('retires every cached atlas even when one texture destroy throws', () => {
    const cache = new ProbeUpdateAtlasTextureCache();
    const first = trackedTexture();
    const second = trackedTexture();
    first.destroy.mockImplementation(() => { throw new Error('hostile destroy'); });
    const device = {
      createTexture: vi.fn()
        .mockReturnValueOnce(first.gpu)
        .mockReturnValueOnce(second.gpu),
    } as unknown as GPUDevice;
    const firstSlot = { width: 1, height: 1 };
    const secondSlot = { width: 1, height: 1 };
    cache.getOrCreateAtlasTexture(device, firstSlot, 'rgba16float');
    cache.getOrCreateAtlasTexture(device, secondSlot, 'rgba16float');

    expect(() => cache.dispose()).not.toThrow();
    expect(first.destroy).toHaveBeenCalledTimes(1);
    expect(second.destroy).toHaveBeenCalledTimes(1);
    expect(cache.getCachedAtlas(firstSlot)).toBeUndefined();
    expect(cache.getCachedAtlas(secondSlot)).toBeUndefined();
  });

  it('retires a displaced grid atlas cohort and later disposes the replacement cohort exactly once', () => {
    const { pass, internal } = makePass();
    const created: TrackedTexture[] = [];
    const device = {
      createTexture: vi.fn((descriptor: GPUTextureDescriptor) => {
        const size = descriptor.size as readonly number[];
        const texture = trackedTexture(Number(size[0]), Number(size[1]));
        created.push(texture);
        return texture.gpu;
      }),
    } as unknown as GPUDevice;

    internal._grid.computeFromBounds({ min: [0, 0, 0], max: [1, 1, 1] }, 1, 16);
    pass.reallocateGridAtlases();
    const oldSlots = [
      internal._grid.irradianceA!,
      internal._grid.irradianceB!,
      internal._grid.visibilityA!,
      internal._grid.visibilityB!,
    ];
    for (const slot of oldSlots) {
      internal._atlasCache.getOrCreateAtlasTexture(device, slot, 'rgba16float');
    }
    const oldTextures = created.slice();

    expect(internal._grid.computeFromBounds(
      { min: [0, 0, 0], max: [5, 5, 5] },
      1,
      16,
    )).toBe(true);
    pass.reallocateGridAtlases();

    for (let i = 0; i < oldSlots.length; i += 1) {
      expect(oldTextures[i]!.destroy).toHaveBeenCalledTimes(1);
      expect(internal._atlasCache.getCachedAtlas(oldSlots[i]!)).toBeUndefined();
    }

    const replacementSlots = [
      internal._grid.irradianceA!,
      internal._grid.irradianceB!,
      internal._grid.visibilityA!,
      internal._grid.visibilityB!,
    ];
    for (const slot of replacementSlots) {
      internal._atlasCache.getOrCreateAtlasTexture(device, slot, 'rgba16float');
    }
    const replacementTextures = created.slice(oldTextures.length);

    pass.dispose();

    for (const texture of oldTextures) {
      expect(texture.destroy).toHaveBeenCalledTimes(1);
    }
    for (const texture of replacementTextures) {
      expect(texture.destroy).toHaveBeenCalledTimes(1);
    }
  });
});

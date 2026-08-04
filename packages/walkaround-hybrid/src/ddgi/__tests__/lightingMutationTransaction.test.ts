import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DDGI } from '../DDGI.js';
import type { DDGILight } from '../types.js';

type BufferMock = GPUBuffer & {
  readonly destroy: ReturnType<typeof vi.fn>;
};

function makeBuffer(label: string, size = 80): BufferMock {
  const bytes = new ArrayBuffer(size);
  return {
    label,
    size,
    destroy: vi.fn(),
    getMappedRange: vi.fn(() => bytes),
    unmap: vi.fn(),
  } as unknown as BufferMock;
}

function makeLiveLightState(size = 96): {
  readonly lightsBuf: GPUBuffer;
  readonly lightsCapacityBytes: number;
} {
  return {
    lightsBuf: makeBuffer('old-lights', size),
    lightsCapacityBytes: size,
  };
}

function light(id: string, intensity: number): DDGILight {
  return { id, kind: 'fixture', intensity, on: true };
}

function passState(ddgi: DDGI) {
  return ddgi.pass as unknown as {
    _lights: DDGILight[];
    _sunIntensityMul: number;
    _emitterTrisData: Float32Array;
    _emitterTrisCount: number;
    _gpu: {
      device: GPUDevice;
      lightsBuf: GPUBuffer;
      lightsCapacityBytes: number;
      emitterTrisBuf: GPUBuffer;
      emitterTrisCount: number;
    } | null;
  };
}

describe('DDGI prepared lighting mutation', () => {
  beforeEach(() => {
    vi.stubGlobal('GPUBufferUsage', { STORAGE: 0x80, COPY_DST: 0x08 });
  });

  it('does not touch live state when candidate allocation fails', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldTris = new Float32Array(20);
    ddgi.setLights([light('old', 1)]);
    ddgi.setSunIntensityMultiplier(2);
    ddgi.setEmitterTris(oldTris, 1);
    const acceptedOldTris = state._emitterTrisData;

    const oldBuffer = makeBuffer('old');
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer: vi.fn(() => { throw new Error('allocation fault'); }) } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    expect(() => ddgi.prepareLightingMutation({
      lights: [light('new', 4)],
      sunIntensityMultiplier: 5,
      emitterTris: new Float32Array(40),
      emitterCount: 2,
    })).toThrow('allocation fault');
    expect(state._lights[0]?.id).toBe('old');
    expect(state._sunIntensityMul).toBe(2);
    expect(state._emitterTrisData).toBe(acceptedOldTris);
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
  });

  it('rejects a public live light replacement above device limits before publication', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldLightBuffer = makeBuffer('old-lights', 16);
    const oldEmitterBuffer = makeBuffer('old-emitter', 80);
    const createBuffer = vi.fn();
    state._gpu = {
      device: {
        limits: {
          maxBufferSize: 64,
          maxStorageBufferBindingSize: 64,
        },
        createBuffer,
      } as unknown as GPUDevice,
      lightsBuf: oldLightBuffer,
      lightsCapacityBytes: oldLightBuffer.size,
      emitterTrisBuf: oldEmitterBuffer,
      emitterTrisCount: 0,
    };

    expect(() => ddgi.setLights([light('too-large', 1)])).toThrow(
      /exceeding device\.limits/,
    );
    expect(state._lights).toEqual([]);
    expect((ddgi as unknown as { _configuredLights: DDGILight[] })._configuredLights)
      .toEqual([]);
    expect(state._gpu.lightsBuf).toBe(oldLightBuffer);
    expect(state._gpu.emitterTrisBuf).toBe(oldEmitterBuffer);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(oldLightBuffer.destroy).not.toHaveBeenCalled();
    expect(oldEmitterBuffer.destroy).not.toHaveBeenCalled();
  });

  it('prepares full-blend invalidation before a standalone lighting publication', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const beforeBlend = ddgi.pass.captureFullBlendState();
    const beforeEpoch = (ddgi as unknown as { _contentEpoch: number })._contentEpoch;
    vi.spyOn(ddgi.pass, 'prepareFullBlendInvalidation').mockImplementation(() => {
      throw new Error('invalidation candidate fault');
    });

    expect(() => ddgi.setLights([light('new', 2)]))
      .toThrow('invalidation candidate fault');
    expect(state._lights).toEqual([]);
    expect((ddgi as unknown as { _configuredLights: DDGILight[] })._configuredLights)
      .toEqual([]);
    expect(ddgi.pass.captureFullBlendState()).toEqual(beforeBlend);
    expect((ddgi as unknown as { _contentEpoch: number })._contentEpoch)
      .toBe(beforeEpoch);
  });

  it('commits and rolls back a prepared invalidation without allocating replacement state', () => {
    const ddgi = new DDGI();
    const beforeBlend = ddgi.pass.captureFullBlendState();
    const mutation = ddgi.prepareLightingMutation({
      lights: [light('new', 2)],
      sunIntensityMultiplier: 1,
      emitterTris: new Float32Array(0),
      emitterCount: 0,
    });
    const latePrepare = vi.spyOn(ddgi.pass, 'prepareFullBlendInvalidation')
      .mockImplementation(() => { throw new Error('late prepare'); });
    const lateRestore = vi.spyOn(ddgi.pass, 'restoreFullBlendState')
      .mockImplementation(() => { throw new Error('late restore'); });

    expect(() => mutation.commit()).not.toThrow();
    expect(() => mutation.rollback()).not.toThrow();
    expect(ddgi.pass.captureFullBlendState()).toEqual(beforeBlend);
    expect(latePrepare).not.toHaveBeenCalled();
    expect(lateRestore).not.toHaveBeenCalled();
  });

  it('completes the configured emitter snapshot before allocating a pass candidate', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldBuffer = makeBuffer('old');
    const createBuffer = vi.fn(() => makeBuffer('candidate', 192));
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };
    const tris = new Float32Array(40);
    Object.defineProperty(tris, 'slice', {
      value: () => {
        throw new Error('configured snapshot fault');
      },
    });

    expect(() => ddgi.prepareLightingMutation({
      lights: [],
      sunIntensityMultiplier: 1,
      emitterTris: tris,
      emitterCount: 2,
    })).toThrow('configured snapshot fault');
    expect(createBuffer).not.toHaveBeenCalled();
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
  });

  it('keeps direct emitter CPU/GPU state atomic when candidate population fails', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const lifecycle = ddgi as unknown as { _frame: number; _ready: boolean };
    const oldTris = new Float32Array(20).fill(1);
    ddgi.setEmitterTris(oldTris, 1);
    const acceptedOldTris = state._emitterTrisData;
    lifecycle._frame = 7;
    lifecycle._ready = true;

    const oldBuffer = makeBuffer('old');
    const candidate = makeBuffer('candidate', 192);
    vi.mocked(candidate.getMappedRange).mockImplementation(() => {
      throw new Error('mapping fault');
    });
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer: vi.fn(() => candidate) } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    expect(() => ddgi.setEmitterTris(new Float32Array(40).fill(2), 2))
      .toThrow('mapping fault');
    expect(state._emitterTrisData).toBe(acceptedOldTris);
    expect(state._emitterTrisCount).toBe(1);
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(state._gpu.emitterTrisCount).toBe(1);
    expect(candidate.destroy).toHaveBeenCalledTimes(1);
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
    expect(ddgi.warmupFrame).toBe(7);
    expect(ddgi.ready).toBe(true);
  });

  it('publishes and retires a direct emitter replacement only after preparation succeeds', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldTris = new Float32Array(20).fill(1);
    ddgi.setEmitterTris(oldTris, 1);
    const oldBuffer = makeBuffer('old');
    const candidate = makeBuffer('candidate', 192);
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer: vi.fn(() => candidate) } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    const replacement = new Float32Array(40).fill(2);
    ddgi.setEmitterTris(replacement, 2);

    expect(state._emitterTrisData.subarray(0, replacement.length)).toEqual(replacement);
    expect(state._emitterTrisData).toHaveLength(48);
    expect(state._emitterTrisData[42]).toBeCloseTo(0.5);
    expect(state._emitterTrisData[46]).toBeCloseTo(0.5);
    expect(state._emitterTrisCount).toBe(2);
    expect(state._gpu.emitterTrisBuf).toBe(candidate);
    expect(state._gpu.emitterTrisCount).toBe(2);
    expect(oldBuffer.destroy).toHaveBeenCalledTimes(1);
    expect(candidate.destroy).not.toHaveBeenCalled();
    expect(ddgi.warmupFrame).toBe(0);
    expect(ddgi.ready).toBe(false);
  });

  it('restores the old generation after commit then rollback', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldTris = new Float32Array(20);
    ddgi.setLights([light('old', 1)]);
    ddgi.setSunIntensityMultiplier(2);
    ddgi.setEmitterTris(oldTris, 1);
    const acceptedOldTris = state._emitterTrisData;

    const oldBuffer = makeBuffer('old');
    const oldLightBuffer = makeBuffer('old-lights', 96);
    const candidate = makeBuffer('candidate', 192);
    const lightCandidate = makeBuffer('light-candidate', 96);
    state._gpu = {
      lightsBuf: oldLightBuffer,
      lightsCapacityBytes: oldLightBuffer.size,
      device: {
        createBuffer: vi.fn((descriptor: GPUBufferDescriptor) =>
          String(descriptor.label).includes('lights') ? lightCandidate : candidate),
      } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    const mutation = ddgi.prepareLightingMutation({
      lights: [light('new', 4)],
      sunIntensityMultiplier: 5,
      emitterTris: new Float32Array(40),
      emitterCount: 2,
    });
    expect(state._lights[0]?.id).toBe('old');
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);

    mutation.commit();
    expect(state._lights[0]?.id).toBe('new');
    expect(state._sunIntensityMul).toBe(5);
    expect(state._emitterTrisCount).toBe(2);
    expect(state._gpu.lightsBuf).toBe(lightCandidate);
    expect(state._gpu.emitterTrisBuf).toBe(candidate);
    expect(oldLightBuffer.destroy).not.toHaveBeenCalled();
    expect(oldBuffer.destroy).not.toHaveBeenCalled();

    mutation.rollback();
    expect(state._lights[0]?.id).toBe('old');
    expect(state._sunIntensityMul).toBe(2);
    expect(state._emitterTrisData).toBe(acceptedOldTris);
    expect(state._gpu.lightsBuf).toBe(oldLightBuffer);
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(lightCandidate.destroy).toHaveBeenCalledTimes(1);
    expect(candidate.destroy).toHaveBeenCalledTimes(1);
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
  });

  it('retires the old emitter buffer only after successful finalize', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    ddgi.setEmitterTris(new Float32Array(20), 1);
    const oldBuffer = makeBuffer('old');
    const candidate = makeBuffer('candidate', 192);
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer: vi.fn(() => candidate) } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    const mutation = ddgi.prepareLightingMutation({
      lights: [],
      sunIntensityMultiplier: 1,
      emitterTris: new Float32Array(40),
      emitterCount: 2,
    });
    mutation.commit();
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
    mutation.finalize();
    expect(oldBuffer.destroy).toHaveBeenCalledTimes(1);
    expect(candidate.destroy).not.toHaveBeenCalled();
  });

  it('rejects an aliased candidate without destroying the live buffer', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldBuffer = makeBuffer('old');
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer: vi.fn(() => oldBuffer) } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    expect(() => ddgi.prepareLightingMutation({
      lights: [],
      sunIntensityMultiplier: 1,
      emitterTris: new Float32Array(20),
      emitterCount: 1,
    })).toThrow(/aliases the live GPU buffer/);
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
  });
  it('keeps the live placeholder buffer across a zero-emitter rollback', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldTris = new Float32Array(20);
    const oldBuffer = makeBuffer('old');
    const oldLightBuffer = makeBuffer('old-lights', 96);
    const lightCandidate = makeBuffer('light-candidate', 96);
    const createBuffer = vi.fn(() => lightCandidate);
    ddgi.setLights([light('old', 1)]);
    ddgi.setSunIntensityMultiplier(2);
    ddgi.setEmitterTris(oldTris, 1);
    const acceptedOldTris = state._emitterTrisData;
    state._gpu = {
      lightsBuf: oldLightBuffer,
      lightsCapacityBytes: oldLightBuffer.size,
      device: { createBuffer } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    const mutation = ddgi.prepareLightingMutation({
      lights: [light('new', 3)],
      sunIntensityMultiplier: 4,
      emitterTris: new Float32Array(0),
      emitterCount: 0,
    });
    expect(createBuffer).toHaveBeenCalledTimes(1);

    mutation.commit();
    expect(state._lights[0]?.id).toBe('new');
    expect(state._emitterTrisData).toHaveLength(0);
    expect(state._emitterTrisCount).toBe(0);
    expect(state._gpu.lightsBuf).toBe(lightCandidate);
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(state._gpu.emitterTrisCount).toBe(0);

    mutation.rollback();
    expect(state._lights[0]?.id).toBe('old');
    expect(state._sunIntensityMul).toBe(2);
    expect(state._emitterTrisData).toBe(acceptedOldTris);
    expect(state._emitterTrisCount).toBe(1);
    expect(state._gpu.lightsBuf).toBe(oldLightBuffer);
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(state._gpu.emitterTrisCount).toBe(1);
    expect(lightCandidate.destroy).toHaveBeenCalledTimes(1);
    expect(oldLightBuffer.destroy).not.toHaveBeenCalled();
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
  });

  it('retains the live placeholder buffer after a finalized zero-emitter transition', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldBuffer = makeBuffer('old');
    const createBuffer = vi.fn();
    ddgi.setEmitterTris(new Float32Array(20), 1);
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    const mutation = ddgi.prepareLightingMutation({
      lights: [],
      sunIntensityMultiplier: 1,
      emitterTris: new Float32Array(0),
      emitterCount: 0,
    });
    mutation.commit();
    mutation.finalize();

    expect(createBuffer).not.toHaveBeenCalled();
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(state._gpu.emitterTrisCount).toBe(0);
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
  });

  it('resets probe warmup on commit and restores it exactly on rollback', () => {
    const ddgi = new DDGI();
    const lifecycle = ddgi as unknown as { _frame: number; _ready: boolean };
    lifecycle._frame = 17;
    lifecycle._ready = true;
    ddgi.setProbeUpdateDivisor(4);
    ddgi.pass.restoreFullBlendState({
      generation: 91,
      stride: 4,
      pendingStrata: [1, 3],
    });
    const priorInvalidation = ddgi.pass.captureFullBlendState();

    const mutation = ddgi.prepareLightingMutation({
      lights: [light('new', 3)],
      sunIntensityMultiplier: 4,
      emitterTris: new Float32Array(0),
      emitterCount: 0,
    });
    mutation.commit();

    expect(ddgi.warmupFrame).toBe(0);
    expect(ddgi.ready).toBe(false);
    expect(ddgi.pass.pendingFullBlend).toBe(true);
    expect(ddgi.pass.captureFullBlendState()).toEqual({
      generation: 92,
      stride: 4,
      pendingStrata: [0, 1, 2, 3],
    });

    mutation.rollback();
    expect(ddgi.warmupFrame).toBe(17);
    expect(ddgi.ready).toBe(true);
    expect(ddgi.pass.captureFullBlendState()).toEqual(priorInvalidation);
  });

  it('retains the emitter GPU buffer for a runtime-only sun/sky mutation', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const tris = new Float32Array(20);
    ddgi.setEmitterTris(tris, 1);
    const oldBuffer = makeBuffer('old');
    const oldLightBuffer = makeBuffer('old-lights', 16);
    const lightCandidate = makeBuffer('light-candidate', 96);
    const createBuffer = vi.fn(() => lightCandidate);
    state._gpu = {
      lightsBuf: oldLightBuffer,
      lightsCapacityBytes: oldLightBuffer.size,
      device: { createBuffer } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    const mutation = ddgi.prepareRuntimeLightingMutation({
      lights: [light('runtime', 3)],
      sunIntensityMultiplier: 4,
      skyTint: [0.1, 0.2, 0.3],
      skyIrradiance: 2,
    });
    mutation.commit();
    mutation.finalize();

    expect(createBuffer).toHaveBeenCalledTimes(1);
    expect(state._gpu.lightsBuf).toBe(lightCandidate);
    expect(state._gpu.emitterTrisBuf).toBe(oldBuffer);
    expect(state._gpu.emitterTrisCount).toBe(1);
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
    expect(oldLightBuffer.destroy).toHaveBeenCalledTimes(1);
    expect(lightCandidate.destroy).not.toHaveBeenCalled();
  });

  it('does not let hostile candidate destruction mask rollback', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldBuffer = makeBuffer('old');
    const candidate = makeBuffer('candidate', 192);
    candidate.destroy.mockImplementation(() => {
      throw new Error('hostile candidate destroy');
    });
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer: vi.fn(() => candidate) } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    const mutation = ddgi.prepareLightingMutation({
      lights: [],
      sunIntensityMultiplier: 1,
      emitterTris: new Float32Array(40),
      emitterCount: 2,
    });

    expect(() => mutation.rollback()).not.toThrow();
    expect(candidate.destroy).toHaveBeenCalledTimes(1);
    expect(oldBuffer.destroy).not.toHaveBeenCalled();
  });

  it('does not report committed publication as failed when old-buffer retirement throws', () => {
    const ddgi = new DDGI();
    const state = passState(ddgi);
    const oldBuffer = makeBuffer('old');
    oldBuffer.destroy.mockImplementation(() => {
      throw new Error('hostile old-buffer destroy');
    });
    const candidate = makeBuffer('candidate', 192);
    state._gpu = {
      ...makeLiveLightState(),
      device: { createBuffer: vi.fn(() => candidate) } as unknown as GPUDevice,
      emitterTrisBuf: oldBuffer,
      emitterTrisCount: 1,
    };

    const mutation = ddgi.prepareLightingMutation({
      lights: [],
      sunIntensityMultiplier: 1,
      emitterTris: new Float32Array(40),
      emitterCount: 2,
    });
    mutation.commit();

    expect(() => mutation.finalize()).not.toThrow();
    expect(state._gpu.emitterTrisBuf).toBe(candidate);
    expect(oldBuffer.destroy).toHaveBeenCalledTimes(1);
  });
});

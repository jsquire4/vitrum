import { describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_FRAME_RESOURCE_BUDGET_BYTES,
  DEFAULT_MAX_RESTIR_RESERVOIR_SCALE,
  assertFrameResourcePlanSupported,
  planFrameResources,
  resolveFrameResourcePlan,
} from './frameResourcePlan.js';
import { createFrameResources } from './resourceManager.js';

describe('frame resource allocation plan', () => {
  const device = {
    limits: {
      maxTextureDimension2D: 8192,
      maxBufferSize: 1 << 30,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
    },
  } as unknown as GPUDevice;
  const defaultGraph = {
    gtaoEnabled: true,
    gtaoDownscale: 2,
    svgfEnabled: false,
    welfordPingPong: true,
    atrousVarianceEstimate: true,
    checkerboard: false,
  } as const;

  it('accounts physical aliases once and the two-channel motion target exactly', () => {
    const plan = planFrameResources(1920, 1080, {
      ...defaultGraph,
    });
    expect(plan.allocations.filter((a) => a.label === 'combined/resolved')).toHaveLength(1);
    expect(plan.allocations.some((a) => a.label === 'resolved-radiance')).toBe(false);
    expect(plan.allocations.find((a) => a.label === 'motion-vectors')?.bytes)
      .toBe(1920 * 1080 * 8);
    expect(plan.persistentBytes).toBeGreaterThan(0);
  });

  it.each([
    [1920, 1080],
    [2560, 1440],
    [3840, 2160],
  ])(
    'auto-resolves %ix%i to the largest deterministic graph below the default ceiling',
    (width, height) => {
      const resolved = resolveFrameResourcePlan(device, width, height, {
        ...defaultGraph,
        resolutionPolicy: 'auto',
      });
      expect(resolved.persistentBytes).toBeLessThanOrEqual(
        DEFAULT_FRAME_RESOURCE_BUDGET_BYTES,
      );
      expect(resolved.effectiveWidth).toBeLessThanOrEqual(width);
      expect(resolved.effectiveHeight).toBeLessThanOrEqual(height);
      expect(resolved.restirReservoirScale).toBeGreaterThanOrEqual(1);
      expect(resolved.restirReservoirScale)
        .toBeLessThanOrEqual(DEFAULT_MAX_RESTIR_RESERVOIR_SCALE);
      expect(resolved.restirDiWidth).toBe(
        Math.max(
          1,
          Math.floor(resolved.effectiveWidth / resolved.restirReservoirScale),
        ),
      );
      expect(resolved.restirGiWidth).toBe(
        Math.max(
          1,
          Math.floor(
            resolved.effectiveWidth / (2 * resolved.restirReservoirScale),
          ),
        ),
      );
      expect(resolveFrameResourcePlan(device, width, height, {
        ...defaultGraph,
        resolutionPolicy: 'auto',
      })).toEqual(resolved);
    },
  );

  it('pins exact native logical footprints at reservoir scales 1 through 4', () => {
    const exact = ([width, height]: readonly [number, number]) =>
      [1, 2, 3, 4].map((reservoirScale) =>
        planFrameResources(width, height, {
          ...defaultGraph,
          reservoirScale,
        }).persistentBytes);
    expect(exact([1920, 1080])).toEqual([
      675_994_296,
      396_058_296,
      344_218_296,
      326_074_296,
    ]);
    expect(exact([2560, 1440])).toEqual([
      1_201_767_096,
      704_103_096,
      611_873_976,
      579_687_096,
    ]);
    expect(exact([3840, 2160])).toEqual([
      2_703_975_096,
      1_584_231_096,
      1_376_871_096,
      1_304_295_096,
    ]);
  });

  it.each([
    {
      request: [1920, 1080] as const,
      effective: [1920, 1080] as const,
      reservoirScale: 2,
      di: [960, 540] as const,
      gi: [480, 270] as const,
      persistentBytes: 396_058_296,
      resolutionDownscale: 1,
    },
    {
      request: [2560, 1440] as const,
      effective: [2134, 1200] as const,
      reservoirScale: 4,
      di: [533, 300] as const,
      gi: [266, 150] as const,
      persistentBytes: 402_634_296,
      resolutionDownscale: 1.2,
    },
    {
      request: [3840, 2160] as const,
      effective: [2134, 1200] as const,
      reservoirScale: 4,
      di: [533, 300] as const,
      gi: [266, 150] as const,
      persistentBytes: 402_634_296,
      resolutionDownscale: 1.8,
    },
  ])(
    'pins exact $request auto selection, reservoir grids, and resize peak',
    ({
      request,
      effective,
      reservoirScale,
      di,
      gi,
      persistentBytes,
      resolutionDownscale,
    }) => {
      const resolved = resolveFrameResourcePlan(
        device,
        request[0],
        request[1],
        { ...defaultGraph, resolutionPolicy: 'auto' },
        persistentBytes,
      );
      expect([resolved.effectiveWidth, resolved.effectiveHeight]).toEqual(effective);
      expect(resolved.restirReservoirScale).toBe(reservoirScale);
      expect([resolved.restirDiWidth, resolved.restirDiHeight]).toEqual(di);
      expect([resolved.restirGiWidth, resolved.restirGiHeight]).toEqual(gi);
      expect(resolved.persistentBytes).toBe(persistentBytes);
      expect(resolved.resizePeakBytes).toBe(2 * persistentBytes);
      expect(resolved.resolutionDownscale).toBeCloseTo(resolutionDownscale, 12);
    },
  );

  it('treats scale 4 as a fixed quality policy and honors explicit scale exactly', () => {
    expect(DEFAULT_MAX_RESTIR_RESERVOIR_SCALE).toBe(4);
    const explicit = planFrameResources(1920, 1080, {
      ...defaultGraph,
      reservoirScale: 3,
    });
    const resolved = resolveFrameResourcePlan(device, 1920, 1080, {
      ...defaultGraph,
      resolutionPolicy: 'native',
      reservoirScale: 3,
      maxPersistentBytes: explicit.persistentBytes,
    });
    expect(resolved.restirReservoirScale).toBe(3);
    expect(resolved.persistentBytes).toBe(explicit.persistentBytes);
    expect(() => planFrameResources(1920, 1080, {
      ...defaultGraph,
      reservoirScale: 5,
    })).toThrow(/reservoirScale must be an integer in \[1, 4\]/);
  });

  it('rejects an explicit scale-1 native request with exact required and limit bytes', () => {
    const native = planFrameResources(1920, 1080, {
      ...defaultGraph,
      reservoirScale: 1,
    });
    expect(() => resolveFrameResourcePlan(device, 1920, 1080, {
      ...defaultGraph,
      resolutionPolicy: 'native',
      reservoirScale: 1,
    })).toThrow(
      `${native.persistentBytes} persistent logical bytes`,
    );
    expect(() => resolveFrameResourcePlan(device, 1920, 1080, {
      ...defaultGraph,
      resolutionPolicy: 'native',
      reservoirScale: 1,
    })).toThrow(
      `host maxPersistentBytes=${DEFAULT_FRAME_RESOURCE_BUDGET_BYTES}`,
    );
  });

  it('honors a raised native ceiling without changing either dimension', () => {
    const native = planFrameResources(1920, 1080, defaultGraph);
    const resolved = resolveFrameResourcePlan(device, 1920, 1080, {
      ...defaultGraph,
      resolutionPolicy: 'native',
      maxPersistentBytes: native.persistentBytes,
    });
    expect(resolved.effectiveWidth).toBe(1920);
    expect(resolved.effectiveHeight).toBe(1080);
    expect(resolved.resolutionDownscale).toBe(1);
    expect(resolved.persistentBytes).toBe(native.persistentBytes);
  });

  it('reports the exact transactional two-generation resize peak', () => {
    const oldGeneration = resolveFrameResourcePlan(device, 1920, 1080, {
      ...defaultGraph,
      resolutionPolicy: 'auto',
    });
    const replacement = resolveFrameResourcePlan(
      device,
      2560,
      1440,
      {
        ...defaultGraph,
        resolutionPolicy: 'auto',
      },
      oldGeneration.persistentBytes,
    );
    expect(replacement.resizePeakBytes).toBe(
      oldGeneration.persistentBytes + replacement.persistentBytes,
    );
    expect(replacement.resizePeakBytes).toBeLessThanOrEqual(
      2 * DEFAULT_FRAME_RESOURCE_BUDGET_BYTES,
    );
  });

  it('raises reservoir scale before whole-frame downscaling for a device storage ceiling', () => {
    const resolved = resolveFrameResourcePlan(device, 3840, 2160, {
      ...defaultGraph,
      resolutionPolicy: 'auto',
      maxPersistentBytes: 4 * 1024 * 1024 * 1024,
    });
    const largestStorage = Math.max(
      ...resolved.footprint.allocations
        .filter((allocation) => allocation.category === 'storage-buffer')
        .map((allocation) => allocation.bytes),
    );
    expect(largestStorage).toBeLessThanOrEqual(128 * 1024 * 1024);
    expect(resolved.effectiveWidth).toBe(3840);
    expect(resolved.effectiveHeight).toBe(2160);
    expect(resolved.restirReservoirScale).toBe(2);
  });

  it('makes disabled GTAO a pair of 1x1 neutral-resource allocations', () => {
    const plan = planFrameResources(3840, 2160, {
      gtaoEnabled: false,
      svgfEnabled: false,
    });
    const gtao = plan.allocations.filter((a) => a.label.startsWith('gtao-'));
    expect(gtao.filter((a) => a.category === 'texture')).toEqual([
      expect.objectContaining({ label: 'gtao-low', width: 1, height: 1, bytes: 8 }),
      expect.objectContaining({ label: 'gtao-full', width: 1, height: 1, bytes: 8 }),
    ]);
  });

  it('rejects a host budget with exact requested bytes', () => {
    const plan = planFrameResources(640, 480, { svgfEnabled: false });
    expect(() => assertFrameResourcePlanSupported(
      { limits: {} } as GPUDevice,
      plan,
      { maxPersistentBytes: plan.persistentBytes - 1 },
    )).toThrow(
      `${plan.persistentBytes} persistent logical bytes`,
    );
  });

  it('rejects an over-limit reservoir before the first GPU allocation', () => {
    const createBuffer = vi.fn();
    const createTexture = vi.fn();
    const device = {
      limits: {
        maxTextureDimension2D: 8192,
        maxBufferSize: 1 << 30,
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
      },
      createBuffer,
      createTexture,
    } as unknown as GPUDevice;
    expect(() => createFrameResources(device, 3840, 2160, {
      svgfEnabled: false,
    })).toThrow(/restir-di-current.*maxStorageBufferBindingSize/);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects texture dimensions before the first GPU allocation', () => {
    const createBuffer = vi.fn();
    const createTexture = vi.fn();
    const device = {
      limits: { maxTextureDimension2D: 4096 },
      createBuffer,
      createTexture,
    } as unknown as GPUDevice;
    expect(() => createFrameResources(device, 4097, 1, {
      svgfEnabled: false,
    })).toThrow(/maxTextureDimension2D=4096.*no resources were allocated/);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createTexture).not.toHaveBeenCalled();
  });
});

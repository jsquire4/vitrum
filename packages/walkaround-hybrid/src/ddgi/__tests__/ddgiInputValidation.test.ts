import { afterEach, describe, expect, it, vi } from 'vitest';
import { SceneBvh } from '@vitrum/shared-bvh';
import { DDGI } from '../DDGI.js';
import { ProbeGrid } from '../probeGrid.js';
import { ProbeUpdatePass } from '../probeUpdatePass.js';
import {
  packProbeUpdateBlendParams,
  packProbeUpdateFrameParams,
} from '../probeUpdateFrameParams.js';
import { packDDGIProbeLights } from '../probeUpdateLights.js';
import type { DDGILight } from '../types.js';

const invalidCadences = [NaN, Infinity, -Infinity, 0, -1, 1.5] as const;
const invalidFiniteScalars = [NaN, Infinity, -Infinity] as const;

function passState(pass: ProbeUpdatePass): {
  _probeUpdateDivisor: number;
  _sunIntensityMul: number;
  _skyTint: [number, number, number];
  _skyIrradiance: number;
  _glassMixScale: number;
  _envRotationY: number;
  _envIntensity: number;
  _lights: DDGILight[];
  _emitterTrisData: Float32Array;
  _emitterTrisCount: number;
  _indirectFeedback: boolean;
  _hasEnv: boolean;
  _initAttempted: boolean;
} {
  return pass as unknown as ReturnType<typeof passState>;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('DDGI public input validation', () => {
  it.each([
    ['probeSpacing', NaN],
    ['probeSpacing', Infinity],
    ['probeSpacing', 0],
    ['probeSpacing', -1],
    ['maxProbesPerAxis', NaN],
    ['maxProbesPerAxis', Infinity],
    ['maxProbesPerAxis', 0],
    ['maxProbesPerAxis', 1.5],
  ] as const)('rejects invalid constructor option %s=%s', (key, value) => {
    expect(() => new DDGI({ [key]: value })).toThrow(RangeError);
  });

  it.each([NaN, Infinity, -Infinity, 1.5])(
    'rejects invalid maxMaterials=%s instead of emitting invalid WGSL',
    (value) => {
      expect(() => new DDGI({ maxMaterials: value })).toThrow(RangeError);
      expect(() => new ProbeUpdatePass(new SceneBvh(), new ProbeGrid(), {
        maxMaterials: value,
      })).toThrow(RangeError);
    },
  );

  it('rejects malformed booleans at constructor and setter boundaries', () => {
    expect(() => new DDGI({ debug: 1 as unknown as boolean })).toThrow(TypeError);
    expect(() => new ProbeUpdatePass(new SceneBvh(), new ProbeGrid(), {
      debug: 'yes' as unknown as boolean,
    })).toThrow(TypeError);

    const ddgi = new DDGI();
    const pass = ddgi.pass;
    const beforeEpoch = (ddgi as unknown as { _contentEpoch: number })._contentEpoch;
    expect(() => ddgi.setIndirectFeedback(1 as unknown as boolean)).toThrow(TypeError);
    expect(() => ddgi.setEnvironment(
      null,
      null,
      0,
      1,
      1 as unknown as boolean,
    )).toThrow(TypeError);
    expect(() => pass.setIndirectFeedback('yes' as unknown as boolean)).toThrow(TypeError);
    expect(() => pass.setEnvironment(
      null,
      null,
      0,
      1,
      'yes' as unknown as boolean,
    )).toThrow(TypeError);
    expect(() => ddgi.setEnvironment(null, null, 0, 1, true)).toThrow(TypeError);
    expect(() => pass.setEnvironment(null, null, 0, 1, true)).toThrow(TypeError);
    expect(passState(pass)._indirectFeedback).toBe(true);
    expect(passState(pass)._hasEnv).toBe(false);
    expect((ddgi as unknown as { _contentEpoch: number })._contentEpoch)
      .toBe(beforeEpoch);
    ddgi.dispose();
  });

  it.each(invalidCadences)(
    'rejects probe divisor %s without changing cadence or content generation',
    (value) => {
      const ddgi = new DDGI();
      const forwarded = vi.spyOn(ddgi.pass, 'setProbeUpdateDivisor');
      const before = {
        stride: ddgi.warmupStride,
        epoch: (ddgi as unknown as { _contentEpoch: number })._contentEpoch,
      };

      expect(() => ddgi.setProbeUpdateDivisor(value)).toThrow(RangeError);

      expect(ddgi.warmupStride).toBe(before.stride);
      expect((ddgi as unknown as { _contentEpoch: number })._contentEpoch)
        .toBe(before.epoch);
      expect(forwarded).not.toHaveBeenCalled();
      ddgi.dispose();
    },
  );

  it.each(invalidCadences)(
    'keeps pass cadence/full-blend state intact when divisor %s is invalid',
    (value) => {
      const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
      pass.requestFullBlend(4);
      const beforeDivisor = passState(pass)._probeUpdateDivisor;
      const beforeBlend = pass.captureFullBlendState();

      expect(() => pass.setProbeUpdateDivisor(value)).toThrow(RangeError);

      expect(passState(pass)._probeUpdateDivisor).toBe(beforeDivisor);
      expect(pass.captureFullBlendState()).toEqual(beforeBlend);
      pass.dispose();
    },
  );

  it.each(invalidCadences)(
    'keeps generation/pending strata intact when full-blend stride %s is invalid',
    (value) => {
      const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
      pass.requestFullBlend(3);
      const before = pass.captureFullBlendState();

      expect(() => pass.requestFullBlend(value)).toThrow(RangeError);
      expect(pass.captureFullBlendState()).toEqual(before);
      pass.dispose();
    },
  );

  it.each([
    { offset: 0, stride: NaN },
    { offset: 0, stride: Infinity },
    { offset: 0, stride: 0 },
    { offset: 0, stride: 1.5 },
    { offset: NaN, stride: 1 },
    { offset: Infinity, stride: 1 },
    { offset: 0.5, stride: 1 },
  ])('runFrame fails closed before initialization for $offset/$stride', async ({ offset, stride }) => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());

    await expect(pass.runFrame({}, offset, stride)).rejects.toThrow(RangeError);
    expect(passState(pass)._initAttempted).toBe(false);
    pass.dispose();
  });

  it.each(invalidFiniteScalars)(
    'rejects non-finite scalar/vector payload %s before facade/pass mutation',
    (value) => {
      const ddgi = new DDGI();
      const pass = ddgi.pass;
      const calls = [
        vi.spyOn(pass, 'setSunIntensityMultiplier'),
        vi.spyOn(pass, 'setSkyParams'),
        vi.spyOn(pass, 'setGlassMixScale'),
        vi.spyOn(pass, 'setEnvironment'),
      ];
      const beforeEpoch = (ddgi as unknown as { _contentEpoch: number })._contentEpoch;

      expect(() => ddgi.setSunIntensityMultiplier(value)).toThrow(RangeError);
      expect(() => ddgi.setSkyParams([value, 0.5, 1], 2)).toThrow(RangeError);
      expect(() => ddgi.setSkyParams([0.4, 0.6, 1], value)).toThrow(RangeError);
      expect(() => ddgi.setGlassMixScale(value)).toThrow(RangeError);
      expect(() => ddgi.setEnvironment(null, null, value, 1, false)).toThrow(RangeError);
      expect(() => ddgi.setEnvironment(null, null, 0, value, false)).toThrow(RangeError);

      expect((ddgi as unknown as { _contentEpoch: number })._contentEpoch)
        .toBe(beforeEpoch);
      for (const call of calls) expect(call).not.toHaveBeenCalled();
      ddgi.dispose();
    },
  );

  it.each(invalidFiniteScalars)(
    'rejects non-finite direct pass payload %s without changing mirrors',
    (value) => {
      const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
      const before = passState(pass);
      const snapshot = {
        sun: before._sunIntensityMul,
        tint: [...before._skyTint],
        irradiance: before._skyIrradiance,
        glass: before._glassMixScale,
        rotation: before._envRotationY,
        envIntensity: before._envIntensity,
      };

      expect(() => pass.setSunIntensityMultiplier(value)).toThrow(RangeError);
      expect(() => pass.setSkyParams([value, 0.6, 1], 2)).toThrow(RangeError);
      expect(() => pass.setSkyParams([0.4, 0.6, 1], value)).toThrow(RangeError);
      expect(() => pass.setGlassMixScale(value)).toThrow(RangeError);
      expect(() => pass.setEnvironment(null, null, value, 1, false)).toThrow(RangeError);
      expect(() => pass.setEnvironment(null, null, 0, value, false)).toThrow(RangeError);

      expect(passState(pass)).toMatchObject({
        _sunIntensityMul: snapshot.sun,
        _skyIrradiance: snapshot.irradiance,
        _glassMixScale: snapshot.glass,
        _envRotationY: snapshot.rotation,
        _envIntensity: snapshot.envIntensity,
      });
      expect(passState(pass)._skyTint).toEqual(snapshot.tint);
      pass.dispose();
    },
  );

  it.each([
    ['negative sun', () => ({ kind: 'sun', intensity: -1, on: true })],
    ['non-boolean on', () => ({ kind: 'fixture', intensity: 1, on: 1 })],
    ['non-boolean shadow', () => ({
      kind: 'fixture', intensity: 1, on: true, castShadow: 'yes',
    })],
    ['non-finite position', () => ({
      kind: 'fixture', intensity: 1, on: true,
      position: { x: Infinity, y: 0, z: 0 },
    })],
    ['zero direction', () => ({
      kind: 'sun', intensity: 1, on: true,
      direction: { x: 0, y: 0, z: 0 },
    })],
    ['negative color', () => ({
      kind: 'fixture', intensity: 1, on: true,
      color: { r: -1, g: 0, b: 0 },
    })],
    ['oversized angular radius', () => ({
      kind: 'sun', intensity: 1, on: true, angularRadius: Math.PI + 0.1,
    })],
    ['negative distance', () => ({
      kind: 'fixture', intensity: 1, on: true, distance: -1,
    })],
    ['negative decay', () => ({
      kind: 'fixture', intensity: 1, on: true, decay: -1,
    })],
    ['partial spot tuple', () => ({
      kind: 'fixture', intensity: 1, on: true,
      spotAxis: { x: 0, y: -1, z: 0 },
    })],
    ['non-finite spot cosine', () => ({
      kind: 'fixture', intensity: 1, on: true,
      spotAxis: { x: 0, y: -1, z: 0 },
      spotCosInner: NaN,
      spotCosOuter: 0.5,
    })],
    ['inverted spot cone', () => ({
      kind: 'fixture', intensity: 1, on: true,
      spotAxis: { x: 0, y: -1, z: 0 },
      spotCosInner: 0.2,
      spotCosOuter: 0.8,
    })],
    ['unknown enumerable light field', () => ({
      kind: 'fixture', intensity: 1, on: true, unsupported: 1,
    })],
    ['blank id', () => ({
      kind: 'fixture', id: '   ', intensity: 1, on: true,
    })],
    ['unknown enumerable vector field', () => ({
      kind: 'fixture', intensity: 1, on: true,
      position: { x: 0, y: 0, z: 0, w: 1 },
    })],
    ['sun position', () => ({
      kind: 'sun', intensity: 1, on: true,
      position: { x: 0, y: 0, z: 0 },
    })],
    ['sun fixture falloff', () => ({
      kind: 'sun', intensity: 1, on: true, distance: 2, decay: 2,
    })],
    ['fixture sun direction', () => ({
      kind: 'fixture', intensity: 1, on: true,
      direction: { x: 0, y: -1, z: 0 },
    })],
    ['fixture angular radius', () => ({
      kind: 'fixture', intensity: 1, on: true, angularRadius: 0.1,
    })],
    ['tea-light spot cone', () => ({
      kind: 'teaLight', intensity: 1, on: true,
      spotAxis: { x: 0, y: -1, z: 0 },
      spotCosInner: 0.8,
      spotCosOuter: 0.5,
    })],
  ] as const)('rejects %s light payload transactionally', (_name, makeLight) => {
    const lights = [makeLight()] as unknown as DDGILight[];
    const ddgi = new DDGI();
    const pass = ddgi.pass;
    const beforeEpoch = (ddgi as unknown as { _contentEpoch: number })._contentEpoch;
    const beforeLights = passState(pass)._lights;

    expect(() => ddgi.setLights(lights)).toThrow();
    expect(() => pass.setLights(lights)).toThrow();
    expect(() => packDDGIProbeLights(lights, 1)).toThrow();
    expect(passState(pass)._lights).toEqual(beforeLights);
    expect((ddgi as unknown as { _contentEpoch: number })._contentEpoch)
      .toBe(beforeEpoch);
    ddgi.dispose();
  });

  it('allows intentional non-enumerable light metadata without publishing it', () => {
    const light = { kind: 'fixture', id: 'fixture-1', intensity: 1, on: true };
    Object.defineProperty(light, 'hostMetadata', {
      enumerable: false,
      value: { owner: 'host' },
    });
    const ddgi = new DDGI();

    expect(() => ddgi.setLights([light] as DDGILight[])).not.toThrow();
    expect(Object.keys(passState(ddgi.pass)._lights[0]!)).not.toContain('hostMetadata');
    ddgi.dispose();
  });

  it.each([-1, 1.5, NaN, Infinity])(
    'rejects invalid emitter count %s before facade/pass mutation',
    (count) => {
      const ddgi = new DDGI();
      const pass = ddgi.pass;
      const payload = new Float32Array(20).fill(1);
      const beforeEpoch = (ddgi as unknown as { _contentEpoch: number })._contentEpoch;

      expect(() => ddgi.setEmitterTris(payload, count)).toThrow(RangeError);
      expect(passState(pass)._emitterTrisCount).toBe(0);
      expect(passState(pass)._emitterTrisData).toHaveLength(0);
      expect((ddgi as unknown as { _contentEpoch: number })._contentEpoch)
        .toBe(beforeEpoch);
      ddgi.dispose();
    },
  );

  it.each([
    ['short payload', new Float32Array(19)],
    ['non-finite field', (() => {
      const out = new Float32Array(20).fill(1);
      out[0] = NaN;
      return out;
    })()],
    ['negative area', (() => {
      const out = new Float32Array(20).fill(1);
      out[15] = -1;
      return out;
    })()],
    ['negative radiance', (() => {
      const out = new Float32Array(20).fill(1);
      out[16] = -1;
      return out;
    })()],
  ] as const)('rejects %s emitter payload without retaining caller bytes', (_name, payload) => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    expect(() => pass.setEmitterTris(payload, 1)).toThrow(RangeError);
    expect(passState(pass)._emitterTrisCount).toBe(0);
    expect(passState(pass)._emitterTrisData).toHaveLength(0);
    pass.dispose();
  });

  it('validates direct frame/blend/light packer inputs', () => {
    const validFrame = {
      frameIndex: 0,
      skyTint: [0.4, 0.6, 1] as const,
      skyIrradiance: 2,
      glassMixScale: 0.7,
    };
    expect(() => packProbeUpdateFrameParams({
      ...validFrame,
      frameIndex: -1,
    })).toThrow(RangeError);
    expect(() => packProbeUpdateFrameParams({
      ...validFrame,
      skyTint: [-1, 0.6, 1],
    })).toThrow(RangeError);
    expect(() => packProbeUpdateFrameParams({
      ...validFrame,
      glassMixScale: 1.1,
    })).toThrow(RangeError);
    expect(() => packProbeUpdateFrameParams({
      ...validFrame,
      hasEnv: 1 as unknown as boolean,
    })).toThrow(TypeError);
    expect(() => packProbeUpdateFrameParams({
      ...validFrame,
      indirectFeedback: 1 as unknown as boolean,
    })).toThrow(TypeError);
    expect(() => packProbeUpdateBlendParams(-0.1)).toThrow(RangeError);
    expect(() => packProbeUpdateBlendParams(Number.NaN)).toThrow(RangeError);
    expect(() => packDDGIProbeLights([], -1)).toThrow(RangeError);
  });
});

describe('ProbeGrid input validation', () => {
  it.each([
    ['non-finite min', { min: [NaN, 0, 0], max: [1, 1, 1] }],
    ['non-finite max', { min: [0, 0, 0], max: [1, Infinity, 1] }],
    ['inverted bounds', { min: [2, 0, 0], max: [1, 1, 1] }],
  ] as const)('rejects %s without mutating the accepted grid', (_name, bounds) => {
    const grid = new ProbeGrid();
    grid.computeFromBounds({ min: [0, 0, 0], max: [2, 3, 4] }, 1, 8);
    const before = {
      dims: { ...grid.dims },
      origin: grid.worldOrigin.clone(),
      spacing: grid.worldSpacing,
      dirty: grid.dirty,
    };

    expect(() => grid.computeFromBounds(bounds, 1, 8)).toThrow(RangeError);
    expect(grid.dims).toEqual(before.dims);
    expect(grid.worldOrigin).toEqual(before.origin);
    expect(grid.worldSpacing).toBe(before.spacing);
    expect(grid.dirty).toBe(before.dirty);
  });

  it.each([NaN, Infinity, 0, -1])(
    'rejects invalid spacing %s without mutating the grid',
    (spacing) => {
      const grid = new ProbeGrid();
      const before = { dims: { ...grid.dims }, spacing: grid.worldSpacing };
      expect(() => grid.computeFromBounds(
        { min: [0, 0, 0], max: [1, 1, 1] },
        spacing,
        8,
      )).toThrow(RangeError);
      expect(grid.dims).toEqual(before.dims);
      expect(grid.worldSpacing).toBe(before.spacing);
    },
  );

  it('keeps a fully degenerate finite AABB finite', () => {
    const grid = new ProbeGrid();
    grid.computeFromBounds({ min: [2, 2, 2], max: [2, 2, 2] });
    expect(grid.worldSpacing).toBe(1);
    expect(grid.dims).toEqual({ x: 3, y: 3, z: 3 });
    expect(Number.isFinite(grid.probeCount)).toBe(true);
  });
});

describe('DDGI deferred payload ownership', () => {
  it('deep-copies lights and sky tuples at the pass boundary', () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    const position = { x: 1, y: 2, z: 3 };
    const color = { r: 0.2, g: 0.3, b: 0.4 };
    const lights: DDGILight[] = [{
      kind: 'fixture', intensity: 2, on: true, position, color,
    }];
    const tint: [number, number, number] = [0.4, 0.6, 1];

    pass.setLights(lights);
    pass.setSkyParams(tint, 2);
    position.x = 99;
    color.r = 99;
    lights.splice(0, 1);
    tint[0] = 99;

    expect(passState(pass)._lights).toEqual([expect.objectContaining({
      position: { x: 1, y: 2, z: 3 },
      color: {
        r: Math.fround(0.2),
        g: Math.fround(0.3),
        b: Math.fround(0.4),
      },
    })]);
    expect(passState(pass)._skyTint).toEqual([
      Math.fround(0.4),
      Math.fround(0.6),
      1,
    ]);
    pass.dispose();
  });

  it('copies pre-init emitter data out of caller-owned storage', () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    const backing = typeof SharedArrayBuffer === 'function'
      ? new SharedArrayBuffer(24 * Float32Array.BYTES_PER_ELEMENT)
      : new ArrayBuffer(24 * Float32Array.BYTES_PER_ELEMENT);
    const caller = new Float32Array(backing, 2 * Float32Array.BYTES_PER_ELEMENT, 20);
    caller.fill(7);

    pass.setEmitterTris(caller, 1);
    caller.fill(99);

    const accepted = passState(pass)._emitterTrisData;
    expect(Array.from(accepted.subarray(0, 20))).toEqual(new Array(20).fill(7));
    expect(Array.from(accepted.subarray(20))).toEqual([1, 0, 1, 0]);
    expect(accepted.buffer).toBeInstanceOf(ArrayBuffer);
    expect(accepted.buffer).not.toBe(caller.buffer);
    pass.dispose();
  });

  it('deep-copies staged lighting before commit', () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    const position = { x: 1, y: 2, z: 3 };
    const spotAxis = { x: 0, y: -1, z: 0 };
    const lights: DDGILight[] = [{
      kind: 'fixture', intensity: 3, on: true, position, spotAxis,
      spotCosInner: 0.9, spotCosOuter: 0.7,
    }];
    const tris = new Float32Array(20).fill(4);
    const mutation = pass.prepareLightingMutation(lights, 2, tris, 1);

    position.x = 77;
    spotAxis.y = 77;
    tris.fill(77);
    mutation.commit();

    expect(passState(pass)._lights[0]).toMatchObject({
      position: { x: 1, y: 2, z: 3 },
      spotAxis: { x: 0, y: -1, z: 0 },
    });
    expect(Array.from(passState(pass)._emitterTrisData.subarray(0, 20)))
      .toEqual(new Array(20).fill(4));
    expect(Array.from(passState(pass)._emitterTrisData.subarray(20)))
      .toEqual([1, 0, 1, 0]);
    mutation.finalize();
    pass.dispose();
  });
});

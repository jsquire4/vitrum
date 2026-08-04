import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { f32ToF16Bits } from '../src/neural/nrc/fusedMlpTrainer.js';
import { NrcSubsystem } from '../src/neural/nrc/nrcSubsystem.js';
import { RIS_GI_WGSL } from '../src/shaders/risGi.wgsl.js';
import { nrcQueryWgsl } from '../src/neural/nrc/wgsl/nrcQuery.wgsl.js';
import { RIS_GI_NRC_BODY } from '../src/shaders/risGiNrc.wgsl.js';

import { buildReservoirGiWgsl } from '../src/shaders/reservoirGi.wgsl.js';
installWebGPUPolyfills();

describe('NRC and guided-RIS estimator hardening', () => {
  it('accepts every finite-positive log contribution without forming an overflowing ratio', () => {
    const defaultCosGuard = 'if (!reservoirGiFinite(cosTheta) || !(cosTheta > 0.0))';
    const defaultCosStart = RIS_GI_WGSL.indexOf(defaultCosGuard);
    const defaultCosReject = RIS_GI_WGSL.slice(
      defaultCosStart,
      RIS_GI_WGSL.indexOf('let bounceRay', defaultCosStart),
    );
    expect(defaultCosReject).toContain(
      'recordInvalidReservoirGICandidate(&r, GI_SAMPLE_SURFACE, currentGrisEpoch);',
    );
    expect(RIS_GI_WGSL).toContain('if (!reservoirGiValidLog(logPHat))');
    expect(RIS_GI_WGSL).toContain('if (!reservoirGiValidLog(logPSrc))');
    expect(RIS_GI_WGSL).toContain('let logWeight = logPHat - logPSrc;');
    expect(RIS_GI_WGSL).not.toContain('r.w_sum');
    expect(RIS_GI_WGSL).not.toContain('pHat < 1e-9');
    expect(RIS_GI_WGSL).not.toContain('pSrc > 1e-12');
    expect(RIS_GI_WGSL).not.toContain('select(0.0, pHat /');

    const nrcCosGuard = 'if (!nrcFinite(cosTheta) || !(cosTheta > 0.0))';
    const cosineReject = RIS_GI_NRC_BODY.slice(
      RIS_GI_NRC_BODY.indexOf(nrcCosGuard),
      RIS_GI_NRC_BODY.indexOf('// WS1 — offset the bounce-ray origin'),
    );
    expect(cosineReject).toContain(
      'recordInvalidReservoirGICandidate(&r, GI_SAMPLE_SURFACE, currentGrisEpoch);',
    );
    expect(RIS_GI_NRC_BODY).toContain('if (!reservoirGiValidLog(logPHat))');
    expect(RIS_GI_NRC_BODY).toContain('if (!reservoirGiValidLog(logPSrc))');
    expect(RIS_GI_NRC_BODY).toContain('let logWeight = logPHat - logPSrc;');
    expect(RIS_GI_NRC_BODY).not.toContain('r.w_sum');
    expect(RIS_GI_NRC_BODY).toContain('nrcRecordInvalidPdf();');
    expect(RIS_GI_NRC_BODY).not.toContain('pHat < 1e-9');
    expect(RIS_GI_NRC_BODY).not.toContain('pSrc > 1e-12');
    expect(RIS_GI_NRC_BODY).not.toContain('select(0.0, pHat / pSrc');

    const reservoir = buildReservoirGiWgsl();
    expect(reservoir).toContain('fn reservoirGiSourceLogW(r: ReservoirPT) -> f32');
    expect(reservoir).toContain('let result = r.H - r.nativeLogPHat;');
    expect(reservoir).not.toContain('log2(r.nativeLogPHat)');
    expect(reservoir).toContain('let cappedLogW = min(logW, logCap);');
    expect(reservoir).not.toMatch(/r\.H\s*=\s*min\(/);
    expect(reservoir).not.toContain('pHatF > 1e-9');
    expect(reservoir).not.toContain('w_sum');
  });

    it('pairs each NRC training input with an independent target from that suffix vertex', () => {
      expect(RIS_GI_NRC_BODY).toContain('if (!nrcFired && nrcTeacherEligible) {');
      expect(RIS_GI_NRC_BODY).toContain('nrcTrackXs = xs;');
      expect(RIS_GI_NRC_BODY).toContain('nrcTrackTarget = nrcTraceIndependentSuffix(');
      expect(RIS_GI_NRC_BODY).not.toContain('nrcTrackTarget = ddgiLo;');

      const recordStart = RIS_GI_NRC_BODY.indexOf('nrcWriteRecord(');
      const recordCall = RIS_GI_NRC_BODY.slice(recordStart, recordStart + 300);
      expect(recordCall).toContain('nrcTrackTarget');
      expect(recordCall).not.toContain('r.Lo');
    });

  it('uses the actual guided/cosine proposal mixture in the spread footprint', () => {
    expect(RIS_GI_NRC_BODY).toContain(
      'logPSrcBounce = reservoirGiLogProposalMixture(',
    );
    expect(RIS_GI_NRC_BODY).toContain(
      'let pSrcBounce = reservoirGiRepresentPositiveLog(logPSrcBounce);',
    );
  });

  it('rejects NRC configurations whose encoded input cannot fit the MLP', () => {
    const device = {} as GPUDevice;
    expect(() => new NrcSubsystem(device, {} as never, { width: 32 })).toThrow(
      /encoded input width 39 exceeds MLP width 32/,
    );
    expect(() => new NrcSubsystem(device, {} as never, { recordCap: 0 })).toThrow(
      /recordCap/,
    );
    expect(() => new NrcSubsystem(device, {} as never, { spreadC: 0 })).not.toThrow();
    expect(() => new NrcSubsystem(
      device,
      {} as never,
      { warmupSteps: 0x1_0000_0000 },
    )).toThrow(/warmupSteps.*u32/);
    expect(() => new NrcSubsystem(
      device,
      {} as never,
      { useF16: 'false' as unknown as boolean },
    )).toThrow(/useF16 must be a boolean/);
  });

  it('zero-pads the MLP input without eagerly indexing past the encoded vector', () => {
    const source = nrcQueryWgsl({
      levels: 8,
      featuresPerEntry: 2,
      oneBlobBins: 8,
      width: 64,
      outWidth: 3,
      hidden: 6,
    });
    expect(source).not.toContain('select(0.0, (*feat)[i], i < NRC_IN_W)');
    expect(source).toContain(
      'for (var i: u32 = 0u; i < NRC_IN_W; i = i + 1u) {\n    actA[i] = (*feat)[i];',
    );
  });

  it('encodes f16 subnormals, ties-to-even, carry, infinities, and NaN correctly', () => {
    const values = new Float32Array([
      2 ** -24,
      2 ** -25,
      1 + 2 ** -11,
      1 + 3 * 2 ** -11,
      1.99951171875,
      65504,
      Number.POSITIVE_INFINITY,
      Number.NaN,
    ]);
    const bits = f32ToF16Bits(values);
    expect(Array.from(bits.slice(0, 7))).toEqual([
      0x0001, 0x0000, 0x3c00, 0x3c02, 0x4000, 0x7bff, 0x7c00,
    ]);
    expect(bits[7]! & 0x7c00).toBe(0x7c00);
    expect(bits[7]! & 0x03ff).not.toBe(0);
  });

  it('uses both projection focal axes for a non-square camera pixel PDF', () => {
    const writeBuffer = vi.fn();
    const subsystem = Object.create(NrcSubsystem.prototype) as NrcSubsystem;
    Object.assign(subsystem, {
      _device: { queue: { writeBuffer } },
      _cfgUbo: { label: 'cfg' },
        _lifecycleState: 'ready',
    });

    const projection = new Float32Array(16);
    projection[0] = 1.5;
    projection[5] = 3;
    subsystem.updateCameraPixelPdf(projection, 1280, 720);

    const [, byteOffset, payload] = writeBuffer.mock.calls[0]!;
    expect(byteOffset).toBe(36);
    expect((payload as Float32Array)[0]).toBeCloseTo(1_036_800, 2);
  });

  it('rejects malformed or non-f32-representable camera PDF inputs', () => {
    const writeBuffer = vi.fn();
    const subsystem = Object.create(NrcSubsystem.prototype) as NrcSubsystem;
    Object.assign(subsystem, {
      _device: { queue: { writeBuffer } },
      _cfgUbo: { label: 'cfg' },
      _lifecycleState: 'ready',
    });
    const projection = new Float32Array(16);
    projection[0] = 1;
    projection[5] = 1;

    expect(() => subsystem.updateCameraPixelPdf(projection.subarray(0, 8), 1, 1))
      .toThrow(/16 elements/);
    projection[0] = Number.NaN;
    expect(() => subsystem.updateCameraPixelPdf(projection, 1, 1))
      .toThrow(/focal terms/);
    projection[0] = 1;
    expect(() => subsystem.updateCameraPixelPdf(projection, 1.5, 1))
      .toThrow(/positive safe integers/);
    projection[0] = 3.4e38;
    projection[5] = 3.4e38;
    expect(() => subsystem.updateCameraPixelPdf(projection, 1, 1))
      .toThrow(/representable as f32/);
    expect(writeBuffer).not.toHaveBeenCalled();
  });

  it('saturates the published trainer step counter instead of wrapping u32', () => {
    const writeBuffer = vi.fn();
    const subsystem = Object.create(NrcSubsystem.prototype) as NrcSubsystem;
    Object.assign(subsystem, {
      _device: { queue: { writeBuffer } },
      _cfgUbo: { label: 'cfg' },
      cfg: { warmupSteps: 8 },
    });

    (subsystem as unknown as {
      _writeTrainingGateState(trainedSteps: number): void;
    })._writeTrainingGateState(0x1_0000_0000);

    const [, byteOffset, payload] = writeBuffer.mock.calls[0]!;
    expect(byteOffset).toBe(40);
    expect(Array.from(payload as Uint32Array)).toEqual([0xffff_ffff, 8]);
  });

    it('keeps one generation-tagged staging buffer while a prior map is pending', () => {
      const copyBufferToBuffer = vi.fn();
      const createBuffer = vi.fn(() => ({ destroy: vi.fn(), mapState: 'unmapped' }));
      const prior = { buffer: { destroy: vi.fn(), mapState: 'pending' }, generation: 3, sequence: 1, destroyed: false };
      const subsystem = Object.create(NrcSubsystem.prototype) as NrcSubsystem;
      Object.assign(subsystem, {
        _device: { createBuffer },
        _lifecycleState: 'ready',
        _readbackState: { kind: 'mapping', ticket: prior },
        _readbackOverlapSkips: 0,
        _generation: 3,
        _readbackSequence: 1,
        _runtimeArena: { size: 1024 },
        _runtimeLayout: {
          recordsByteOffset: 256,
          diagnosticsByteOffset: 512,
        },
        _recordByteSize: 64,
        _readbackByteSize: 84,
      });
      const encoder = { copyBufferToBuffer } as unknown as GPUCommandEncoder;

      subsystem.recordCopyForReadback(encoder);
      expect(createBuffer).not.toHaveBeenCalled();
      expect(copyBufferToBuffer).not.toHaveBeenCalled();
      expect((subsystem as unknown as { _readbackOverlapSkips: number })._readbackOverlapSkips).toBe(1);

      Object.assign(subsystem, { _readbackState: { kind: 'idle' } });
      subsystem.recordCopyForReadback(encoder);
      expect(createBuffer).toHaveBeenCalledTimes(1);
      expect(copyBufferToBuffer).toHaveBeenCalledTimes(2);
      expect((subsystem as unknown as { _readbackState: { kind: string } })._readbackState.kind).toBe('copy-recorded');
    });

    it('drops a mapped training batch when scene reset changes its generation', async () => {
      let resolveMap!: () => void;
      const mapAsync = vi.fn(() => new Promise<void>((resolve) => { resolveMap = resolve; }));
      const unmap = vi.fn();
      const destroy = vi.fn();
      const setBatch = vi.fn();
      const recordTrainStep = vi.fn();
      const recordTableStep = vi.fn();
      const mapped = new ArrayBuffer(48);
      new Float32Array(mapped, 0, 7).set([1, 2, 3, 4, 5, 6, 7]);
      const readback = {
        mapState: 'pending' as GPUBufferMapState,
        mapAsync,
        getMappedRange: () => mapped,
        unmap,
        destroy,
      };
      const ticket = { buffer: readback, generation: 4, sequence: 1, destroyed: false };
      const subsystem = Object.create(NrcSubsystem.prototype) as NrcSubsystem;
      Object.assign(subsystem, {
        _lifecycleState: 'ready',
        _readbackState: { kind: 'copy-recorded', ticket },
        _generation: 4,
        _staleReadbacks: 0,
        _trainingFailures: 0,
        _recordStride: 7,
        _recordByteSize: 28,
        _readbackByteSize: 48,
        _inW: 1,
        cfg: { recordCap: 1, learningRate: 0.01 },
        _lastGpuDiagnostics: new Uint32Array(5),
        _batchX: new Float32Array(1),
        _batchY: new Float32Array(3),
        _batchPos: new Float32Array(3),
        _trainer: { setBatch, recordTrainStep },
        _tableTrainer: { recordStep: recordTableStep },
      });

      const training = subsystem.trainFromRecords();
      Object.assign(subsystem, { _generation: 5 });
      resolveMap();
      await training;

      expect(setBatch).not.toHaveBeenCalled();
      expect(recordTrainStep).not.toHaveBeenCalled();
      expect(recordTableStep).not.toHaveBeenCalled();
      expect(unmap).toHaveBeenCalledOnce();
      expect(destroy).toHaveBeenCalledOnce();
      expect((subsystem as unknown as { _staleReadbacks: number })._staleReadbacks).toBe(1);
      expect((subsystem as unknown as { _readbackState: { kind: string } })._readbackState.kind).toBe('idle');
    });
});

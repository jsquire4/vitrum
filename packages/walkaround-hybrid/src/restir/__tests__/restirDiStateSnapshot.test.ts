import { describe, expect, it } from 'vitest';
import {
  assertRestirDISnapshot,
  coldMigrateLegacyRestirDISnapshot,
  isValidRestirDISnapshot,
  type RestirDISnapshot,
} from '../restirDiStateSnapshot.js';
import { RESERVOIR_DI_STRIDE_U32 } from '../reservoirDiLayout.js';

function words(width = 2, height = 2): Uint32Array {
  return new Uint32Array(
    Math.max(64, width * height * RESERVOIR_DI_STRIDE_U32),
  );
}

function setFiniteEmitterRecord(
  data: Uint32Array,
  record: number,
  areaM = 3,
  envM = 2,
): void {
  const base = record * RESERVOIR_DI_STRIDE_U32;
  const floats = new Float32Array(data.buffer);
  data[base] = 7;
  data[base + 1] = areaM + envM;
  floats[base + 2] = 4;
  floats[base + 3] = 0.5;
  floats[base + 4] = 0.25;
  floats[base + 5] = 0.75;
  data[base + 6] = areaM;
  data[base + 7] = envM;
}

function snapshot(): RestirDISnapshot {
  const current = words();
  const previous = words();
  const spatial = words();
  setFiniteEmitterRecord(current, 0);
  setFiniteEmitterRecord(previous, 1);
  setFiniteEmitterRecord(spatial, 2);
  return {
    representationVersion: 1,
    width: 2,
    height: 2,
    strideU32: RESERVOIR_DI_STRIDE_U32,
    current,
    previous,
    spatial,
  };
}

describe('ReSTIR-DI state snapshot validation', () => {
  it('accepts valid logical records and zeroed allocation padding', () => {
    const value = snapshot();
    expect(() => assertRestirDISnapshot(value)).not.toThrow();
    expect(isValidRestirDISnapshot(value)).toBe(true);
  });

  it('accepts both cold-zero and canonical live-empty records', () => {
    const value = snapshot();
    value.current.fill(0, 0, RESERVOIR_DI_STRIDE_U32);
    const floats = new Float32Array(value.current.buffer);
    const logZero = -Math.fround(3.4028234663852886e38);
    floats[2] = logZero;
    floats[3] = logZero;
    expect(() => assertRestirDISnapshot(value)).not.toThrow();

    floats[3] = 0;
    expect(() => assertRestirDISnapshot(value)).toThrow(/non-empty/);
  });

  it('rejects a missing or stale semantic representation marker', () => {
    expect(
      isValidRestirDISnapshot({
        ...snapshot(),
        representationVersion: 0,
      }),
    ).toBe(false);
  });

  it('rejects shape, non-finite, inconsistent-log, and non-zero-padding corruption', () => {
    expect(
      isValidRestirDISnapshot({
        ...snapshot(),
        current: new Uint32Array(63),
      }),
    ).toBe(false);

    const nonFinite = snapshot();
    new Float32Array(nonFinite.current.buffer)[2] = Number.NaN;
    expect(() => assertRestirDISnapshot(nonFinite)).toThrow(/non-finite/);

    const inconsistentLog = snapshot();
    new Float32Array(inconsistentLog.current.buffer)[3] = -Math.fround(
      3.4028234663852886e38,
    );
    expect(() => assertRestirDISnapshot(inconsistentLog)).toThrow(
      /inconsistent logarithmic/,
    );

    const padding = snapshot();
    padding.current[63] = 1;
    expect(() => assertRestirDISnapshot(padding)).toThrow(/padding/);
  });

  it('rejects inconsistent support counts and selected-domain support', () => {
    const countMismatch = snapshot();
    countMismatch.current[1] = 4;
    expect(() => assertRestirDISnapshot(countMismatch)).toThrow(
      /support counts/,
    );

    const missingAreaSupport = snapshot();
    missingAreaSupport.current[1] = 2;
    missingAreaSupport.current[6] = 0;
    missingAreaSupport.current[7] = 2;
    expect(() => assertRestirDISnapshot(missingAreaSupport)).toThrow(
      /outside its represented support/,
    );

    const missingEnvironmentSupport = snapshot();
    missingEnvironmentSupport.current[0] = 0xffff_ffff;
    missingEnvironmentSupport.current[1] = 3;
    missingEnvironmentSupport.current[6] = 3;
    missingEnvironmentSupport.current[7] = 0;
    expect(() => assertRestirDISnapshot(missingEnvironmentSupport)).toThrow(
      /outside its represented support/,
    );
  });

  it('accepts zero-valued estimates with attempted support but rejects dirty empty records', () => {
    const zeroEstimate = snapshot();
    const f = new Float32Array(zeroEstimate.current.buffer);
    zeroEstimate.current[0] = 0;
    zeroEstimate.current[1] = 2;
    f[2] = 0;
    f[3] = 0;
    f[4] = 0;
    f[5] = 0;
    zeroEstimate.current[6] = 2;
    zeroEstimate.current[7] = 0;
    expect(() => assertRestirDISnapshot(zeroEstimate)).not.toThrow();

    const dirtyEmpty = snapshot();
    dirtyEmpty.current.fill(0, 0, RESERVOIR_DI_STRIDE_U32);
    new Float32Array(dirtyEmpty.current.buffer)[4] = 0.5;
    expect(() => assertRestirDISnapshot(dirtyEmpty)).toThrow(/non-empty/);
  });

  it('uses saturating support arithmetic at the uint32 boundary', () => {
    const saturated = snapshot();
    saturated.current[1] = 0xffff_ffff;
    saturated.current[6] = 0xffff_fffe;
    saturated.current[7] = 2;
    expect(() => assertRestirDISnapshot(saturated)).not.toThrow();
  });

  it('cold-migrates a structurally valid legacy cohort without reusing weights', () => {
    const legacy = snapshot();
    const migrated = coldMigrateLegacyRestirDISnapshot({
      width: legacy.width,
      height: legacy.height,
      strideU32: legacy.strideU32,
      current: legacy.current,
      previous: legacy.previous,
      spatial: legacy.spatial,
    });
    expect(migrated.representationVersion).toBe(1);
    expect([...migrated.current]).toEqual(new Array(64).fill(0));
    expect([...migrated.previous]).toEqual(new Array(64).fill(0));
    expect([...migrated.spatial]).toEqual(new Array(64).fill(0));
  });
});

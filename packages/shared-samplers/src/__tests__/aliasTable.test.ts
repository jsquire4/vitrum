import { describe, expect, it } from 'vitest';
import {
  ALIAS_TABLE_ENTRY_BYTES,
  aliasColumnFromU32,
  buildAliasTable,
} from '../aliasTable.js';

function decode(table: ReturnType<typeof buildAliasTable>) {
  const view = new DataView(table.data);
  return Array.from({ length: table.count }, (_, index) => {
    const base = index * ALIAS_TABLE_ENTRY_BYTES;
    return {
      q: view.getFloat32(base, true),
      alias: view.getUint32(base + 4, true),
      pmf: view.getFloat32(base + 8, true),
      pad: view.getUint32(base + 12, true),
    };
  });
}

function representedPmf(entries: ReturnType<typeof decode>): number[] {
  const pmf = Array.from({ length: entries.length }, () => 0);
  const invCount = 1 / entries.length;
  for (let column = 0; column < entries.length; column += 1) {
    const entry = entries[column]!;
    pmf[column] = pmf[column]! + entry.q * invCount;
    pmf[entry.alias] = pmf[entry.alias]! + (1 - entry.q) * invCount;
  }
  return pmf;
}

describe('buildAliasTable', () => {
  it('keeps the final identity reachable above the f32 24-bit domain', () => {
    const count = (1 << 24) + 1;
    expect(() => aliasColumnFromU32(0, count)).toThrow(/rejection threshold/);
    expect(aliasColumnFromU32(count - 1, count)).toBe(count - 1);
  });

  it('maps accepted non-power-of-two words uniformly after rejection', () => {
    expect(() => aliasColumnFromU32(5, 10)).toThrow(/rejection threshold 6/);
    expect(Array.from({ length: 10 }, (_, index) => aliasColumnFromU32(index + 6, 10)))
      .toEqual([6, 7, 8, 9, 0, 1, 2, 3, 4, 5]);
  });

  it('stores the PMF represented by its quantized wire thresholds', () => {
    const table = buildAliasTable([1, 3, 0, 2]);
    const entries = decode(table);
    const represented = representedPmf(entries);

    expect(table.data.byteLength).toBe(4 * ALIAS_TABLE_ENTRY_BYTES);
    expect(entries.every((entry) => entry.alias < entries.length && entry.pad === 0)).toBe(true);
    for (let index = 0; index < entries.length; index += 1) {
      expect(entries[index]!.pmf).toBeCloseTo(represented[index]!, 7);
    }
    expect(entries[0]!.pmf).toBeGreaterThan(0);
    expect(entries[1]!.pmf).toBeGreaterThan(0);
    expect(entries[2]!.pmf).toBe(0);
    expect(entries[3]!.pmf).toBeGreaterThan(0);
  });

  it('uses a uniform self-aliasing distribution for all-zero weights', () => {
    const entries = decode(buildAliasTable([0, 0, 0, 0]));
    expect(entries).toEqual([
      { q: 1, alias: 0, pmf: 0.25, pad: 0 },
      { q: 1, alias: 1, pmf: 0.25, pad: 0 },
      { q: 1, alias: 2, pmf: 0.25, pad: 0 },
      { q: 1, alias: 3, pmf: 0.25, pad: 0 },
    ]);
  });

  it('preserves exact zero support when positive weights exist', () => {
    const entries = decode(buildAliasTable([0, 4, 0]));
    expect(entries[0]!.pmf).toBe(0);
    expect(entries[1]!.pmf).toBe(1);
    expect(entries[2]!.pmf).toBe(0);
  });

  it('accepts typed arrays and finite weights whose direct sum would overflow', () => {
    const table = buildAliasTable(
      new Float64Array([Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE]),
    );
    for (const entry of decode(table)) expect(entry.pmf).toBeCloseTo(1 / 3, 7);
  });

  it('retains represented support across extreme finite dynamic range', () => {
    const entries = decode(buildAliasTable([Number.MIN_VALUE, Number.MAX_VALUE]));
    expect(entries[0]!.pmf).toBeGreaterThan(0);
    expect(entries[1]!.pmf).toBeGreaterThan(0);
    expect(entries[0]!.pmf + entries[1]!.pmf).toBeCloseTo(1, 7);
  });

  it('rejects negative and non-finite weights', () => {
    expect(() => buildAliasTable([1, -1])).toThrow(/nonnegative/);
    expect(() => buildAliasTable([1, Number.NaN])).toThrow(/finite/);
    expect(() => buildAliasTable([1, Number.POSITIVE_INFINITY])).toThrow(/finite/);
  });
});

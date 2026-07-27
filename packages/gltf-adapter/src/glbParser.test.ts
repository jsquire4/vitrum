import { describe, expect, it } from 'vitest';
import { parseGlb } from './glbParser.js';

function buildGlbWithBin(): ArrayBuffer {
  const jsonBytes = new TextEncoder().encode('{"asset":{"version":"2.0"}}');
  const paddedJsonLength = Math.ceil(jsonBytes.length / 4) * 4;
  const binLength = 4;
  const totalLength = 12 + 8 + paddedJsonLength + 8 + binLength;
  const buffer = new ArrayBuffer(totalLength);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, totalLength, true);
  view.setUint32(12, paddedJsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(0x20, 20, 20 + paddedJsonLength);
  bytes.set(jsonBytes, 20);

  const binHeaderOffset = 20 + paddedJsonLength;
  view.setUint32(binHeaderOffset, binLength, true);
  view.setUint32(binHeaderOffset + 4, 0x004e4942, true);
  bytes.set([1, 2, 3, 4], binHeaderOffset + 8);
  return buffer;
}

describe('parseGlb host-owned ArrayBuffer handling', () => {
  it('uses intrinsic length and slicing when own properties shadow ArrayBuffer APIs', () => {
    const input = buildGlbWithBin();
    Object.defineProperty(input, 'byteLength', { value: 0 });
    Object.defineProperty(input, 'slice', {
      value: () => {
        throw new Error('shadowed slice must not run');
      },
    });

    const parsed = parseGlb(input);

    expect(parsed.json.asset.version).toBe('2.0');
    expect(Array.from(new Uint8Array(parsed.binChunk!))).toEqual([1, 2, 3, 4]);
  });

  it('runs BIN allocation preflight before the intrinsic slice', () => {
    const input = buildGlbWithBin();
    const stopBeforeCopy = new Error('stop before BIN copy');
    let observed: unknown;

    try {
      parseGlb(input, {
        beforeBinChunkCopy: (info) => {
          observed = info;
          throw stopBeforeCopy;
        },
      });
    } catch (error) {
      expect(error).toBe(stopBeforeCopy);
    }

    expect(observed).toEqual({
      byteOffset: input.byteLength - 4,
      byteLength: 4,
    });
  });

  it('reports a detached input as a typed short-header failure', () => {
    const input = buildGlbWithBin();
    structuredClone(input, { transfer: [input] });

    expect(() => parseGlb(input)).toThrow(expect.objectContaining({
      reason: 'glb-header-too-small',
      actualLength: 0,
    }));
  });

  it('rejects a declared total length smaller than the actual container', () => {
    const input = buildGlbWithBin();
    new DataView(input).setUint32(8, input.byteLength - 4, true);

    expect(() => parseGlb(input)).toThrow(expect.objectContaining({
      reason: 'glb-declared-length-mismatch',
    }));
  });

  it('rejects BIN before JSON', () => {
    const input = buildGlbWithBin();
    new DataView(input).setUint32(16, 0x004e4942, true);

    expect(() => parseGlb(input)).toThrow(expect.objectContaining({
      reason: 'glb-invalid-chunk-order',
    }));
  });

  it('rejects duplicate JSON and BIN chunks', () => {
    const duplicateJson = buildGlbWithBin();
    const jsonLength = new DataView(duplicateJson).getUint32(12, true);
    new DataView(duplicateJson).setUint32(20 + jsonLength + 4, 0x4e4f534a, true);
    expect(() => parseGlb(duplicateJson)).toThrow(expect.objectContaining({
      reason: 'glb-duplicate-chunk',
    }));

    const original = buildGlbWithBin();
    const duplicateBin = new Uint8Array(original.byteLength + 12);
    duplicateBin.set(new Uint8Array(original));
    const duplicateView = new DataView(duplicateBin.buffer);
    duplicateView.setUint32(8, duplicateBin.byteLength, true);
    duplicateView.setUint32(original.byteLength, 4, true);
    duplicateView.setUint32(original.byteLength + 4, 0x004e4942, true);
    duplicateBin.set([5, 6, 7, 8], original.byteLength + 8);
    expect(() => parseGlb(duplicateBin.buffer)).toThrow(expect.objectContaining({
      reason: 'glb-duplicate-chunk',
    }));
  });

  it('rejects non-aligned chunk lengths', () => {
    const input = buildGlbWithBin();
    const view = new DataView(input);
    view.setUint32(12, view.getUint32(12, true) - 1, true);

    expect(() => parseGlb(input)).toThrow(expect.objectContaining({
      reason: 'glb-invalid-chunk-alignment',
    }));
  });

  it('rejects a trailing fragment shorter than a chunk header', () => {
    const original = buildGlbWithBin();
    const withTail = new Uint8Array(original.byteLength + 4);
    withTail.set(new Uint8Array(original));
    new DataView(withTail.buffer).setUint32(8, withTail.byteLength, true);

    expect(() => parseGlb(withTail.buffer)).toThrow(expect.objectContaining({
      reason: 'glb-trailing-bytes',
    }));
  });

  it('rejects malformed UTF-8 in the JSON chunk instead of replacing it', () => {
    const input = buildGlbWithBin();
    const bytes = new Uint8Array(input);
    const versionDigit = bytes.indexOf(0x32, 20);
    expect(versionDigit).toBeGreaterThanOrEqual(20);
    bytes[versionDigit] = 0xff;

    expect(() => parseGlb(input)).toThrow(expect.objectContaining({
      reason: 'glb-json-parse-failed',
    }));
  });
});

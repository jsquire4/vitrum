/**
 * restirPtParamsPin.test.ts — I4.4
 *
 * Structural pin for the `RestirPtParams` WGSL struct ↔ host packer layout.
 *
 * The `GpuResources.writeReservoirParams` host packer writes an 8-field × 4-byte
 * UBO (32 bytes total).  The WGSL struct `RestirPtParams` in
 * `reservoirPtHero.wgsl.ts` must mirror that layout exactly or GPU reads
 * will silently fetch the wrong values.
 *
 * This test asserts:
 *   (a) The WGSL struct fields parse out in the order declared in
 *       RESTIR_PT_PARAMS_FIELDS.
 *   (b) The total byte size encoded in RESTIR_PT_PARAMS_BYTES equals 32.
 *   (c) The WGSL struct has exactly as many fields as RESTIR_PT_PARAMS_FIELDS.
 *   (d) Each field's name and type in RESTIR_PT_PARAMS_FIELDS matches the WGSL.
 */
import { describe, expect, it } from 'vitest';
import {
  RESTIR_PT_PARAMS_WGSL,
  RESTIR_PT_PARAMS_FIELDS,
  RESTIR_PT_PARAMS_BYTES,
} from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';

// ── helpers ───────────────────────────────────────────────────────────────────

interface ParsedField {
  name: string;
  type: string;
}

/** Parse `struct NAME { fields }` from a WGSL string. */
function parseWgslStructFields(wgsl: string, structName: string): ParsedField[] {
  const re = new RegExp(`struct\\s+${structName}\\s*\\{([\\s\\S]*?)\\}`, 'm');
  const match = wgsl.match(re);
  if (match == null) throw new Error(`struct ${structName} not found in WGSL`);
  const body = match[1] ?? '';
  const fields: ParsedField[] = [];
  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('//')) continue;
    // Strip inline comment and trailing comma.
    const noComment = (line.split('//')[0] ?? line).replace(/,\s*$/, '').trim();
    if (!noComment) continue;
    const m = noComment.match(/^(\w+)\s*:\s*(\w+)$/);
    if (m == null) continue;
    fields.push({ name: m[1]!, type: m[2]! });
  }
  return fields;
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe('RestirPtParams WGSL struct ↔ host packer pin (I4.4)', () => {
  const wgslFields = parseWgslStructFields(RESTIR_PT_PARAMS_WGSL, 'RestirPtParams');

  it('RESTIR_PT_PARAMS_BYTES is 32 (8 × 4-byte fields)', () => {
    expect(RESTIR_PT_PARAMS_BYTES).toBe(32);
  });

  it('WGSL struct has exactly the same number of fields as RESTIR_PT_PARAMS_FIELDS', () => {
    expect(wgslFields.length).toBe(RESTIR_PT_PARAMS_FIELDS.length);
  });

  it('WGSL struct field names match RESTIR_PT_PARAMS_FIELDS in declaration order', () => {
    const wgslNames = wgslFields.map((f) => f.name);
    const pinNames = RESTIR_PT_PARAMS_FIELDS.map((f) => f.name);
    expect(wgslNames).toEqual(pinNames);
  });

  it('WGSL struct field types match RESTIR_PT_PARAMS_FIELDS', () => {
    for (let i = 0; i < RESTIR_PT_PARAMS_FIELDS.length; i++) {
      const pin = RESTIR_PT_PARAMS_FIELDS[i]!;
      const wgsl = wgslFields[i]!;
      expect(wgsl.type).toBe(pin.type);
    }
  });

  it('RESTIR_PT_PARAMS_FIELDS byte offsets are contiguous 4-byte steps', () => {
    // All fields are scalar (u32 or f32 = 4 bytes), so offsets must be
    // 0, 4, 8, 12, 16, 20, 24, 28 — no gaps.
    for (let i = 0; i < RESTIR_PT_PARAMS_FIELDS.length; i++) {
      expect(RESTIR_PT_PARAMS_FIELDS[i]!.byteOffset).toBe(i * 4);
    }
  });

  it('last field ends at RESTIR_PT_PARAMS_BYTES', () => {
    const last = RESTIR_PT_PARAMS_FIELDS[RESTIR_PT_PARAMS_FIELDS.length - 1]!;
    expect(last.byteOffset + 4).toBe(RESTIR_PT_PARAMS_BYTES);
  });
});

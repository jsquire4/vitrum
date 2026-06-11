/**
 * sppmStatsPin.test.ts — I4.5
 *
 * Structural pin for the `SppmStats` WGSL struct ↔ host packer layout.
 *
 * The `GpuResources.writeSppmStats` host packer writes an 8-field × 4-byte
 * UBO (32 bytes total).  The WGSL struct `SppmStats` inside
 * `SPPM_GROUP4_BINDINGS_WGSL` (sppmBindings.wgsl.ts) must mirror that layout
 * exactly or GPU reads will silently fetch the wrong values.
 *
 * This test asserts:
 *   (a) The WGSL struct fields parse out in the order declared in
 *       SPPM_STATS_FIELDS.
 *   (b) The total byte size SPPM_STATS_BYTES equals 32.
 *   (c) The WGSL struct has exactly as many fields as SPPM_STATS_FIELDS.
 *   (d) Each field's name and type match between the two representations.
 */
import { describe, expect, it } from 'vitest';
import {
  SPPM_GROUP4_BINDINGS_WGSL,
  SPPM_STATS_BYTES,
  SPPM_STATS_FIELDS,
} from '../wgsl/pathTrace/sppmBindings.wgsl.js';

// ── helpers ───────────────────────────────────────────────────────────────────

interface ParsedField {
  name: string;
  type: string;
}

/** Parse `struct NAME { fields }` from a WGSL template string. */
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
    // Field: `name  : type`  (possibly extra whitespace)
    const m = noComment.match(/^(\w+)\s*:\s*(\w+)$/);
    if (m == null) continue;
    fields.push({ name: m[1]!, type: m[2]! });
  }
  return fields;
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe('SppmStats WGSL struct ↔ host packer pin (I4.5)', () => {
  const wgslFields = parseWgslStructFields(SPPM_GROUP4_BINDINGS_WGSL, 'SppmStats');

  it('SPPM_STATS_BYTES is 32 (8 × 4-byte fields)', () => {
    expect(SPPM_STATS_BYTES).toBe(32);
  });

  it('WGSL struct has exactly the same number of fields as SPPM_STATS_FIELDS', () => {
    expect(wgslFields.length).toBe(SPPM_STATS_FIELDS.length);
  });

  it('WGSL struct field names match SPPM_STATS_FIELDS in declaration order', () => {
    const wgslNames = wgslFields.map((f) => f.name);
    const pinNames = SPPM_STATS_FIELDS.map((f) => f.name);
    expect(wgslNames).toEqual(pinNames);
  });

  it('WGSL struct field types match SPPM_STATS_FIELDS', () => {
    for (let i = 0; i < SPPM_STATS_FIELDS.length; i++) {
      const pin = SPPM_STATS_FIELDS[i]!;
      const wgsl = wgslFields[i]!;
      expect(wgsl.type).toBe(pin.type);
    }
  });

  it('SPPM_STATS_FIELDS byte offsets are contiguous 4-byte steps', () => {
    // All fields are scalar (f32 or u32 = 4 bytes), so offsets must be
    // 0, 4, 8, 12, 16, 20, 24, 28 — no gaps, no padding between fields.
    for (let i = 0; i < SPPM_STATS_FIELDS.length; i++) {
      expect(SPPM_STATS_FIELDS[i]!.byteOffset).toBe(i * 4);
    }
  });

  it('last field ends at SPPM_STATS_BYTES', () => {
    const last = SPPM_STATS_FIELDS[SPPM_STATS_FIELDS.length - 1]!;
    expect(last.byteOffset + 4).toBe(SPPM_STATS_BYTES);
  });
});

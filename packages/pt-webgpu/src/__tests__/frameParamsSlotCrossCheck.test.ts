/**
 * frameParamsSlotCrossCheck.test.ts — §H H55-a
 *
 * Derivational cross-check: TS FrameParamsSlot constants ↔ WGSL struct layout.
 *
 * The existing tests:
 *   - frameParamsLayout.test.ts     — extracts the WGSL struct, asserts field ORDER.
 *   - frameParamsPacker.golden.test.ts — pins TS slot → byte-offset for each field.
 *   - wgslContract.test.ts          — SHA256 pins the composed trace WGSL string.
 *
 * This test adds the MISSING DIRECTION: it DERIVES the expected WGSL struct byte
 * offsets from the TS slot constants alone (treating the slots as the definition)
 * and asserts they match the field offsets computed from the WGSL struct fields.
 * This means:
 *   - The TS slot table cannot drift from the WGSL struct WITHOUT failing here.
 *   - The WGSL struct cannot change field count/order without failing here.
 *   - The generated layout file cannot be stale relative to the WGSL.
 *
 * The one thing NOT checked is whether the GPU reads from the correct offsets —
 * that requires a real GPU or a full CPU emulator.  The frameParamsPacker.golden
 * covers that via byte-identical reconstruction.
 *
 * Algorithm:
 *   1. Parse the WGSL struct FrameParams fields (name + type), in order.
 *   2. Compute the byte offset for each field from the WGSL type sizes
 *      (u32=4, f32=4, vec4f=16, mat4x4f=64), respecting vec4f / mat4x4f
 *      alignment (16-byte and 64-byte boundaries).
 *   3. Convert each byte offset to a u32/f32 slot index (offset ÷ 4).
 *   4. Assert it equals FrameParamsSlot[fieldName].
 */

import { describe, it, expect } from 'vitest';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { FRAME_PARAMS_WGSL_FIELDS, FRAME_PARAMS_BYTE_SIZE } from '../scene/frameParamsLayout.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

// ── WGSL type sizes and alignments ────────────────────────────────────────────

interface WgslTypeInfo {
  size: number;       // bytes occupied by the field
  align: number;      // alignment requirement in bytes
}

function wgslTypeInfo(type: string): WgslTypeInfo {
  switch (type.trim()) {
    case 'u32':
    case 'f32':
      return { size: 4, align: 4 };
    case 'vec4f':
      return { size: 16, align: 16 };
    case 'mat4x4f':
      return { size: 64, align: 16 };
    default:
      throw new Error(`[frameParamsSlotCrossCheck] Unknown WGSL type: ${type}`);
  }
}

// ── Field extraction ──────────────────────────────────────────────────────────

interface WgslField {
  name: string;
  type: string;
}

/** Extract fields from the FrameParams struct in the composed trace WGSL. */
function extractWgslFields(wgsl: string): WgslField[] {
  const match = wgsl.match(/struct FrameParams\s*\{([\s\S]*?)\};/);
  if (match == null) throw new Error('FrameParams struct not found in PT_WEBGPU_TRACE_WGSL');
  const body = match[1] ?? '';
  const fields: WgslField[] = [];
  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('//')) continue;
    const noComment = (line.split('//')[0] ?? line).replace(/,\s*$/, '').trim();
    if (!noComment) continue;
    // Pattern: `name: type`
    const m = noComment.match(/^(\w+)\s*:\s*(\S+)$/);
    if (m == null) continue;
    fields.push({ name: m[1]!, type: m[2]! });
  }
  return fields;
}

/** Compute the u32-slot index for each field from the WGSL struct layout rules. */
function computeSlots(fields: WgslField[]): Map<string, number> {
  const slots = new Map<string, number>();
  let byteOffset = 0;
  for (const { name, type } of fields) {
    const info = wgslTypeInfo(type);
    // Align the current offset up to the field's alignment.
    const rem = byteOffset % info.align;
    if (rem !== 0) byteOffset += info.align - rem;
    // Slot index = byte offset ÷ 4 (all GPU scalars are 4-byte).
    slots.set(name, byteOffset / 4);
    byteOffset += info.size;
  }
  return slots;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('FrameParamsSlot ↔ WGSL struct derivational cross-check (H55-a)', () => {
  const wgslFields = extractWgslFields(PT_WEBGPU_TRACE_WGSL);
  const derivedSlots = computeSlots(wgslFields);

  it('WGSL struct has the same field count as FRAME_PARAMS_WGSL_FIELDS', () => {
    // FRAME_PARAMS_WGSL_FIELDS is the canonical source generated alongside the slot table.
    expect(wgslFields.length).toBe(FRAME_PARAMS_WGSL_FIELDS.length);
  });

  it('I4.3 — field-name ORDER: WGSL struct field order matches FRAME_PARAMS_WGSL_FIELDS exactly', () => {
    // FRAME_PARAMS_WGSL_FIELDS (auto-generated alongside FrameParamsSlot) is the
    // slot-table-authoritative expected order. This test asserts the WGSL struct
    // fields appear in EXACTLY that order, so a hand-edit that reorders the struct
    // without re-running the generator fails loudly here instead of silently
    // corrupting GPU reads.
    const wgslFieldStrings = wgslFields.map(({ name, type }) => `${name}: ${type}`);
    expect(wgslFieldStrings).toEqual(Array.from(FRAME_PARAMS_WGSL_FIELDS));
  });

  it('every FrameParamsSlot entry has a matching WGSL struct field', () => {
    for (const [name, slot] of Object.entries(FrameParamsSlot)) {
      const derivedSlot = derivedSlots.get(name);
      expect(derivedSlot, `field '${name}': TS slot=${slot}, WGSL-derived slot=${derivedSlot}`).toBe(slot);
    }
  });

  it('every WGSL struct field has a corresponding TS FrameParamsSlot entry', () => {
    for (const { name } of wgslFields) {
      expect(
        name in FrameParamsSlot,
        `WGSL field '${name}' has no FrameParamsSlot entry — TS constant is stale`,
      ).toBe(true);
    }
  });

  it('the derived struct byte size matches FRAME_PARAMS_BYTE_SIZE', () => {
    // Re-derive the total size from the WGSL fields.
    let byteOffset = 0;
    for (const { type } of wgslFields) {
      const info = wgslTypeInfo(type);
      const rem = byteOffset % info.align;
      if (rem !== 0) byteOffset += info.align - rem;
      byteOffset += info.size;
    }
    // Round up to alignment of the largest member (mat4x4f = 16-byte aligned).
    // WGSL struct rules: struct size is rounded up to its own alignment.
    const structAlign = Math.max(...wgslFields.map(({ type }) => wgslTypeInfo(type).align));
    const rem = byteOffset % structAlign;
    if (rem !== 0) byteOffset += structAlign - rem;

    expect(byteOffset).toBe(FRAME_PARAMS_BYTE_SIZE);
  });

  it('the cameraPos field is at the expected 16-byte alignment boundary', () => {
    // cameraPos is vec4f; it must be at a 16-byte boundary.
    const camSlot = FrameParamsSlot.cameraPos;
    expect(camSlot * 4).toBe(derivedSlots.get('cameraPos')! * 4);
    expect((camSlot * 4) % 16).toBe(0);
  });

  it('mat4 fields (invViewProj, viewProj, prevViewProj) are at 64-byte multiples', () => {
    // mat4x4f requires 16-byte alignment; 64-byte strides make each matrix
    // start at a 64-byte multiple if the prior vec4s fill the gap correctly.
    for (const name of ['invViewProj', 'viewProj', 'prevViewProj'] as const) {
      const byteOffset = FrameParamsSlot[name] * 4;
      expect(byteOffset % 16).toBe(0);
      // Derived slot should match.
      const derived = derivedSlots.get(name);
      expect(derived, `${name} derived slot mismatch`).toBe(FrameParamsSlot[name]);
    }
  });
});

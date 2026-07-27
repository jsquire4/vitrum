/**
 * Unit coverage for @vitrum/stained-glass-extensions.
 *
 * Two of the three modules are pure ID/key registries (wire-level enums) whose
 * VALUES are themselves the contract — drift means silent GPU corruption, so
 * the tests pin every entry exactly. The third, `packCameUBO`, is a std140 UBO
 * wire-packer: it gets an exact byte-layout pin (byteLength + per-offset value +
 * padding), since a layout drift the consuming shader can't see is silent
 * corruption.
 *
 * Authoritative came layout (from cameUniformUploader.ts):
 *   SEGMENT_FLOATS = 16  → each segment occupies a 64-byte std140 slot (mat4),
 *                          but only floats 0..8 carry data; 9..15 are padding.
 *     [0] startWorld.x  [1] startWorld.y  [2] startWorld.z  [3] railWidth
 *     [4] endWorld.x    [5] endWorld.y    [6] endWorld.z    [7] blockHeight
 *     [8] webThickness  [9..15] = 0 (pad)
 *   NODE_FLOATS = 4      → each node occupies a 16-byte vec4 slot.
 *     [0] position.x  [1] position.y  [2] position.z  [3] radius
 */
import { describe, expect, it, vi } from 'vitest';
import {
  SURFACE_TEXTURE_ID,
  packCameUBO,
} from '../src/index.js';
import type { CameNode, CameSegment } from '../src/index.js';

const SEGMENT_FLOATS = 16;
const NODE_FLOATS = 4;
const F32 = Float32Array.BYTES_PER_ELEMENT; // 4

describe('SURFACE_TEXTURE_ID — wire-level texture enum', () => {
  it('pins every entry to its exact integer (renumbering = silent shader-switch corruption)', () => {
    expect(SURFACE_TEXTURE_ID).toEqual({
      smooth: 0,
      hammered: 1,
      ripple: 2,
      granite: 3,
      baroque: 4,
      waterglass: 5,
      catspaw: 6,
      flemish: 7,
    });
  });

  it('is a dense, gapless 0..N-1 sequence (the "next unused integer" rule)', () => {
    const values = Object.values(SURFACE_TEXTURE_ID).sort((a, b) => a - b);
    expect(values).toEqual(values.map((_, i) => i));
    expect(new Set(values).size).toBe(values.length); // no duplicate IDs
  });

  it('starts at 0 (smooth is the default/zero glass surface)', () => {
    expect(SURFACE_TEXTURE_ID.smooth).toBe(0);
  });
});

// ── Fixtures with distinct, easily-tracked values per field ───────────────────
const SEG: CameSegment = {
  startWorld: [1, 2, 3],
  endWorld: [4, 5, 6],
  railWidth: 0.25,
  blockHeight: 0.5,
  webThickness: 0.125,
};
const NODE: CameNode = {
  position: [7, 8, 9],
  radius: 0.75,
};

describe('packCameUBO — GPU std140 wire contract (EXACT byte layout)', () => {
  it('a single segment occupies exactly one 64-byte std140 slot', () => {
    const { segments } = packCameUBO([SEG], []);
    // 16 floats × 4 bytes = 64 bytes = one mat4 std140 slot.
    expect(segments.length).toBe(SEGMENT_FLOATS);
    expect(segments.byteLength).toBe(SEGMENT_FLOATS * F32);
    expect(segments.byteLength).toBe(64);
  });

  it('packs each segment field at its authoritative offset (values 0..8)', () => {
    const { segments } = packCameUBO([SEG], []);
    // vec4 #0: startWorld.xyz + railWidth in .w
    expect(segments[0]).toBe(1); // startWorld.x  @ byte 0
    expect(segments[1]).toBe(2); // startWorld.y  @ byte 4
    expect(segments[2]).toBe(3); // startWorld.z  @ byte 8
    expect(segments[3]).toBe(0.25); // railWidth   @ byte 12 (.w lane)
    // vec4 #1: endWorld.xyz + blockHeight in .w
    expect(segments[4]).toBe(4); // endWorld.x    @ byte 16
    expect(segments[5]).toBe(5); // endWorld.y    @ byte 20
    expect(segments[6]).toBe(6); // endWorld.z    @ byte 24
    expect(segments[7]).toBe(0.5); // blockHeight  @ byte 28 (.w lane)
    // vec4 #2: webThickness in .x
    expect(segments[8]).toBe(0.125); // webThickness @ byte 32
  });

  it('zeroes the std140 padding lanes (floats 9..15 / bytes 36..63)', () => {
    const { segments } = packCameUBO([SEG], []);
    for (let i = 9; i < SEGMENT_FLOATS; i++) {
      expect(segments[i]).toBe(0);
    }
  });

  it('reads the same offsets back through a DataView (byte-exact, little-endian)', () => {
    const { segments } = packCameUBO([SEG], []);
    const dv = new DataView(segments.buffer);
    expect(dv.getFloat32(0 * F32, true)).toBe(1); // startWorld.x
    expect(dv.getFloat32(3 * F32, true)).toBe(0.25); // railWidth @ byte 12
    expect(dv.getFloat32(7 * F32, true)).toBe(0.5); // blockHeight @ byte 28
    expect(dv.getFloat32(8 * F32, true)).toBe(0.125); // webThickness @ byte 32
  });

  it('lays the Nth segment at base offset N·16 (stride = one std140 slot)', () => {
    const seg2: CameSegment = {
      startWorld: [10, 20, 30],
      endWorld: [40, 50, 60],
      railWidth: 0.05,
      blockHeight: 0.07,
      webThickness: 0.02,
    };
    const { segments, segmentCount } = packCameUBO([SEG, seg2], []);
    expect(segmentCount).toBe(2);
    expect(segments.length).toBe(2 * SEGMENT_FLOATS);
    expect(segments.byteLength).toBe(128);
    const b = SEGMENT_FLOATS; // second segment base = 16
    expect(segments[b + 0]).toBe(10);
    // 0.05 / 0.07 / 0.02 aren't exactly representable in float32 → close, not eq.
    expect(segments[b + 3]).toBeCloseTo(0.05, 6);
    expect(segments[b + 4]).toBe(40);
    expect(segments[b + 7]).toBeCloseTo(0.07, 6);
    expect(segments[b + 8]).toBeCloseTo(0.02, 6);
    // First segment is untouched by the second write.
    expect(segments[0]).toBe(1);
    expect(segments[8]).toBe(0.125);
  });

  it('a single node occupies exactly one 16-byte vec4 slot', () => {
    const { nodes } = packCameUBO([], [NODE]);
    expect(nodes.length).toBe(NODE_FLOATS);
    expect(nodes.byteLength).toBe(NODE_FLOATS * F32);
    expect(nodes.byteLength).toBe(16);
  });

  it('packs node position.xyz + radius.w at offsets 0..3', () => {
    const { nodes } = packCameUBO([], [NODE]);
    expect(nodes[0]).toBe(7); // position.x @ byte 0
    expect(nodes[1]).toBe(8); // position.y @ byte 4
    expect(nodes[2]).toBe(9); // position.z @ byte 8
    expect(nodes[3]).toBe(0.75); // radius     @ byte 12
    const dv = new DataView(nodes.buffer);
    expect(dv.getFloat32(3 * F32, true)).toBe(0.75);
  });

  it('lays the Nth node at base offset N·4 (vec4 stride)', () => {
    const node2: CameNode = { position: [11, 12, 13], radius: 0.9 };
    const { nodes, nodeCount } = packCameUBO([], [NODE, node2]);
    expect(nodeCount).toBe(2);
    expect(nodes.byteLength).toBe(32);
    expect(nodes[4]).toBe(11);
    expect(nodes[5]).toBe(12);
    expect(nodes[6]).toBe(13);
    expect(nodes[7]).toBeCloseTo(0.9, 6); // 0.9 not exactly representable in float32
  });

  it('reports counts and keeps the two buffers independent (no cross-contamination)', () => {
    const { segments, nodes, segmentCount, nodeCount } = packCameUBO([SEG], [NODE]);
    expect(segmentCount).toBe(1);
    expect(nodeCount).toBe(1);
    expect(segments.byteLength).toBe(64);
    expect(nodes.byteLength).toBe(16);
    expect(nodes[0]).toBe(7); // node buffer is not polluted by segment data
    expect(segments[0]).toBe(1);
  });

  it('handles empty inputs with zero-length buffers (no allocation past need)', () => {
    const empty = packCameUBO([], []);
    expect(empty.segments.length).toBe(0);
    expect(empty.nodes.length).toBe(0);
    expect(empty.segmentCount).toBe(0);
    expect(empty.nodeCount).toBe(0);
    // One-sided empties.
    expect(packCameUBO([SEG], []).nodes.length).toBe(0);
    expect(packCameUBO([], [NODE]).segments.length).toBe(0);
  });
});

describe('packCameUBO — strict caps and explicit truncation', () => {
  it('defaults to 500 segment / 200 node caps (no warning under the cap)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const segs = Array.from({ length: 500 }, (): CameSegment => SEG);
      const nodes = Array.from({ length: 200 }, (): CameNode => NODE);
      const packed = packCameUBO(segs, nodes);
      expect(packed.segmentCount).toBe(500);
      expect(packed.nodeCount).toBe(200);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects overflow by default without silently packing a partial geometry set', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      expect(() => packCameUBO([SEG, SEG], [NODE], {
        maxSegments: 1,
        maxNodes: 1,
      })).toThrow(/pass \{ overflow: 'truncate' \}/);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('truncates only under the explicit policy and warns once per over-capped input', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const segs = Array.from({ length: 4 }, (_, i): CameSegment => ({
        startWorld: [i, 0, 0],
        endWorld: [i + 1, 0, 0],
        railWidth: 1,
        blockHeight: 2,
        webThickness: 3,
      }));
      const nodes = Array.from({ length: 3 }, (_, i): CameNode => ({
        position: [i, 0, 0],
        radius: 1,
      }));
      const packed = packCameUBO(segs, nodes, {
        maxSegments: 2,
        maxNodes: 1,
        overflow: 'truncate',
      });
      expect(packed.segmentCount).toBe(2);
      expect(packed.nodeCount).toBe(1);
      expect(packed.segments.byteLength).toBe(2 * 64);
      expect(packed.nodes.byteLength).toBe(16);
      // The retained entries are the FIRST ones (excess discarded from the tail).
      expect(packed.segments[0]).toBe(0); // first segment startWorld.x
      expect(packed.segments[SEGMENT_FLOATS]).toBe(1); // second segment startWorld.x
      expect(packed.nodes[0]).toBe(0);
      // One warning per capped collection (segments + nodes).
      expect(warn).toHaveBeenCalledTimes(2);
    } finally {
      warn.mockRestore();
    }
  });

  it('does not warn when input length exactly equals the cap', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      packCameUBO([SEG, SEG], [NODE], { maxSegments: 2, maxNodes: 1 });
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it.each([
    ['maxSegments', Number.NaN],
    ['maxSegments', Number.POSITIVE_INFINITY],
    ['maxSegments', 0],
    ['maxSegments', -1],
    ['maxSegments', 1.5],
    ['maxSegments', Number.MAX_SAFE_INTEGER + 1],
    ['maxNodes', Number.NaN],
    ['maxNodes', Number.POSITIVE_INFINITY],
    ['maxNodes', 0],
    ['maxNodes', -1],
    ['maxNodes', 1.5],
    ['maxNodes', Number.MAX_SAFE_INTEGER + 1],
  ])('rejects unsafe allocation cap %s=%s', (key, value) => {
    expect(() => packCameUBO([], [], { [key]: value })).toThrow(
      /positive safe integer/,
    );
  });

  it('rejects caps that could authorize an unreasonably large typed-array allocation', () => {
    expect(() => packCameUBO([], [], { maxSegments: 4_194_305 })).toThrow(
      /larger than 268435456 bytes/,
    );
    expect(() => packCameUBO([], [], { maxNodes: 16_777_217 })).toThrow(
      /larger than 268435456 bytes/,
    );
  });

  it('enforces one combined allocation budget across both output buffers', () => {
    expect(() => packCameUBO([], [], {
      maxSegments: 4_194_303,
      maxNodes: 5,
    })).toThrow(/combined bytes.*total allocation budget is 268435456 bytes/);
  });

  it('rejects unknown overflow policies', () => {
    expect(() => packCameUBO([], [], { overflow: 'drop' } as never)).toThrow(
      /overflow must be either 'error' or 'truncate'/,
    );
  });

  it('rejects unknown option keys instead of silently ignoring misspellings', () => {
    expect(() => packCameUBO([], [], {
      maxSegments: 1,
      maxNode: 1,
    } as never)).toThrow(/unknown option maxNode/);
    expect(() => packCameUBO([], [], {
      [Symbol('extra')]: true,
    } as never)).toThrow(/unknown option Symbol\(extra\)/);
  });

  it('publishes truncation warnings only after a complete pack succeeds', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      expect(() => packCameUBO(
        [SEG, SEG],
        [{ ...NODE, radius: 0 }],
        { maxSegments: 1, maxNodes: 1, overflow: 'truncate' },
      )).toThrow(/nodes\[0\]\.radius/);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });
});

describe('packCameUBO — runtime geometry validation', () => {
  it.each([
    ['railWidth', 0],
    ['railWidth', -1],
    ['railWidth', Number.MIN_VALUE],
    ['blockHeight', Number.NaN],
    ['blockHeight', Number.POSITIVE_INFINITY],
    ['webThickness', Number.MAX_VALUE],
  ])('rejects invalid segment dimension %s=%s', (field, value) => {
    expect(() => packCameUBO([
      { ...SEG, [field]: value },
    ] as CameSegment[], [])).toThrow(new RegExp(`segments\\[0\\]\\.${field}`));
  });

  it.each([0, -1, Number.MIN_VALUE, Number.NaN, Number.POSITIVE_INFINITY, Number.MAX_VALUE])(
    'rejects invalid node radius %s',
    (radius) => {
      expect(() => packCameUBO([], [{ ...NODE, radius }])).toThrow(/nodes\[0\]\.radius/);
    },
  );

  it('rejects malformed and non-finite coordinate vectors before packing', () => {
    expect(() => packCameUBO([
      { ...SEG, startWorld: [1, 2] },
    ] as unknown as CameSegment[], [])).toThrow(/startWorld must contain exactly three/);
    expect(() => packCameUBO([
      { ...SEG, endWorld: [1, Number.NaN, 3] },
    ], [])).toThrow(/endWorld\[1\] must be a finite float32/);
    expect(() => packCameUBO([], [
      { ...NODE, position: [1, Number.MAX_VALUE, 3] },
    ])).toThrow(/position\[1\] must be a finite float32/);
  });

  it('rejects zero-length segments, including endpoints that collapse in float32', () => {
    expect(() => packCameUBO([{
      ...SEG,
      endWorld: SEG.startWorld,
    }], [])).toThrow(/startWorld and endWorld must remain distinct/);
    expect(() => packCameUBO([{
      ...SEG,
      startWorld: [1, 2, 3],
      endWorld: [1 + Number.EPSILON, 2, 3],
    }], [])).toThrow(/startWorld and endWorld must remain distinct/);
  });

  it('rejects non-array collection payloads at the public boundary', () => {
    expect(() => packCameUBO({ length: 0 } as never, [])).toThrow(/segments must be an array/);
    expect(() => packCameUBO([], { length: 0 } as never)).toThrow(/nodes must be an array/);
  });
});

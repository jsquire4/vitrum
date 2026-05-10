/**
 * cameUniformUploader.test.ts — Unit tests for Sprint 5 CameUBO packer.
 *
 * Covers:
 *  - 1 segment + 1 node: verifies exact Float32Array indices
 *  - Cap enforcement (maxSegments, maxNodes)
 *  - Padding slots are zero
 *  - Empty inputs produce zero-length buffers
 */

import { describe, it, expect, vi } from 'vitest';
import { packCameUBO } from '../cameUniformUploader.js';
import type { CameSegment, CameNode } from '../cameUniformUploader.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

const SEG: CameSegment = {
  startWorld: [1, 2, 3],
  endWorld: [4, 5, 6],
  railWidth: 0.008,
  blockHeight: 0.010,
  webThickness: 0.001,
};

const NODE: CameNode = {
  position: [7, 8, 9],
  radius: 0.006,
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('packCameUBO', () => {
  describe('segment packing — 16 floats per segment (std140)', () => {
    it('packs 1 segment into a 16-float buffer', () => {
      const result = packCameUBO([SEG], []);
      expect(result.segments).toHaveLength(16);
      expect(result.segmentCount).toBe(1);
    });

    it('startWorld.xyz at indices 0, 1, 2', () => {
      const { segments } = packCameUBO([SEG], []);
      expect(segments[0]).toBeCloseTo(SEG.startWorld[0]);
      expect(segments[1]).toBeCloseTo(SEG.startWorld[1]);
      expect(segments[2]).toBeCloseTo(SEG.startWorld[2]);
    });

    it('railWidth at index 3 (shares vec4 slot with startWorld)', () => {
      const { segments } = packCameUBO([SEG], []);
      expect(segments[3]).toBeCloseTo(SEG.railWidth);
    });

    it('endWorld.xyz at indices 4, 5, 6', () => {
      const { segments } = packCameUBO([SEG], []);
      expect(segments[4]).toBeCloseTo(SEG.endWorld[0]);
      expect(segments[5]).toBeCloseTo(SEG.endWorld[1]);
      expect(segments[6]).toBeCloseTo(SEG.endWorld[2]);
    });

    it('blockHeight at index 7 (shares vec4 slot with endWorld)', () => {
      const { segments } = packCameUBO([SEG], []);
      expect(segments[7]).toBeCloseTo(SEG.blockHeight);
    });

    it('webThickness at index 8', () => {
      const { segments } = packCameUBO([SEG], []);
      expect(segments[8]).toBeCloseTo(SEG.webThickness);
    });

    it('padding slots 9–15 are zero', () => {
      const { segments } = packCameUBO([SEG], []);
      for (let i = 9; i < 16; i++) {
        expect(segments[i]).toBe(0);
      }
    });

    it('packs 2 segments correctly (second segment at offset 16)', () => {
      const seg2: CameSegment = {
        startWorld: [10, 20, 30],
        endWorld: [40, 50, 60],
        railWidth: 0.005,
        blockHeight: 0.007,
        webThickness: 0.002,
      };
      const { segments, segmentCount } = packCameUBO([SEG, seg2], []);
      expect(segmentCount).toBe(2);
      expect(segments).toHaveLength(32);

      // Second segment at base offset 16
      expect(segments[16]).toBeCloseTo(seg2.startWorld[0]);
      expect(segments[17]).toBeCloseTo(seg2.startWorld[1]);
      expect(segments[18]).toBeCloseTo(seg2.startWorld[2]);
      expect(segments[19]).toBeCloseTo(seg2.railWidth);
      expect(segments[20]).toBeCloseTo(seg2.endWorld[0]);
      expect(segments[21]).toBeCloseTo(seg2.endWorld[1]);
      expect(segments[22]).toBeCloseTo(seg2.endWorld[2]);
      expect(segments[23]).toBeCloseTo(seg2.blockHeight);
      expect(segments[24]).toBeCloseTo(seg2.webThickness);
    });
  });

  describe('node packing — 4 floats per node (vec4)', () => {
    it('packs 1 node into a 4-float buffer', () => {
      const result = packCameUBO([], [NODE]);
      expect(result.nodes).toHaveLength(4);
      expect(result.nodeCount).toBe(1);
    });

    it('position.xyz at indices 0, 1, 2', () => {
      const { nodes } = packCameUBO([], [NODE]);
      expect(nodes[0]).toBeCloseTo(NODE.position[0]);
      expect(nodes[1]).toBeCloseTo(NODE.position[1]);
      expect(nodes[2]).toBeCloseTo(NODE.position[2]);
    });

    it('radius at index 3', () => {
      const { nodes } = packCameUBO([], [NODE]);
      expect(nodes[3]).toBeCloseTo(NODE.radius);
    });
  });

  describe('combined 1 segment + 1 node', () => {
    it('both buffers populated correctly', () => {
      const result = packCameUBO([SEG], [NODE]);
      expect(result.segmentCount).toBe(1);
      expect(result.nodeCount).toBe(1);
      expect(result.segments).toHaveLength(16);
      expect(result.nodes).toHaveLength(4);
      // Spot-check cross-contamination: segment data not leaking into node buf
      expect(result.nodes[0]).toBeCloseTo(NODE.position[0]);
      expect(result.nodes[3]).toBeCloseTo(NODE.radius);
    });
  });

  describe('cap enforcement', () => {
    it('respects maxSegments and discards excess (with console.warn)', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const segs = Array.from({ length: 5 }, (_, i): CameSegment => ({
        startWorld: [i, 0, 0],
        endWorld: [i + 1, 0, 0],
        railWidth: 0.01,
        blockHeight: 0.01,
        webThickness: 0.001,
      }));
      const result = packCameUBO(segs, [], { maxSegments: 3 });
      expect(result.segmentCount).toBe(3);
      expect(result.segments).toHaveLength(48); // 3 × 16
      expect(warnSpy).toHaveBeenCalledOnce();
      warnSpy.mockRestore();
    });

    it('respects maxNodes and discards excess (with console.warn)', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const ns = Array.from({ length: 5 }, (_, i): CameNode => ({
        position: [i, 0, 0],
        radius: 0.005,
      }));
      const result = packCameUBO([], ns, { maxNodes: 2 });
      expect(result.nodeCount).toBe(2);
      expect(result.nodes).toHaveLength(8); // 2 × 4
      expect(warnSpy).toHaveBeenCalledOnce();
      warnSpy.mockRestore();
    });
  });

  describe('empty inputs', () => {
    it('empty segments → zero-length segment buffer', () => {
      const result = packCameUBO([], [NODE]);
      expect(result.segments).toHaveLength(0);
      expect(result.segmentCount).toBe(0);
    });

    it('empty nodes → zero-length node buffer', () => {
      const result = packCameUBO([SEG], []);
      expect(result.nodes).toHaveLength(0);
      expect(result.nodeCount).toBe(0);
    });

    it('both empty → zero-length buffers', () => {
      const result = packCameUBO([], []);
      expect(result.segments).toHaveLength(0);
      expect(result.nodes).toHaveLength(0);
      expect(result.segmentCount).toBe(0);
      expect(result.nodeCount).toBe(0);
    });
  });
});

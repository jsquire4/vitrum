import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { resolveReSTIRBvhMode } from '../src/restir/sceneBvhFromCore.js';

function meshScene(count: number): Scene {
  const primitives = Array.from({ length: count }, (_, i) => ({
    id: `m${i}`,
    kind: 'mesh' as const,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
    material: { baseColor: [1, 1, 1] as const, roughness: 0.5, metallic: 0 },
  }));
  return {
    primitives,
    emitters: [],
    environment: { kind: 'none' as const },
  };
}

describe('resolveReSTIRBvhMode', () => {
  it('defaults single mesh to merged', () => {
    expect(resolveReSTIRBvhMode(meshScene(1))).toBe('merged');
  });

  it('defaults multi-mesh to tlas', () => {
    expect(resolveReSTIRBvhMode(meshScene(2))).toBe('tlas');
  });

  it('honours explicit override', () => {
    expect(resolveReSTIRBvhMode(meshScene(2), 'merged')).toBe('merged');
    expect(resolveReSTIRBvhMode(meshScene(1), 'tlas')).toBe('tlas');
  });
});

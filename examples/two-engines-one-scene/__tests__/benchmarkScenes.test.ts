import { describe, expect, it } from 'vitest';
import {
  buildBenchmark200kThreeScene,
  buildTlas10InstThreeScene,
  countThreeSceneTriangles,
} from '@vitrum-examples/shared';
import * as THREE from 'three';

describe('PR-6 benchmark scenes', () => {
  it('bench200k scene is within 5% of 200k triangles', () => {
    const scene = buildBenchmark200kThreeScene(200_000);
    const tris = countThreeSceneTriangles(scene);
    expect(tris).toBeGreaterThanOrEqual(190_000);
    expect(tris).toBeLessThanOrEqual(210_000);
  });

  it('tlas10inst scene has one InstancedMesh with count 10', () => {
    const scene = buildTlas10InstThreeScene();
    let instCount = 0;
    scene.traverse((obj) => {
      if (obj instanceof THREE.InstancedMesh) {
        instCount += 1;
        expect(obj.count).toBe(10);
      }
    });
    expect(instCount).toBe(1);
  });
});

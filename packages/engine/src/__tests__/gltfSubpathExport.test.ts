import { describe, expect, it } from 'vitest';
import {
  loadGltfForEngine,
  loadGltfWithEngine,
  loadGltfWithProgressiveEngine,
  type LoadGltfWithProgressiveEngineOptions,
  type LoadGltfWithEngineOptions,
} from '../gltf.js';

describe('@vitrum/engine/gltf subpath', () => {
  it('exports the adapter bridge and createEngine-backed convenience wrapper', () => {
    const opts: LoadGltfWithEngineOptions = {
      compatibilityMode: 'best-effort',
    };

    expect(opts.compatibilityMode).toBe('best-effort');
    expect(typeof loadGltfForEngine).toBe('function');
    expect(typeof loadGltfWithEngine).toBe('function');
    expect(typeof loadGltfWithProgressiveEngine).toBe('function');
  });

  it('exports the createProgressiveEngine-backed glTF convenience wrapper types', () => {
    const canvas = {} as HTMLCanvasElement;
    const opts: LoadGltfWithProgressiveEngineOptions = {
      engineOptions: { canvas },
    };

    expect(opts.engineOptions.canvas).toBe(canvas);
  });
});

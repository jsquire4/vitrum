import { BufferAttribute } from 'three';
import { describe, expect, it, vi } from 'vitest';

// JS-only absorbed fork source; vitest resolves the alias to packages/three-gpu-pathtracer.
// @ts-expect-error - no package subpath declaration for fork internals.
import { PathTracingSceneGenerator } from '@vitrum-pathtracer/src/core/PathTracingSceneGenerator.js';
// @ts-expect-error - no package subpath declaration for fork internals.
import { GEOMETRY_REBUILT } from '@vitrum-pathtracer/src/core/utils/StaticGeometryGenerator.js';

function deferred<T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe('PathTracingSceneGenerator.generateAsync', () => {
  it('resolves queued regeneration after an in-flight async BVH build completes', async () => {
    const generator = new PathTracingSceneGenerator([]);
    const material = { uuid: 'mat-queued-async' };
    const firstBvh = deferred<{ readonly id: 'first' }>();
    const secondBvh = deferred<{ readonly id: 'second' }>();
    const generateBvh = vi
      .fn()
      .mockReturnValueOnce(firstBvh.promise)
      .mockReturnValueOnce(secondBvh.promise);

    generator.setBVHWorker({ generate: generateBvh });
    generator.staticGeometryGenerator = {
      objects: [],
      attributes: [],
      generate(targetGeometry: { setAttribute: (name: string, attribute: BufferAttribute) => void }) {
        targetGeometry.setAttribute(
          'position',
          new BufferAttribute(new Float32Array([0, 0, 0]), 3),
        );
        return {
          changeType: GEOMETRY_REBUILT,
          materials: [material],
        };
      },
    };

    const firstGenerate = generator.generateAsync();
    const queuedGenerate = generator.generateAsync();

    expect(generateBvh).toHaveBeenCalledTimes(1);

    firstBvh.resolve({ id: 'first' });
    const firstResult = await firstGenerate;

    expect(firstResult.bvh).toEqual({ id: 'first' });
    expect(generateBvh).toHaveBeenCalledTimes(2);

    secondBvh.resolve({ id: 'second' });
    await expect(queuedGenerate).resolves.toMatchObject({
      bvh: { id: 'second' },
      bvhChanged: true,
    });
  });
});

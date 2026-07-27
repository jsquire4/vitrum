import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

// A5 — BDPT host-driver loop. Verifies the per-column light-subpath passes are
// actually ISSUED (the inert-warn failure mode: bdpt:true but no passes). We record
// the ORDERED sequence of uBdptVertexCol / uBdptLightSubpathPass sets + draw calls
// so the loop structure is observable without a GPU. mockGl's name-keyed recorder
// only keeps last-write, so we use an ordered log.

const GREY: MaterialSpec = { baseColor: [0.6, 0.6, 0.6], roughness: 1, metallic: 0 };
function mesh(id: string, y: number): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([-1, y, -1, 1, y, -1, 1, y, 1, -1, y, 1]),
    normals: new Float32Array([0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: GREY,
  };
}
function sceneWithAnalyticLight(): Scene {
  // Use a finite rect-area light so the BDPT driver has an analytic endpoint source.
  return {
    primitives: [mesh('floor', 0)],
    emitters: [
      {
        kind: 'rect-area',
        id: 'L',
        position: [0, 3, 0],
        color: [1, 1, 1],
        intensity: 5,
        uAxis: [1, 0, 0],
        vAxis: [0, 0, 1],
      },
    ],
    environment: { kind: 'none' },
  };
}
function sceneWithMeshAreaLight(): Scene {
  const panel: MeshPrimitive = {
    kind: 'mesh',
    id: 'panel',
    positions: new Float32Array([-1, 3, -1, 1, 3, -1, 1, 3, 1, -1, 3, 1]),
    normals: new Float32Array([0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: GREY,
  };
  return {
    primitives: [mesh('floor', 0), panel],
    emitters: [{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 5 }],
    environment: { kind: 'none' },
  };
}
function sceneWithEnvironmentLight(): Scene {
  return {
    primitives: [mesh('floor', 0)],
    emitters: [],
    environment: {
      kind: 'hdri',
      hdri: {
        width: 1,
        height: 1,
        data: new Float32Array([1, 1, 1, 1]),
      },
      intensity: 1,
    },
  };
}
function sceneWithAnalyticAndMeshAreaLight(): Scene {
  const base = sceneWithMeshAreaLight();
  return {
    ...base,
    emitters: [
      ...base.emitters,
      ...sceneWithAnalyticLight().emitters,
    ],
  };
}
function sceneWithAnalyticAndEnvironmentLight(): Scene {
  const analytic = sceneWithAnalyticLight();
  return {
    ...sceneWithEnvironmentLight(),
    emitters: analytic.emitters,
  };
}
function frame(): FrameInput {
  const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
  const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
  return {
    viewMatrix: view as never,
    projMatrix: proj as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 16, height: 16, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: 1 },
  };
}

/** A GL mock that appends an ordered op-log of the BDPT-relevant calls. */
function orderedGl(log: { op: string; v?: unknown }[]): WebGLRenderingContext {
  const base = createMockGl();
  return new Proxy(base, {
    get(t, prop): unknown {
      if (prop === 'getUniformLocation') return (_p: unknown, name: string) => ({ __u: name });
      if (prop === 'uniform1i') {
        return (loc: { __u?: string }, v: number) => {
          const n = loc?.__u;
          if (n === 'uBdptVertexCol' || n === 'uBdptLightSubpathPass') log.push({ op: n, v });
        };
      }
      if (prop === 'uniform2f') {
        return (loc: { __u?: string }, x: number, y: number) => {
          const n = loc?.__u;
          if (n === 'resolution') log.push({ op: n, v: [x, y] });
        };
      }
      if (prop === 'drawArrays' || prop === 'drawElements') {
        return () => log.push({ op: 'draw' });
      }
      return (t as unknown as Record<string, unknown>)[prop as string];
    },
  });
}

describe('A5 BDPT host driver', () => {
  it('defaults bdpt:true to four light vertices in the 8x8 path buffer, then eye flag=0', async () => {
    const log: { op: string; v?: unknown }[] = [];
    const gl = orderedGl(log) as unknown as WebGL2RenderingContext;
    const engine = await createPTEngine_WebGL2({ device: gl, bdpt: true });
    engine.setScene(sceneWithAnalyticLight());
    engine.renderFrame(frame());

    const cols = log.filter((e) => e.op === 'uBdptVertexCol').map((e) => e.v);
    expect(cols).toEqual([0, 1, 2, 3]);
    expect(log.some((e) => e.op === 'resolution' && Array.isArray(e.v) && e.v[0] === 8 && e.v[1] === 8)).toBe(true);

    const passFlags = log.filter((e) => e.op === 'uBdptLightSubpathPass').map((e) => e.v);
    expect(passFlags).toContain(1);
    expect(passFlags[passFlags.length - 1]).toBe(0); // eye pass last
    const firstEyeFlagIdx = log.findIndex((e) => e.op === 'uBdptLightSubpathPass' && e.v === 0);
    const lastColIdx = log.map((e) => e.op).lastIndexOf('uBdptVertexCol');
    expect(firstEyeFlagIdx).toBeGreaterThan(lastColIdx); // eye flag set AFTER all columns
  });

  it('accepts an explicitly shorter two-vertex light subpath', async () => {
    const log: { op: string; v?: unknown }[] = [];
    const engine = await createPTEngine_WebGL2({
      device: orderedGl(log) as unknown as WebGL2RenderingContext,
      bdpt: true,
      bdptOptions: { maxLightBounces: 2 },
    });
    engine.setScene(sceneWithAnalyticLight());
    engine.renderFrame(frame());

    expect(log.filter((e) => e.op === 'uBdptVertexCol').map((e) => e.v)).toEqual([0, 1]);
    engine.dispose();
  });

  it('accepts the maximum eight-vertex subpath and rejects larger depths', async () => {
    const log: { op: string; v?: unknown }[] = [];
    const engine = await createPTEngine_WebGL2({
      device: orderedGl(log) as unknown as WebGL2RenderingContext,
      bdpt: true,
      bdptOptions: { maxLightBounces: 8 },
    });
    engine.setScene(sceneWithAnalyticLight());
    engine.renderFrame(frame());
    expect(log.filter((e) => e.op === 'uBdptVertexCol').map((e) => e.v)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    engine.dispose();

    const error = await createPTEngine_WebGL2({
        device: orderedGl([]) as unknown as WebGL2RenderingContext,
        bdpt: true,
        bdptOptions: { maxLightBounces: 9 },
      }).then(
        () => undefined,
        (reason: unknown) => reason,
      );
    expect(error).toBeInstanceOf(RangeError);
  });

  it('rejects non-positive, fractional, and non-finite BDPT depths', async () => {
    await expect(
      createPTEngine_WebGL2({
        device: orderedGl([]) as unknown as WebGL2RenderingContext,
        bdpt: true,
        bdptOptions: { maxLightBounces: 0 },
      }),
    ).rejects.toThrow('supported range 1..8');

    await expect(
      createPTEngine_WebGL2({
        device: orderedGl([]) as unknown as WebGL2RenderingContext,
        bdpt: true,
        bdptOptions: { maxLightBounces: 2.75 },
      }),
    ).rejects.toThrow('supported range 1..8');

    await expect(
      createPTEngine_WebGL2({
        device: orderedGl([]) as unknown as WebGL2RenderingContext,
        bdpt: true,
        bdptOptions: { maxLightBounces: Number.POSITIVE_INFINITY },
      }),
    ).rejects.toThrow('supported range 1..8');
  });

  it('does NOT issue any light-subpath pass when bdpt:false (unidirectional invariant)', async () => {
    const log: { op: string; v?: unknown }[] = [];
    const gl = orderedGl(log) as unknown as WebGL2RenderingContext;
    const engine = await createPTEngine_WebGL2({ device: gl, bdpt: false });
    engine.setScene(sceneWithAnalyticLight());
    engine.renderFrame(frame());

    // bdpt:false → the BDPT uniforms are never touched at all.
    expect(log.filter((e) => e.op === 'uBdptVertexCol')).toHaveLength(0);
    expect(log.filter((e) => e.op === 'uBdptLightSubpathPass')).toHaveLength(0);
  });

  it('skips the subpath build when there are no lights (nothing to connect to)', async () => {
    const log: { op: string; v?: unknown }[] = [];
    const gl = orderedGl(log) as unknown as WebGL2RenderingContext;
    const engine = await createPTEngine_WebGL2({ device: gl, bdpt: true });
    // No emitters and no emissive mesh → lightCount 0 → no subpath passes, eye pass still runs.
    engine.setScene({ primitives: [mesh('floor', 0)], emitters: [], environment: { kind: 'none' } });
    engine.renderFrame(frame());
    expect(log.filter((e) => e.op === 'uBdptVertexCol' && e.v !== undefined)).toHaveLength(0);
    // Main, candidate replay, and no-loop resolve each receive the inert flag.
    const passFlags = log.filter((e) => e.op === 'uBdptLightSubpathPass').map((e) => e.v);
    expect(passFlags).toEqual([0, 0, 0]);
  });

  it('builds BDPT light subpaths for mesh-area-only light sources', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const log: { op: string; v?: unknown }[] = [];
    try {
      const engine = await createPTEngine_WebGL2({
        device: orderedGl(log) as unknown as WebGL2RenderingContext,
        bdpt: true,
        onWarning: (w) => structured.push(w),
      });

      engine.setScene(sceneWithMeshAreaLight());
      engine.renderFrame(frame());

      expect(warn.mock.calls.some((c) =>
        String(c[0]).includes('BDPT connections fall back to the unidirectional'),
      )).toBe(false);
      expect(structured.some((w) => w.code === 'pt-webgl2.bdpt-source-partition')).toBe(false);
      expect(log.filter((e) => e.op === 'uBdptVertexCol').map((e) => e.v)).toEqual([0, 1, 2, 3]);
    } finally {
      warn.mockRestore();
    }
  });

  it('builds BDPT light subpaths for environment-only light sources', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const log: { op: string; v?: unknown }[] = [];
    try {
      const engine = await createPTEngine_WebGL2({
        device: orderedGl(log) as unknown as WebGL2RenderingContext,
        bdpt: true,
        onWarning: (w) => structured.push(w),
      });

      engine.setScene(sceneWithEnvironmentLight());
      engine.renderFrame(frame());

      expect(structured.some((w) => w.code === 'pt-webgl2.bdpt-source-partition')).toBe(false);
      expect(warn.mock.calls.some((c) => String(c[0]).includes('Ordinary NEE'))).toBe(false);
      expect(log.filter((e) => e.op === 'uBdptVertexCol').map((e) => e.v)).toEqual([0, 1, 2, 3]);
    } finally {
      warn.mockRestore();
    }
  });

  it('builds BDPT light subpaths when analytic and mesh-area sources are mixed', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const log: { op: string; v?: unknown }[] = [];
    try {
      const engine = await createPTEngine_WebGL2({
        device: orderedGl(log) as unknown as WebGL2RenderingContext,
        bdpt: true,
        onWarning: (w) => structured.push(w),
      });

      engine.setScene(sceneWithAnalyticAndMeshAreaLight());
      engine.renderFrame(frame());

      expect(warn.mock.calls.some((c) =>
        String(c[0]).includes('BDPT connections fall back to the unidirectional'),
      )).toBe(false);
      expect(structured.some((w) => w.code === 'pt-webgl2.bdpt-source-partition')).toBe(false);
      expect(log.filter((e) => e.op === 'uBdptVertexCol').map((e) => e.v)).toEqual([0, 1, 2, 3]);
    } finally {
      warn.mockRestore();
    }
  });

  it('builds BDPT light subpaths when analytic and environment sources are mixed', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const log: { op: string; v?: unknown }[] = [];
    try {
      const engine = await createPTEngine_WebGL2({
        device: orderedGl(log) as unknown as WebGL2RenderingContext,
        bdpt: true,
        onWarning: (w) => structured.push(w),
      });

      engine.setScene(sceneWithAnalyticAndEnvironmentLight());
      engine.setScene(sceneWithAnalyticAndEnvironmentLight());
      engine.renderFrame(frame());

      expect(structured.some((w) => w.code === 'pt-webgl2.bdpt-source-partition')).toBe(false);
      expect(warn.mock.calls.filter((c) => String(c[0]).includes('Ordinary NEE'))).toHaveLength(0);
      expect(log.filter((e) => e.op === 'uBdptVertexCol').map((e) => e.v)).toEqual([0, 1, 2, 3]);
    } finally {
      warn.mockRestore();
    }
  });
});

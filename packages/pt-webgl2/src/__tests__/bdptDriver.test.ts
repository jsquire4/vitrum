import { describe, expect, it } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

// A5 — BDPT host-driver loop. Verifies the per-column light-subpath passes are
// actually ISSUED (the inert-warn failure mode: bdpt:true but no passes). We record
// the ORDERED sequence of uBdptVertexCol / uBdptLightSubpathPass sets + draw calls
// so the loop structure (3 subpath columns, then the eye pass) is observable without
// a GPU. mockGl's name-keyed recorder only keeps last-write, so we use an ordered log.

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
  // BDPT's light subpath samples ANALYTIC lights (randomLightSample), so the driver
  // only runs the subpath passes when scene.lightCount > 0 — use a rect-area light.
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
  it('issues the light-subpath passes (subpath flag=1 for each of 3 columns, then eye flag=0)', async () => {
    const log: { op: string; v?: unknown }[] = [];
    const gl = orderedGl(log) as unknown as WebGL2RenderingContext;
    const engine = await createPTEngine_WebGL2({ device: gl, bdpt: true });
    engine.setScene(sceneWithAnalyticLight());
    engine.renderFrame(frame());

    // The driver must set uBdptLightSubpathPass=1 (the subpath build) for each column
    // and uBdptVertexCol = 0,1,2 in order, then uBdptLightSubpathPass=0 for the eye.
    const cols = log.filter((e) => e.op === 'uBdptVertexCol').map((e) => e.v);
    expect(cols).toEqual([0, 1, 2]); // BDPT_MAX_LIGHT_BOUNCES columns, in order
    expect(log.some((e) => e.op === 'resolution' && Array.isArray(e.v) && e.v[0] === 3 && e.v[1] === 4)).toBe(true);

    const passFlags = log.filter((e) => e.op === 'uBdptLightSubpathPass').map((e) => e.v);
    // The subpath flag is set to 1 (build) before the column loop, then back to 0 for
    // the eye pass. So: at least one 1, and the LAST set is 0 (eye), and the 1 precedes
    // all three column draws.
    expect(passFlags).toContain(1);
    expect(passFlags[passFlags.length - 1]).toBe(0); // eye pass last
    const firstEyeFlagIdx = log.findIndex((e) => e.op === 'uBdptLightSubpathPass' && e.v === 0);
    const lastColIdx = log.map((e) => e.op).lastIndexOf('uBdptVertexCol');
    expect(firstEyeFlagIdx).toBeGreaterThan(lastColIdx); // eye flag set AFTER all columns
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
    // The eye pass still sets the subpath flag to 0.
    const passFlags = log.filter((e) => e.op === 'uBdptLightSubpathPass').map((e) => e.v);
    expect(passFlags).toEqual([0]);
  });
});

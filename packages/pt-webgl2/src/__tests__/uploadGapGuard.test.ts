import { describe, expect, it } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

// ── Upload-gap regression GUARD (items_to_fix §H H1/H2/H3) ───────────────────
//
// The §H deep audit's root finding for pt-webgl2 was a repeating class of bug:
// scene data is computed/packed but the UNIFORM that drives it is never uploaded,
// so the feature is silently inert —
//   H1: `lights.count` never set    → all analytic lights dead (NEE/forward/BDPT).
//   H2: CMF tables never set         → `spectral` renders black.
//   H3: `backgroundAlpha` never set  → directly-visible env never accumulates.
// The existing mock-GL suite was BLIND to this (anonymous uniform locations, no-op
// setters), which is exactly why the class shipped. This guard records uniform
// sets by NAME and asserts the load-bearing uniforms ARE uploaded — so removing an
// upload (or the setUint/setFloatArray path) fails here, at unit speed, without a GPU.

const GREY: MaterialSpec = { baseColor: [0.6, 0.6, 0.6], roughness: 1, metallic: 0 };
function tri(id: string): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
}
function sceneWithPointLight(): Scene {
  return {
    primitives: [tri('tri')],
    emitters: [{ kind: 'point', id: 'p', position: [1, 2, 3], color: [1, 1, 1], intensity: 2 }],
    environment: { kind: 'none' },
  } as Scene;
}
function sceneNoEmitters(): Scene {
  return { primitives: [tri('tri')], emitters: [], environment: { kind: 'none' } } as Scene;
}
function frame(spp: number): FrameInput {
  const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
  const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
  return {
    viewMatrix: view as never,
    projMatrix: proj as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 32, height: 32, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: spp },
  };
}

async function renderAndRecord(
  scene: Scene,
  engineOpts: Record<string, unknown> = {},
): Promise<Map<string, unknown>> {
  const record = new Map<string, unknown>();
  const gl = createMockGl(record);
  const engine = await createPTEngine_WebGL2({ device: gl, ...engineOpts } as never);
  engine.setScene(scene);
  engine.renderFrame(frame(4));
  return record;
}

describe('pt-webgl2 upload-gap guard — load-bearing uniforms ARE uploaded', () => {
  it('H1: lights.count is uploaded and equals the analytic-light count', async () => {
    const withLight = await renderAndRecord(sceneWithPointLight());
    // The regression that shipped: NO setter for lights.count → key ABSENT here.
    expect(withLight.has('lights.count')).toBe(true);
    expect(withLight.get('lights.count')).toBe(1);

    const noLight = await renderAndRecord(sceneNoEmitters());
    expect(noLight.has('lights.count')).toBe(true);
    expect(noLight.get('lights.count')).toBe(0);
  });

  it('H2: the CIE CMF tables/CDFs/integrals are uploaded when spectral is enabled', async () => {
    const spectral = await renderAndRecord(sceneNoEmitters(), { spectral: true });
    for (const name of ['uCmfX', 'uCmfY', 'uCmfZ', 'uXCmfCdf', 'uYCmfCdf', 'uZCmfCdf']) {
      expect(spectral.has(name), `${name} must be uploaded`).toBe(true);
      expect(ArrayBuffer.isView(spectral.get(name)), `${name} is a float array`).toBe(true);
    }
    for (const name of ['uXCmfIntegral', 'uYCmfIntegral', 'uZCmfIntegral']) {
      expect(spectral.has(name), `${name} must be uploaded`).toBe(true);
      expect(spectral.get(name)).toBeGreaterThan(0);
    }
    // ...and NOT uploaded when spectral is off (gated → no wasted per-frame work).
    const nonSpectral = await renderAndRecord(sceneNoEmitters(), { spectral: false });
    expect(nonSpectral.has('uCmfX')).toBe(false);
  });

  it('H3: backgroundAlpha is uploaded (default opaque = 1)', async () => {
    const rec = await renderAndRecord(sceneNoEmitters());
    expect(rec.has('backgroundAlpha')).toBe(true);
    expect(rec.get('backgroundAlpha')).toBe(1);
  });

  it('the recording mock actually distinguishes set-vs-unset (meta-check)', async () => {
    // A uniform the engine never sets must be ABSENT — proves the guard can FAIL
    // (i.e. the H1/H2/H3 asserts above are meaningful, not vacuously true).
    const rec = await renderAndRecord(sceneNoEmitters());
    expect(rec.has('uniformThatDoesNotExist')).toBe(false);
  });
});

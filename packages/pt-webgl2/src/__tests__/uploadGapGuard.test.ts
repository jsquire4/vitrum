import { describe, expect, it, vi } from 'vitest';
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
  };
}
function sceneNoEmitters(): Scene {
  return { primitives: [tri('tri')], emitters: [], environment: { kind: 'none' } };
}
function sceneWithMeshAreaLight(): Scene {
  // A separate emissive panel mesh referenced by a mesh-area emitter (B4 NEE).
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
    primitives: [tri('tri'), panel],
    emitters: [{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 5 }],
    environment: { kind: 'none' },
  };
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
  const engine = await createPTEngine_WebGL2({ device: gl, ...engineOpts });
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

  it('B4: mesh-area NEE uniforms are uploaded (count + Σ area) for an emissive mesh', async () => {
    const rec = await renderAndRecord(sceneWithMeshAreaLight());
    expect(rec.has('uMeshLightCount')).toBe(true);
    // The emissive panel is 2 triangles → 2 triangle lights.
    expect(rec.get('uMeshLightCount')).toBe(2);
    expect(rec.has('uTotalEmissiveArea')).toBe(true);
    // Panel spans [-1,1]×[-1,1] (area 4) → two tris of total area 4.
    expect(rec.get('uTotalEmissiveArea')).toBeCloseTo(4, 5);
  });

  it('B4: mesh-area NEE uniforms are inert (count 0) when no mesh-area emitter', async () => {
    const rec = await renderAndRecord(sceneNoEmitters());
    expect(rec.has('uMeshLightCount')).toBe(true);
    expect(rec.get('uMeshLightCount')).toBe(0);
    expect(rec.get('uTotalEmissiveArea')).toBe(0);
  });

  it('D11: materialLodDepth is explicitly uploaded as 0 by default (texture LOD disabled)', async () => {
    const rec = await renderAndRecord(sceneNoEmitters());
    expect(rec.has('materialLodDepth')).toBe(true);
    expect(rec.get('materialLodDepth')).toBe(0);
  });

  it('D11: materialLodDepth option is uploaded when the host opts into texture LOD', async () => {
    const rec = await renderAndRecord(sceneNoEmitters(), { materialLodDepth: 2 });
    expect(rec.has('materialLodDepth')).toBe(true);
    expect(rec.get('materialLodDepth')).toBe(2);
  });

  it('A5: BDPT host-driver uniforms are uploaded when bdpt:true', async () => {
    const rec = await renderAndRecord(sceneWithMeshAreaLight(), { bdpt: true });
    // The eye pass sets the light-subpath pass flag to 0 and uploads the bounce count.
    expect(rec.has('uBdptLightSubpathPass')).toBe(true);
    expect(rec.has('uBdptMaxLightBounces')).toBe(true);
    expect(rec.get('uBdptMaxLightBounces')).toBe(3);
  });

  it('A5: BDPT uniforms are NOT touched when bdpt:false (byte-identical invariant)', async () => {
    const rec = await renderAndRecord(sceneWithMeshAreaLight(), { bdpt: false });
    expect(rec.has('uBdptLightSubpathPass')).toBe(false);
    expect(rec.has('uBdptMaxLightBounces')).toBe(false);
  });

  it('H2 follow-on: Cauchy IOR coefficients are uploaded (non-zero) when spectral:true', async () => {
    const rec = await renderAndRecord(sceneNoEmitters(), { spectral: true });
    expect(rec.has('iorCauchyA')).toBe(true);
    expect(rec.get('iorCauchyA')).toBeGreaterThan(1); // Crown Glass A ≈ 1.5046
    expect(rec.get('iorCauchyB')).toBeGreaterThan(0);
    // Non-spectral: no dispersion (the GLSL cauchyEnabled fast-path → byte-identical).
    const off = await renderAndRecord(sceneNoEmitters(), { spectral: false });
    expect(off.get('iorCauchyB')).toBe(0);
  });

  it('flag-plumbing: dof uniforms are uploaded when dof is set, absent otherwise', async () => {
    const withDof = await renderAndRecord(sceneNoEmitters(), {
      dof: { focusDistance: 5, bokehSize: 2 },
    });
    expect(withDof.has('physicalCamera.focusDistance')).toBe(true);
    expect(withDof.get('physicalCamera.focusDistance')).toBe(5);
    const noDof = await renderAndRecord(sceneNoEmitters());
    expect(noDof.has('physicalCamera.focusDistance')).toBe(false);
  });

  it('tonemap present-pass: uTonemapMode/uExposure/uOutputColorSpace are uploaded per frame', async () => {
    // Default quality (no quality field) → aces(0), exposure=1.0, srgb(0).
    const rec = await renderAndRecord(sceneNoEmitters());
    expect(rec.has('uTonemapMode')).toBe(true);
    expect(rec.get('uTonemapMode')).toBe(0); // aces
    expect(rec.has('uExposure')).toBe(true);
    expect(rec.get('uExposure')).toBeCloseTo(1.0, 6);
    expect(rec.has('uOutputColorSpace')).toBe(true);
    expect(rec.get('uOutputColorSpace')).toBe(0); // srgb
  });

  it('tonemap present-pass: quality.tonemap=agx → uTonemapMode=1', async () => {
    const gl = createMockGl(new Map());
    const rec = new Map<string, unknown>();
    const gl2 = createMockGl(rec);
    const engine = await createPTEngine_WebGL2({ device: gl2 });
    engine.setScene(sceneNoEmitters());
    const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
    const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
    engine.renderFrame({
      viewMatrix: view as never, projMatrix: proj as never,
      cameraPosition: [0, 0, 5] as never,
      viewport: { width: 32, height: 32, devicePixelRatio: 1 },
      frameIndex: 0, frameSeed: 0,
      quality: { samplesTarget: 4, tonemap: 'agx' },
    });
    expect(rec.get('uTonemapMode')).toBe(1); // agx
    void gl;
  });

  it('tonemap present-pass: quality.exposure=2.5 → uExposure=2.5', async () => {
    const rec = new Map<string, unknown>();
    const gl = createMockGl(rec);
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(sceneNoEmitters());
    const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
    const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
    engine.renderFrame({
      viewMatrix: view as never, projMatrix: proj as never,
      cameraPosition: [0, 0, 5] as never,
      viewport: { width: 32, height: 32, devicePixelRatio: 1 },
      frameIndex: 0, frameSeed: 0,
      quality: { samplesTarget: 4, exposure: 2.5 },
    });
    expect(rec.get('uExposure')).toBeCloseTo(2.5, 5);
  });

  it("tonemap present-pass: quality.outputColorSpace='linear' → uOutputColorSpace=1", async () => {
    const rec = new Map<string, unknown>();
    const gl = createMockGl(rec);
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(sceneNoEmitters());
    const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
    const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
    engine.renderFrame({
      viewMatrix: view as never, projMatrix: proj as never,
      cameraPosition: [0, 0, 5] as never,
      viewport: { width: 32, height: 32, devicePixelRatio: 1 },
      frameIndex: 0, frameSeed: 0,
      quality: { samplesTarget: 4, outputColorSpace: 'linear' },
    });
    expect(rec.get('uOutputColorSpace')).toBe(1); // linear: OETF skipped
  });

  it("tonemap present-pass: default outputColorSpace='srgb' → uOutputColorSpace=0", async () => {
    const rec = await renderAndRecord(sceneNoEmitters());
    // Default is 'srgb' — OETF applied, uOutputColorSpace must be 0.
    expect(rec.get('uOutputColorSpace')).toBe(0);
  });

  it('the recording mock actually distinguishes set-vs-unset (meta-check)', async () => {
    // A uniform the engine never sets must be ABSENT — proves the guard can FAIL
    // (i.e. the H1/H2/H3 asserts above are meaningful, not vacuously true).
    const rec = await renderAndRecord(sceneNoEmitters());
    expect(rec.has('uniformThatDoesNotExist')).toBe(false);
  });

  // ── Item 22 — DOF × equirectangular regime guard ────────────────────────────
  //
  // Thin-lens DOF applied to equirectangular projection is physically undefined:
  // the aperture offset in camera space has no consistent meaning for a full-sphere
  // projection.  The engine must (a) warn once and (b) force FEATURE_DOF=0 (i.e.
  // physicalCamera.focusDistance is NOT uploaded) even when dof is supplied.

  it('item22: equirect + dof emits a console.warn naming the regime mismatch', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await renderAndRecord(sceneNoEmitters(), {
        cameraType: 'equirectangular',
        dof: { focusDistance: 5, bokehSize: 2 },
      });
      const equirectDofWarns = warn.mock.calls.filter((a) =>
        String(a[0]).includes('equirectangular'),
      );
      expect(equirectDofWarns.length).toBeGreaterThan(0);
      expect(String(equirectDofWarns[0]![0])).toContain('dof');
      expect(String(equirectDofWarns[0]![0])).toContain('pt-webgl2');
    } finally {
      warn.mockRestore();
    }
  });

  it('item22: equirect + dof forces FEATURE_DOF=0 — physicalCamera.focusDistance is NOT uploaded', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const rec = await renderAndRecord(sceneNoEmitters(), {
        cameraType: 'equirectangular',
        dof: { focusDistance: 5, bokehSize: 2 },
      });
      // Even though dof was supplied, the engine must NOT upload the DOF uniform
      // (FEATURE_DOF must be 0 — same as if dof were absent).
      expect(rec.has('physicalCamera.focusDistance')).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('item22: orthographic + dof is accepted (FEATURE_DOF=1, no warn)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const rec = await renderAndRecord(sceneNoEmitters(), {
        cameraType: 'orthographic',
        dof: { focusDistance: 5, bokehSize: 2 },
      });
      // Orthographic DOF is physically coherent (tilt-shift model); no warn, DOF on.
      const equirectWarns = warn.mock.calls.filter((a) =>
        String(a[0]).includes('equirectangular'),
      );
      expect(equirectWarns).toHaveLength(0);
      expect(rec.has('physicalCamera.focusDistance')).toBe(true);
      expect(rec.get('physicalCamera.focusDistance')).toBe(5);
    } finally {
      warn.mockRestore();
    }
  });

  it('item22: perspective + dof is accepted (FEATURE_DOF=1, no warn)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const rec = await renderAndRecord(sceneNoEmitters(), {
        cameraType: 'perspective',
        dof: { focusDistance: 10, bokehSize: 1.5 },
      });
      const equirectWarns = warn.mock.calls.filter((a) =>
        String(a[0]).includes('equirectangular'),
      );
      expect(equirectWarns).toHaveLength(0);
      expect(rec.has('physicalCamera.focusDistance')).toBe(true);
      expect(rec.get('physicalCamera.focusDistance')).toBe(10);
    } finally {
      warn.mockRestore();
    }
  });

  it('H6: environmentRotation is identity for rotationY=0 (zero-rotation invariant)', async () => {
    // Zero-rotation invariant: when rotationY is absent (default), the uploaded
    // environmentRotation matrix must equal the identity — byte-identical behaviour
    // to the pre-H6 IDENTITY_MAT4 constant.
    const scene: Scene = {
      ...sceneNoEmitters(),
      environment: { kind: 'hdri', hdri: {}, intensity: 1 },
    };
    const rec = await renderAndRecord(scene);
    const mat = rec.get('environmentRotation');
    expect(mat).toBeDefined();
    expect(ArrayBuffer.isView(mat)).toBe(true);
    const f = new Float32Array((mat as Float32Array).buffer, (mat as Float32Array).byteOffset, 16);
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    for (let i = 0; i < 16; i++) {
      expect(f[i]).toBeCloseTo(identity[i]!, 10);
    }
  });

  it('H6: environmentRotation is non-identity for rotationY=π/2', async () => {
    // When rotationY = π/2, the matrix must be RY(−π/2) (not identity).
    // The packer uses makeRotationYMat4(-rotationY).
    // For RY(-π/2): cos(-π/2)=0, sin(-π/2)=-1.
    // Column-major: col0=[0,0,1,0], col1=[0,1,0,0], col2=[-1,0,0,0], col3=[0,0,0,1].
    // Check: m[0]=cos=0, m[2]=-sin=1, m[8]=sin=-1, m[10]=cos=0.
    const scene: Scene = {
      ...sceneNoEmitters(),
      environment: { kind: 'hdri', hdri: {}, intensity: 1, rotationY: Math.PI / 2 },
    };
    const rec = await renderAndRecord(scene);
    const mat = rec.get('environmentRotation');
    expect(mat).toBeDefined();
    expect(ArrayBuffer.isView(mat)).toBe(true);
    const f = new Float32Array((mat as Float32Array).buffer, (mat as Float32Array).byteOffset, 16);
    // Must differ from identity (i.e. the rotation was applied).
    expect(Math.abs(f[0]! - 1)).toBeGreaterThan(0.5); // cos(-π/2)=0, not 1
    // col0[0] = cos(-π/2) ≈ 0
    expect(f[0]).toBeCloseTo(0, 5);
    // col2[0] = sin(-π/2) ≈ -1  (index 8)
    expect(f[8]).toBeCloseTo(-1, 5);
  });
});

import { describe, expect, it } from 'vitest';
import {
  BACKEND_PROMISE_LEDGER,
  ENGINE_DENOISER_MODES,
  MATERIAL_SPEC_FIELDS,
  asBackendTexture,
  asBackendTextureFormat,
  narrowToBackendTexture,
  narrowToBackendTextureFormat,
} from '../index.js';
import { asMat4, isMat4 } from '../scene/math.js';

describe('backend promise ledger', () => {
  it('contains all expected backend IDs', () => {
    expect(Object.keys(BACKEND_PROMISE_LEDGER).sort()).toEqual([
      'pt-webgl2',
      'pt-webgpu',
      'walkaround-hybrid',
    ]);
  });

  it('keeps incremental patch support exhaustive for every backend', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(typeof rec.incrementalPatchSupport.transform).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.positions).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.material).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.emitter).toBe('boolean');
      expect(typeof rec.incrementalPatchSupport.topology).toBe('boolean');
    }
  });

  it('declares add/remove-primitive support exhaustively and consistently', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(typeof rec.supportsAddRemovePrimitive).toBe('boolean');
      expect(typeof rec.methodPromises.addPrimitive).toBe('boolean');
      expect(typeof rec.methodPromises.removePrimitive).toBe('boolean');
      // The capability flag and the two method promises must agree — a backend
      // that advertises add/remove support must expose both methods, and vice
      // versa (no half-implemented surface).
      expect(rec.methodPromises.addPrimitive).toBe(rec.supportsAddRemovePrimitive);
      expect(rec.methodPromises.removePrimitive).toBe(rec.supportsAddRemovePrimitive);
    }
  });

  it('pins the per-backend add/remove-primitive truth table', () => {
    expect(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportsAddRemovePrimitive).toBe(true);
    expect(BACKEND_PROMISE_LEDGER['pt-webgl2'].supportsAddRemovePrimitive).toBe(true);
    expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportsAddRemovePrimitive).toBe(true);
  });

  it('keeps supportDetails aligned with coarse supported-kind sets', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      const primitiveKinds = new Set<string>(rec.supportedPrimitiveKinds);
      for (const [kind, mode] of Object.entries(rec.supportDetails.primitives)) {
        expect(primitiveKinds.has(kind)).toBe(mode !== 'unsupported');
      }

      const emitterKinds = new Set<string>(rec.supportedEmitterKinds);
      for (const [kind, mode] of Object.entries(rec.supportDetails.emitters)) {
        expect(emitterKinds.has(kind)).toBe(mode !== 'unsupported');
      }

      const environmentKinds = new Set<string>(rec.supportedEnvironmentKinds);
      for (const [kind, mode] of Object.entries(rec.supportDetails.environments)) {
        expect(environmentKinds.has(kind)).toBe(mode !== 'unsupported');
      }

      const analyticShapes = new Set<string>(rec.supportedAnalyticShapes);
      for (const [shape, mode] of Object.entries(rec.supportDetails.analyticShapes)) {
        expect(analyticShapes.has(shape)).toBe(mode !== 'unsupported');
      }
    }
  });

  it('keeps supportDetails mutation rows aligned with promised optional methods', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(rec.supportDetails.mutations.emitter !== 'unsupported').toBe(
        rec.methodPromises.updateEmitter,
      );
      expect(rec.supportDetails.mutations.environment !== 'unsupported').toBe(
        rec.methodPromises.updateEnvironment,
      );
      expect(rec.supportDetails.mutations.addPrimitive !== 'unsupported').toBe(
        rec.methodPromises.addPrimitive,
      );
      expect(rec.supportDetails.mutations.removePrimitive !== 'unsupported').toBe(
        rec.methodPromises.removePrimitive,
      );
      expect(rec.supportDetails.mutations.resize !== 'unsupported').toBe(
        rec.methodPromises.setSize,
      );
      expect(rec.supportDetails.mutations.lighting !== 'unsupported').toBe(
        rec.methodPromises.updateLighting,
      );
    }
  });

  it('marks reserved displacement material fields unsupported on every shipping backend', () => {
    for (const rec of Object.values(BACKEND_PROMISE_LEDGER)) {
      expect(rec.supportDetails.materials.displacementMap).toBe('unsupported');
      expect(rec.supportDetails.materials.displacementScale).toBe('unsupported');
      expect(rec.supportDetails.materials.displacementBias).toBe('unsupported');
    }
  });

  // CAP-01 — the material support matrix is EXHAUSTIVE over MaterialSpec keys
  // for every shipping backend. MATERIAL_SPEC_FIELDS itself is compile-time
  // checked against `keyof MaterialSpec` in both directions (promiseLedger.ts),
  // so a new MaterialSpec field fails typecheck AND this runtime pin.
  it('keeps the per-field material support matrix exhaustive over MaterialSpec for every backend', () => {
    expect(MATERIAL_SPEC_FIELDS.length).toBeGreaterThanOrEqual(63);
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      const matrixKeys = Object.keys(rec.supportDetails.materials).sort();
      expect(matrixKeys, `materials matrix keys for ${id}`).toEqual(
        [...MATERIAL_SPEC_FIELDS].sort(),
      );
      for (const field of MATERIAL_SPEC_FIELDS) {
        const mode = rec.supportDetails.materials[field];
        expect(['native', 'approximate', 'unsupported'], `materials.${field} for ${id}`).toContain(
          mode,
        );
      }
    }
  });

  it('pins the spot-checked material matrix rows derived from code reads (CAP-01)', () => {
    const wa = BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.materials;
    const gl2 = BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.materials;
    const gpu = BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials;

    // walkaround: quantized scalar model plus first baseColorMap atlas slice;
    // other image maps and Disney lobes remain unsupported.
    expect(wa.baseColor).toBe('approximate');
    expect(wa.emissive).toBe('native');
    expect(wa.shadingModel).toBe('approximate');
    expect(wa.baseColorMap).toBe('approximate');
    expect(wa.sheen).toBe('unsupported');
    expect(wa.extensions).toBe('native');

    // pt-webgl2: scalar anisotropy plus KHR_materials_anisotropy map are packed/sampled.
    expect(gl2.anisotropy).toBe('native');
    expect(gl2.anisotropyRotation).toBe('native');
    expect(gl2.anisotropyMap).toBe('native');
    expect(gl2.scatteringCoefficientRGB).toBe('approximate');
    expect(gl2.shadingModel).toBe('approximate');
    expect(gl2.thickness).toBe('approximate');
    expect(gl2.thicknessMap).toBe('approximate');
    expect(gl2.thinFilmStack).toBe('native');

    // pt-webgpu: KHR_materials_specular scalars are consumed by the ordinary PT
    // BRDF/PDF path; extension maps are full-tier-megakernel approximate until
    // specialized reuse/source-lobe sampler paths carry the same modulation.
    // anisotropy trio IS native; scatteringCoefficientRGB is genuine σ_s.
    expect(gpu.specularIntensity).toBe('approximate');
    expect(gpu.specularColor).toBe('approximate');
    expect(gpu.shadingModel).toBe('approximate');
    expect(gpu.clearcoatMap).toBe('approximate');
    expect(gpu.sheenColorMap).toBe('approximate');
    expect(gpu.clearcoatNormalMap).toBe('approximate');
    expect(gpu.clearcoatNormalScale).toBe('native');
    expect(gpu.thickness).toBe('unsupported');
    expect(gpu.transmissionMap).toBe('native');
    expect(gpu.alphaMap).toBe('native');
    expect(gpu.emissiveMap).toBe('native');
    expect(gpu.aoMap).toBe('native');
    expect(gpu.anisotropyMap).toBe('native');
    expect(gpu.bumpMap).toBe('native');
    expect(gpu.lightMap).toBe('native');
    expect(gpu.normalMap).toBe('native');
    expect(gpu.normalScale).toBe('native');
    expect(gpu.anisotropy).toBe('native');
    expect(gpu.scatteringCoefficientRGB).toBe('native');
    expect(gpu.roughnessMap).toBe('native');
    expect(gpu.metallicMap).toBe('native');
  });

  // SHADOW-01 — shadow-flag support rows are exhaustive + pinned per backend.
  it('keeps the shadow support matrix exhaustive and pinned for every backend (SHADOW-01)', () => {
    const SHADOW_KEYS = ['primitiveCastShadow', 'emitterCastShadow', 'receiveShadow'];
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      const keys = Object.keys(rec.supportDetails.shadows).sort();
      expect(keys, `shadows matrix keys for ${id}`).toEqual([...SHADOW_KEYS].sort());
      // receiveShadow is non-physical for GI — unsupported EVERYWHERE; backends
      // emit a structured *.reserved-receive-shadow warning when set false.
      expect(rec.supportDetails.shadows.receiveShadow, `receiveShadow for ${id}`).toBe(
        'unsupported',
      );
    }
    const wa = BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.shadows;
    const gl2 = BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.shadows;
    const gpu = BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.shadows;
    // walkaround: primitive flag honored by DI, ReSTIR-GI, DDGI, GRIS, and RC
    // shadow visibility; emitter flag is honored by direct, area-emitter,
    // DDGI fixture/sun, RC fixture/sun, and direct-sun paths.
    expect(wa.primitiveCastShadow).toBe('native');
    expect(wa.emitterCastShadow).toBe('native');
    // pt-webgl2: material-lane shadow-ray gate (native); emitter flag honored
    // for analytic NEE lights only (mesh-area + forward/BDPT do not consume it).
    expect(gl2.primitiveCastShadow).toBe('native');
    expect(gl2.emitterCastShadow).toBe('approximate');
    // pt-webgpu: every any-hit occlusion traversal skips the flag (native);
    // emitter flag honored by default-kernel NEE + BSDF-MIS connections only.
    expect(gpu.primitiveCastShadow).toBe('native');
    expect(gpu.emitterCastShadow).toBe('approximate');
  });

  it('keeps denoiser support rows exhaustive and host-readable', () => {
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(
        Object.keys(rec.supportDetails.denoisers).sort(),
        `denoisers matrix keys for ${id}`,
      ).toEqual([...ENGINE_DENOISER_MODES].sort());
      for (const mode of ENGINE_DENOISER_MODES) {
        expect(['native', 'approximate', 'unsupported'], `denoisers.${mode} for ${id}`).toContain(
          rec.supportDetails.denoisers[mode],
        );
      }
    }

    expect(BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.denoisers).toEqual({
      none: 'native',
      atrous: 'unsupported',
      'atrous-variance': 'unsupported',
      'svgf-real': 'unsupported',
      bmfr: 'unsupported',
      'oidn-final': 'unsupported',
      neural: 'unsupported',
    });
    expect(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.denoisers['oidn-final']).toBe(
      'native',
    );
    expect(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.denoisers['svgf-real']).toBe(
      'unsupported',
    );
    expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.denoisers['svgf-real']).toBe(
      'native',
    );
    expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.denoisers.bmfr).toBe(
      'native',
    );
  });

  it('pins onError: true for all three shipping backends (item 28 — GPU error surface)', () => {
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(rec.methodPromises.onError).toBe(true);
      // This is also an exhaustive contract check so new backends are forced
      // to decide rather than inherit a silent false default.
      expect(typeof rec.methodPromises.onError).toBe('boolean');
      void id;
    }
  });

  it('pins onWarning: true for all three shipping backends (ENGINE-01 warning surface)', () => {
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(rec.methodPromises.onWarning).toBe(true);
      expect(typeof rec.methodPromises.onWarning).toBe('boolean');
      void id;
    }
  });

  it('pins captureFrame: true for all three shipping backends (item 31 — pixel readback)', () => {
    for (const [id, rec] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(rec.methodPromises.captureFrame).toBe(true);
      expect(typeof rec.methodPromises.captureFrame).toBe('boolean');
      void id;
    }
  });

  it('pins walkaround mutation fidelity rows that differ from the boolean patch surface', () => {
    const rec = BACKEND_PROMISE_LEDGER['walkaround-hybrid'];

    expect(rec.incrementalPatchSupport).toEqual({
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    });
    expect(rec.supportDetails.mutations).toEqual({
      transform: 'native',
      positions: 'native',
      material: 'native',
      emitter: 'native',
      topology: 'fallback-rebuild',
      addPrimitive: 'fallback-rebuild',
      removePrimitive: 'fallback-rebuild',
      environment: 'approximate',
      resize: 'native',
      lighting: 'native',
    });
    expect(rec.methodPromises.setSize).toBe(true);
    expect(rec.methodPromises.updateLighting).toBe(true);
  });

  it('pins PT backend resize/lighting method promises truthfully', () => {
    const webgl2 = BACKEND_PROMISE_LEDGER['pt-webgl2'];
    expect(webgl2.presentationMode).toBe('offscreen-texture');
    expect(webgl2.frameInputPromises).toEqual({
      honorsViewportPerFrame: true,
      requiresSwapChainView: false,
      honorsPerFrameBounces: true,
    });
    expect(webgl2.supportDetails.mutations.resize).toBe('native');
    expect(webgl2.supportDetails.mutations.lighting).toBe('unsupported');
    expect(webgl2.methodPromises.setSize).toBe(true);
    expect(webgl2.methodPromises.updateLighting).toBe(false);

    const webgpu = BACKEND_PROMISE_LEDGER['pt-webgpu'];
    expect(webgpu.presentationMode).toBe('offscreen-texture');
    expect(webgpu.frameInputPromises).toEqual({
      honorsViewportPerFrame: true,
      requiresSwapChainView: false,
      honorsPerFrameBounces: true,
    });
    expect(webgpu.supportDetails.mutations.resize).toBe('unsupported');
    expect(webgpu.supportDetails.mutations.lighting).toBe('unsupported');
    expect(webgpu.methodPromises.setSize).toBe(false);
    expect(webgpu.methodPromises.updateLighting).toBe(false);
  });

  it('pins environment fidelity rows for DDGI SH walkaround versus path tracers', () => {
    expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.environments).toEqual({
      none: 'native',
      // Wave 4 (2026-06-10): promoted approximate→native — env DI NEE candidate
      // in RIS, HDRI in DDGI probe misses + risGiNrc + RC, runtime CDF rebuild
      // on updateEnvironment. Radiometric A/B pending V28-B.
      hdri: 'native',
      // procedural-sky degrades to scalar tint via resolveHybridEnvironment
      // (mode: 'procedural-sky-approx'; turbidity/rayleigh/mie ignored; warn emitted).
      // 'approximate' reflects the degraded-but-functional reality. Item 18c.
      'procedural-sky': 'approximate',
    });
    expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportedEnvironmentKinds).toEqual([
      'none',
      'hdri',
      'procedural-sky',
    ]);

    expect(BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.environments).toEqual({
      none: 'native',
      hdri: 'native',
      'procedural-sky': 'approximate',
    });
    expect(BACKEND_PROMISE_LEDGER['pt-webgl2'].supportedEnvironmentKinds).toEqual([
      'none',
      'hdri',
      'procedural-sky',
    ]);

    // procedural-sky is 'approximate' on PT backends: both bake the Preetham
    // model to a finite equirect map and route it through the HDRI sampling path.
    expect(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.environments).toEqual({
      none: 'native',
      hdri: 'native',
      'procedural-sky': 'approximate',
    });
    expect(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportedEnvironmentKinds).toEqual([
      'none',
      'hdri',
      'procedural-sky',
    ]);
  });
});

describe('mat4 branding', () => {
  it('accepts exactly 16-element matrices', () => {
    const m = asMat4(new Float32Array(16));
    expect(isMat4(m)).toBe(true);
  });

  it('rejects non-16-element arrays', () => {
    expect(() => asMat4(new Float32Array(9))).toThrow(/Mat4 requires 16 elements/);
  });
});

describe('backend texture branding', () => {
  it('round-trips branded backend texture handles through narrow helpers', () => {
    const h = { texture: true as const };
    const branded = asBackendTexture<'webgpu', { texture: true }>(h);
    expect(narrowToBackendTexture<'webgpu', { texture: true }>(branded)).toBe(h);
  });

  it('round-trips branded backend texture formats through narrow helpers', () => {
    const f = 'rgba16float';
    const branded = asBackendTextureFormat<'webgpu', string>(f);
    expect(narrowToBackendTextureFormat<'webgpu', string>(branded)).toBe(f);
  });
});

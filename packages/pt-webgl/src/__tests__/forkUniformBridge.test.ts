import { describe, expect, it } from 'vitest';
import { BoxGeometry, Mesh, MeshPhysicalMaterial, Scene } from 'three';
import { driveForkMaterialUniforms } from '../forkUniformBridge.js';
import { ForkAccess, makeForkPathTracerStubForTests } from '../forkAccess.js';

/** Build a path-tracer stub with a fresh uniforms map. Construction goes
 *  through {@link makeForkPathTracerStubForTests} so individual test files
 *  never bind to the fork-private navigation shape directly — that lives
 *  inside `forkAccess.ts`. */
function makeStubPathTracer() {
  return makeForkPathTracerStubForTests({
    material: {
      uniforms: {
        u_volumeDensity: { value: 123 },
        uCausticStrategy: { value: -1 },
        uMneeMaxIterations: { value: 0 },
        uMneeMaxChainLength: { value: 0 },
        uCmfX: { value: null as unknown },
        uCmfY: { value: null as unknown },
        uCmfZ: { value: null as unknown },
        uYCmfCdf: { value: null as unknown },
        uYCmfIntegral: { value: 0 },
        uSpectralRendering: { value: -1 },
        uRadianceClamp: { value: -1 },
        // Sprint 10c — BDPT uniforms
        uBdptEnabled: { value: false as unknown },
        uBdptMaxLightBounces: { value: 0 },
        uBdptLightPathTex: { value: null as unknown },
      },
    },
  });
}

/** Read-back helper: resolves uniforms via the same accessor production
 *  code uses. Asserts non-null so tests fail loudly rather than chasing
 *  optional-chain undefineds when the stub shape drifts. */
function uniformsOf(pathTracer: unknown): Record<string, { value: unknown }> {
  const material = ForkAccess.getMaterial(pathTracer);
  if (material == null || material.uniforms == null) {
    throw new Error('test bug: stub path tracer has no material/uniforms');
  }
  return material.uniforms;
}

describe('driveForkMaterialUniforms', () => {
  it('uploads CMF and CDF tables when scene is set', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0x99ccff, ior: 1.58 });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'manifold-nee',
      mneeMaxIterations: 12,
      mneeMaxChainLength: 4,
    });
    const uniforms = uniformsOf(pathTracer);

    expect(uniforms.uCmfX!.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uCmfY!.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uCmfZ!.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfCdf!.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfIntegral!.value).toBeCloseTo(106.857);
    expect(uniforms.uSpectralRendering!.value).toBe(0);
    expect(uniforms.uRadianceClamp!.value).toBe(0);
    expect(uniforms.uCausticStrategy!.value).toBe(1);
    expect(uniforms.uMneeMaxIterations!.value).toBe(12);
    expect(uniforms.uMneeMaxChainLength!.value).toBe(4);
  });

  it('does not override per-material scalar uniforms', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0xffffff });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer);
    const uniforms = uniformsOf(pathTracer);

    expect(uniforms.u_volumeDensity!.value).toBe(123);
    expect(uniforms.uYCmfIntegral!.value).toBeCloseTo(106.857);
  });

  it('maps strategy "none" to zero code', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0xffffff });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
    });
    const uniforms = uniformsOf(pathTracer);

    expect(uniforms.uCausticStrategy!.value).toBe(0);
    expect(uniforms.uMneeMaxIterations!.value).toBe(6);
    expect(uniforms.uMneeMaxChainLength!.value).toBe(2);
  });

  it('enables experimental hero-wavelength reconstruction when requested', () => {
    const scene = new Scene();
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), new MeshPhysicalMaterial({ color: 0xffffff })));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
      spectralRendering: true,
      radianceClamp: 8,
    });

    const u = uniformsOf(pathTracer);
    expect(u.uSpectralRendering!.value).toBe(1);
    expect(u.uRadianceClamp!.value).toBe(8);
  });

  // ── Sprint 10c — BDPT option flow-through tests ─────────────────────────

  it('Sprint 10c: BDPT disabled by default when no bdptOptions provided', () => {
    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
    });
    const uniforms = uniformsOf(pathTracer);
    // When bdptOptions is not supplied, uBdptEnabled must be false.
    expect(uniforms.uBdptEnabled!.value).toBe(false);
    // The lightPathTex uniform is not touched by the bridge (null = no bind).
    // We only assert uBdptEnabled = false as the structural "off" guarantee.
  });

  it('Sprint 10c: BDPT enabled=true + valid lightPathTex threads through to uniforms', () => {
    const pathTracer = makeStubPathTracer();
    // Simulate a valid WebGL texture object (opaque; test only checks reference equality).
    const fakeTexture = { isTex: true };
    driveForkMaterialUniforms(
      pathTracer,
      { strategy: 'none', mneeMaxIterations: 6, mneeMaxChainLength: 2 },
      { enabled: true, maxLightBounces: 2, lightPathTex: fakeTexture },
    );
    const uniforms = uniformsOf(pathTracer);
    expect(uniforms.uBdptEnabled!.value).toBe(true);
    expect(uniforms.uBdptMaxLightBounces!.value).toBe(2);
    expect(uniforms.uBdptLightPathTex!.value).toBe(fakeTexture);
  });

  it('Sprint 10c: BDPT enabled=true + null lightPathTex forces enabled=false (safety guard)', () => {
    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(
      pathTracer,
      { strategy: 'none', mneeMaxIterations: 6, mneeMaxChainLength: 2 },
      { enabled: true, maxLightBounces: 3, lightPathTex: null },
    );
    const uniforms = uniformsOf(pathTracer);
    // Safety guard: null texture → force disabled to prevent sampling unbound slot.
    expect(uniforms.uBdptEnabled!.value).toBe(false);
  });

  it('handles mixed-material scenes without clobbering scalar controls', () => {
    const scene = new Scene();
    const base = new MeshPhysicalMaterial({ color: 0xffffff, transmission: 1, ior: 1.52 });
    base.userData['vitrumScatteringCoefficient'] = 0.2;
    base.userData['vitrumScatteringAnisotropy'] = 0.4;
    base.userData['vitrumDispersionAbbeNumber'] = 32;
    base.userData['vitrumFrontLayer'] = { transmission: [0.8, 0.85, 0.9], roughness: 0.25 };
    base.userData['vitrumBackLayer'] = { transmission: [0.95, 0.95, 0.95], roughness: 0.1 };
    base.userData['vitrumSpectralAttenuation'] = {
      samples: [
        [380, 0.2],
        [550, 0.5],
        [780, 0.8],
      ],
    };
    base.userData['vitrumThinFilmStack'] = {
      layers: [
        { ior: 1.33, thicknessNm: 120 },
        { ior: 1.5, thicknessNm: 250 },
      ],
    };
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), base));

    const second = new MeshPhysicalMaterial({ color: 0xccddff, roughness: 0.4, metalness: 0.1 });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), second));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'photon-map',
      mneeMaxIterations: 10,
      mneeMaxChainLength: 5,
    });
    const uniforms = uniformsOf(pathTracer);

    expect(uniforms.u_volumeDensity!.value).toBe(123);
    expect(uniforms.uCausticStrategy!.value).toBe(2);
    expect(uniforms.uMneeMaxIterations!.value).toBe(10);
    expect(uniforms.uMneeMaxChainLength!.value).toBe(5);
    expect(uniforms.uCmfX!.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfIntegral!.value).toBeCloseTo(106.857);
  });
});

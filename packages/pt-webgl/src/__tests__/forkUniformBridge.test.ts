import { describe, expect, it } from 'vitest';
import { BoxGeometry, Mesh, MeshPhysicalMaterial, Scene, Vector3 } from 'three';
import { driveForkMaterialUniforms } from '../legacy/three/forkUniformBridge.js';
import { rgbToSpectralCoefficients, spectralCoefficientsToRGB } from '@vitrum/shared-samplers';

function makeStubPathTracer() {
  return {
    _pathTracer: {
      material: {
        uniforms: {
          u_volumeDensity: { value: 123 },
          uCausticStrategy: { value: -1 },
          uMneeMaxIterations: { value: 0 },
          uMneeMaxChainLength: { value: 0 },
          uCmfX: { value: null as unknown },
          uCmfY: { value: null as unknown },
          uCmfZ: { value: null as unknown },
          uXCmfCdf: { value: null as unknown },
          uYCmfCdf: { value: null as unknown },
          uZCmfCdf: { value: null as unknown },
          uXCmfIntegral: { value: 0 },
          uYCmfIntegral: { value: 0 },
          uZCmfIntegral: { value: 0 },
          uSpectralRendering: { value: -1 },
          uRadianceClamp: { value: -1 },
          // Real Jakob & Hanika RGB→spectrum coefficient uniform (THREE.Vector3).
          u_jakobCoeffs: { value: new Vector3(0, 0, 0) },
          // Sprint 10c — BDPT uniforms
          uBdptEnabled: { value: false as unknown },
          uBdptMaxLightBounces: { value: 0 },
          uBdptLightPathTex: { value: null as unknown },
        },
        defines: { FEATURE_BDPT: 0 as number },
        setDefine( name: string, value: number ) {

          ( this.defines as Record<string, number> )[ name ] = value;

        },
      },
    },
  };
}

function makeCountingUniform<T>(initialValue: T) {
  let current = initialValue;
  let writes = 0;
  return {
    get value(): T {
      return current;
    },
    set value(next: T) {
      writes += 1;
      current = next;
    },
    get writes(): number {
      return writes;
    },
  };
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
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.uCmfX.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uCmfY.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uCmfZ.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfCdf.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
    expect(uniforms.uSpectralRendering.value).toBe(0);
    expect(uniforms.uRadianceClamp.value).toBe(0);
    expect(uniforms.uCausticStrategy.value).toBe(1);
    expect(uniforms.uMneeMaxIterations.value).toBe(12);
    expect(uniforms.uMneeMaxChainLength.value).toBe(4);
  });

  it('uploads static spectral tables once per fork material while dynamic uniforms keep updating', () => {
    const pathTracer = makeStubPathTracer();
    const uniforms = pathTracer._pathTracer.material.uniforms;
    const cmfX = makeCountingUniform<unknown>(null);
    const cmfY = makeCountingUniform<unknown>(null);
    const cmfZ = makeCountingUniform<unknown>(null);
    const xCdf = makeCountingUniform<unknown>(null);
    const yCdf = makeCountingUniform<unknown>(null);
    const zCdf = makeCountingUniform<unknown>(null);
    const xIntegral = makeCountingUniform(0);
    const yIntegral = makeCountingUniform(0);
    const zIntegral = makeCountingUniform(0);
    const spectralRendering = makeCountingUniform(-1);
    uniforms.uCmfX = cmfX;
    uniforms.uCmfY = cmfY;
    uniforms.uCmfZ = cmfZ;
    uniforms.uXCmfCdf = xCdf;
    uniforms.uYCmfCdf = yCdf;
    uniforms.uZCmfCdf = zCdf;
    uniforms.uXCmfIntegral = xIntegral;
    uniforms.uYCmfIntegral = yIntegral;
    uniforms.uZCmfIntegral = zIntegral;
    uniforms.uSpectralRendering = spectralRendering;

    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
      spectralRendering: false,
    });
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
      spectralRendering: true,
    });

    expect(cmfX.writes).toBe(1);
    expect(cmfY.writes).toBe(1);
    expect(cmfZ.writes).toBe(1);
    expect(xCdf.writes).toBe(1);
    expect(yCdf.writes).toBe(1);
    expect(zCdf.writes).toBe(1);
    expect(xIntegral.writes).toBe(1);
    expect(yIntegral.writes).toBe(1);
    expect(zIntegral.writes).toBe(1);
    expect(spectralRendering.writes).toBe(2);
    expect(spectralRendering.value).toBe(1);
  });

  it('does not override per-material scalar uniforms', () => {
    const scene = new Scene();
    const mat = new MeshPhysicalMaterial({ color: 0xffffff });
    scene.add(new Mesh(new BoxGeometry(1, 1, 1), mat));

    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer);
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.u_volumeDensity.value).toBe(123);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
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
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.uCausticStrategy.value).toBe(0);
    expect(uniforms.uMneeMaxIterations.value).toBe(6);
    expect(uniforms.uMneeMaxChainLength.value).toBe(2);
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

    expect(pathTracer._pathTracer.material.uniforms.uSpectralRendering.value).toBe(1);
    expect(pathTracer._pathTracer.material.uniforms.uRadianceClamp.value).toBe(8);
  });

  // ── Real Jakob & Hanika RGB→spectrum coefficient upload ──────────────────

  it('uploads real Jakob–Hanika coefficients to u_jakobCoeffs when spectralAlbedo + spectralRendering are set', () => {
    const pathTracer = makeStubPathTracer();
    const albedo: [number, number, number] = [0.7, 0.2, 0.45];
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
      spectralRendering: true,
      spectralAlbedo: albedo,
    });

    const coeffs = pathTracer._pathTracer.material.uniforms.u_jakobCoeffs.value as Vector3;
    const [c0, c1, c2] = rgbToSpectralCoefficients(...albedo);
    // The bridge must upload exactly the genuine solver output.
    expect(coeffs.x).toBe(c0);
    expect(coeffs.y).toBe(c1);
    expect(coeffs.z).toBe(c2);
    // And those coefficients must round-trip back to the source albedo — i.e.
    // the uploaded uniform really is the paper-accurate upsampling, not a flat
    // placeholder.
    const [rr, gg, bb] = spectralCoefficientsToRGB([coeffs.x, coeffs.y, coeffs.z]);
    expect(rr).toBeCloseTo(albedo[0], 2);
    expect(gg).toBeCloseTo(albedo[1], 2);
    expect(bb).toBeCloseTo(albedo[2], 2);
  });

  it('leaves u_jakobCoeffs at its flat (0,0,0) default when no spectralAlbedo is provided', () => {
    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
      spectralRendering: true,
    });
    const coeffs = pathTracer._pathTracer.material.uniforms.u_jakobCoeffs.value as Vector3;
    expect(coeffs.x).toBe(0);
    expect(coeffs.y).toBe(0);
    expect(coeffs.z).toBe(0);
  });

  it('does not upload coefficients when spectralRendering is off, even if spectralAlbedo is provided', () => {
    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
      spectralRendering: false,
      spectralAlbedo: [0.9, 0.1, 0.3],
    });
    const coeffs = pathTracer._pathTracer.material.uniforms.u_jakobCoeffs.value as Vector3;
    expect(coeffs.x).toBe(0);
    expect(coeffs.y).toBe(0);
    expect(coeffs.z).toBe(0);
  });

  // ── Sprint 10c — BDPT option flow-through tests ─────────────────────────

  it('Sprint 10c: BDPT disabled by default when no bdptOptions provided', () => {
    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(pathTracer, {
      strategy: 'none',
      mneeMaxIterations: 6,
      mneeMaxChainLength: 2,
    });
    const uniforms = pathTracer._pathTracer.material.uniforms;
    // When bdptOptions is not supplied, uBdptEnabled must be false.
    expect(uniforms.uBdptEnabled.value).toBe(false);
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
    const uniforms = pathTracer._pathTracer.material.uniforms;
    expect(uniforms.uBdptEnabled.value).toBe(true);
    expect(uniforms.uBdptMaxLightBounces.value).toBe(2);
    expect(uniforms.uBdptLightPathTex.value).toBe(fakeTexture);
  });

  it('Sprint 10c: BDPT enabled=true + null lightPathTex forces enabled=false (safety guard)', () => {
    const pathTracer = makeStubPathTracer();
    driveForkMaterialUniforms(
      pathTracer,
      { strategy: 'none', mneeMaxIterations: 6, mneeMaxChainLength: 2 },
      { enabled: true, maxLightBounces: 3, lightPathTex: null },
    );
    const uniforms = pathTracer._pathTracer.material.uniforms;
    const material = pathTracer._pathTracer.material;
    // Safety guard: null texture → eye-path connections off; shader still compiles BDPT.
    expect(uniforms.uBdptEnabled.value).toBe(false);
    expect(uniforms.uBdptLightPathTex.value).toBeNull();
    expect(material.defines.FEATURE_BDPT).toBe(1);
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
    const uniforms = pathTracer._pathTracer.material.uniforms;

    expect(uniforms.u_volumeDensity.value).toBe(123);
    expect(uniforms.uCausticStrategy.value).toBe(2);
    expect(uniforms.uMneeMaxIterations.value).toBe(10);
    expect(uniforms.uMneeMaxChainLength.value).toBe(5);
    expect(uniforms.uCmfX.value).toBeInstanceOf(Float32Array);
    expect(uniforms.uYCmfIntegral.value).toBeCloseTo(106.857);
  });
});

import { describe, expect, it } from 'vitest';
import {
  asMat4,
  type AnalyticPrimitive,
  type MaterialSpec,
  type MeshPrimitive,
  type Scene,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';
import {
  analyzeOpticalMediumTopology,
  assertOpticalMediumTopology,
  lowerTransmissiveAnalyticPrimitives,
  materialDefinesBulkOpticalMedium,
  OpticalMediumTopologyError,
} from '../opticalMediumTopology.js';
import {
  analyzeShaderF32LinearOrientation,
  applyMatrix4MergedWorldF32,
  applyMatrix4ShaderF32,
} from '../worldTransforms.js';

const BULK: MaterialSpec = {
  baseColor: [1, 1, 1],
  roughness: 0,
  metallic: 0,
  transmission: 1,
  thickness: 1,
};

const CUBE_POSITIONS = new Float32Array([
  -1, -1, -1,
   1, -1, -1,
   1,  1, -1,
  -1,  1, -1,
  -1, -1,  1,
   1, -1,  1,
   1,  1,  1,
  -1,  1,  1,
]);

const CUBE_INDICES = new Uint32Array([
  0, 2, 1, 0, 3, 2,
  4, 5, 6, 4, 6, 7,
  0, 1, 5, 0, 5, 4,
  3, 7, 6, 3, 6, 2,
  0, 4, 7, 0, 7, 3,
  1, 2, 6, 1, 6, 5,
]);

const CUBE_NORMALS = new Float32Array(CUBE_POSITIONS.length).fill(1);

function matrix(scale: number, x = 0, y = 0, z = 0) {
  return asMat4(new Float32Array([
    scale, 0, 0, 0,
    0, scale, 0, 0,
    0, 0, scale, 0,
    x, y, z, 1,
  ]));
}

function affine(
  sx: number,
  sy: number,
  sz: number,
  angle = 0,
  x = 0,
  y = 0,
  z = 0,
) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return asMat4(new Float32Array([
    c * sx, s * sx, 0, 0,
    -s * sy, c * sy, 0, 0,
    0, 0, sz, 0,
    x, y, z, 1,
  ]));
}

function previousPositiveF32(value: number): number {
  const floats = new Float32Array([value]);
  const bits = new Uint32Array(floats.buffer);
  bits[0] = bits[0]! - 1;
  return floats[0]!;
}

function cube(id: string, scale = 1, x = 0, material = BULK): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: CUBE_POSITIONS,
    normals: CUBE_NORMALS,
    indices: CUBE_INDICES,
    material,
    transform: matrix(scale, x),
  };
}

function scene(primitives: Scene['primitives']): Scene {
  return { primitives, emitters: [], environment: { kind: 'none' } };
}

describe('bulk optical material activation', () => {
  it('matches thickness, effective scattering override, absorption, and spectral production lanes', () => {
    const material = (patch: Partial<MaterialSpec>): MaterialSpec => ({
      ...BULK,
      thickness: 0,
      ...patch,
    });
    expect(materialDefinesBulkOpticalMedium(material({ thickness: 0 }))).toBe(false);
    expect(materialDefinesBulkOpticalMedium(material({ thickness: 0.1 }))).toBe(true);
    expect(materialDefinesBulkOpticalMedium(material({ scatteringCoefficient: 0.2 }))).toBe(true);
    expect(materialDefinesBulkOpticalMedium(material({ scatteringCoefficientRGB: [0, 0.2, 0] }))).toBe(true);
    expect(materialDefinesBulkOpticalMedium(material({
      attenuationColor: [0.7, 0.8, 0.9],
      attenuationDistance: 2,
    }))).toBe(true);
    expect(materialDefinesBulkOpticalMedium(material({
      attenuationColor: [1, 1, 1],
      attenuationDistance: 2,
    }))).toBe(false);
    expect(materialDefinesBulkOpticalMedium(material({
      attenuationColor: [0.2, 0.3, 0.4],
      attenuationDistance: Infinity,
    }))).toBe(false);
    expect(materialDefinesBulkOpticalMedium(material({
      attenuationColor: [0.2, 0.3, 0.4],
    }))).toBe(false);
    expect(materialDefinesBulkOpticalMedium(material({
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0, 0.1, 0]),
      },
    }))).toBe(true);
    expect(materialDefinesBulkOpticalMedium(material({
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0, 0, 0]),
      },
    }))).toBe(true);
    expect(materialDefinesBulkOpticalMedium(material({
      thicknessMap: { handle: { id: 'thickness-only' } },
    }))).toBe(false);
    expect(materialDefinesBulkOpticalMedium({ ...material({ thickness: 1 }), transmission: 0 })).toBe(false);
  });

  it('honors RGB-scattering override semantics over the scalar lane', () => {
    expect(materialDefinesBulkOpticalMedium({
      ...BULK,
      thickness: 0,
      scatteringCoefficient: 1,
      scatteringCoefficientRGB: [0, 0, 0],
    })).toBe(false);
  });
});

describe('optical topology transform arithmetic', () => {
  const cancellationMatrix = asMat4(new Float32Array([
    1.2345670461654663, 0, 0, 0,
    -0.9876539707183838, 1, 0, 0,
    0.33333298563957214, 0, 1, 0,
    43731892, 0, 0, 1,
  ]));

  it('pins the cancellation case where TLAS f32 and merged-world packing differ', () => {
    const shader = applyMatrix4ShaderF32(
      cancellationMatrix,
      33856616,
      -23691874,
      19167270,
    );
    const merged = applyMatrix4MergedWorldF32(
      cancellationMatrix,
      33856616,
      -23691874,
      19167270,
    );
    expect(shader.point).toEqual([115318616, -23691876, 19167272]);
    expect(merged.point).toEqual([115318608, -23691874, 19167270]);
    expect(shader.uncertainty).toBeGreaterThan(0);
    expect(merged.uncertainty).toBe(0);
  });

  it('validates the exact represented stream selected by the backend', () => {
    const positions = new Float32Array([
      33856616, -23691874, 19167270,
      33856620, -23691874, 19167270,
      33856620, -23691872, 19167270,
      33856616, -23691872, 19167270,
      33856616, -23691874, 19167272,
      33856620, -23691874, 19167272,
      33856620, -23691872, 19167272,
      33856616, -23691872, 19167272,
    ]);
    const primitive: MeshPrimitive = {
      kind: 'mesh',
      id: 'cancellation-cube',
      positions,
      normals: new Float32Array(positions.length).fill(1),
      indices: CUBE_INDICES,
      material: BULK,
      transform: cancellationMatrix,
    };

    expect(analyzeOpticalMediumTopology(scene([primitive]), {
      transformArithmetic: 'merged-world-f64-to-f32',
    }).componentCount).toBe(1);
    expect(() => analyzeOpticalMediumTopology(scene([primitive]), {
      transformArithmetic: 'tlas-shader-f32',
    })).toThrow(/degenerate after position welding/);
  });

  it('rejects unknown transform-arithmetic modes', () => {
    expect(() => analyzeOpticalMediumTopology(scene([]), {
      transformArithmetic: 'f64' as never,
    })).toThrow(/transformArithmetic/);
  });

  it('rejects a core-valid near-singular transform when live TLAS orientation orders disagree', () => {
    const transform = asMat4(new Float32Array([
      -0.9997833371162415, -0.181410014629364, 0.8804394006729126, 0,
      -0.790558934211731, 0.6959825158119202, -0.493493914604187, 0,
      -1.7903428077697754, 0.5145728588104248, 0.38694533705711365, 0,
      0, 0, 0, 1,
    ]));
    const orientation = analyzeShaderF32LinearOrientation(transform);
    expect(orientation.sharedDeterminant).toBe(0);
    expect(orientation.ptDeterminant).toBe(-5.960464477539063e-8);
    expect(orientation.reliable).toBe(false);

    expect(() => analyzeOpticalMediumTopology(scene([{
      ...cube('ill-conditioned'),
      transform,
    }]), {
      transformArithmetic: 'tlas-shader-f32',
    })).toThrow(/ill-conditioned transform.*orientation is ambiguous/);
  });
});

describe('transmissive represented-range contact preflight', () => {
  const thinMaterial: MaterialSpec = {
    baseColor: [1, 1, 1],
    roughness: 0,
    metallic: 0,
    transmission: 1,
  };

  function thinTriangle(id: string, z = 0, transform?: ReturnType<typeof matrix>): MeshPrimitive {
    return {
      kind: 'mesh',
      id,
      positions: new Float32Array([0, 0, z, 4, 0, z, 0, 4, z]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: thinMaterial,
      ...(transform != null ? { transform } : {}),
    };
  }

  it('preserves an ordinary duplicated-index seam within one thin range', () => {
    const seam: MeshPrimitive = {
      kind: 'mesh',
      id: 'duplicated-index-seam',
      positions: new Float32Array([
        0, 0, 0, 1, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 0, 0, 1, 0,
      ]),
      normals: new Float32Array(18).fill(1),
      material: thinMaterial,
    };
    expect(() => analyzeOpticalMediumTopology(scene([seam]))).not.toThrow();
  });

  it('rejects two distinct coincident thin-sheet ranges before a first ray can tie', () => {
    expect(() => analyzeOpticalMediumTopology(scene([
      thinTriangle('coincident-a'),
      thinTriangle('coincident-b'),
    ]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );
  });

  it('detects contact after the selected merged-world f32 transform arithmetic', () => {
    const hugeTranslation = matrix(1, 0, 0, 2 ** 24);
    expect(() => analyzeOpticalMediumTopology(scene([
      thinTriangle('world-f32-a', 0, hugeTranslation),
      thinTriangle('world-f32-b', 0.25, hugeTranslation),
    ]), {
      transformArithmetic: 'merged-world-f64-to-f32',
    })).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );
  });
});

describe('optical medium topology', () => {
  it('accepts one outward closed manifold and reports one live medium', () => {
    expect(analyzeOpticalMediumTopology(scene([cube('glass')]))).toEqual({
      componentCount: 1,
      maxNestedMedia: 1,
      components: [{
        boundaryId: 0,
        primitiveId: 'glass',
        instanceIndex: 0,
        componentIndex: 0,
        representation: {
          kind: 'triangle-mesh',
          sourceTriangles: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        enclosingDepth: 0,
      }],
    });
  });

  it('accepts coplanar adjacent manifold faces and mirrored world transforms', () => {
    const mirrored = { ...cube('mirror'), transform: affine(-2, 1, 0.5) };
    const analysis = analyzeOpticalMediumTopology(scene([mirrored]));
    expect(analysis.componentCount).toBe(1);
    expect(analysis.maxNestedMedia).toBe(1);
  });

  it('is scale robust and does not infer contact from overlapping AABBs', () => {
    const longA = { ...cube('long-a'), transform: affine(2, 0.05, 0.05, Math.PI / 4) };
    const longB = {
      ...cube('long-b'),
      transform: affine(2, 0.05, 0.05, Math.PI / 4, -0.5, 0.5),
    };
    const broadPhaseOnly = analyzeOpticalMediumTopology(scene([longA, longB]));
    expect(broadPhaseOnly.componentCount).toBe(2);
    expect(broadPhaseOnly.maxNestedMedia).toBe(1);

    const tiny = analyzeOpticalMediumTopology(scene([
      cube('tiny-outer', 1e-12),
      cube('tiny-inner', 2e-13),
      cube('tiny-disjoint', 1e-13, 3e-12),
    ]));
    expect(tiny.maxNestedMedia).toBe(2);
  });

  it('counts exact disjoint-or-contained nesting, not overlapping AABBs', () => {
    const nested = analyzeOpticalMediumTopology(scene([
      cube('outer', 4),
      cube('middle', 2),
      cube('inner', 0.5),
      cube('disjoint', 1, 20),
    ]));
    expect(nested.maxNestedMedia).toBe(3);
    expect(nested.components.map((entry) => entry.enclosingDepth)).toEqual([0, 1, 2, 0]);
    expect(() => assertOpticalMediumTopology(scene([
      cube('outer', 4), cube('middle', 2), cube('inner', 0.5),
    ]), { maxNestedMedia: 2, backend: '@vitrum/test', method: 'setScene' })).toThrow(
      /requires 3 simultaneously live.*limit 2/,
    );
  });

  it('expands instanced transforms into separately validated components', () => {
    const { transform: _unusedTransform, ...base } = cube('instances');
    const instanced = {
      ...base,
      kind: 'instanced-mesh' as const,
      instances: [matrix(1, -4), matrix(1, 4)],
    };
    const analysis = analyzeOpticalMediumTopology(scene([instanced]));
    expect(analysis.componentCount).toBe(2);
    expect(analysis.maxNestedMedia).toBe(1);
    expect(analysis.components.map((entry) => entry.instanceIndex)).toEqual([0, 1]);
  });

  it('solves the current skinned pose before topology analysis', () => {
    const base = cube('skin');
    const skin: SkinnedMeshPrimitive = {
      ...base,
      kind: 'skinned-mesh',
      skinIndices: new Uint32Array(8 * 4),
      skinWeights: new Float32Array(8 * 4).map((_, index) => index % 4 === 0 ? 1 : 0),
      bones: matrix(1, 5),
      boneInverses: matrix(1),
      morphTargets: [new Float32Array(CUBE_POSITIONS.length).map(
        (_, index) => index % 3 === 0 ? 2.5 : 0,
      )],
      morphWeights: new Float32Array([1]),
    };
    // Rest pose is disjoint from the outer shell. Bone translation alone would
    // make the two cubes touch at x=6 and fail; applying the authored morph as
    // well moves the live shell wholly inside [6,9].
    const analysis = analyzeOpticalMediumTopology(scene([
      cube('skin-outer', 1.5, 7.5),
      skin,
    ]));
    expect(analysis.componentCount).toBe(2);
    expect(analysis.maxNestedMedia).toBe(2);
    expect(analysis.components.map((entry) => entry.enclosingDepth)).toEqual([0, 1]);
  });

  it('validates generated analytic orientation and displacement-resolved geometry', () => {
    const analytic = {
      kind: 'analytic' as const,
      id: 'analytic-sphere',
      shape: 'sphere' as const,
      params: new Float32Array([0, 0, 0, 1]),
      material: BULK,
      transform: matrix(1, -4),
    };
    const displaced = {
      ...cube('displaced', 1, 4),
      uvs: new Float32Array(8 * 2),
      material: {
        ...BULK,
        displacementMap: {
          handle: { width: 1, height: 1, data: new Float32Array([1]), channels: 1 },
          mipFilter: 'none' as const,
        },
        // Without this resolved displacement the cube touches the neighbor at
        // x=5. The negative constant height translates it away before topology
        // analysis, proving the validator consumes represented geometry.
        displacementScale: -0.2,
      },
    };
    const analysis = analyzeOpticalMediumTopology(scene([
      analytic,
      displaced,
      cube('displacement-neighbor', 1, 6),
    ]));
    expect(analysis.componentCount).toBe(3);
    expect(analysis.maxNestedMedia).toBe(1);
  });

  it.each([
    ['sphere', [0, 0, 0, 1]],
    ['box', [0, 0, 0, 1, 1, 1]],
    ['capsule', [0, -1, 0, 0, 1, 0, 0.5]],
    ['cylinder', [0, 0, 0, 1, 1]],
    ['h-channel-came', [4, 1, 2, 0.2]],
  ] satisfies Array<readonly [AnalyticPrimitive['shape'], number[]]>) (
    'accepts the generated %s analytic as a closed outward bulk boundary',
    (shape, params) => {
      const primitive: AnalyticPrimitive = {
        kind: 'analytic',
        id: `analytic-${shape}`,
        shape,
        params: new Float32Array(params),
        material: BULK,
      };
      const analysis = analyzeOpticalMediumTopology(scene([primitive]));
      expect(analysis.componentCount).toBe(1);
      expect(analysis.maxNestedMedia).toBe(1);
    },
  );

  it('selects fallback or generated-triangle analytic topology explicitly', () => {
    const analytic: AnalyticPrimitive = {
      kind: 'analytic',
      id: 'analytic-with-unrelated-open-fallback',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: BULK,
      fallbackMesh: {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        indices: new Uint32Array([0, 1, 2]),
      },
    };
    expect(() => analyzeOpticalMediumTopology(scene([analytic]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({
        code: 'open-or-nonmanifold-edge',
      }),
    );
    const generatedAnalysis = analyzeOpticalMediumTopology(scene([analytic]), {
      analyticGeometry: 'generated-triangle',
    });
    expect(generatedAnalysis.componentCount).toBe(1);
    expect(generatedAnalysis.components[0]).toMatchObject({
      boundaryId: 0,
      primitiveId: analytic.id,
      instanceIndex: 0,
      componentIndex: 0,
      representation: { kind: 'generated-analytic-triangle' },
    });
    expect(generatedAnalysis.components[0]!.representation).toHaveProperty('sourceTriangles');
    expect(() => assertOpticalMediumTopology(scene([analytic]), {
      maxNestedMedia: 8,
      analyticGeometry: 'generated-triangle',
    })).not.toThrow();
  });

  it('lowers every transmissive analytic to the exact triangle transport scene', () => {
    const bulk: AnalyticPrimitive = {
      kind: 'analytic',
      id: 'bulk-analytic',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: BULK,
    };
    const thin: AnalyticPrimitive = {
      ...bulk,
      id: 'thin-analytic',
      material: { ...BULK, thickness: 0 },
      transform: matrix(1, 4),
    };
    const opaque: AnalyticPrimitive = {
      ...bulk,
      id: 'opaque-analytic',
      material: { ...BULK, transmission: 0 },
    };
    const lowered = lowerTransmissiveAnalyticPrimitives(scene([bulk, thin, opaque]));
    expect(lowered.primitives[0]?.kind).toBe('mesh');
    expect(lowered.primitives[1]?.kind).toBe('mesh');
    expect(lowered.primitives[2]).toBe(opaque);
    const analysis = analyzeOpticalMediumTopology(lowered);
    expect(analysis.components[0]?.representation.kind).toBe('triangle-mesh');
  });

  it('rejects topology that only separates before the represented f32 position pack', () => {
    const largeOrigin = 2 ** 20;
    const shiftedPositions = new Float32Array(CUBE_POSITIONS);
    for (let vertex = 0; vertex < shiftedPositions.length / 3; vertex += 1) {
      shiftedPositions[vertex * 3] = shiftedPositions[vertex * 3]! + 2.05;
    }
    expect(() => analyzeOpticalMediumTopology(scene([
      cube('f32-a', 1, largeOrigin),
      { ...cube('f32-b', 1, largeOrigin), positions: shiftedPositions },
    ]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );
  });

  it('fails closed for adjacent-f32 nested shells under a nonuniform TLAS transform', () => {
    const outer = {
      ...cube('nonuniform-outer'),
      transform: affine(2, 3, 4),
    };
    const adjacentInner = {
      ...cube('nonuniform-inner'),
      transform: affine(previousPositiveF32(2), 2, 3),
    };
    expect(() => analyzeOpticalMediumTopology(scene([
      outer,
      adjacentInner,
    ]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );

    const safelySeparatedInner = {
      ...adjacentInner,
      id: 'nonuniform-inner-safe',
      transform: affine(1.5, 2, 3),
    };
    expect(analyzeOpticalMediumTopology(scene([
      outer,
      safelySeparatedInner,
    ])).maxNestedMedia).toBe(2);
  });

  it('inherits the core affine-only transform contract before topology proof', () => {
    const perspective = matrix(1);
    perspective[3] = 0.01;
    expect(() => analyzeOpticalMediumTopology(scene([{
      ...cube('perspective-bulk'),
      transform: perspective,
    }]))).toThrow(/affine column-major matrix/);
  });

  it('rejects open, reversed, degenerate, touching, and fractional boundaries', () => {
    const open = { ...cube('open'), indices: CUBE_INDICES.slice(0, -3) };
    expect(() => analyzeOpticalMediumTopology(scene([open]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'open-or-nonmanifold-edge' }),
    );

    const inconsistentIndices = new Uint32Array(CUBE_INDICES);
    [inconsistentIndices[1], inconsistentIndices[2]] = [
      inconsistentIndices[2]!, inconsistentIndices[1]!,
    ];
    expect(() => analyzeOpticalMediumTopology(scene([{
      ...cube('inconsistent-edge'),
      indices: inconsistentIndices,
    }]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({
        code: 'inconsistent-edge-orientation',
      }),
    );

    const reversed = { ...cube('reversed'), indices: new Uint32Array([...CUBE_INDICES].reverse()) };
    expect(() => analyzeOpticalMediumTopology(scene([reversed]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'reversed-or-zero-volume' }),
    );

    const degenerateIndices = new Uint32Array(CUBE_INDICES);
    degenerateIndices[0] = degenerateIndices[1]!;
    expect(() => analyzeOpticalMediumTopology(scene([{ ...cube('degenerate'), indices: degenerateIndices }]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'degenerate-triangle' }),
    );

    expect(() => analyzeOpticalMediumTopology(scene([{
      kind: 'mesh',
      id: 'duplicate-opposite-faces',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 2, 1, 0]),
      material: BULK,
    }]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'self-contact' }),
    );

    expect(() => analyzeOpticalMediumTopology(scene([
      cube('a'), cube('touching', 1, 2),
    ]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );

    expect(() => analyzeOpticalMediumTopology(scene([
      cube('edge-a'),
      { ...cube('edge-touch'), transform: affine(1, 1, 1, 0, 2, 2) },
    ]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );

    expect(() => analyzeOpticalMediumTopology(scene([
      cube('vertex-a'),
      { ...cube('vertex-touch'), transform: affine(1, 1, 1, 0, 2, 2, 2) },
    ]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );

    expect(() => analyzeOpticalMediumTopology(scene([
      cube('a'), cube('interpenetrating', 1, 1),
    ]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );

    const selfIntersectingPositions = new Float32Array(CUBE_POSITIONS);
    selfIntersectingPositions.set([-0.5, -0.5, -2], 6 * 3);
    expect(() => analyzeOpticalMediumTopology(scene([{
      ...cube('self-intersecting'),
      positions: selfIntersectingPositions,
    }]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'self-contact' }),
    );

    // Two tetrahedral lobes joined by an oriented triangular tube form one
    // edge-connected closed surface. Their otherwise unrelated apex vertices
    // occupy the same represented position, producing two disconnected fans
    // at one welded vertex without first creating a nonmanifold edge.
    const splitFanPositions = new Float32Array([
      -3, 0, 0,
      -2, -1, -1,
      -2, 1, -1,
      -2, 0, 1,
      -3, 0, 0,
      2, -1, -1,
      2, 1, -1,
      2, 0, 1,
    ]);
    const splitFanIndices = new Uint32Array([
      0, 2, 1, 0, 3, 2, 0, 1, 3,
      4, 6, 5, 4, 7, 6, 4, 5, 7,
      1, 2, 5, 1, 5, 6,
      2, 3, 7, 2, 7, 5,
      3, 1, 6, 3, 6, 7,
    ]);
    expect(() => analyzeOpticalMediumTopology(scene([{
      kind: 'mesh',
      id: 'split-vertex-fan',
      positions: splitFanPositions,
      normals: new Float32Array(splitFanPositions.length).fill(1),
      indices: splitFanIndices,
      material: BULK,
    }]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'nonmanifold-vertex' }),
    );

    const { transform: _instanceTransform, ...instanceBase } = cube('overlap-instances');
    expect(() => analyzeOpticalMediumTopology(scene([{
      ...instanceBase,
      kind: 'instanced-mesh',
      instances: [matrix(1), matrix(1, 1)],
    }]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'component-contact' }),
    );

    expect(() => analyzeOpticalMediumTopology(scene([
      cube('masked', 1, 0, { ...BULK, alphaMode: 'mask' }),
    ]))).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({ code: 'nondeterministic-boundary' }),
    );
  });
});

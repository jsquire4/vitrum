/**
 * Exact host-side topology preflight for bounded optical media.
 *
 * GPU medium stacks assume that every participating/absorbing transmissive
 * surface is a consistently outward-oriented, closed two-manifold and that
 * distinct media are either disjoint or wholly nested.  Failing that contract
 * in a shader can silently strand a medium-stack entry and black out transport,
 * so every backend calls this module synchronously before publishing a scene or
 * mutation.
 *
 * The broad phase is deliberately only a rejection accelerator: AABBs never
 * establish containment or non-intersection.  Triangle contact is decided by
 * convex separating axes, and containment by the closed-mesh solid angle.
 */

import {
  analyticPrimitiveToMesh,
  solveSkin,
  validateScene,
  type Mat4,
  type MaterialSpec,
  type MeshPrimitive,
  type Scene,
  type ScenePrimitive,
} from '@vitrum/core';
import { resolveDisplacedGeometry, type DisplaceablePrimitive } from './vertexDisplacement.js';
import {
  IDENTITY_MAT4,
  analyzeShaderF32LinearOrientation,
  applyMatrix4MergedWorldF32,
  applyMatrix4ShaderF32,
  determinant4,
} from './worldTransforms.js';
import {
  coplanarTrianglesContactBeyondSharedVertex,
  nonCoplanarTrianglesContactBeyondSharedVertex,
} from './triangleSharedVertexContact.js';

type V3 = readonly [number, number, number];

interface Bounds3 {
  min: [number, number, number];
  max: [number, number, number];
}

interface WeldedTriangle {
  readonly sourceTriangle: number;
  readonly vertices: readonly [number, number, number];
  readonly bounds: Bounds3;
  readonly centroid: V3;
}

interface TriangleTree {
  readonly bounds: Bounds3;
  readonly count: number;
  readonly left?: TriangleTree;
  readonly right?: TriangleTree;
  readonly triangles?: readonly WeldedTriangle[];
}

interface MutableComponent {
  readonly primitiveId: string;
  readonly instanceIndex: number;
  readonly componentIndex: number;
  readonly vertices: readonly V3[];
  readonly triangles: readonly WeldedTriangle[];
  readonly bounds: Bounds3;
  readonly tree: TriangleTree;
  readonly samplePoint: V3;
  /** Conservative shader-representation coordinate error for this component. */
  readonly representationUncertainty: number;
  readonly representation: OpticalMediumBoundaryRepresentation;
}

/** Stable diagnostic codes suitable for host telemetry and tests. */
export type OpticalMediumTopologyErrorCode =
  | 'nondeterministic-boundary'
  | 'degenerate-triangle'
  | 'open-or-nonmanifold-edge'
  | 'inconsistent-edge-orientation'
  | 'nonmanifold-vertex'
  | 'reversed-or-zero-volume'
  | 'self-contact'
  | 'component-contact'
  | 'capacity-exceeded';

export class OpticalMediumTopologyError extends RangeError {
  readonly code: OpticalMediumTopologyErrorCode;

  constructor(code: OpticalMediumTopologyErrorCode, message: string) {
    super(message);
    this.name = 'OpticalMediumTopologyError';
    this.code = code;
  }
}

export type OpticalMediumBoundaryRepresentation =
  | {
    /** The backend intersects the exact resolved/diced triangle stream. */
    readonly kind: 'triangle-mesh';
    /** Source triangle ordinals belonging to this deterministic component. */
    readonly sourceTriangles: readonly number[];
  }
  | {
    /** The backend intersects this exact canonical analytic triangulation. */
    readonly kind: 'generated-analytic-triangle';
    readonly sourceTriangles: readonly number[];
  };

export interface OpticalMediumComponentAnalysis {
  /**
   * Dense, deterministic scene-local boundary identity. This is the value
   * triangle/analytic hit records pack for medium-stack matching. It is valid
   * only for the analyzed scene representation and must be rebuilt on scene or
   * represented-geometry mutation.
   */
  readonly boundaryId: number;
  readonly primitiveId: string;
  readonly instanceIndex: number;
  readonly componentIndex: number;
  /** Explicitly excludes closed-form analytic identity from bulk transport. */
  readonly representation: OpticalMediumBoundaryRepresentation;
  /** Number of enclosing components, excluding this component itself. */
  readonly enclosingDepth: number;
}

export interface OpticalMediumTopologyAnalysis {
  readonly componentCount: number;
  /** Maximum number of simultaneously live bulk media at any point. */
  readonly maxNestedMedia: number;
  readonly components: readonly OpticalMediumComponentAnalysis[];
}

export interface AnalyzeOpticalMediumTopologyOptions {
  /**
   * Which triangle representation an analytic primitive contributes. Bulk
   * optical analytics may never use a closed-form hit against a finite-mesh
   * proof: a backend choosing generated triangles must transport those same
   * triangles (normally via {@link lowerTransmissiveAnalyticPrimitives}).
   */
  readonly analyticGeometry?: 'prefer-fallback' | 'generated-triangle';
  /** Exact arithmetic of the geometry representation the backend traverses. */
  readonly transformArithmetic?:
    | 'tlas-shader-f32'
    | 'merged-world-f64-to-f32';
}

export interface AssertOpticalMediumTopologyOptions
  extends AnalyzeOpticalMediumTopologyOptions {
  readonly maxNestedMedia: number;
  /** Package/backend label included in deterministic diagnostics. */
  readonly backend?: string;
  /** Calling method included in deterministic diagnostics. */
  readonly method?: string;
}

interface DetailedTopologyAnalysis {
  readonly analysis: OpticalMediumTopologyAnalysis;
}

/**
 * True when a material creates a bounded bulk medium rather than a thin sheet.
 * This deliberately mirrors the production path-tracer activation contract:
 * transmission plus authored thickness, scattering, RGB scattering, a
 * positive RGB absorption coefficient derived from finite
 * attenuationColor/attenuationDistance, or spectral attenuation.
 */
export function materialDefinesBulkOpticalMedium(material: MaterialSpec): boolean {
  if (!((material.transmission ?? 0) > 0)) return false;
  if ((material.thickness ?? 0) > 0) return true;
  const rgbScattering = material.scatteringCoefficientRGB;
  if (rgbScattering != null) {
    if (rgbScattering[0] > 0 || rgbScattering[1] > 0 || rgbScattering[2] > 0) {
      return true;
    }
  } else if ((material.scatteringCoefficient ?? 0) > 0) {
    return true;
  }
  if (
    material.attenuationColor != null &&
    material.attenuationDistance != null &&
    Number.isFinite(material.attenuationDistance) &&
    material.attenuationDistance > 0 &&
    (
      material.attenuationColor[0] < 1 ||
      material.attenuationColor[1] < 1 ||
      material.attenuationColor[2] < 1
    )
  ) {
    return true;
  }
  return material.spectralAttenuation != null;
}

/**
 * Replace every authored-transmissive analytic primitive with the canonical
 * generated mesh used by optical transport. This includes thin sheets: their
 * transmitted continuation needs the same exact triangle source-feature
 * exclusion as bulk boundaries. Closed-form analytics remain only for
 * nontransmissive materials. Backends must run validation and triangle packing
 * against this returned scene.
 */
export function lowerTransmissiveAnalyticPrimitives(scene: Scene): Scene {
  let changed = false;
  const primitives = scene.primitives.map((primitive) => {
    if (
      primitive.kind !== 'analytic' ||
      !((primitive.material.transmission ?? 0) > 0)
    ) {
      return primitive;
    }
    changed = true;
    return analyticPrimitiveToMesh(primitive, { preferFallbackMesh: false });
  });
  return changed ? { ...scene, primitives } : scene;
}

/** Analyze every bulk-optical component and reject malformed topology. */
export function analyzeOpticalMediumTopology(
  scene: Scene,
  options: AnalyzeOpticalMediumTopologyOptions = {},
): OpticalMediumTopologyAnalysis {
  return analyzeOpticalMediumTopologyDetailed(scene, options).analysis;
}

function analyzeOpticalMediumTopologyDetailed(
  scene: Scene,
  options: AnalyzeOpticalMediumTopologyOptions,
): DetailedTopologyAnalysis {
  validateAnalyzeOptions(options);
  validateScene(scene);
  const components: MutableComponent[] = [];
  for (const primitive of scene.primitives) {
    if (!materialDefinesBulkOpticalMedium(primitive.material)) continue;
    assertDeterministicallySolidBoundary(primitive);
    appendPrimitiveComponents(primitive, components, options);
  }

  for (const component of components) {
    const contact = findSelfContact(component);
    if (contact != null) {
      throw topologyError(
        'self-contact',
        component,
        `triangles ${contact[0]} and ${contact[1]} intersect or touch outside their shared manifold boundary`,
      );
    }
    validateComponentSignedVolume(component);
  }

  for (let a = 0; a < components.length; a += 1) {
    const componentA = components[a]!;
    for (let b = a + 1; b < components.length; b += 1) {
      const componentB = components[b]!;
      if (!boundsOverlap(
        componentA.bounds,
        componentB.bounds,
        representationContactTolerance(componentA, componentB),
      )) continue;
      const contact = findTreeContact(componentA, componentB);
      if (contact != null) {
        throw new OpticalMediumTopologyError(
          'component-contact',
          `${componentLabel(componentA)} and ${componentLabel(componentB)} ` +
            `intersect or touch at source triangles ${contact[0]} and ${contact[1]}; ` +
            'bulk media must be disjoint or wholly nested.',
        );
      }
    }
  }

  const analysis: OpticalMediumComponentAnalysis[] = [];
  let maxNestedMedia = 0;
  for (let innerIndex = 0; innerIndex < components.length; innerIndex += 1) {
    const inner = components[innerIndex]!;
    let enclosingDepth = 0;
    for (let outerIndex = 0; outerIndex < components.length; outerIndex += 1) {
      if (innerIndex === outerIndex) continue;
      const outer = components[outerIndex]!;
      if (!pointWithinBounds(inner.samplePoint, outer.bounds)) continue;
      if (pointInsideClosedComponent(inner.samplePoint, outer)) {
        enclosingDepth += 1;
      }
    }
    maxNestedMedia = Math.max(maxNestedMedia, enclosingDepth + 1);
    analysis.push({
      boundaryId: innerIndex,
      primitiveId: inner.primitiveId,
      instanceIndex: inner.instanceIndex,
      componentIndex: inner.componentIndex,
      representation: inner.representation,
      enclosingDepth,
    });
  }

  // Preserve the more specific closed-bulk manifold diagnostics above, then
  // apply the broader represented-range ambiguity preflight to thin and bulk
  // transmissive geometry alike.
  assertTransmissiveRepresentedRangesDoNotContact(scene, options);

  return {
    analysis: {
      componentCount: components.length,
      maxNestedMedia: components.length === 0 ? 0 : maxNestedMedia,
      components: analysis,
    },
  };
}

/** Analyze and enforce a backend's advertised live-medium capacity. */
export function assertOpticalMediumTopology(
  scene: Scene,
  options: AssertOpticalMediumTopologyOptions,
): OpticalMediumTopologyAnalysis {
  if (options == null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('assertOpticalMediumTopology: options must be an object.');
  }
  if (!Number.isSafeInteger(options.maxNestedMedia) || options.maxNestedMedia <= 0) {
    throw new RangeError(
      `assertOpticalMediumTopology: maxNestedMedia must be a positive safe integer (got ${String(options.maxNestedMedia)}).`,
    );
  }
  if (options.backend !== undefined && typeof options.backend !== 'string') {
    throw new TypeError('assertOpticalMediumTopology: backend must be a string when supplied.');
  }
  if (options.method !== undefined && typeof options.method !== 'string') {
    throw new TypeError('assertOpticalMediumTopology: method must be a string when supplied.');
  }
  const detailed = analyzeOpticalMediumTopologyDetailed(scene, options);
  const analysis = detailed.analysis;
  if (analysis.maxNestedMedia > options.maxNestedMedia) {
    const prefix = [options.backend, options.method].filter(Boolean).join(' ');
    throw new OpticalMediumTopologyError(
      'capacity-exceeded',
      `${prefix.length > 0 ? `[${prefix}] ` : ''}` +
      `scene requires ${analysis.maxNestedMedia} simultaneously live bulk optical media, ` +
        `above the advertised limit ${options.maxNestedMedia}.`,
    );
  }
  return analysis;
}

function validateAnalyzeOptions(options: AnalyzeOpticalMediumTopologyOptions): void {
  if (options == null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('analyzeOpticalMediumTopology: options must be an object.');
  }
  if (
    options.analyticGeometry !== undefined &&
    options.analyticGeometry !== 'prefer-fallback' &&
    options.analyticGeometry !== 'generated-triangle'
  ) {
    throw new RangeError(
      'analyzeOpticalMediumTopology: analyticGeometry must be ' +
        '"prefer-fallback" or "generated-triangle" when supplied.',
    );
  }
  if (
    options.transformArithmetic !== undefined &&
    options.transformArithmetic !== 'tlas-shader-f32' &&
    options.transformArithmetic !== 'merged-world-f64-to-f32'
  ) {
    throw new RangeError(
      'analyzeOpticalMediumTopology: transformArithmetic must be ' +
        '"tlas-shader-f32" or "merged-world-f64-to-f32" when supplied.',
    );
  }
}

function assertDeterministicallySolidBoundary(primitive: ScenePrimitive): void {
  const material = primitive.material;
  if ((material.alphaMode ?? 'opaque') === 'opaque') return;
  throw new OpticalMediumTopologyError(
    'nondeterministic-boundary',
    `Bulk optical primitive "${primitive.id}" uses alphaMode ` +
      `"${material.alphaMode}". Participating/absorbing boundaries must be ` +
      'deterministically solid; alpha mask/blend coverage cannot define a closed medium.',
  );
}

/**
 * A continuation source token is scoped to one represented primitive-instance
 * range. If two distinct transmissive ranges touch, an ordinary first ray can
 * select either equal-t range by traversal order and the later token cannot
 * recover the one that was hidden. Reject that ambiguous representation before
 * publication. Within one range, exact-coordinate welding preserves ordinary
 * indexed and duplicated-index triangle fans while rejecting non-adjacent
 * self-contact.
 */
function assertTransmissiveRepresentedRangesDoNotContact(
  scene: Scene,
  options: AnalyzeOpticalMediumTopologyOptions,
): void {
  const ranges: MutableComponent[] = [];
  for (const primitive of scene.primitives) {
    if (!((primitive.material.transmission ?? 0) > 0)) continue;
    appendTransmissiveRepresentedRanges(primitive, ranges, options);
  }
  for (const range of ranges) {
    const contact = findSelfContact(range);
    if (contact != null) {
      throw new OpticalMediumTopologyError(
        'self-contact',
        `Transmissive represented range "${range.primitiveId}" instance ` +
          `${range.instanceIndex} has non-adjacent self-contact at source ` +
          `triangles ${contact[0]} and ${contact[1]}.`,
      );
    }
    validateOpenRangeVertexFans(range);
  }
  for (let a = 0; a < ranges.length; a += 1) {
    for (let b = a + 1; b < ranges.length; b += 1) {
      const rangeA = ranges[a]!;
      const rangeB = ranges[b]!;
      if (!boundsOverlap(
        rangeA.bounds,
        rangeB.bounds,
        representationContactTolerance(rangeA, rangeB),
      )) continue;
      const contact = findTreeContact(rangeA, rangeB);
      if (contact == null) continue;
      throw new OpticalMediumTopologyError(
        'component-contact',
        `Transmissive represented ranges "${rangeA.primitiveId}" instance ` +
          `${rangeA.instanceIndex} and "${rangeB.primitiveId}" instance ` +
          `${rangeB.instanceIndex} intersect or touch at source triangles ` +
          `${contact[0]} and ${contact[1]}; distinct transmissive ranges must ` +
          'not share an exact first-hit event.',
      );
    }
  }
}

function appendTransmissiveRepresentedRanges(
  primitive: ScenePrimitive,
  destination: MutableComponent[],
  options: AnalyzeOpticalMediumTopologyOptions,
): void {
  const generatedAnalytic = primitive.kind === 'analytic' &&
    (
      options.analyticGeometry === 'generated-triangle' ||
      primitive.fallbackMesh == null
    );
  const mesh = resolveMeshPrimitive(primitive, options);
  const resolved = resolveDisplacedGeometry(mesh, () => {});
  const positions = resolved.sourcePositions;
  const sourceVertexCount = Math.floor(positions.length / 3);
  const sourceIndices: ArrayLike<number> = resolved.baseIndicesSource ??
    sequentialIndices(sourceVertexCount);
  const transforms: ReadonlyArray<ArrayLike<number> | undefined> =
    primitive.kind === 'instanced-mesh' ? primitive.instances : [mesh.transform];
  const transformArithmetic =
    options.transformArithmetic ?? 'tlas-shader-f32';

  for (let instanceIndex = 0; instanceIndex < transforms.length; instanceIndex += 1) {
    const transform = transforms[instanceIndex] ?? IDENTITY_MAT4;
    const worldPositions: V3[] = [];
    let representationUncertainty = 0;
    for (let vertex = 0; vertex < sourceVertexCount; vertex += 1) {
      const transformed = transformArithmetic === 'merged-world-f64-to-f32'
        ? applyMatrix4MergedWorldF32(
          transform,
          positions[vertex * 3]!,
          positions[vertex * 3 + 1]!,
          positions[vertex * 3 + 2]!,
        )
        : applyMatrix4ShaderF32(
          transform,
          positions[vertex * 3]!,
          positions[vertex * 3 + 1]!,
          positions[vertex * 3 + 2]!,
        );
      if (!finite3(transformed.point)) {
        throw new OpticalMediumTopologyError(
          'degenerate-triangle',
          `Transmissive primitive "${primitive.id}" instance ${instanceIndex} ` +
            'has a non-finite transformed vertex.',
        );
      }
      worldPositions.push(transformed.point);
      representationUncertainty = Math.max(
        representationUncertainty,
        transformed.uncertainty,
      );
    }

    const weldedVertices: V3[] = [];
    const weld = new Map<string, number>();
    const sourceToWelded = new Uint32Array(sourceVertexCount);
    for (let vertex = 0; vertex < sourceVertexCount; vertex += 1) {
      const point = worldPositions[vertex]!;
      const key = `${canonicalCoordinate(point[0])},${canonicalCoordinate(point[1])},${canonicalCoordinate(point[2])}`;
      let welded = weld.get(key);
      if (welded === undefined) {
        welded = weldedVertices.length;
        weld.set(key, welded);
        weldedVertices.push(point);
      }
      sourceToWelded[vertex] = welded;
    }

    let flip: boolean;
    if (transformArithmetic === 'tlas-shader-f32') {
      const orientation = analyzeShaderF32LinearOrientation(transform);
      if (!orientation.reliable) {
        throw new OpticalMediumTopologyError(
          'degenerate-triangle',
          `Transmissive primitive "${primitive.id}" instance ${instanceIndex} ` +
            'has an ill-conditioned transform whose live TLAS f32 orientation is ambiguous.',
        );
      }
      flip = orientation.sign < 0;
    } else {
      flip = determinant4(transform) < 0;
    }

    const triangles: WeldedTriangle[] = [];
    const triangleCount = Math.floor(sourceIndices.length / 3);
    for (let triangle = 0; triangle < triangleCount; triangle += 1) {
      const i0 = sourceIndices[triangle * 3]!;
      const i1 = sourceIndices[triangle * 3 + 1]!;
      const i2 = sourceIndices[triangle * 3 + 2]!;
      if (i0 >= sourceVertexCount || i1 >= sourceVertexCount || i2 >= sourceVertexCount) {
        throw new OpticalMediumTopologyError(
          'degenerate-triangle',
          `Transmissive primitive "${primitive.id}" instance ${instanceIndex} ` +
            `triangle ${triangle} has an out-of-range index.`,
        );
      }
      const raw: [number, number, number] = [
        sourceToWelded[i0]!, sourceToWelded[i1]!, sourceToWelded[i2]!,
      ];
      const vertexIds: [number, number, number] = flip
        ? [raw[2], raw[1], raw[0]]
        : raw;
      const a = weldedVertices[vertexIds[0]]!;
      const b = weldedVertices[vertexIds[1]]!;
      const c = weldedVertices[vertexIds[2]]!;
      if (
        vertexIds[0] === vertexIds[1] ||
        vertexIds[1] === vertexIds[2] ||
        vertexIds[2] === vertexIds[0] ||
        lengthSquared(cross(sub(b, a), sub(c, a))) === 0
      ) {
        throw new OpticalMediumTopologyError(
          'degenerate-triangle',
          `Transmissive primitive "${primitive.id}" instance ${instanceIndex} ` +
            `triangle ${triangle} is degenerate after position welding.`,
        );
      }
      triangles.push({
        sourceTriangle: triangle,
        vertices: vertexIds,
        bounds: boundsFromPoints(a, b, c),
        centroid: [
          (a[0] + b[0] + c[0]) / 3,
          (a[1] + b[1] + c[1]) / 3,
          (a[2] + b[2] + c[2]) / 3,
        ],
      });
    }
    if (triangles.length === 0) continue;
    const bounds = boundsFromTriangles(triangles);
    destination.push({
      primitiveId: primitive.id,
      instanceIndex,
      componentIndex: 0,
      vertices: weldedVertices,
      triangles,
      bounds,
      tree: buildTriangleTree(triangles),
      samplePoint: weldedVertices[triangles[0]!.vertices[0]]!,
      representationUncertainty,
      representation: generatedAnalytic
        ? { kind: 'generated-analytic-triangle', sourceTriangles: triangles.map((t) => t.sourceTriangle) }
        : { kind: 'triangle-mesh', sourceTriangles: triangles.map((t) => t.sourceTriangle) },
    });
  }
}

function validateOpenRangeVertexFans(range: MutableComponent): void {
  const incident = new Map<number, number[]>();
  const edgeUses = new Map<string, number[]>();
  for (let triangle = 0; triangle < range.triangles.length; triangle += 1) {
    const vertices = range.triangles[triangle]!.vertices;
    for (const vertex of vertices) {
      const uses = incident.get(vertex) ?? [];
      uses.push(triangle);
      incident.set(vertex, uses);
    }
    for (const [a, b] of [
      [vertices[0], vertices[1]],
      [vertices[1], vertices[2]],
      [vertices[2], vertices[0]],
    ] as const) {
      const key = edgeKey(a, b);
      const uses = edgeUses.get(key) ?? [];
      uses.push(triangle);
      edgeUses.set(key, uses);
    }
  }
  for (const [vertex, triangles] of incident) {
    if (triangles.length <= 1) continue;
    const triangleSet = new Set(triangles);
    const adjacency = new Map<number, Set<number>>(
      triangles.map((triangle) => [triangle, new Set<number>()]),
    );
    for (const [key, uses] of edgeUses) {
      const [a, b] = key.split(':').map(Number);
      if (a !== vertex && b !== vertex) continue;
      for (const first of uses) {
        if (!triangleSet.has(first)) continue;
        for (const second of uses) {
          if (first !== second && triangleSet.has(second)) {
            adjacency.get(first)!.add(second);
          }
        }
      }
    }
    const visited = new Set<number>();
    const pending = [triangles[0]!];
    while (pending.length > 0) {
      const triangle = pending.pop()!;
      if (visited.has(triangle)) continue;
      visited.add(triangle);
      for (const neighbor of adjacency.get(triangle) ?? []) pending.push(neighbor);
    }
    if (visited.size !== triangles.length) {
      throw new OpticalMediumTopologyError(
        'self-contact',
        `Transmissive represented range "${range.primitiveId}" instance ` +
          `${range.instanceIndex} has disconnected triangle fans at welded ` +
          `vertex ${vertex}.`,
      );
    }
  }
}

function appendPrimitiveComponents(
  primitive: ScenePrimitive,
  destination: MutableComponent[],
  options: AnalyzeOpticalMediumTopologyOptions,
): void {
  const generatedAnalytic = primitive.kind === 'analytic' &&
    (
      options.analyticGeometry === 'generated-triangle' ||
      primitive.fallbackMesh == null
    );
  const mesh = resolveMeshPrimitive(primitive, options);
  const resolved = resolveDisplacedGeometry(mesh, () => {});
  const positions = resolved.sourcePositions;
  const vertexCount = Math.floor(positions.length / 3);
  const sourceIndices: ArrayLike<number> = resolved.baseIndicesSource ??
    sequentialIndices(vertexCount);
  const transforms: ReadonlyArray<ArrayLike<number> | undefined> =
    primitive.kind === 'instanced-mesh' ? primitive.instances : [mesh.transform];
  const firstComponent = destination.length;
  for (let instanceIndex = 0; instanceIndex < transforms.length; instanceIndex += 1) {
    appendRealizedMeshComponents(
      primitive.id,
      instanceIndex,
      positions,
      sourceIndices,
      transforms[instanceIndex] ?? IDENTITY_MAT4,
      destination,
      options.transformArithmetic ?? 'tlas-shader-f32',
      generatedAnalytic
        ? { kind: 'generated-analytic-triangle', sourceTriangles: [] }
        : { kind: 'triangle-mesh', sourceTriangles: [] },
    );
  }
  if (generatedAnalytic && destination.length !== firstComponent + 1) {
    throw new OpticalMediumTopologyError(
      'open-or-nonmanifold-edge',
      `Generated analytic bulk primitive "${primitive.id}" must represent exactly one ` +
        `closed component (resolved ${destination.length - firstComponent}).`,
    );
  }
}

function resolveMeshPrimitive(
  primitive: ScenePrimitive,
  options: AnalyzeOpticalMediumTopologyOptions,
): MeshPrimitive | DisplaceablePrimitive & {
  readonly transform?: Mat4;
} {
  if (primitive.kind === 'analytic') {
    return analyticPrimitiveToMesh(primitive, {
      preferFallbackMesh: options.analyticGeometry !== 'generated-triangle',
    });
  }
  if (primitive.kind !== 'skinned-mesh') return primitive;
  const solved = solveSkin(primitive);
  return {
    ...primitive,
    positions: solved.positions,
    normals: solved.normals,
    ...(solved.tangents != null ? { tangents: solved.tangents } : {}),
    ...(solved.uvs != null ? { uvs: solved.uvs } : {}),
    ...(solved.uv1 != null ? { uv1: solved.uv1 } : {}),
    ...(solved.uvSets != null ? { uvSets: solved.uvSets } : {}),
    ...(solved.colors != null ? { colors: solved.colors } : {}),
    ...(solved.colorSets != null ? { colorSets: solved.colorSets } : {}),
  };
}

function sequentialIndices(vertexCount: number): Uint32Array {
  const out = new Uint32Array(vertexCount);
  for (let index = 0; index < vertexCount; index += 1) out[index] = index;
  return out;
}

function appendRealizedMeshComponents(
  primitiveId: string,
  instanceIndex: number,
  positions: Float32Array,
  indices: ArrayLike<number>,
  transform: ArrayLike<number>,
  destination: MutableComponent[],
  transformArithmetic:
    | 'tlas-shader-f32'
    | 'merged-world-f64-to-f32',
  representation: OpticalMediumBoundaryRepresentation,
): void {
  const sourceVertexCount = Math.floor(positions.length / 3);
  const worldPositions: V3[] = [];
  let representationUncertainty = 0;
  for (let vertex = 0; vertex < sourceVertexCount; vertex += 1) {
    const transformed = transformArithmetic === 'merged-world-f64-to-f32'
      ? applyMatrix4MergedWorldF32(
        transform,
        positions[vertex * 3]!,
        positions[vertex * 3 + 1]!,
        positions[vertex * 3 + 2]!,
      )
      : applyMatrix4ShaderF32(
      transform,
      positions[vertex * 3]!,
      positions[vertex * 3 + 1]!,
      positions[vertex * 3 + 2]!,
    );
    // Match the exact selected representation. TLAS uncertainty is carried
    // into contact tests; merged vertices are already exact packed f32 values.
    const point = transformed.point;
    representationUncertainty = Math.max(
      representationUncertainty,
      transformed.uncertainty,
    );
    if (!finite3(point)) {
      throw new OpticalMediumTopologyError(
        'degenerate-triangle',
        `Bulk optical primitive "${primitiveId}" instance ${instanceIndex} has a non-finite transformed vertex.`,
      );
    }
    worldPositions[vertex] = point;
  }

  const weldedVertices: V3[] = [];
  const weld = new Map<string, number>();
  const sourceToWelded = new Uint32Array(sourceVertexCount);
  for (let vertex = 0; vertex < sourceVertexCount; vertex += 1) {
    const point = worldPositions[vertex]!;
    const key = `${canonicalCoordinate(point[0])},${canonicalCoordinate(point[1])},${canonicalCoordinate(point[2])}`;
    let welded = weld.get(key);
    if (welded === undefined) {
      welded = weldedVertices.length;
      weld.set(key, welded);
      weldedVertices.push(point);
    }
    sourceToWelded[vertex] = welded;
  }

  const triangleCount = Math.floor(indices.length / 3);
  const triangles: WeldedTriangle[] = [];
  let flip: boolean;
  if (transformArithmetic === 'tlas-shader-f32') {
    const orientation = analyzeShaderF32LinearOrientation(transform);
    if (!orientation.reliable) {
      throw new OpticalMediumTopologyError(
        'degenerate-triangle',
        `Bulk optical primitive "${primitiveId}" instance ${instanceIndex} ` +
          'has an ill-conditioned transform whose live TLAS f32 orientation ' +
          `is ambiguous (${orientation.sharedDeterminant} vs ` +
          `${orientation.ptDeterminant}).`,
      );
    }
    flip = orientation.sign < 0;
  } else {
    // mergeWorldSpaceFromCore reverses indices from the JS-f64 determinant.
    flip = determinant4(transform) < 0;
  }
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const i0 = indices[triangle * 3]!;
    const i1 = indices[triangle * 3 + 1]!;
    const i2 = indices[triangle * 3 + 2]!;
    if (i0 >= sourceVertexCount || i1 >= sourceVertexCount || i2 >= sourceVertexCount) {
      throw new OpticalMediumTopologyError(
        'degenerate-triangle',
        `Bulk optical primitive "${primitiveId}" instance ${instanceIndex} triangle ${triangle} has an out-of-range index.`,
      );
    }
    const raw: [number, number, number] = [
      sourceToWelded[i0]!, sourceToWelded[i1]!, sourceToWelded[i2]!,
    ];
    const vertexIds: [number, number, number] = flip
      ? [raw[2], raw[1], raw[0]]
      : raw;
    const a = weldedVertices[vertexIds[0]]!;
    const b = weldedVertices[vertexIds[1]]!;
    const c = weldedVertices[vertexIds[2]]!;
    if (
      vertexIds[0] === vertexIds[1] ||
      vertexIds[1] === vertexIds[2] ||
      vertexIds[2] === vertexIds[0] ||
      lengthSquared(cross(sub(b, a), sub(c, a))) === 0
    ) {
      throw new OpticalMediumTopologyError(
        'degenerate-triangle',
        `Bulk optical primitive "${primitiveId}" instance ${instanceIndex} triangle ${triangle} is degenerate after position welding.`,
      );
    }
    triangles.push({
      sourceTriangle: triangle,
      vertices: vertexIds,
      bounds: boundsFromPoints(a, b, c),
      centroid: [
        (a[0] + b[0] + c[0]) / 3,
        (a[1] + b[1] + c[1]) / 3,
        (a[2] + b[2] + c[2]) / 3,
      ],
    });
  }
  if (triangles.length === 0) return;

  const parent = triangles.map((_, index) => index);
  const edgeUses = new Map<string, Array<{ triangle: number; from: number; to: number }>>();
  for (let triangle = 0; triangle < triangles.length; triangle += 1) {
    const [a, b, c] = triangles[triangle]!.vertices;
    for (const [from, to] of [[a, b], [b, c], [c, a]] as const) {
      const key = edgeKey(from, to);
      const uses = edgeUses.get(key) ?? [];
      if (uses.length > 0) union(parent, triangle, uses[0]!.triangle);
      uses.push({ triangle, from, to });
      edgeUses.set(key, uses);
    }
  }

  const byRoot = new Map<number, WeldedTriangle[]>();
  for (let triangle = 0; triangle < triangles.length; triangle += 1) {
    const root = find(parent, triangle);
    const list = byRoot.get(root) ?? [];
    list.push(triangles[triangle]!);
    byRoot.set(root, list);
  }

  let componentIndex = 0;
  for (const componentTriangles of byRoot.values()) {
    validateComponentManifold(
      primitiveId,
      instanceIndex,
      componentIndex,
      componentTriangles,
    );
    const bounds = boundsFromTriangles(componentTriangles);
    const component: MutableComponent = {
      primitiveId,
      instanceIndex,
      componentIndex,
      vertices: weldedVertices,
      triangles: componentTriangles,
      bounds,
      tree: buildTriangleTree(componentTriangles),
      samplePoint: weldedVertices[componentTriangles[0]!.vertices[0]]!,
      representationUncertainty,
      representation: {
        kind: representation.kind,
        sourceTriangles: componentTriangles.map((triangle) => triangle.sourceTriangle),
      },
    };
    destination.push(component);
    componentIndex += 1;
  }
}

function validateComponentManifold(
  primitiveId: string,
  instanceIndex: number,
  componentIndex: number,
  triangles: readonly WeldedTriangle[],
): void {
  const componentStub = {
    primitiveId,
    instanceIndex,
    componentIndex,
  };
  const edgeUses = new Map<string, Array<{ from: number; to: number; triangle: number }>>();
  const vertexLinks = new Map<number, Array<readonly [number, number]>>();
  for (let localTriangle = 0; localTriangle < triangles.length; localTriangle += 1) {
    const [a, b, c] = triangles[localTriangle]!.vertices;
    for (const [from, to] of [[a, b], [b, c], [c, a]] as const) {
      const key = edgeKey(from, to);
      const uses = edgeUses.get(key) ?? [];
      uses.push({ from, to, triangle: triangles[localTriangle]!.sourceTriangle });
      edgeUses.set(key, uses);
    }
    for (const [vertex, linkA, linkB] of [[a, b, c], [b, c, a], [c, a, b]] as const) {
      const links = vertexLinks.get(vertex) ?? [];
      links.push([linkA, linkB]);
      vertexLinks.set(vertex, links);
    }
  }

  for (const [key, uses] of edgeUses) {
    if (uses.length !== 2) {
      throw topologyError(
        'open-or-nonmanifold-edge',
        componentStub,
        `welded edge ${key} is used ${uses.length} times; every edge must be used exactly twice`,
      );
    }
    if (!(uses[0]!.from === uses[1]!.to && uses[0]!.to === uses[1]!.from)) {
      throw topologyError(
        'inconsistent-edge-orientation',
        componentStub,
        `welded edge ${key} is not traversed once in each direction`,
      );
    }
  }

  for (const [vertex, linkEdges] of vertexLinks) {
    const adjacency = new Map<number, number[]>();
    for (const [a, b] of linkEdges) {
      const aLinks = adjacency.get(a) ?? [];
      aLinks.push(b);
      adjacency.set(a, aLinks);
      const bLinks = adjacency.get(b) ?? [];
      bLinks.push(a);
      adjacency.set(b, bLinks);
    }
    if ([...adjacency.values()].some((neighbors) => neighbors.length !== 2)) {
      throw topologyError(
        'nonmanifold-vertex',
        componentStub,
        `welded vertex ${vertex} does not have one closed manifold fan`,
      );
    }
    const first = adjacency.keys().next().value;
    if (first === undefined) continue;
    const visited = new Set<number>([first]);
    const pending = [first];
    while (pending.length > 0) {
      const current = pending.pop()!;
      for (const neighbor of adjacency.get(current) ?? []) {
        if (visited.has(neighbor)) continue;
        visited.add(neighbor);
        pending.push(neighbor);
      }
    }
    if (visited.size !== adjacency.size) {
      throw topologyError(
        'nonmanifold-vertex',
        componentStub,
        `welded vertex ${vertex} has multiple disconnected incident fans`,
      );
    }
  }

}

function validateComponentSignedVolume(component: MutableComponent): void {
  const reference = component.vertices[component.triangles[0]!.vertices[0]]!;
  let volume6 = 0;
  let correction = 0;
  for (const triangle of component.triangles) {
    const a = sub(component.vertices[triangle.vertices[0]]!, reference);
    const b = sub(component.vertices[triangle.vertices[1]]!, reference);
    const c = sub(component.vertices[triangle.vertices[2]]!, reference);
    const term = dot(a, cross(b, c));
    const adjusted = term - correction;
    const sum = volume6 + adjusted;
    correction = (sum - volume6) - adjusted;
    volume6 = sum;
  }
  if (!(volume6 > 0) || !Number.isFinite(volume6)) {
    throw topologyError(
      'reversed-or-zero-volume',
      component,
      `signed volume must be finite and positive for outward winding (got ${String(volume6 / 6)})`,
    );
  }
}

function findSelfContact(component: MutableComponent): readonly [number, number] | null {
  return findSelfTreeContact(component.tree, component);
}

function findSelfTreeContact(
  node: TriangleTree,
  component: MutableComponent,
): readonly [number, number] | null {
  if (node.triangles != null) {
    for (let a = 0; a < node.triangles.length; a += 1) {
      for (let b = a + 1; b < node.triangles.length; b += 1) {
        const contact = selfPairHasInvalidContact(node.triangles[a]!, node.triangles[b]!, component);
        if (contact) return [node.triangles[a]!.sourceTriangle, node.triangles[b]!.sourceTriangle];
      }
    }
    return null;
  }
  const leftContact = findSelfTreeContact(node.left!, component);
  if (leftContact != null) return leftContact;
  const crossContact = findTreePairContact(node.left!, node.right!, component, component, true);
  if (crossContact != null) return crossContact;
  return findSelfTreeContact(node.right!, component);
}

function findTreeContact(
  a: MutableComponent,
  b: MutableComponent,
): readonly [number, number] | null {
  return findTreePairContact(a.tree, b.tree, a, b, false);
}

function findTreePairContact(
  a: TriangleTree,
  b: TriangleTree,
  componentA: MutableComponent,
  componentB: MutableComponent,
  self: boolean,
): readonly [number, number] | null {
  if (!boundsOverlap(
    a.bounds,
    b.bounds,
    representationContactTolerance(componentA, componentB),
  )) return null;
  if (a.triangles != null && b.triangles != null) {
    for (const triangleA of a.triangles) {
      for (const triangleB of b.triangles) {
        if (self && triangleA === triangleB) continue;
        if (!boundsOverlap(
          triangleA.bounds,
          triangleB.bounds,
          representationContactTolerance(componentA, componentB),
        )) continue;
        const invalid = self
          ? selfPairHasInvalidContact(triangleA, triangleB, componentA)
          : trianglesContact(triangleA, componentA, triangleB, componentB);
        if (invalid) return [triangleA.sourceTriangle, triangleB.sourceTriangle];
      }
    }
    return null;
  }
  if (b.triangles != null || (a.triangles == null && a.count >= b.count)) {
    return findTreePairContact(a.left!, b, componentA, componentB, self) ??
      findTreePairContact(a.right!, b, componentA, componentB, self);
  }
  return findTreePairContact(a, b.left!, componentA, componentB, self) ??
    findTreePairContact(a, b.right!, componentA, componentB, self);
}

function selfPairHasInvalidContact(
  a: WeldedTriangle,
  b: WeldedTriangle,
  component: MutableComponent,
): boolean {
  const shared = a.vertices.filter((vertex) => b.vertices.includes(vertex));
  // Every face in one component is produced by the same indexed f32 stream and
  // transform expression, so its arithmetic choice is correlated. Applying
  // the independent-component margin inside that one represented mesh would
  // falsely reject ordinary manifold fans at large world translations. The
  // exact emulated f32 coordinates remain the source of truth for self-contact.
  const representationTolerance = 0;
  if (!boundsOverlap(
    a.bounds,
    b.bounds,
    representationTolerance,
  )) return false;
  if (shared.length === 0) {
    return trianglesContact(a, component, b, component, false);
  }
  // Opposite-wound duplicate faces can satisfy the two-use/opposite-direction
  // edge counts, but they enclose no volume and are self-contact by definition.
  // Handle this before looking for a non-shared third vertex.
  if (shared.length === 3) return true;
  const pointsA = trianglePoints(a, component);
  const pointsB = trianglePoints(b, component);
  const normalA = cross(sub(pointsA[1], pointsA[0]), sub(pointsA[2], pointsA[0]));
  const normalB = cross(sub(pointsB[1], pointsB[0]), sub(pointsB[2], pointsB[0]));
  const tolerance = contactTolerance(
    pointsA,
    pointsB,
    representationTolerance,
  );
  const unitNormalA = normalize(normalA);
  const unitNormalB = normalize(normalB);
  const coplanar = Math.sqrt(lengthSquared(cross(unitNormalA, unitNormalB))) <=
      Number.EPSILON * 64 &&
    Math.abs(planeDistance(pointsB[0], pointsA[0], unitNormalA)) <= tolerance;
  if (shared.length >= 2) {
    if (!coplanar) return false;
    const edgeA = component.vertices[shared[0]!]!;
    const edgeB = component.vertices[shared[1]!]!;
    const thirdA = component.vertices[a.vertices.find((v) => !shared.includes(v))!]!;
    const thirdB = component.vertices[b.vertices.find((v) => !shared.includes(v))!]!;
    const axis = cross(normalA, sub(edgeB, edgeA));
    const sideA = dot(sub(thirdA, edgeA), axis);
    const sideB = dot(sub(thirdB, edgeA), axis);
    return sideA === 0 || sideB === 0 || Math.sign(sideA) === Math.sign(sideB);
  }
  if (!trianglesContact(a, component, b, component)) return false;
  if (coplanar) {
    return coplanarTrianglesContactBeyondSharedVertex(
      pointsA,
      pointsB,
      component.vertices[shared[0]!]!,
    );
  }
  const sharedPoint = component.vertices[shared[0]!]!;
  return nonCoplanarTrianglesContactBeyondSharedVertex(
    pointsA,
    pointsB,
    sharedPoint,
    tolerance,
  );
}

function trianglesContact(
  triangleA: WeldedTriangle,
  componentA: MutableComponent,
  triangleB: WeldedTriangle,
  componentB: MutableComponent,
  includeIndependentRepresentationUncertainty = true,
): boolean {
  const a = trianglePoints(triangleA, componentA);
  const b = trianglePoints(triangleB, componentB);
  const edgesA = [sub(a[1], a[0]), sub(a[2], a[1]), sub(a[0], a[2])] as const;
  const edgesB = [sub(b[1], b[0]), sub(b[2], b[1]), sub(b[0], b[2])] as const;
  const normalA = cross(edgesA[0], sub(a[2], a[0]));
  const normalB = cross(edgesB[0], sub(b[2], b[0]));
  const axes: V3[] = [normalA, normalB];
  for (const edgeA of edgesA) {
    axes.push(cross(normalA, edgeA));
    for (const edgeB of edgesB) axes.push(cross(edgeA, edgeB));
  }
  for (const edgeB of edgesB) axes.push(cross(normalB, edgeB));
  const tolerance = contactTolerance(
    a,
    b,
    includeIndependentRepresentationUncertainty
      ? representationContactTolerance(componentA, componentB)
      : 0,
  );
  for (const rawAxis of axes) {
    if (lengthSquared(rawAxis) === 0) continue;
    const axis = normalize(rawAxis);
    const projectionA = projectPoints(a, axis);
    const projectionB = projectPoints(b, axis);
    if (
      projectionA.max < projectionB.min - tolerance ||
      projectionB.max < projectionA.min - tolerance
    ) {
      return false;
    }
  }
  return true;
}

function pointInsideClosedComponent(point: V3, component: MutableComponent): boolean {
  let solidAngle = 0;
  let correction = 0;
  for (const triangle of component.triangles) {
    const [pa, pb, pc] = trianglePoints(triangle, component);
    const a = sub(pa, point);
    const b = sub(pb, point);
    const c = sub(pc, point);
    const la = Math.sqrt(lengthSquared(a));
    const lb = Math.sqrt(lengthSquared(b));
    const lc = Math.sqrt(lengthSquared(c));
    const numerator = dot(a, cross(b, c));
    const denominator = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    const term = 2 * Math.atan2(numerator, denominator);
    const adjusted = term - correction;
    const sum = solidAngle + adjusted;
    correction = (sum - solidAngle) - adjusted;
    solidAngle = sum;
  }
  return Math.abs(solidAngle) > Math.PI * 2;
}

function buildTriangleTree(triangles: readonly WeldedTriangle[]): TriangleTree {
  const bounds = boundsFromTriangles(triangles);
  if (triangles.length <= 8) return { bounds, count: triangles.length, triangles };
  const extents: V3 = [
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  ];
  const axis = extents[1] > extents[0]
    ? (extents[2] > extents[1] ? 2 : 1)
    : (extents[2] > extents[0] ? 2 : 0);
  const sorted = [...triangles].sort(
    (a, b) => a.centroid[axis] - b.centroid[axis] || a.sourceTriangle - b.sourceTriangle,
  );
  const midpoint = Math.floor(sorted.length / 2);
  return {
    bounds,
    count: triangles.length,
    left: buildTriangleTree(sorted.slice(0, midpoint)),
    right: buildTriangleTree(sorted.slice(midpoint)),
  };
}

function trianglePoints(
  triangle: WeldedTriangle,
  component: MutableComponent,
): readonly [V3, V3, V3] {
  return [
    component.vertices[triangle.vertices[0]]!,
    component.vertices[triangle.vertices[1]]!,
    component.vertices[triangle.vertices[2]]!,
  ];
}

function boundsFromPoints(a: V3, b: V3, c: V3): Bounds3 {
  return {
    min: [Math.min(a[0], b[0], c[0]), Math.min(a[1], b[1], c[1]), Math.min(a[2], b[2], c[2])],
    max: [Math.max(a[0], b[0], c[0]), Math.max(a[1], b[1], c[1]), Math.max(a[2], b[2], c[2])],
  };
}

function boundsFromTriangles(triangles: readonly WeldedTriangle[]): Bounds3 {
  const bounds: Bounds3 = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const triangle of triangles) {
    for (const axis of [0, 1, 2] as const) {
      bounds.min[axis] = Math.min(bounds.min[axis], triangle.bounds.min[axis]);
      bounds.max[axis] = Math.max(bounds.max[axis], triangle.bounds.max[axis]);
    }
  }
  return bounds;
}

function boundsOverlap(a: Bounds3, b: Bounds3, tolerance = 0): boolean {
  return a.min[0] <= b.max[0] + tolerance && b.min[0] <= a.max[0] + tolerance &&
    a.min[1] <= b.max[1] + tolerance && b.min[1] <= a.max[1] + tolerance &&
    a.min[2] <= b.max[2] + tolerance && b.min[2] <= a.max[2] + tolerance;
}

function pointWithinBounds(point: V3, bounds: Bounds3): boolean {
  return point[0] >= bounds.min[0] && point[0] <= bounds.max[0] &&
    point[1] >= bounds.min[1] && point[1] <= bounds.max[1] &&
    point[2] >= bounds.min[2] && point[2] <= bounds.max[2];
}

function edgeKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function find(parent: number[], value: number): number {
  let root = value;
  while (parent[root] !== root) root = parent[root]!;
  while (parent[value] !== value) {
    const next = parent[value]!;
    parent[value] = root;
    value = next;
  }
  return root;
}

function union(parent: number[], a: number, b: number): void {
  const rootA = find(parent, a);
  const rootB = find(parent, b);
  if (rootA !== rootB) parent[rootB] = rootA;
}

function canonicalCoordinate(value: number): number {
  return Object.is(value, -0) ? 0 : value;
}

function finite3(value: readonly number[]): value is V3 {
  return value.length >= 3 && Number.isFinite(value[0]) &&
    Number.isFinite(value[1]) && Number.isFinite(value[2]);
}

function sub(a: V3, b: V3): V3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(value: V3, scalar: number): V3 {
  return [value[0] * scalar, value[1] * scalar, value[2] * scalar];
}

function dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: V3, b: V3): V3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function lengthSquared(value: V3): number {
  return dot(value, value);
}

function normalize(value: V3): V3 {
  return scale(value, 1 / Math.sqrt(lengthSquared(value)));
}

function planeDistance(point: V3, planePoint: V3, normal: V3): number {
  return dot(sub(point, planePoint), normal);
}

function contactTolerance(
  a: readonly [V3, V3, V3],
  b: readonly [V3, V3, V3],
  representationTolerance = 0,
): number {
  let scaleValue = Number.MIN_VALUE;
  for (const point of [...a, ...b]) {
    scaleValue = Math.max(scaleValue, Math.abs(point[0]), Math.abs(point[1]), Math.abs(point[2]));
  }
  return Math.max(
    Number.EPSILON * scaleValue * 64,
    representationTolerance,
  );
}

function representationContactTolerance(
  a: MutableComponent,
  b: MutableComponent,
): number {
  // A normalized projection axis can sum three coordinate errors. sqrt(3)
  // converts each component's per-coordinate envelope into a Euclidean one;
  // summing both sides bounds their possible relative movement.
  return Math.sqrt(3) * (
    a.representationUncertainty + b.representationUncertainty
  );
}

function projectPoints(points: readonly [V3, V3, V3], axis: V3): { min: number; max: number } {
  const p0 = dot(points[0], axis);
  const p1 = dot(points[1], axis);
  const p2 = dot(points[2], axis);
  return { min: Math.min(p0, p1, p2), max: Math.max(p0, p1, p2) };
}

function componentLabel(
  component: Pick<
    MutableComponent,
    'primitiveId' | 'instanceIndex' | 'componentIndex'
  >,
): string {
  return `Bulk optical primitive "${component.primitiveId}" instance ` +
    `${component.instanceIndex} component ${component.componentIndex}`;
}

function topologyError(
  code: OpticalMediumTopologyErrorCode,
  component: Pick<
    MutableComponent,
    'primitiveId' | 'instanceIndex' | 'componentIndex'
  >,
  detail: string,
): OpticalMediumTopologyError {
  return new OpticalMediumTopologyError(code, `${componentLabel(component)} ${detail}.`);
}

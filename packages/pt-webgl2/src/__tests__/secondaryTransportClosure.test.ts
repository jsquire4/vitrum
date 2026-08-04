import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { effectiveMaterialIor } from '@vitrum/core';
import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';
import { NEE_RESOLVE_MAIN } from '../glsl/neeResolveMain.glsl.js';
import { RENDER_MAIN } from '../glsl/renderMain.glsl.js';
import { ATTENUATE_HIT_SCALAR_RICH_GLSL } from '../glsl/render/attenuate_hit_scalar_rich.glsl.js';
import { GET_SURFACE_RECORD_SCALAR_RICH_GLSL } from '../glsl/render/get_surface_record_scalar_rich.glsl.js';
import { FOG_MATERIAL_GLSL } from '../glsl/shader/structs/fog_material.glsl.js';
import { BSDF_BASIC_GLSL } from '../glsl/shader/bsdf/bsdf_basic.glsl.js';
import { THREE_COMMON_SHIM } from '../glsl/common/threeCommonShim.js';
import * as AttenuateModule from '../glsl/render/attenuate_hit_function.glsl.js';
import * as BdptConnectionModule from '../glsl/render/bdpt_connection.glsl.js';
import * as BdptLightSubpathModule from '../glsl/render/bdpt_light_subpath.glsl.js';
import * as BsdfModule from '../glsl/shader/bsdf/bsdf_functions.glsl.js';
import * as ContainmentModule from '../glsl/shader/bvh/inside_fog_volume_function.glsl.js';
import * as MappedSurfaceModule from '../glsl/render/get_surface_record_function.glsl.js';
import * as TraceSceneModule from '../glsl/render/trace_scene_function.glsl.js';
import * as UtilModule from '../glsl/shader/common/util_functions.glsl.js';

function chunk(module: Record<string, unknown>, name: string): string {
  const value = module[name];
  expect(typeof value, `${name} GLSL export`).toBe('string');
  return value as string;
}

const attenuation = chunk(
  AttenuateModule,
  'attenuate_hit_function',
);
const bdptConnection = chunk(
  BdptConnectionModule,
  'bdpt_connection',
);
const bdptLightSubpath = chunk(
  BdptLightSubpathModule,
  'bdpt_light_subpath',
);
const bsdf = chunk(BsdfModule, 'bsdf_functions');
const containment = chunk(
  ContainmentModule,
  'inside_fog_volume_function',
);
const mappedSurface = chunk(
  MappedSurfaceModule,
  'get_surface_record_function',
);
const traceScene = chunk(
  TraceSceneModule,
  'trace_scene_function',
);
const util = chunk(UtilModule, 'util_functions');

type CoverageStatus = 'hole' | 'solid' | 'fractional';

function classifyCoverage(
  materialSide: number,
  hitSide: number,
  alphaTest: number,
  transparent: boolean,
  coverage: number,
): CoverageStatus {
  if (materialSide !== 0 && hitSide !== materialSide) return 'hole';
  if (!Number.isFinite(coverage)) return 'fractional';
  const c = Math.max(0, Math.min(1, coverage));
  if (alphaTest !== 0) return c < alphaTest ? 'hole' : 'solid';
  if (!transparent) return 'solid';
  if (c <= 0) return 'hole';
  if (c >= 1) return 'solid';
  return 'fractional';
}

interface BoundaryEvent {
  readonly materialId: number;
  readonly boundaryId: number;
  readonly side: 1 | -1;
}

type Point3 = readonly [number, number, number];

/** CPU oracle for the shader's inclusive +Z projected-edge acceptance rule. */
function includesPositiveZRayHit(
  triangle: readonly [Point3, Point3, Point3],
  origin: readonly [number, number] = [0, 0],
): boolean {
  const projected = triangle.map((point) =>
    [point[0] - origin[0], point[1] - origin[1]] as const);
  const [a, b, c] = projected;
  let u = c![0] * b![1] - c![1] * b![0];
  let v = a![0] * c![1] - a![1] * c![0];
  let w = b![0] * a![1] - b![1] * a![0];
  let determinant = u + v + w;
  if (determinant === 0) return false;
  if (determinant < 0) {
    u = -u;
    v = -v;
    w = -w;
    determinant = -determinant;
  }
  return determinant > 0 && u >= 0 && v >= 0 && w >= 0;
}

function resolveExactTieSides(sides: readonly (1 | -1)[]): 1 | -1 | 0 | null {
  const fronts = sides.filter((side) => side === 1).length;
  const backs = sides.length - fronts;
  if (fronts > 0 && backs === 0) return 1;
  if (backs > 0 && fronts === 0) return -1;
  if (fronts === backs) return 0;
  return null;
}

function buildContainingStack(
  events: readonly BoundaryEvent[],
  options: { readonly capacity?: number } = {},
): number[] | null {
  const capacity = options.capacity ?? 8;
  const pending: BoundaryEvent[] = [];
  const containingInnerToOuter: number[] = [];
  for (const event of events) {
    if (event.side === 1) {
      if (pending.length >= capacity) return null;
      pending.push(event);
    } else {
      if (pending.length > 0) {
        if (pending.at(-1)?.boundaryId !== event.boundaryId) return null;
        pending.pop();
      } else {
        if (containingInnerToOuter.length >= capacity) return null;
        containingInnerToOuter.push(event.materialId);
      }
    }
  }
  if (pending.length !== 0) return null;
  return containingInnerToOuter.reverse();
}

function cauchyIor(wavelengthNm: number, iorAtD: number, bNm2: number): number {
  const d = 589.3;
  return iorAtD + bNm2 * (1 / (wavelengthNm ** 2) - 1 / (d ** 2));
}

interface OpticalMedium {
  readonly id: number;
  readonly ior: number;
  readonly dispersion: number;
}

function interfaceEta(
  stack: readonly OpticalMedium[],
  surface: OpticalMedium,
  frontFace: boolean,
  opticalVolume: boolean,
  wavelengthNm: number,
): number | null {
  const atHero = (m: OpticalMedium): number =>
    cauchyIor(wavelengthNm, m.ior, m.dispersion);
  const incident = stack.length === 0 ? 1 : atHero(stack[stack.length - 1]!);
  let transmitted: number;
  if (opticalVolume && !frontFace) {
    if (stack.length === 0 || stack[stack.length - 1]!.id !== surface.id) return null;
    transmitted = stack.length > 1 ? atHero(stack[stack.length - 2]!) : 1;
  } else {
    transmitted = atHero(surface);
  }
  return incident / transmitted;
}

function transition(
  stack: number[],
  materialId: number,
  frontFace: boolean,
  opticalVolume: boolean,
  crossedBoundary: boolean,
): boolean {
  if (!opticalVolume || !crossedBoundary) return true;
  if (frontFace) {
    if (stack.length >= 8) return false;
    stack.push(materialId);
    return true;
  }
  if (stack.at(-1) !== materialId) return false;
  stack.pop();
  return true;
}

function extinctionTransmittance(extinction: number, distance: number): number {
  if (
    Number.isNaN(extinction) || extinction < 0 ||
    Number.isNaN(distance) || distance < 0
  ) return 0;
  if (distance === 0) return 1;
  if (distance === Number.POSITIVE_INFINITY) return extinction === 0 ? 1 : 0;
  if (extinction === Number.POSITIVE_INFINITY) return 0;
  return Math.exp(-extinction * distance);
}

function extinctionCollisionDensity(extinction: number, distance: number): number {
  if (
    Number.isNaN(extinction) || extinction < 0 ||
    Number.isNaN(distance) || distance < 0
  ) return 0;
  if (distance === 0 && extinction === Number.POSITIVE_INFINITY) {
    return Number.POSITIVE_INFINITY;
  }
  if (
    extinction === Number.POSITIVE_INFINITY ||
    distance === Number.POSITIVE_INFINITY
  ) return 0;
  return extinction * extinctionTransmittance(extinction, distance);
}

function refract2d(
  incident: readonly [number, number],
  normal: readonly [number, number],
  eta: number,
): readonly [number, number] | null {
  const dotNI = normal[0] * incident[0] + normal[1] * incident[1];
  const discriminant = 1 - eta * eta * (1 - dotNI * dotNI);
  if (discriminant < 0) return null;
  const normalScale = eta * dotNI + Math.sqrt(discriminant);
  return [
    eta * incident[0] - normalScale * normal[0],
    eta * incident[1] - normalScale * normal[1],
  ];
}

function f32(value: number): number {
  return Math.fround(value);
}

const RNG_LATTICE_SIZE = 2 ** 24;
const REPRESENTED_EQUAL_RGB: readonly [number, number, number] = [
  5_592_406 / RNG_LATTICE_SIZE,
  5_592_406 / RNG_LATTICE_SIZE,
  5_592_404 / RNG_LATTICE_SIZE,
];

function representedBernoulliProbability(probability: number): number {
  const p = f32(probability);
  if (!(p > 0)) return 0;
  if (p >= 1) return 1;
  return Math.min(
    RNG_LATTICE_SIZE - 1,
    Math.max(1, Math.floor(f32(f32(p * RNG_LATTICE_SIZE) + 0.5))),
  ) / RNG_LATTICE_SIZE;
}

function consumeMediumBudget(
  remainingDistance: number,
  geometricDistance: number,
): { readonly effectiveDistance: number; readonly remainingDistance: number } {
  if (
    Number.isNaN(remainingDistance) || remainingDistance < 0 ||
    Number.isNaN(geometricDistance) || geometricDistance < 0
  ) return { effectiveDistance: -1, remainingDistance };
  if (remainingDistance === Number.POSITIVE_INFINITY) {
    return { effectiveDistance: geometricDistance, remainingDistance };
  }
  const effectiveDistance = Math.min(geometricDistance, remainingDistance);
  return {
    effectiveDistance,
    remainingDistance: Math.max(remainingDistance - effectiveDistance, 0),
  };
}

function reconcileBidirectionalMediumBudget(
  initialDistance: number,
  eyeRemaining: number,
  lightRemaining: number,
): number | null {
  if (
    !Number.isFinite(initialDistance) || initialDistance < 0 ||
    eyeRemaining < 0 || eyeRemaining > initialDistance ||
    lightRemaining < 0 || lightRemaining > initialDistance
  ) return null;
  const lightConsumed = initialDistance - lightRemaining;
  return Math.max(eyeRemaining - lightConsumed, 0);
}

interface InitialTieCandidate {
  readonly componentId: number;
  readonly representedRangeId: number;
  readonly transmissive: boolean;
}

function initialTieIsUnambiguous(
  candidates: readonly InitialTieCandidate[],
): boolean {
  const selected = candidates[0];
  if (selected == null) return true;
  return candidates.slice(1).every((candidate) => {
    const distinctIdentity =
      candidate.componentId !== selected.componentId ||
      candidate.representedRangeId !== selected.representedRangeId;
    return !distinctIdentity || (!selected.transmissive && !candidate.transmissive);
  });
}

function nextPositiveF32(value: number): number {
  const lane = new Float32Array([value]);
  const words = new Uint32Array(lane.buffer);
  words[0] = words[0]! + 1;
  return lane[0]!;
}

describe('pt-webgl2 secondary-transport closure', () => {
  it('defines finite infinite-segment Beer limits without zero-times-infinity NaNs', () => {
    expect(extinctionTransmittance(0, Number.POSITIVE_INFINITY)).toBe(1);
    expect(extinctionTransmittance(0.25, Number.POSITIVE_INFINITY)).toBe(0);
    expect(extinctionTransmittance(Number.POSITIVE_INFINITY, 2)).toBe(0);
    expect(extinctionTransmittance(Number.POSITIVE_INFINITY, 0)).toBe(1);
    expect(extinctionTransmittance(Number.NaN, 2)).toBe(0);
    expect(extinctionTransmittance(0.25, Number.NaN)).toBe(0);
    expect(extinctionTransmittance(0.25, -1)).toBe(0);
    expect(extinctionTransmittance(0.25, 2)).toBeCloseTo(Math.exp(-0.5));

    expect(util).toContain('float extinctionTransmittance( float extinction, float dist )');
    expect(util).toContain('if ( vitrumIsInfiniteDistance( dist ) )');
    expect(util).toContain('return extinction == 0.0 ? 1.0 : 0.0;');
    expect(util).toContain('return extinctionTransmittance( ot, dist );');
    expect(util).toContain('return extinctionTransmittance( muLambda, dist );');
    for (const source of [bsdf, BSDF_BASIC_GLSL]) {
      const start = source.indexOf('vec3 fogSegmentTransmittance(');
      const end = source.indexOf('float fogProposalSurvival(', start);
      const segment = source.slice(start, end);
      expect(start).toBeGreaterThanOrEqual(0);
      expect(end).toBeGreaterThan(start);
      expect(segment).toContain('return extinctionTransmittance(');
      expect(segment).not.toContain('return exp(');
    }
  });

  it('preserves independent zero-color Beer and scattering lanes under the RGB mixture', () => {
    const attenuationColor = [0, 0.5, 1] as const;
    const attenuationDistance = 2;
    const sigmaS = [0.2, 0.1, 0.05] as const;
    const sigmaT = attenuationColor.map((color, channel) =>
      (color === 0 ? Number.POSITIVE_INFINITY : -Math.log(color) / attenuationDistance) +
        sigmaS[channel]!,
    );
    for (const distance of [0, 3]) {
      const survival = sigmaT.map((extinction) =>
        extinctionTransmittance(extinction, distance));
      const proposalSurvival = survival.reduce(
        (sum, value, channel) => sum + value * REPRESENTED_EQUAL_RGB[channel]!,
        0,
      );
      const surfaceWeight = survival.map((value) => value / proposalSurvival);
      expect(surfaceWeight.every((value) => !Number.isNaN(value))).toBe(true);
      expect(Math.max(...surfaceWeight)).toBeLessThanOrEqual(
        1 / Math.min(...REPRESENTED_EQUAL_RGB),
      );
      for (let channel = 0; channel < 3; channel += 1) {
        expect(proposalSurvival * surfaceWeight[channel]!).toBeCloseTo(
          survival[channel]!,
          14,
        );
      }

      const collisionLanes = sigmaT.map((extinction) =>
        extinctionCollisionDensity(extinction, distance));
      const proposalCollision = collisionLanes.reduce(
        (sum, value, channel) =>
          sum + value * REPRESENTED_EQUAL_RGB[channel]!,
        0,
      );
      const collisionScatterWeight = survival.map((value, channel) =>
        Number.isFinite(proposalCollision) && proposalCollision > 0
          ? sigmaS[channel]! * value / proposalCollision
          : 0);
      expect(collisionScatterWeight.every((value) => !Number.isNaN(value))).toBe(true);
      if (Number.isFinite(proposalCollision) && proposalCollision > 0) {
        for (let channel = 0; channel < 3; channel += 1) {
          expect(proposalCollision * collisionScatterWeight[channel]!).toBeCloseTo(
            sigmaS[channel]! * survival[channel]!,
            14,
          );
        }
      } else {
        // At d=0 the +Infinity extinction lane is an immediate absorbing atom.
        expect(distance).toBe(0);
        expect(collisionScatterWeight).toEqual([0, 0, 0]);
      }
    }

    for (const source of [bsdf, BSDF_BASIC_GLSL]) {
      expect(source).toContain('float fogProposalSurvival(');
      expect(source).toContain('float fogProposalCollisionDensity(');
      expect(source).toContain('vec3 fogFreeFlightSurvivalWeight(');
      expect(source).toContain('vec3 fogFreeFlightCollisionWeight(');
      expect(source).toContain('representedEqualThreeWayProbabilities()');
      expect(source).toContain('dot( channelProbability, survival )');
      expect(source).toContain('dot( channelProbability, density )');
      expect(source).not.toContain('/ 3.0');
    }
    expect(RENDER_MAIN).toContain('float bdptCollisionDensity =');
    expect(bdptLightSubpath).toContain('float survival = fogProposalSurvival(');
    expect(FOG_MATERIAL_GLSL).toContain('! isinf( s15.r )');
  });

  it('forms the RGB collision marginal without overflowing finite f32 lanes', () => {
    // At d=0 each collision density is sigmaT. Every lane below is a valid
    // finite f32 value, but summing the lanes before dividing overflows.
    const maxFiniteF32 = f32(3.4028234663852886e38);
    const lanes = [maxFiniteF32, maxFiniteF32, maxFiniteF32] as const;
    const overflowingMean = f32(
      f32(f32(lanes[0] + lanes[1]) + lanes[2]) / 3,
    );
    const weightedLanes = lanes.map((lane, channel) =>
      f32(lane * REPRESENTED_EQUAL_RGB[channel]!));
    const stableMean = f32(
      f32(f32(weightedLanes[0]! + weightedLanes[1]!) + weightedLanes[2]!),
    );
    expect(overflowingMean).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isFinite(stableMean)).toBe(true);
    expect(stableMean).toBeLessThanOrEqual(maxFiniteF32);
    expect((maxFiniteF32 - stableMean) / maxFiniteF32).toBeLessThan(1e-6);

    // The same ordering remains finite for a tiny positive distance whose
    // rounded Beer survival is still one.
    const tinyDistance = f32(1.401298464324817e-45);
    expect(tinyDistance).toBeGreaterThan(0);
    const densityAtTinyDistance = lanes.map((sigmaT) =>
      f32(sigmaT * f32(Math.exp(f32(-sigmaT * tinyDistance)))));
    const stableTinyMean = f32(densityAtTinyDistance.reduce(
      (sum, lane, channel) =>
        f32(sum + f32(lane * REPRESENTED_EQUAL_RGB[channel]!)),
      0,
    ));
    expect(densityAtTinyDistance.every(Number.isFinite)).toBe(true);
    expect(Number.isFinite(stableTinyMean)).toBe(true);
  });

  it('uses RNG-lattice-realised probabilities for every dynamic transport branch', () => {
    const oneBucket = 1 / RNG_LATTICE_SIZE;
    expect(representedBernoulliProbability(0)).toBe(0);
    expect(representedBernoulliProbability(1)).toBe(1);
    expect(representedBernoulliProbability(2 ** -149)).toBe(oneBucket);
    expect(representedBernoulliProbability(f32(1 - 2 ** -25))).toBe(1);
    expect(REPRESENTED_EQUAL_RGB.reduce((sum, value) => sum + value, 0)).toBe(1);
    expect(REPRESENTED_EQUAL_RGB.every((value) => value > 0)).toBe(true);

    expect(THREE_COMMON_SHIM).toContain(
      'float representedBernoulliProbabilityF32( float probability )',
    );
    expect(THREE_COMMON_SHIM).toContain(
      'void representedCategoricalProbabilities4(',
    );
    expect(bsdf).toContain('representedCategoricalProbabilities4(');
    expect(BSDF_BASIC_GLSL).toContain(
      'diffuseWeight = representedBernoulliProbabilityF32( diffuseWeight );',
    );
    expect(RENDER_MAIN).toContain(
      'rrProb = representedBernoulliProbabilityF32( rrProb );',
    );
    expect(RENDER_MAIN).toContain('rand( 8 ) >= rrProb');
    for (const source of [mappedSurface, GET_SURFACE_RECORD_SCALAR_RICH_GLSL]) {
      expect(source).toContain('representedBernoulliProbabilityF32( coverage )');
    }
    for (const source of [attenuation, ATTENUATE_HIT_SCALAR_RICH_GLSL]) {
      expect(source).toContain('representedBernoulliProbabilityF32( albedo.a )');
    }
  });

  it('consumes one finite medium-distance budget across surfaces and collisions', () => {
    let remaining = 0.5;
    const first = consumeMediumBudget(remaining, 0.2);
    remaining = first.remainingDistance;
    const second = consumeMediumBudget(remaining, 0.4);
    remaining = second.remainingDistance;
    const exhausted = consumeMediumBudget(remaining, 3);
    expect([first.effectiveDistance, second.effectiveDistance]).toEqual([0.2, 0.3]);
    expect(remaining).toBe(0);
    expect(exhausted).toEqual({ effectiveDistance: 0, remainingDistance: 0 });
    expect(
      extinctionTransmittance(2, first.effectiveDistance) *
      extinctionTransmittance(2, second.effectiveDistance),
    ).toBeCloseTo(extinctionTransmittance(2, 0.5), 14);

    // A collision consumes the sampled interior distance; a later segment can
    // use only what remains even though geometry traversal continues normally.
    const collision = consumeMediumBudget(1, 0.25);
    const afterScatter = consumeMediumBudget(collision.remainingDistance, 0.8);
    expect(collision).toEqual({ effectiveDistance: 0.25, remainingDistance: 0.75 });
    expect(afterScatter).toEqual({ effectiveDistance: 0.75, remainingDistance: 0 });

    // Unbounded media keep their sentinel and never form Infinity - Infinity.
    expect(consumeMediumBudget(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY))
      .toEqual({
        effectiveDistance: Number.POSITIVE_INFINITY,
        remainingDistance: Number.POSITIVE_INFINITY,
      });

    expect(FOG_MATERIAL_GLSL).toContain('float mediumEffectiveSegmentDistance(');
    expect(FOG_MATERIAL_GLSL).toContain('bool consumeMediumSegmentDistance(');
    expect(FOG_MATERIAL_GLSL).toContain(
      'stack.attenuationThicknesses[ top ] - effectiveDistance',
    );
    expect(RENDER_MAIN).toContain('consumeMediumSegmentDistance(');
    expect(bdptLightSubpath).toContain('effectiveSolidMediumDistance');
    for (const source of [attenuation, ATTENUATE_HIT_SCALAR_RICH_GLSL]) {
      expect(source).toContain('float effectiveSegmentDistance =');
      expect(source).toContain('consumeMediumSegmentDistance(');
    }
  });

  it('reconciles two partial BDPT histories against one immutable finite cap', () => {
    const initial = 10;
    const eyeRemaining = 8; // eye half consumed 2
    const lightRemaining = 7; // light half consumed 3
    const joined = reconcileBidirectionalMediumBudget(
      initial,
      eyeRemaining,
      lightRemaining,
    );
    expect(joined).toBe(5);
    expect(joined).not.toBe(Math.min(eyeRemaining, lightRemaining));
    expect(reconcileBidirectionalMediumBudget(initial, 1, 2)).toBe(0);
    expect(reconcileBidirectionalMediumBudget(initial, 11, 2)).toBeNull();

    expect(FOG_MATERIAL_GLSL).toContain(
      'float initialAttenuationThicknesses[ MEDIUM_STACK_CAPACITY ];',
    );
    expect(bdptLightSubpath).toContain('out vec4 row12');
    expect(bdptLightSubpath).toContain('out vec4 row13');
    expect(bdptLightSubpath).toContain(
      'stack.initialAttenuationThicknesses[ i ]',
    );
    expect(bdptConnection).toContain('bdptReconcileConnectionMediumStacks(');
    expect(bdptConnection).toContain(
      'float lightConsumed = initialThickness - lightRemaining;',
    );
    expect(bdptConnection).toContain('eyeRemaining - lightConsumed');
    expect(bdptConnection).toContain('lightVisibilityStack');
  });

  it('uses a strict unbiased free-flight ordering at surfaces and exhausted caps', () => {
    const surfaceDistance = 1;
    const nearSurfaceParticle = surfaceDistance - 1e-8;
    expect(nearSurfaceParticle).toBeLessThan(surfaceDistance);
    expect(nearSurfaceParticle < surfaceDistance).toBe(true);
    expect(surfaceDistance < surfaceDistance).toBe(false); // exact tie survives

    // Infinite extinction samples the immediate atom at zero. A positive cap
    // absorbs there, but an exhausted cap has a zero segment and therefore
    // owns survival identity instead of a collision.
    const immediateInfiniteExtinctionParticle = 0;
    expect(immediateInfiniteExtinctionParticle < 0.25).toBe(true);
    expect(immediateInfiniteExtinctionParticle < 0).toBe(false);
    expect(extinctionTransmittance(Number.POSITIVE_INFINITY, 0)).toBe(1);

    expect(traceScene).toContain('float segmentLimit = mediumEffectiveSegmentDistance(');
    expect(traceScene).toContain('if ( particleDist < segmentLimit )');
    expect(traceScene).not.toContain('particleDist + RAY_OFFSET');
  });

  it('classifies a fixed-origin outward segment in exact LIFO order', () => {
    // The launch is inside A. A is observed as an unmatched back before a
    // later disjoint B front/back pair.
    expect(buildContainingStack([
      { materialId: 1, boundaryId: 1, side: -1 },
      { materialId: 2, boundaryId: 2, side: 1 },
      { materialId: 2, boundaryId: 2, side: -1 },
    ])).toEqual([1]);

    // Unmatched backs arrive inner-to-outer and are reversed for the live
    // outer-to-inner stack. Distinct boundaries may share a material slot.
    expect(buildContainingStack([
      { materialId: 7, boundaryId: 12, side: -1 },
      { materialId: 7, boundaryId: 11, side: -1 },
    ])).toEqual([7, 7]);
    expect(buildContainingStack([
      { materialId: 1, boundaryId: 1, side: -1 },
      { materialId: 2, boundaryId: 2, side: -1 },
    ], { capacity: 1 })).toBeNull();
    expect(buildContainingStack([
      { materialId: 1, boundaryId: 1, side: 1 },
      { materialId: 2, boundaryId: 2, side: -1 },
    ])).toBeNull();

    expect(containment).toContain('uint queryLimit = uSceneTriangleCount + 1u;');
    expect(containment).toContain('queryIndex < queryLimit');
    expect(containment).toContain('float minimumDistanceExclusive = 0.0;');
    expect(containment).toContain('minimumDistanceExclusive = boundaryHit.dist;');
    expect(containment).not.toContain('outsideDistance');
    expect(containment).not.toContain('stepRayOrigin(');
    expect(containment).toContain('enterMedium(');
    expect(containment).toContain('if ( pendingCount != 0 ) return false;');
    expect(containment).toContain('containingCount - 1 - outputIndex');
  });

  it('assigns exact welded diagonal, vertex, and concave-edge hits once', () => {
    const planarDiagonal = [
      [[-1, -1, 1], [1, -1, 1], [1, 1, 1]],
      [[-1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    ] as const;
    expect(planarDiagonal.filter((triangle) =>
      includesPositiveZRayHit(triangle)).length).toBe(2);

    const center: Point3 = [0, 0, 1];
    const corners = [
      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ] as const;
    const vertexFan = corners.map((corner, index) =>
      [center, corner, corners[(index + 1) % corners.length]!] as const);
    expect(vertexFan.filter((triangle) =>
      includesPositiveZRayHit(triangle)).length).toBe(4);

    const edgeBottom: Point3 = [0, -1, 1];
    const edgeTop: Point3 = [0, 1, 1];
    const concaveEdge = [
      [edgeBottom, [-1, 0, 2], edgeTop],
      [edgeBottom, edgeTop, [1, 0, 2]],
    ] as const;
    expect(concaveEdge.filter((triangle) =>
      includesPositiveZRayHit(triangle)).length).toBe(2);

    expect(resolveExactTieSides([1, 1, 1, 1])).toBe(1);
    expect(resolveExactTieSides([-1, -1])).toBe(-1);
    expect(resolveExactTieSides([1, -1])).toBe(0); // silhouette tangent edge
    expect(resolveExactTieSides([1, -1, 1, -1])).toBe(0); // tangent vertex fan
    expect(resolveExactTieSides([1, 1, -1])).toBeNull();

    expect(containment).toContain('bool intersectsOpticalBoundaryTriangle(');
    expect(containment).toContain('if ( u < 0.0 || v < 0.0 || w < 0.0 )');
    expect(containment).toContain('float opticalSharedEdgeDistance(');
    expect(containment).toContain('localDist == closestDistance');
    expect(containment).toContain('localComponentId != componentId');
    expect(containment).toContain('frontCount == backCount');
  });

  it('keeps extremely close distinct bulk boundaries and ignores nearby opaque faces', () => {
    const nextF32AfterOne = new Float32Array(
      new Uint32Array([0x3f800001]).buffer,
    )[0]!;
    const representedHits = [
      { t: 0.99999994, optical: false, componentId: 0 },
      { t: 1, optical: true, componentId: 11 },
      { t: nextF32AfterOne, optical: true, componentId: 12 },
    ];
    let minimumExclusive = -1;
    const events: number[] = [];
    while (true) {
      const next = representedHits
        .filter((hit) => hit.optical && hit.t > minimumExclusive)
        .sort((a, b) => a.t - b.t)[0];
      if (next == null) break;
      events.push(next.componentId);
      minimumExclusive = next.t;
    }
    expect(events).toEqual([11, 12]);
    expect(nextF32AfterOne).toBeGreaterThan(1);
    expect(containment).toContain('control.opticalVolume != ( localComponentId != 0u )');
    expect(containment).toContain('! ( localDist > minimumDistanceExclusive )');
  });

  it('keeps stochastic coverage in surface transport and out of bulk topology', () => {
    expect(classifyCoverage(0, 1, 0.5, false, 0.2)).toBe('hole');
    expect(classifyCoverage(0, 1, 0.5, false, 0.8)).toBe('solid');
    expect(classifyCoverage(0, 1, 0, true, 0)).toBe('hole');
    expect(classifyCoverage(0, 1, 0, true, 1)).toBe('solid');
    expect(classifyCoverage(0, 1, 0, true, 0.4)).toBe('fractional');

    for (const surface of [mappedSurface, GET_SURFACE_RECORD_SCALAR_RICH_GLSL]) {
      expect(surface.match(/float evaluateSurfaceCoverage\(/g)).toHaveLength(1);
      expect(surface.match(/evaluateSurfaceCoverage\(/g)).toHaveLength(2);
      expect(surface).toContain('classifySurfaceCoverage(');
      expect(surface.match(/rand\( 3 \)/g)).toHaveLength(1);
    }
    expect(containment).not.toContain('boundaryCoverage');
    expect(containment).not.toContain('evaluateSurfaceCoverage(');
    expect(containment).toContain('control.opticalVolume != ( localComponentId != 0u )');
    expect(containment).not.toContain('transmissionMap');
    expect(containment).not.toContain('rand(');
    expect(FOG_MATERIAL_GLSL).toContain('if ( clampedCoverage <= 0.0 )');
    expect(FOG_MATERIAL_GLSL).toContain('if ( clampedCoverage >= 1.0 )');
  });

  it('resolves both sides of nested dispersive interfaces and mutates only on transmission', () => {
    const outer = { id: 10, ior: 1.4, dispersion: 2_000 };
    const inner = { id: 20, ior: 1.7, dispersion: 8_000 };
    const wavelength = 450;
    expect(interfaceEta([outer], inner, true, true, wavelength)).toBeCloseTo(
      cauchyIor(wavelength, outer.ior, outer.dispersion) /
        cauchyIor(wavelength, inner.ior, inner.dispersion),
      12,
    );
    expect(interfaceEta([outer, inner], inner, false, true, wavelength)).toBeCloseTo(
      cauchyIor(wavelength, inner.ior, inner.dispersion) /
        cauchyIor(wavelength, outer.ior, outer.dispersion),
      12,
    );

    const stack = [10];
    expect(transition(stack, 20, true, true, false)).toBe(true); // reflection
    expect(stack).toEqual([10]);
    expect(transition(stack, 20, true, false, true)).toBe(true); // thin sheet
    expect(stack).toEqual([10]);
    expect(transition(stack, 20, true, true, true)).toBe(true);
    expect(stack).toEqual([10, 20]);
    expect(transition(stack, 20, false, true, true)).toBe(true);
    expect(stack).toEqual([10]);
    expect(transition(stack, 99, false, true, true)).toBe(false);

    expect(bsdf).toContain('incidentIor = opticalMaterialIorAtHero(');
    expect(bsdf).toContain('transmittedIor = opticalMaterialIorAtHero(');
    expect(bsdf).toContain('incidentStack.count - 2');
    expect(bsdf).toContain('surf.eta = incidentIor / transmittedIor;');
  });

  it('keeps the authored IOR-zero endpoint perfectly reflective in float32 transport', () => {
    const effective = effectiveMaterialIor(0);
    expect(effective).toBe(100_000_000);
    const eta = Math.fround(1 / Math.fround(effective));
    const ratio = Math.fround(
      Math.fround(eta - 1) / Math.fround(eta + 1),
    );
    const f0 = Math.fround(ratio * ratio);
    expect(f0).toBe(1);
    expect(Math.fround(1 - f0)).toBe(0);
    expect(bsdf).toContain('surf.f0 = surf.eta == 0.0 ? 1.0 : iorRatioToF0( surf.eta );');
    expect(bsdf).not.toContain('incidentIor = max( incidentIor, 1.0 )');
  });

  it('charges the completed edge to the incident medium before terminal or local shading', () => {
    const beer = RENDER_MAIN.lastIndexOf(
      'state.throughput *= opticalPathSegmentThroughput(',
    );
    const unlit = RENDER_MAIN.indexOf('if ( activeMaterialUnlit )');
    const sampling = RENDER_MAIN.indexOf('scatterRec = bsdfSample(');
    expect(beer).toBeGreaterThanOrEqual(0);
    expect(beer).toBeLessThan(unlit);
    expect(unlit).toBeLessThan(sampling);
    expect(RENDER_MAIN).toContain('pendingSolidOpticalDistance += opticalSegmentDistance;');
    expect(bsdf).toContain('stack.hasAttenuationThicknesses[ top ]');
    expect(bsdf).toContain('stack.attenuationThicknesses[ top ]');
    expect(RENDER_MAIN).not.toContain('useExitSurfaceThickness');
    expect(RENDER_MAIN).not.toContain('activeMaterialCastShadow');
    expect(RENDER_MAIN).not.toContain('state.isShadowRay = rand(');
  });

  it('uses thickness maps only to scale a positive authored Beer-distance cap', () => {
    const start = mappedSurface.indexOf('float evaluateAttenuationThickness(');
    const end = mappedSurface.indexOf('int getSurfaceRecord(', start);
    const evaluator = mappedSurface.slice(start, end);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    expect(evaluator).toContain(
      'hasAttenuationThickness = material.thickness > 0.0;',
    );
    expect(evaluator).not.toContain(
      'material.thickness > 0.0 || material.thicknessMap != - 1',
    );
    expect(evaluator).toContain('float attenuationThickness = material.thickness;');
    expect(evaluator).toContain('vec4 thicknessSample;');
    expect(evaluator).toContain('sampleMappedMaterialTexture(');
    expect(evaluator).toContain(') attenuationThickness *= thicknessSample.g;');
    expect(evaluator).toContain('return max( attenuationThickness, 0.0 );');

    // The glTF texture is multiplicative: it cannot create a cap from a zero
    // factor, and a black texel may locally reduce a real factor to zero while
    // preserving the fact that this entry owns an authored cap.
    expect(0 * 0.75).toBe(0);
    expect(0.35 * 0).toBe(0);
  });

  it('reconstructs the incident stack for NEE replay before evaluating the surface', () => {
    const build = NEE_RESOLVE_MAIN.indexOf('bvhBuildMediumStack(');
    const trace = NEE_RESOLVE_MAIN.indexOf('int hitType = traceScene(');
    const configure = NEE_RESOLVE_MAIN.indexOf('configureSurfaceOpticalInterface(');
    const result = NEE_RESOLVE_MAIN.indexOf('float bsdfPdf = bsdfResult(');
    expect(build).toBeGreaterThanOrEqual(0);
    expect(build).toBeLessThan(trace);
    expect(trace).toBeLessThan(configure);
    expect(configure).toBeLessThan(result);
  });

  it('attenuates active media but never collapses a physical interface in straight visibility', () => {
    for (const source of [attenuation, ATTENUATE_HIT_SCALAR_RICH_GLSL]) {
      const segment = source.indexOf('opticalVisibilitySegmentTransmittance(');
      const blocker = source.lastIndexOf('result = true;');
      expect(source).toContain('filterShadowMediumStack(');
      expect(source).toMatch(/! material\.castShadow && (?:isShadowRay|state\.isShadowRay)/);
      expect(source).toContain('continue;');
      expect(segment).toBeGreaterThanOrEqual(0);
      expect(segment).toBeLessThan(blocker);
      expect(source).not.toContain('enterMedium(');
      expect(source).not.toContain('leaveMedium(');
    }
    expect(FOG_MATERIAL_GLSL).toContain('if ( candidate.castShadow )');
    expect(containment).not.toContain('castShadow');
  });

  it('collapses only an exact-delta compound sheet with one TMM and reciprocal exit eta', () => {
    const incidentAngle = Math.PI / 3;
    const ior = 1.52;
    const entryStoredEta = 1 / ior;
    const effectiveEta = (storedEta: number, localWoZ: number): number =>
      localWoZ >= 0 ? storedEta : 1 / storedEta;
    const entryEta = effectiveEta(entryStoredEta, 1);
    // The virtual exit reverses the basis but retains the stored interface
    // ratio. Its wo is below that basis, so the evaluator reciprocates once.
    const exitEta = effectiveEta(entryStoredEta, -1);
    expect(entryEta).toBeCloseTo(1 / ior, 15);
    expect(exitEta).toBeCloseTo(ior, 15);
    expect(entryEta ** 2 * exitEta ** 2).toBeCloseTo(1, 15);

    const internalAngle = Math.asin(Math.sin(incidentAngle) / ior);
    const exitAngle = Math.asin(ior * Math.sin(internalAngle));
    expect(exitAngle).toBeCloseTo(incidentAngle, 12);
    const incidentDirection = [
      Math.sin(incidentAngle),
      -Math.cos(incidentAngle),
    ] as const;
    const interfaceNormal = [0, 1] as const;
    const internalDirection = refract2d(
      incidentDirection, interfaceNormal, entryEta,
    );
    expect(internalDirection).not.toBeNull();
    const exitDirection = refract2d(
      internalDirection!, interfaceNormal, exitEta,
    );
    expect(exitDirection).not.toBeNull();
    expect(exitDirection![0]).toBeCloseTo(incidentDirection[0], 12);
    expect(exitDirection![1]).toBeCloseTo(incidentDirection[1], 12);

    const criticalAngle = Math.asin(1 / ior);
    const aboveCriticalInternalAngle = criticalAngle + 0.01;
    const tirDirection = [
      Math.sin(aboveCriticalInternalAngle),
      -Math.cos(aboveCriticalInternalAngle),
    ] as const;
    expect(refract2d(tirDirection, interfaceNormal, exitEta)).toBeNull();

    const compoundStart = bsdf.indexOf(
      'ThinSheetTransmissionSample sampleThinSheetTransmission(',
    );
    const compoundEnd = bsdf.indexOf(
      'float bsdfDeltaPdfLocal(',
      compoundStart,
    );
    const compound = bsdf.slice(compoundStart, compoundEnd);
    expect(compound).toContain('oppositeFacingSurface( surf, false )');
    expect(compound).toContain('exitWo lies below the reversed basis');
    expect(compound).toContain('exitSurf.thinFilmEnabled = 0.0;');
    expect(compound).toContain('exitSurf.thinFilmLayerCount = 0.0;');
    expect(compound).toContain('activeLayerThroughput( surf, heroWavelength )');
    expect(compound).toContain('oppositeLayerThroughput( surf, heroWavelength )');

    for (const source of [attenuation, ATTENUATE_HIT_SCALAR_RICH_GLSL]) {
      expect(source).toContain('thinSheetExactVisibilityTransmission(');
      expect(source).toContain('material.roughness == 0.0');
      expect(source).toContain('material.frontLayerRoughness == 0.0');
      expect(source).toContain('material.backLayerRoughness == 0.0');
      expect(source).toContain('color *= sheetAttenuation;');
      expect(source).toContain('setExactRayRangeFromSurfaceHit( ray, surfaceHit )');
    }
    expect(mappedSurface).toContain('bool frontFaceHit = surfaceHit.side == 1.0;');
    expect(GET_SURFACE_RECORD_SCALAR_RICH_GLSL).toContain(
      'bool frontFaceHit = surfaceHit.side == 1.0;',
    );
  });

  it('uses the dynamically resolved bulk eta once on a geometric back face', () => {
    const ior = 1.52;
    // configureSurfaceOpticalInterface resolves a back-face bulk crossing as
    // incident glass / transmitted air. The hit normal is already flipped to
    // face the ray, so local wo.z is positive and no further reciprocal is due.
    const configuredBackFaceEta = ior / 1;
    const localWoZ = 1;
    const evaluatedEta = localWoZ >= 0
      ? configuredBackFaceEta
      : 1 / configuredBackFaceEta;
    expect(evaluatedEta).toBeCloseTo(ior, 15);
    expect(bsdf).toContain('return localWo.z >= 0.0 ? surf.eta : 1.0 / surf.eta;');
    expect(bsdf).toContain('surf.eta = incidentIor / transmittedIor;');
    expect(mappedSurface).toContain('normal *= surfaceHit.side;');
  });

  it('carries exact source features through bulk and sheet continuation without gaps', () => {
    expect(RENDER_MAIN).toContain('exactTransmissionContinuation');
    expect(RENDER_MAIN).toContain('materialControl.opticalVolume || surf.thinFilm');
    expect(RENDER_MAIN).toContain('setExactRayRangeFromSurfaceHit( ray, surfaceHit )');
    expect(RENDER_MAIN).not.toContain('vec3 hitPoint = stepRayOrigin(');
    expect(bdptLightSubpath).toContain('exactPreviousTransmission');
    expect(bdptLightSubpath).toContain('crossedPreviousOpticalBoundary ||');
    expect(bdptLightSubpath).toContain('prevSurf.thinFilm');
    expect(containment).toContain('canonicalizeSelectedSurfaceHit(');
    expect(containment).toContain('sourceFeatureKind = 2u;');
    expect(containment).toContain('sourceFeatureKind = 3u;');
    expect(containment).toContain('opticalBitwiseEqualVec3(');
    expect(containment).toContain('candidatePrimitiveInstanceId != sourcePrimitiveInstanceId');
    expect(containment).toContain('candidateBoundaryId != sourceBoundaryId');
    expect(containment).toContain('! ( localDist > minimumDistanceExclusive )');
  });

  it('uses one canonical watertight edge solve across unlike adjacent faces', () => {
    const sharedA: Point3 = [-1, 0, 5];
    const sharedB: Point3 = [1, 0, 5];
    const compactThird: Point3 = [0, 1, 5];
    const extremeThird: Point3 = [0, -1e30, 5];
    expect(includesPositiveZRayHit([sharedA, sharedB, compactThird])).toBe(true);
    expect(includesPositiveZRayHit([sharedB, sharedA, extremeThird])).toBe(true);

    expect(containment).toContain('float opticalProjectedEdgeFunction(');
    expect(containment).toContain('float opticalRayParameterAtDelta(');
    expect(containment).toContain('vec3 scaledDirection = rayDirection / directionScale;');
    expect(containment).toContain('cross( edge0 / edgeScale, edge1 / edgeScale )');
    expect(containment).not.toContain('coordinateScale');
    expect(containment).not.toContain('TRI_INTERSECT_ANGULAR_EPSILON');
  });

  it('canonicalizes signed zero and rejects tied distinct sheet ranges', () => {
    expect(Object.is(-0, 0)).toBe(false);
    expect(-0 === 0).toBe(true);
    expect(containment).toContain('a.x == 0.0 ? 0u : floatBitsToUint( a.x )');
    expect(containment).toContain('localPrimitiveInstanceId != primitiveInstanceId');
    expect(containment).toContain('uint localPrimitiveInstanceId = identity.b;');
  });

  it('groups the ordinary first ray canonically and fails closed only for transmissive ambiguity', () => {
    const sameSheet = [
      { componentId: 0, representedRangeId: 7, transmissive: true },
      { componentId: 0, representedRangeId: 7, transmissive: true },
    ] as const;
    const distinctSheets = [
      sameSheet[0],
      { componentId: 0, representedRangeId: 8, transmissive: true },
    ] as const;
    const sheetAndOpaque = [
      sameSheet[0],
      { componentId: 0, representedRangeId: 8, transmissive: false },
    ] as const;
    const distinctOpaque = [
      { componentId: 0, representedRangeId: 3, transmissive: false },
      { componentId: 0, representedRangeId: 4, transmissive: false },
    ] as const;
    expect(initialTieIsUnambiguous(sameSheet)).toBe(true);
    expect(initialTieIsUnambiguous(distinctSheets)).toBe(false);
    expect(initialTieIsUnambiguous(sheetAndOpaque)).toBe(false);
    expect(initialTieIsUnambiguous(distinctOpaque)).toBe(true);

    expect(containment).toContain('bool bvhIntersectCanonicalInitialFirstHit(');
    expect(containment).toContain('selectedTransmissive || localTransmissive');
    expect(traceScene).toContain('bvhIntersectCanonicalInitialFirstHit(');
    for (const source of [attenuation, ATTENUATE_HIT_SCALAR_RICH_GLSL]) {
      expect(source).toContain('bvhIntersectCanonicalInitialFirstHit(');
    }
  });

  it('anchors exact continuation on the represented triangle across a one-ULP adjacent layer', () => {
    const firstLayer = f32(1);
    const adjacentLayer = nextPositiveF32(firstLayer);
    expect(adjacentLayer).toBeGreaterThan(firstLayer);

    // A cancellation-heavy independent ray reconstruction does not preserve
    // even the selected layer coordinate. The represented-triangle anchor does.
    const rayOrigin = f32(-16_777_216);
    const rayDistance = f32(firstLayer - rayOrigin);
    const independentlyRounded = f32(rayOrigin + rayDistance);
    expect(independentlyRounded).not.toBe(firstLayer);
    expect(independentlyRounded).not.toBe(adjacentLayer);
    const anchoredFirst = f32(firstLayer + f32(0.25 * f32(firstLayer - firstLayer)));
    const anchoredAdjacent = f32(
      adjacentLayer + f32(0.25 * f32(adjacentLayer - adjacentLayer)),
    );
    expect(anchoredFirst).toBe(firstLayer);
    expect(anchoredAdjacent).toBe(adjacentLayer);

    expect(containment).toContain('vec3 exactOrigin = sourceHit.point;');
    expect(containment).toContain('canonicalPoint = a +');
    expect(containment).toContain('representedP1 - representedP0');
    expect(containment).not.toContain(
      'vec3 exactOrigin = ray.origin + ray.direction * sourceHit.dist;',
    );
  });

  it('stores and reconstructs incident stacks throughout BDPT forward/reverse evaluation', () => {
    expect(bdptLightSubpath).toContain(
      'p5, p6, p7, p8, p9, p10, p11, p12, p13,',
    );
    expect(bdptLightSubpath).toContain('stack.boundaryIds[ i ] = uint( packedBoundaryId );');
    expect(bdptLightSubpath).toContain(
      'stack.attenuationThicknesses[ i ] = max( thicknessCap, 0.0 );',
    );
    expect(bdptLightSubpath).toContain('MediumStack previousIncidentStack = mediumStack;');
    expect(bdptLightSubpath).toContain('previousIncidentStack, prevSurf');
    expect(bdptLightSubpath).toContain('pendingSolidMediumDistance += mediumDistance;');
    expect(bdptLightSubpath).toContain('segmentRatioWeight *= opticalPathSegmentThroughput(');
    expect(bdptLightSubpath).toContain('if ( crossedPreviousOpticalBoundary )');
    expect(bdptConnection).toContain('lightIncidentStack, lightSurf');
    expect(bdptConnection).toContain('firstIncidentStack');
  });

  it('has exactly one reachable bulk-scattering estimator and no dormant surface SSS lane', () => {
    const materialsPacker = readFileSync(
      fileURLToPath(new URL('../scene/materialsTexture.ts', import.meta.url)),
      'utf8',
    );
    expect(bsdf).not.toContain('sssSample(');
    expect(BSDF_BASIC_GLSL).not.toContain('sssSample(');
    expect(bsdf).not.toContain('sampleExponentialDistance(');
    expect(bsdf).not.toContain('TRANSLUCENT_BIT');
    expect(BSDF_BASIC_GLSL).not.toContain('TRANSLUCENT_BIT');
    expect(materialsPacker).not.toContain('TRANSLUCENT_BIT');
    expect(FOG_MATERIAL_GLSL).toContain('fog.sigmaS = max( s16.rgb, vec3( 0.0 ) );');
    expect(FOG_MATERIAL_GLSL).toContain('fog.opacity = sigmaT;');
    expect(bsdf).toContain('vec3 fogFreeFlightSurvivalWeight(');
    expect(bsdf).toContain('vec3 fogFreeFlightCollisionWeight(');
    expect(bsdf).toContain('sampleRec.direction = sampleMediumPhase(');
    expect(RENDER_MAIN).toContain('setFogSurfaceRecord( state.fogMaterial, surf );');
  });

  it('orders the factored coverage evaluator before containment in every composed rich tier', () => {
    const mapped = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
    const scalar = composeTraceGlsl({
      ...DEFAULT_TRACE_FEATURES,
      mappedRichMaterials: false,
      scalarRichMaterials: true,
    });
    for (const source of [mapped, scalar]) {
      const evaluator = source.indexOf('float evaluateSurfaceCoverage(');
      const builder = source.indexOf('bool bvhBuildMediumStack(');
      expect(evaluator).toBeGreaterThanOrEqual(0);
      expect(evaluator).toBeLessThan(builder);
    }
  });
});

/**
 * Production path-replay pass.
 *
 * The certified domain is intentionally small: RGB `MaterialSpec.emissive` on
 * the opaque, triangle-backed primary hit of a one-bounce render. The derivative
 * is analytic:
 *
 *   d rendered.rgb / d emissive.rgb = emissiveIntensity
 *
 * Visibility is replayed with the frozen primary-ray samples. There are no
 * local finite differences, BSDF derivatives, light estimators, emitter
 * derivatives, or transport derivatives in this shader. Unsupported requests
 * are rejected by InverseSession and AdjointPass before pipeline creation.
 */
import {
  composePtWebgpuRngWgsl,
  type PtWebgpuSamplingMode,
} from '../common.wgsl.js';
import { MOLLER_TRUMBORE_WGSL } from '@vitrum/shared-bvh';

/** Sole field code accepted by the production descriptor validator and shader. */
export const ADJOINT_FIELD_EMISSIVE = 2;

/** mat4 + camera vec4 + two u32 vec4 rows. */
export const ADJOINT_PARAMS_UBO_BYTES = 64 + 16 + 16 + 16;

export function composePtWebgpuAdjointPassWgsl(
  sampling: PtWebgpuSamplingMode = 'pcg',
): string {
  return /* wgsl */ `
const MATERIAL_VEC4_STRIDE = 29u;
const ADJOINT_FIELD_EMISSIVE = ${ADJOINT_FIELD_EMISSIVE}u;
const ADJOINT_FROZEN_SEED_BASE = 0x5eed5eedu;

struct AdjointParams {
  invViewProj: mat4x4f,
  cameraPos: vec4f,
  width: u32,
  height: u32,
  triangleCount: u32,
  paramCount: u32,
  channels: u32,
  sampleCount: u32,
  gradientScale: f32,
  _pad1: u32,
}

struct Ray {
  origin: vec3f,
  direction: vec3f,
}

struct Hit {
  valid: bool,
  t: f32,
  triangle: u32,
}

${MOLLER_TRUMBORE_WGSL}

@group(0) @binding(0) var<uniform> params: AdjointParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> indices: array<vec4u>;
@group(0) @binding(3) var<storage, read> triMaterialIds: array<vec2u>;
@group(0) @binding(4) var<storage, read> materials: array<vec4f>;
@group(0) @binding(5) var<storage, read> dLossDRendered: array<f32>;
@group(0) @binding(6) var<storage, read_write> gradAccum: array<atomic<i32>>;
// One vec4u per parameter:
// { materialId, ADJOINT_FIELD_EMISSIVE, gradientOffset, emissiveIntensityBits }.
@group(0) @binding(7) var<storage, read> adjointParamDescs: array<vec4u>;

${composePtWebgpuRngWgsl(sampling)}

fn safeNormalize(v: vec3f) -> vec3f {
  let scale = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (!(scale > 0.0) || scale > 3.402823e38) {
    return vec3f(0.0, 1.0, 0.0);
  }
  let scaled = v / scale;
  return scaled / length(scaled);
}

fn generatePrimaryRay(px: u32, py: u32, jitter: vec2f) -> Ray {
  let uv =
    (vec2f(f32(px), f32(py)) + jitter) /
    vec2f(f32(params.width), f32(params.height));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  var farH = params.invViewProj * vec4f(ndc, 1.0, 1.0);
  var nearH = params.invViewProj * vec4f(ndc, -1.0, 1.0);
  var ray: Ray;
  ray.origin = vec3f(0.0);
  ray.direction = vec3f(0.0);
  let farScale =
    max(max(abs(farH.x), abs(farH.y)), max(abs(farH.z), abs(farH.w)));
  let nearScale =
    max(max(abs(nearH.x), abs(nearH.y)), max(abs(nearH.z), abs(nearH.w)));
  if (
    !(farScale > 0.0) || farScale > 3.402823e38 ||
    !(nearScale > 0.0) || nearScale > 3.402823e38
  ) {
    return ray;
  }
  farH /= farScale;
  nearH /= nearScale;
  if (nearH.w == 0.0) {
    return ray;
  }
  let nearPoint = nearH.xyz / nearH.w;
  var orientation = 1.0;
  if (farH.w != 0.0) {
    orientation = sign(farH.w * nearH.w);
  }
  let directionNumerator =
    (farH.xyz * nearH.w - nearH.xyz * farH.w) * orientation;
  let directionScale = max(
    abs(directionNumerator.x),
    max(abs(directionNumerator.y), abs(directionNumerator.z)),
  );
  if (
    !all(nearPoint == nearPoint) ||
    any(abs(nearPoint) > vec3f(3.402823e38)) ||
    !(directionScale > 0.0) ||
    directionScale > 3.402823e38
  ) {
    return ray;
  }
  ray.origin = nearPoint;
  ray.direction = safeNormalize(directionNumerator);
  return ray;
}

fn materialDoubleSided(materialId: u32) -> bool {
  if (materialId > 0xffffffffu / MATERIAL_VEC4_STRIDE) {
    return false;
  }
  let materialBase = materialId * MATERIAL_VEC4_STRIDE;
  if (
    materialBase > arrayLength(&materials) ||
    MATERIAL_VEC4_STRIDE > arrayLength(&materials) - materialBase
  ) {
    return false;
  }
  let flagsValue = materials[materialBase + 26u].w;
  if (
    flagsValue != flagsValue ||
    abs(flagsValue) > 3.402823466e38 ||
    flagsValue < 0.0 ||
    flagsValue != floor(flagsValue) ||
    flagsValue >= 8.0
  ) {
    return false;
  }
  let flags = u32(flagsValue);
  return (flags & 4u) != 0u;
}

// Brute-force replay is deliberate: transformed/skinned/instanced scenes are
// flattened to the same world-space triangle stream by the host. Analytic
// primitives are rejected before this function can be dispatched.
fn closestOpaqueTriangleHit(ray: Ray) -> Hit {
  var best: Hit;
  best.valid = false;
  best.t = 3.402823466e38;
  best.triangle = 0u;

  for (var triangle = 0u; triangle < params.triangleCount; triangle += 1u) {
    if (triangle >= arrayLength(&indices) || triangle >= arrayLength(&triMaterialIds)) {
      break;
    }
    let idx = indices[triangle];
    if (
      idx.x >= arrayLength(&positions) ||
      idx.y >= arrayLength(&positions) ||
      idx.z >= arrayLength(&positions)
    ) {
      continue;
    }

    let v0 = positions[idx.x].xyz;
    let triHit = mollerTrumboreCore(
      ray.origin,
      ray.direction,
      v0,
      positions[idx.y].xyz,
      positions[idx.z].xyz,
      0.0,
    );
    if (!triHit.hit) {
      continue;
    }

    // Forward traversal ignores opaque single-sided back faces. Transmissive
    // back faces are outside the certified domain and rejected on the CPU.
    let materialId = triMaterialIds[triangle].x;
    if (triHit.det < 0.0 && !materialDoubleSided(materialId)) {
      continue;
    }

    if (triHit.t > 0.0 && triHit.t < best.t) {
      best.valid = true;
      best.t = triHit.t;
      best.triangle = triangle;
    }
  }
  return best;
}

fn adjointScatter(index: u32, value: f32) {
  if (
    index >= arrayLength(&gradAccum) ||
    value != value ||
    abs(value) > 3.402823e38
  ) {
    return;
  }
  let scaled = value * params.gradientScale;
  let rounded = round(scaled);
  if (
    rounded != rounded ||
    rounded < -2147483648.0 ||
    rounded > 2147483520.0
  ) {
    return;
  }
  atomicAdd(&gradAccum[index], i32(rounded));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let pixel = gid.y * params.width + gid.x;
  let lossBase = pixel * params.channels;
  if (lossBase + 2u >= arrayLength(&dLossDRendered)) {
    return;
  }
  let dLoss = vec3f(
    dLossDRendered[lossBase],
    dLossDRendered[lossBase + 1u],
    dLossDRendered[lossBase + 2u],
  );

  let replaySamples = max(params.sampleCount, 1u);
  let invReplaySamples = 1.0 / f32(replaySamples);
  // Aggregate all replay samples locally, then perform one atomic add per pixel,
  // parameter and channel. This bounds fixed-point rounding accumulation by the
  // pixel count rather than pixelCount * sampleCount.
  for (var parameter = 0u; parameter < params.paramCount; parameter += 1u) {
    if (parameter >= arrayLength(&adjointParamDescs)) {
      break;
    }
    let descriptor = adjointParamDescs[parameter];
    if (descriptor.y != ADJOINT_FIELD_EMISSIVE) {
      continue;
    }
    let emissiveIntensity = bitcast<f32>(descriptor.w);
    var matchingHitCount = 0u;
    for (var sampleIndex = 0u; sampleIndex < replaySamples; sampleIndex += 1u) {
      let frameSeed = ADJOINT_FROZEN_SEED_BASE + sampleIndex;
      var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(frameSeed, 0u));
      let ray = generatePrimaryRay(
        gid.x,
        gid.y,
        vec2f(rand_f32(&rng), rand_f32(&rng)),
      );
      let hit = closestOpaqueTriangleHit(ray);
      if (
        hit.valid &&
        triMaterialIds[hit.triangle].x == descriptor.x
      ) {
        matchingHitCount += 1u;
      }
    }
    // The emissive partial is constant for every matching primary hit. Count
    // hits exactly in u32 and perform one f32 ratio/product instead of repeatedly
    // accumulating O(sampleCount) rounded f32 contributions.
    let hitFraction = f32(matchingHitCount) * invReplaySamples;
    let gradient = dLoss * emissiveIntensity * hitFraction;
    let gradientOffset = descriptor.z;
    if (gradientOffset > 0xfffffffdu) { continue; }
    adjointScatter(gradientOffset, gradient.x);
    adjointScatter(gradientOffset + 1u, gradient.y);
    adjointScatter(gradientOffset + 2u, gradient.z);
  }
}
`;
}

/** Default PCG composition used by the engine and static shader gates. */
export const PT_WEBGPU_ADJOINT_PASS_WGSL = composePtWebgpuAdjointPassWgsl();

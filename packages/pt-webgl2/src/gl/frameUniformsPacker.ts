// frameUniformsPacker — pure builder of the per-frame `FrameUniforms` payload for
// the pt-webgl2 accumulation draw (extracted from index.ts #frameUniforms, T3-D /
// D11-1). The caller supplies its config + scene state as a plain record so this
// module stays free of the engine class.

import type { FrameInput, Scene } from '@vitrum/core';
import type { FrameUniforms } from './glResources.js';
import type { PTEngineWebGL2Options } from '../options.js';
import {
  canonicalizeEnvironmentRotationF32,
  TONEMAP_MODE_INDEX,
} from '@vitrum/shared-samplers';
import { invertMat4, makeRotationYMat4 } from '../mat4.js';

import { sharedBdptWavelengthForSeed } from './sharedBdptWavelength.js';
import {
  validateWebgl2CameraTransportDomain,
  webgl2RayOriginBias,
  type Webgl2CoordinateBounds,
  type Webgl2TransportBounds,
} from '../scene/sceneScalePolicy.js';
import {
  WEBGL2_F32_MAX,
  multiplyNonNegativeFloat32,
  requireFiniteFloat32,
  requireNonNegativeFloat32,
} from '../scene/float32Policy.js';
const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const WEBGL2_F32_MIN_SUBNORMAL = 2 ** -149;
const WEBGL2_F32_ARITHMETIC_EPSILON = 1.192092896e-7;
type Vec4 = readonly [number, number, number, number];
type Vec3 = readonly [number, number, number];
type PackedDofUniforms = Exclude<FrameUniforms['dof'], null>;
type SampleIndependentFrameUniforms = Omit<
  FrameUniforms,
  'bdptSharedWavelengthNm' | 'bdptSharedWavelengthPdf'
>;

/** Engine-config + scene-state slice consumed by `packFrameUniforms`. */
export interface FrameUniformsConfig {
  readonly scene: Scene | null;
  readonly hasEnvMap: boolean;
  readonly materialLodDepth: number;
  readonly backgroundBlur: number;
  readonly spectralEnabled: boolean;
  readonly backgroundAlpha: number;
  readonly bdpt: boolean;
  readonly bdptMaxLightBounces: number;
  readonly cameraType: 0 | 1 | 2;
  readonly transportBounds: Webgl2TransportBounds;
  readonly dof: PTEngineWebGL2Options['dof'];
}

/**
 * Fully validated, sample-independent payload. The renderer computes this
 * before program preparation or target allocation; only the deterministic
 * shared BDPT wavelength remains to be appended after accumulation resets.
 */
export interface FrameUniformsPreflight {
  readonly frameSeed: number;
  readonly uniforms: SampleIndependentFrameUniforms;
}

function multiplyMat4Vec4F32(
  matrix: Float32Array,
  vector: Vec4,
  context: string,
): Vec4 {
  const packedVector = vector.map((component, index) =>
    requireFiniteFloat32(component, `${context} input[${index}]`),
  ) as [number, number, number, number];
  const result = [0, 0, 0, 0] as [number, number, number, number];
  for (let row = 0; row < 4; row += 1) {
    const products = [0, 1, 2, 3].map((column) =>
      Math.fround(
        matrix[column * 4 + row]! * packedVector[column]!,
      ),
    );
    const absoluteTermSum = products.reduce(
      (sum, product) => sum + Math.abs(product),
      0,
    );
    requireFiniteFloat32(
      absoluteTermSum,
      `${context} row ${row} absolute term sum`,
    );
    let sum = products[0]!;
    for (let column = 1; column < 4; column += 1) {
      sum = Math.fround(sum + products[column]!);
      if (!Number.isFinite(sum)) {
        throw new RangeError(
          `${context} row ${row} overflows shader float32 addition.`,
        );
      }
    }
    result[row] = sum;
  }
  return result;
}

function divideHomogeneousOrigin(
  homogeneous: Vec4,
  context: string,
): readonly [number, number, number] {
  const w = homogeneous[3];
  if (w === 0) {
    throw new RangeError(
      `${context} has zero homogeneous w and cannot define a primary-ray origin.`,
    );
  }
  return [0, 1, 2].map((axis) =>
    requireFiniteFloat32(
      Math.fround(homogeneous[axis]! / w),
      `${context}[${axis}]`,
    ),
  ) as [number, number, number];
}

function requireFiniteNonZeroDirection(
  direction: readonly [number, number, number],
  context: string,
): void {
  const scale = Math.max(...direction.map(Math.abs));
  if (!(scale > 0) || scale > WEBGL2_F32_MAX) {
    throw new RangeError(`${context} must remain finite and non-zero in float32.`);
  }
}

function float32Gamma(operationCount: number): number {
  return (
    operationCount * WEBGL2_F32_ARITHMETIC_EPSILON /
    (1 - operationCount * WEBGL2_F32_ARITHMETIC_EPSILON)
  );
}

/**
 * Round a non-negative real upper bound outward to the next binary32 value
 * when nearest-even rounding would move it inward.
 */
function outwardFloat32Upper(value: number, context: string): number {
  const rounded = requireNonNegativeFloat32(value, context);
  if (!(rounded < value)) return rounded;

  const storage = new ArrayBuffer(4);
  const view = new DataView(storage);
  view.setFloat32(0, rounded, false);
  view.setUint32(0, view.getUint32(0, false) + 1, false);
  const outward = view.getFloat32(0, false);
  if (!Number.isFinite(outward)) {
    throw new RangeError(`${context} overflows its conservative float32 bound.`);
  }
  return outward;
}

function packDofUniforms(
  dof: PTEngineWebGL2Options['dof'],
): PackedDofUniforms | null {
  if (dof == null) return null;

  const focusDistance = requireNonNegativeFloat32(
    dof.focusDistance,
    'pt-webgl2 dof.focusDistance',
  );
  if (!(focusDistance > 0)) {
    throw new RangeError('pt-webgl2 dof.focusDistance must be positive.');
  }
  const bokehSize = requireNonNegativeFloat32(
    dof.bokehSize,
    'pt-webgl2 dof.bokehSize',
  );
  const anamorphicRatio = requireNonNegativeFloat32(
    dof.anamorphicRatio ?? 1,
    'pt-webgl2 dof.anamorphicRatio',
  );
  if (!(anamorphicRatio > 0)) {
    throw new RangeError('pt-webgl2 dof.anamorphicRatio must be positive.');
  }

  // Mirror camera_util_functions' branch: ratios below one never evaluate a
  // reciprocal; ratios at/above one produce a finite squeeze in (0, 1].
  if (bokehSize > 0 && anamorphicRatio >= 1) {
    const squeeze = requireFiniteFloat32(
      1 / anamorphicRatio,
      'pt-webgl2 dof.anamorphic squeeze',
    );
    if (!(squeeze > 0 && squeeze <= 1)) {
      throw new RangeError(
        'pt-webgl2 dof.anamorphic squeeze must remain in (0, 1].',
      );
    }
  }

  return {
    focusDistance,
    bokehSize,
    apertureBlades: dof.apertureBlades ?? 0,
    apertureRotation: requireFiniteFloat32(
      dof.apertureRotation ?? 0,
      'pt-webgl2 dof.apertureRotation',
    ),
    anamorphicRatio,
  };
}

/**
 * Bound the exact active-DOF chain in camera_util_functions:
 *
 * sampleAperture -> (bokehSize * 0.5) * f32(1e-3) -> rotateVector
 * -> bounded anamorphic branch -> cameraWorldMatrix * vec4(sample, 0, 0).
 *
 * The scale's two shader multiplications are mirrored separately. The
 * remaining forward-error factors include polygon/circle sampling, two-term
 * rotation and matrix dot products, and an absolute subnormal allowance.
 * A rotation whose independently rounded sin/cos components each reach one
 * has operator norm at most sqrt(2), so the bound does not assume exact
 * trigonometric orthogonality.
 */
function dofLensExtentBounds(
  cameraWorldMatrix: Float32Array,
  dof: PackedDofUniforms,
): Vec3 {
  if (!(dof.bokehSize > 0)) return [0, 0, 0];

  const halfBokehSize = multiplyNonNegativeFloat32(
    dof.bokehSize,
    0.5,
    'pt-webgl2 realized DOF half-bokeh scale',
  );
  const apertureScale = multiplyNonNegativeFloat32(
    halfBokehSize,
    Math.fround(1e-3),
    'pt-webgl2 realized DOF aperture scale',
  );

  const apertureSampleNormUpper =
    1 + float32Gamma(12);
  const trigComponentUpper =
    1 + WEBGL2_F32_ARITHMETIC_EPSILON;
  const rotatedApertureNormUpper = outwardFloat32Upper(
    apertureScale *
      apertureSampleNormUpper *
      Math.SQRT2 *
      trigComponentUpper *
      (1 + float32Gamma(4)) *
      (1 + float32Gamma(1)) +
      12 * WEBGL2_F32_MIN_SUBNORMAL,
    'pt-webgl2 rotated DOF aperture norm',
  );

  return [0, 1, 2].map((axis) => {
    const cameraBasisLength = Math.hypot(
      cameraWorldMatrix[axis]!,
      cameraWorldMatrix[4 + axis]!,
    );
    return outwardFloat32Upper(
      rotatedApertureNormUpper *
        cameraBasisLength *
        (1 + float32Gamma(2)) +
        4 * WEBGL2_F32_MIN_SUBNORMAL,
      `pt-webgl2 DOF lens-origin extent[${axis}]`,
    );
  }) as [number, number, number];
}

/**
 * Mirror the scale-stable refocus subtraction in camera_util_functions. For
 * every sample its common scale is max(focusDistance, maxAbs(lensOffset)), so
 * both division results are finite and at most one in magnitude. The
 * subtraction therefore remains bounded independently of the authored
 * aperture/focus ratio; no physical-camera restriction is imposed here.
 */
function validateDofScaledFocusArithmetic(
  dof: PackedDofUniforms,
  lensExtents: Vec3,
): void {
  if (!(dof.bokehSize > 0)) return;

  const maximumLensComponent = Math.max(...lensExtents);
  const maximumCommonScale = requireFiniteFloat32(
    Math.max(dof.focusDistance, maximumLensComponent),
    'pt-webgl2 DOF relative-focus common scale bound',
  );
  if (!(maximumCommonScale > 0)) {
    throw new RangeError(
      'pt-webgl2 DOF relative-focus common scale must remain positive.',
    );
  }

  // baseDirection is the result of vitrumNormalizeVec3, so allow its complete
  // normalization forward error plus one multiply and one subtraction.
  outwardFloat32Upper(
    (
      (1 + float32Gamma(8)) *
        (1 + float32Gamma(1)) +
      1
    ) *
      (1 + float32Gamma(1)) +
      WEBGL2_F32_MIN_SUBNORMAL,
    'pt-webgl2 scaled DOF relative focus-direction component bound',
  );
}

/**
 * Prove the packed camera 3x3 maps every unit direction to a finite, non-zero
 * float32 vector. The forward error bound is the standard γ₃ dot-product bound;
 * comparing it with 1/||A⁻¹||F rejects only transforms for which float32 cannot
 * prove a non-zero result after cancellation.
 */
function validateCameraDirectionTransform(cameraWorldMatrix: Float32Array): void {
  const rows = [0, 1, 2].map((row) => [
    cameraWorldMatrix[row]!,
    cameraWorldMatrix[4 + row]!,
    cameraWorldMatrix[8 + row]!,
  ] as const);
  const unitRoundoff = 1.192092896e-7;
  const gamma3 = (3 * unitRoundoff) / (1 - 3 * unitRoundoff);
  const rowNorms = rows.map((row, index) =>
    requireFiniteFloat32(
      Math.hypot(...row) * (1 + gamma3),
      `pt-webgl2 camera direction transform row ${index} bound`,
    ),
  );
  const frobeniusNorm = Math.hypot(...rows.flat());

  const a00 = rows[0]![0], a01 = rows[0]![1], a02 = rows[0]![2];
  const a10 = rows[1]![0], a11 = rows[1]![1], a12 = rows[1]![2];
  const a20 = rows[2]![0], a21 = rows[2]![1], a22 = rows[2]![2];
  const c00 = a11 * a22 - a12 * a21;
  const c01 = a02 * a21 - a01 * a22;
  const c02 = a01 * a12 - a02 * a11;
  const c10 = a12 * a20 - a10 * a22;
  const c11 = a00 * a22 - a02 * a20;
  const c12 = a02 * a10 - a00 * a12;
  const c20 = a10 * a21 - a11 * a20;
  const c21 = a01 * a20 - a00 * a21;
  const c22 = a00 * a11 - a01 * a10;
  const determinant = a00 * c00 + a01 * c10 + a02 * c20;
  if (!Number.isFinite(determinant) || determinant === 0) {
    throw new RangeError(
      'pt-webgl2 camera direction transform is singular after float32 packing.',
    );
  }
  const inverseFrobeniusNorm = Math.hypot(
    c00 / determinant, c01 / determinant, c02 / determinant,
    c10 / determinant, c11 / determinant, c12 / determinant,
    c20 / determinant, c21 / determinant, c22 / determinant,
  );
  const minimumOutputBound = 1 / inverseFrobeniusNorm;
  const roundingErrorBound = gamma3 * frobeniusNorm;
  if (
    !Number.isFinite(minimumOutputBound) ||
    minimumOutputBound < WEBGL2_F32_MIN_SUBNORMAL ||
    !(minimumOutputBound > roundingErrorBound) ||
    rowNorms.some((bound) => bound >= WEBGL2_F32_MAX)
  ) {
    throw new RangeError(
      'pt-webgl2 camera direction transform cannot preserve every unit ' +
      'direction as a finite non-zero float32 vector.',
    );
  }
}

function validatePrimaryDirectionDomain(
  cameraWorldMatrix: Float32Array,
  invProjectionMatrix: Float32Array,
  cameraType: 0 | 1 | 2,
  ndcX: readonly [number, number],
  ndcY: readonly [number, number],
  requireLensPlaneSeparation: boolean,
): void {
  // An invertible, numerically resolvable camera 3x3 also proves that its lens
  // plane (columns X/Y) cannot contain the local-Z direction component.
  if (cameraType === 2 || requireLensPlaneSeparation) {
    validateCameraDirectionTransform(cameraWorldMatrix);
  }
  const cameraForward4 = multiplyMat4Vec4F32(
    cameraWorldMatrix,
    [0, 0, -1, 0],
    'pt-webgl2 camera-forward direction',
  );
  requireFiniteNonZeroDirection(
    [cameraForward4[0], cameraForward4[1], cameraForward4[2]],
    'pt-webgl2 camera-forward direction',
  );
  if (cameraType !== 0) return;

  const perspectiveDirections: Vec3[] = [];
  const perspectiveViewDirections: Vec3[] = [];
  for (const x of ndcX) {
    for (const y of ndcY) {
      const viewDirection = multiplyMat4Vec4F32(
        invProjectionMatrix,
        [x, y, 0, 1],
        'pt-webgl2 inverse-projection direction numerator',
      );
      const worldDirection = multiplyMat4Vec4F32(
        cameraWorldMatrix,
        [viewDirection[0], viewDirection[1], viewDirection[2], 0],
        'pt-webgl2 perspective direction numerator',
      );
      perspectiveViewDirections.push([
        viewDirection[0],
        viewDirection[1],
        viewDirection[2],
      ]);
      const xyz = [
        worldDirection[0],
        worldDirection[1],
        worldDirection[2],
      ] as const;
      requireFiniteNonZeroDirection(
        xyz,
        'pt-webgl2 perspective direction numerator',
      );
      perspectiveDirections.push(xyz);
    }
  }
  const arithmeticEpsilon = 1.192092896e-7;
  const gamma3 =
    (3 * arithmeticEpsilon) / (1 - 3 * arithmeticEpsilon);
  const gamma4 =
    (4 * arithmeticEpsilon) / (1 - 4 * arithmeticEpsilon);
  const maximumNdcX = Math.max(...ndcX.map(Math.abs));
  const maximumNdcY = Math.max(...ndcY.map(Math.abs));
  const viewDirectionBounds = [0, 1, 2].map((row) =>
    Math.abs(invProjectionMatrix[row]!) * maximumNdcX +
    Math.abs(invProjectionMatrix[4 + row]!) * maximumNdcY +
    Math.abs(invProjectionMatrix[12 + row]!),
  );
  const viewDirectionErrors = viewDirectionBounds.map(
    (bound) => gamma4 * bound,
  );
  if (requireLensPlaneSeparation) {
    const localZMinimum = Math.min(
      ...perspectiveViewDirections.map((direction) => direction[2]),
    );
    const localZMaximum = Math.max(
      ...perspectiveViewDirections.map((direction) => direction[2]),
    );
    const localZError =
      viewDirectionErrors[2]! + WEBGL2_F32_MIN_SUBNORMAL;
    if (
      !(
        localZMinimum > localZError ||
        localZMaximum < -localZError
      )
    ) {
      throw new RangeError(
        'pt-webgl2 perspective direction domain crosses the camera lens plane.',
      );
    }
  }
  const hasStableNonZeroComponent = [0, 1, 2].some((axis) => {
    const minimum = Math.min(
      ...perspectiveDirections.map((direction) => direction[axis]!),
    );
    const maximum = Math.max(
      ...perspectiveDirections.map((direction) => direction[axis]!),
    );
    const worldAbsoluteTermBound = [0, 1, 2].reduce(
      (sum, column) =>
        sum +
        Math.abs(cameraWorldMatrix[column * 4 + axis]!) *
          viewDirectionBounds[column]!,
      0,
    );
    const propagatedViewError = [0, 1, 2].reduce(
      (sum, column) =>
        sum +
        Math.abs(cameraWorldMatrix[column * 4 + axis]!) *
          viewDirectionErrors[column]!,
      0,
    );
    const errorBound =
      propagatedViewError + gamma3 * worldAbsoluteTermBound;
    requireFiniteFloat32(
      worldAbsoluteTermBound + errorBound,
      `pt-webgl2 perspective direction domain component ${axis} bound`,
    );
    const nonZeroMargin = errorBound + WEBGL2_F32_MIN_SUBNORMAL;
    return (
      minimum > nonZeroMargin ||
      maximum < -nonZeroMargin
    );
  });
  if (!hasStableNonZeroComponent) {
    throw new RangeError(
      'pt-webgl2 perspective direction domain cannot prove a finite non-zero ' +
      'float32 numerator across the jittered viewport.',
    );
  }
}

function primaryRayOriginBounds(
  cameraWorldMatrix: Float32Array,
  invProjectionMatrix: Float32Array,
  cameraType: 0 | 1 | 2,
  width: number,
  height: number,
  dof: PackedDofUniforms | null,
): Webgl2CoordinateBounds {
  const ndcX = [
    Math.fround(-1 - 1 / width),
    Math.fround(1 + 1 / width),
  ] as const;
  const ndcY = [
    Math.fround(-1 - 1 / height),
    Math.fround(1 + 1 / height),
  ] as const;
  validatePrimaryDirectionDomain(
    cameraWorldMatrix,
    invProjectionMatrix,
    cameraType,
    ndcX,
    ndcY,
    dof != null && dof.bokehSize > 0,
  );
  let origins: Array<readonly [number, number, number]>;
  if (cameraType === 2) {
    const worldOrigin = multiplyMat4Vec4F32(
      cameraWorldMatrix,
      [0, 0, 0, 1],
      'pt-webgl2 equirectangular camera origin',
    );
    origins = [
      divideHomogeneousOrigin(
        worldOrigin,
        'pt-webgl2 equirectangular camera origin',
      ),
    ];
  } else {
    // gl_FragCoord spans [0.5, size-0.5] and tentFilter spans [-1, 1].
    // Therefore the full logical NDC domain, including tiled NEE passes, is
    // [-1-1/size, 1+1/size] on each axis.
    const homogeneousOrigins: Vec4[] = [];
    origins = [];
    for (const x of ndcX) {
      for (const y of ndcY) {
        const viewOrigin = multiplyMat4Vec4F32(
          invProjectionMatrix,
          [x, y, -1, 1],
          'pt-webgl2 inverse-projection camera origin',
        );
        const worldOrigin = multiplyMat4Vec4F32(
          cameraWorldMatrix,
          viewOrigin,
          'pt-webgl2 world-space camera origin',
        );
        homogeneousOrigins.push(worldOrigin);
        origins.push(
          divideHomogeneousOrigin(
            worldOrigin,
            'pt-webgl2 primary-ray origin',
          ),
        );
      }
    }
    const homogeneousSign = Math.sign(homogeneousOrigins[0]![3]);
    if (
      homogeneousSign === 0 ||
      homogeneousOrigins.some(
        (origin) => Math.sign(origin[3]) !== homogeneousSign,
      )
    ) {
      throw new RangeError(
        'pt-webgl2 primary-ray origin domain crosses zero homogeneous w.',
      );
    }
  }

  const min = [0, 1, 2].map((axis) =>
    Math.min(...origins.map((origin) => origin[axis]!)),
  ) as [number, number, number];
  const max = [0, 1, 2].map((axis) =>
    Math.max(...origins.map((origin) => origin[axis]!)),
  ) as [number, number, number];

  if (dof != null && dof.bokehSize > 0) {
    const lensExtents = dofLensExtentBounds(cameraWorldMatrix, dof);
    for (let axis = 0; axis < 3; axis += 1) {
      const lensExtent = lensExtents[axis]!;
      min[axis] = requireFiniteFloat32(
        min[axis]! - lensExtent,
        `pt-webgl2 DOF lens-origin minimum[${axis}]`,
      );
      max[axis] = requireFiniteFloat32(
        max[axis]! + lensExtent,
        `pt-webgl2 DOF lens-origin maximum[${axis}]`,
      );
    }
    validateDofScaledFocusArithmetic(dof, lensExtents);
  }

  return { min, max };
}

/**
 * Validate and pack every sample-independent frame value. This is intentionally
 * pure and must run before any program/target preparation.
 */
export function preflightFrameUniforms(
  input: FrameInput,
  bounces: number,
  w: number,
  h: number,
  cfg: FrameUniformsConfig,
): FrameUniformsPreflight {
  const cameraWorldMatrix = invertMat4(input.viewMatrix);
  const invProjectionMatrix = invertMat4(input.projMatrix);
  if (cameraWorldMatrix == null || invProjectionMatrix == null) {
    throw new Error('renderFrame: singular view/projection matrix');
  }
  const packedDof = packDofUniforms(cfg.dof);
  if (
    packedDof != null &&
    packedDof.bokehSize > 0 &&
    cfg.cameraType === 2
  ) {
    throw new RangeError(
      'pt-webgl2 active dof is unsupported for an equirectangular camera.',
    );
  }
  const cameraOrigins = primaryRayOriginBounds(
    cameraWorldMatrix,
    invProjectionMatrix,
    cfg.cameraType,
    w,
    h,
    packedDof,
  );
  validateWebgl2CameraTransportDomain(cameraOrigins, cfg.transportBounds);
  // H6 FIX (2026-06-09): honour the HDRI environment's `intensity` contract field
  // (was hardcoded to 1, so `environment.intensity` was silently ignored).
  // Mirrors pt-webgpu (environmentPacking.ts:54: `env.intensity ?? 1`).
  const env = cfg.scene?.environment;
  const envIntensity =
    env != null && env.kind === 'hdri'
      ? requireNonNegativeFloat32(
          env.intensity ?? 1,
          'pt-webgl2 HDRI environment intensity',
        )
      : 1;
  const packedSceneCenter = [0, 1, 2].map((axis) =>
    requireFiniteFloat32(
      cfg.transportBounds.center[axis]!,
      `pt-webgl2 transport center[${axis}]`,
    ),
  ) as [number, number, number];
  const packedSceneRadius = requireNonNegativeFloat32(
    cfg.transportBounds.radius,
    'pt-webgl2 transport radius',
  );
  if (!(packedSceneRadius > 0)) {
    throw new RangeError('pt-webgl2 transport radius must be positive.');
  }
  const environmentRotationY =
    env?.kind === 'hdri'
      ? canonicalizeEnvironmentRotationF32(
          env.rotationY ?? 0,
          'pt-webgl2 HDRI environment rotationY',
        )
      : 0;
  const uniforms: SampleIndependentFrameUniforms = {
    resolution: [w, h],
    bounces,
    transmissiveBounces: bounces,
    filterGlossyFactor: requireNonNegativeFloat32(
      input.quality?.filteredGlossyFactor ?? 0,
      'pt-webgl2 filteredGlossyFactor',
    ),
    materialLodDepth: cfg.materialLodDepth,
    cameraWorldMatrix,
    invProjectionMatrix,
    environmentIntensity: cfg.hasEnvMap ? envIntensity : 0,
    // H6 FIX (2026-06-09): honour HdriEnvironment.rotationY (CCW env dome rotation
    // around +Y, radians).  Convention: a world-space direction `d` looks up the
    // UNROTATED map at `RY(−rotationY) * d`, so the uniform matrix is
    // makeRotationYMat4(−rotationY).  The GLSL then evaluates:
    //   envRotation3x3 = mat3(environmentRotation)   → RY(−rotationY)
    //   lookupDir      = envRotation3x3 * worldDir   → RY(−rotationY) * d ✓
    // rotationY = 0 → identity → byte-identical to the pre-H6 IDENTITY_MAT4 path.
    environmentRotation: environmentRotationY !== 0
      ? makeRotationYMat4(-environmentRotationY)
      : IDENTITY_MAT4,
    backgroundBlur: requireNonNegativeFloat32(
      cfg.backgroundBlur,
      'pt-webgl2 backgroundBlur',
    ),
    spectralEnabled: cfg.spectralEnabled,
    backgroundAlpha: requireNonNegativeFloat32(
      cfg.backgroundAlpha,
      'pt-webgl2 backgroundAlpha',
    ),
    rayOriginBias: requireNonNegativeFloat32(
      webgl2RayOriginBias(packedSceneRadius),
      'pt-webgl2 ray origin bias',
    ),
    // A5 — BDPT host-driver inputs (no-op when bdpt:false).
    bdpt: cfg.bdpt,
    bdptMaxLightBounces: cfg.bdptMaxLightBounces,
    bdptSceneCenter: packedSceneCenter,
    bdptSceneRadius: packedSceneRadius,
    dof: packedDof,
    // ── Tonemap / present-pass dials (2026-06-10) ─────────────────────────
    // Matches the contract (FrameQualitySettings) and the walkaround-hybrid
    // orchestrator wiring (HybridEngineFrameOrchestrator.ts:764).
    // Default: aces(0) @ 1.0 @ srgb(0) — same as walkaround and the contract.
    //
    // CONTRACT-DEFAULT TENSION: pt-webgl2 previously returned raw linear HDR
    // (no present pass).  Adding the present pass with default aces+srgb
    // changes the default visual output.  Hosts that relied on the raw HDR
    // should pass quality.tonemap='none' + quality.outputColorSpace='linear'.
    tonemapMode:      TONEMAP_MODE_INDEX[input.quality?.tonemap ?? 'aces'],
    exposure: requireNonNegativeFloat32(
      input.quality?.exposure ?? 1.0,
      'pt-webgl2 exposure',
    ),
    outputColorSpace: input.quality?.outputColorSpace === 'linear' ? 1 : 0,
  };
  return {
    frameSeed: input.frameSeed,
    uniforms,
  };
}

/** Append the only accumulated-sample-dependent frame values after resets. */
export function finalizeFrameUniforms(
  preflight: FrameUniformsPreflight,
  accumulatedSample = 0,
): FrameUniforms {
  const sharedBdptWavelength = sharedBdptWavelengthForSeed(
    preflight.frameSeed,
    accumulatedSample,
  );
  return {
    ...preflight.uniforms,
    bdptSharedWavelengthNm: sharedBdptWavelength.wavelengthNm,
    bdptSharedWavelengthPdf: sharedBdptWavelength.pdf,
  };
}

/**
 * Standalone convenience wrapper retained for pure callers/tests. Engine
 * rendering uses the explicit preflight/finalize split to preserve ordering.
 */
export function packFrameUniforms(
  input: FrameInput,
  bounces: number,
  w: number,
  h: number,
  cfg: FrameUniformsConfig,
  accumulatedSample = 0,
): FrameUniforms {
  return finalizeFrameUniforms(
    preflightFrameUniforms(input, bounces, w, h, cfg),
    accumulatedSample,
  );
}

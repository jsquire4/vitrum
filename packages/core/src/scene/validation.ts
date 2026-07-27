import { validateAnalyticParams } from './analyticParams.js';
import type {
  DirectionalEmitter,
  DiscAreaEmitter,
  MeshAreaEmitter,
  PointEmitter,
  RectAreaEmitter,
  SceneEmitter,
  SpotEmitter,
} from './emitters.js';
import type {
  HdriEnvironment,
  NoneEnvironment,
  ProceduralSkyEnvironment,
  SceneEnvironment,
} from './environment.js';
import type {
  MaterialSpec,
  SpectralCurve,
  SurfaceAbsorptionLayer,
  TextureRef,
  ThinFilmLayer,
  ThinFilmStack,
  UvTransform,
} from './material.js';
import type { Mat4 } from './math.js';
import type {
  AnalyticPrimitive,
  InstancedMeshPrimitive,
  MeshPrimitive,
  ScenePrimitive,
  SkinnedMeshPrimitive,
} from './primitives.js';
import { sparseArrayOwnIndices } from './primitives.js';
import type { Scene } from './index.js';

const FLOAT32_BRAND = 'Float32Array';
const UINT16_BRAND = 'Uint16Array';
const UINT32_BRAND = 'Uint32Array';
const TYPED_ARRAY_BRAND_DESCRIPTOR = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Float32Array.prototype) as object,
  Symbol.toStringTag,
);
const TYPED_ARRAY_LENGTH_DESCRIPTOR = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Float32Array.prototype) as object,
  'length',
);

type AnalyticFallbackMesh = NonNullable<AnalyticPrimitive['fallbackMesh']>;

type ExhaustiveKeyList<T, Keys extends readonly (keyof T)[]> =
  Exclude<keyof T, Keys[number]> extends never ? Keys : never;

/**
 * Both runtime allow-list and compile-time exhaustiveness pin. Adding a public
 * field to any contract below fails core typecheck until validation is updated.
 */
function exhaustiveKnownKeys<T>() {
  return <const Keys extends readonly (keyof T)[]>(
    keys: ExhaustiveKeyList<T, Keys>,
  ): ReadonlySet<string> => new Set(keys.map((key) => String(key)));
}

type ExhaustiveValueList<T, Values extends readonly T[]> =
  Exclude<T, Values[number]> extends never ? Values : never;

function exhaustiveValues<T>() {
  return <const Values extends readonly T[]>(values: ExhaustiveValueList<T, Values>): Values => values;
}

const SCENE_KEYS = exhaustiveKnownKeys<Scene>()([
  'primitives', 'emitters', 'environment',
]);

const MESH_KEYS = exhaustiveKnownKeys<MeshPrimitive>()([
  'kind', 'id', 'positions', 'normals', 'uvs', 'uv1', 'uvSets', 'tangents',
  'colors', 'colorSets', 'indices', 'material', 'transform', 'castShadow',
]);
const INSTANCED_MESH_KEYS = exhaustiveKnownKeys<InstancedMeshPrimitive>()([
  'kind', 'id', 'positions', 'normals', 'uvs', 'uv1', 'uvSets', 'tangents',
  'colors', 'colorSets', 'indices', 'material', 'instances', 'castShadow',
]);
const SKINNED_MESH_KEYS = exhaustiveKnownKeys<SkinnedMeshPrimitive>()([
  'kind', 'id', 'positions', 'normals', 'uvs', 'uv1', 'uvSets', 'tangents',
  'colors', 'colorSets', 'indices', 'skinIndices', 'skinWeights',
  'skinInfluencesPerVertex', 'bones', 'boneInverses', 'bindMatrix',
  'bindMatrixInverse', 'morphTargets', 'morphTargetNormals',
  'morphTargetTangents', 'morphTargetUvs', 'morphTargetUv1s',
  'morphTargetUvSets', 'morphWeights', 'material', 'transform', 'castShadow',
]);
const ANALYTIC_KEYS = exhaustiveKnownKeys<AnalyticPrimitive>()([
  'kind', 'id', 'shape', 'params', 'material', 'transform', 'castShadow',
  'fallbackMesh',
]);
const ANALYTIC_FALLBACK_MESH_KEYS = exhaustiveKnownKeys<AnalyticFallbackMesh>()([
  'positions', 'normals', 'uvs', 'uv1', 'uvSets', 'tangents', 'colors',
  'colorSets', 'indices', 'castShadow',
]);

const DIRECTIONAL_EMITTER_KEYS = exhaustiveKnownKeys<DirectionalEmitter>()([
  'kind', 'id', 'color', 'intensity', 'castShadow', 'direction',
  'angularDiameter',
]);
const DISC_AREA_EMITTER_KEYS = exhaustiveKnownKeys<DiscAreaEmitter>()([
  'kind', 'id', 'color', 'intensity', 'castShadow', 'position', 'normal',
  'radius',
]);
const RECT_AREA_EMITTER_KEYS = exhaustiveKnownKeys<RectAreaEmitter>()([
  'kind', 'id', 'color', 'intensity', 'castShadow', 'position', 'uAxis',
  'vAxis',
]);
const POINT_EMITTER_KEYS = exhaustiveKnownKeys<PointEmitter>()([
  'kind', 'id', 'color', 'intensity', 'castShadow', 'position', 'distance',
  'decay',
]);
const SPOT_EMITTER_KEYS = exhaustiveKnownKeys<SpotEmitter>()([
  'kind', 'id', 'color', 'intensity', 'castShadow', 'position', 'direction',
  'angle', 'penumbra', 'distance', 'decay',
]);
const MESH_AREA_EMITTER_KEYS = exhaustiveKnownKeys<MeshAreaEmitter>()([
  'kind', 'id', 'color', 'intensity', 'castShadow', 'meshId',
]);

const HDRI_ENVIRONMENT_KEYS = exhaustiveKnownKeys<HdriEnvironment>()([
  'kind', 'hdri', 'intensity', 'rotationY',
]);
const PROCEDURAL_SKY_ENVIRONMENT_KEYS = exhaustiveKnownKeys<ProceduralSkyEnvironment>()([
  'kind', 'sunDirection', 'turbidity', 'rayleigh', 'mieCoefficient',
  'mieDirectionalG', 'intensity',
]);
const NONE_ENVIRONMENT_KEYS = exhaustiveKnownKeys<NoneEnvironment>()(['kind']);

const MATERIAL_KEYS = exhaustiveKnownKeys<MaterialSpec>()([
  'baseColor', 'roughness', 'metallic', 'emissive', 'emissiveIntensity',
  'shadingModel', 'alphaMode', 'alphaCutoff', 'opacity', 'doubleSided',
  'transmission', 'ior', 'attenuationColor', 'attenuationDistance', 'thickness',
  'baseColorMap', 'normalMap', 'normalScale', 'roughnessMap', 'metallicMap',
  'transmissionMap', 'thicknessMap', 'emissiveMap', 'alphaMap', 'aoMap',
  'aoMapIntensity', 'clearcoatMap', 'clearcoatRoughnessMap',
  'clearcoatNormalMap', 'clearcoatNormalScale', 'sheenColorMap',
  'sheenRoughnessMap', 'iridescenceMap', 'iridescenceThicknessMap',
  'anisotropyMap', 'specularColorMap', 'specularIntensityMap', 'bumpMap',
  'bumpScale', 'displacementMap', 'displacementScale', 'displacementBias',
  'displacementSubdivisions', 'lightMap', 'lightMapIntensity', 'sheen',
  'sheenColor', 'sheenRoughness', 'clearcoat', 'clearcoatRoughness',
  'iridescence', 'iridescenceIor', 'iridescenceThicknessRange',
  'specularIntensity', 'specularColor', 'envMapIntensity',
  'spectralAttenuation', 'dispersionAbbeNumber', 'scatteringCoefficient',
  'scatteringAnisotropy', 'scatteringCoefficientRGB', 'frontLayer', 'backLayer',
  'thinFilmStack', 'anisotropy', 'anisotropyRotation', 'extensions',
]);
const TEXTURE_REF_KEYS = exhaustiveKnownKeys<TextureRef>()([
  'handle', 'texCoord', 'transform', 'wrapS', 'wrapT', 'magFilter', 'minFilter',
  'mipFilter',
]);
const UV_TRANSFORM_KEYS = exhaustiveKnownKeys<UvTransform>()([
  'offset', 'scale', 'rotation',
]);
const SURFACE_LAYER_KEYS = exhaustiveKnownKeys<SurfaceAbsorptionLayer>()([
  'transmission', 'roughness', 'normalMap', 'normalScale',
]);
const THIN_FILM_STACK_KEYS = exhaustiveKnownKeys<ThinFilmStack>()([
  'layers', 'incidentIor', 'angleDependent',
]);
const THIN_FILM_LAYER_KEYS = exhaustiveKnownKeys<ThinFilmLayer>()([
  'ior', 'extinctionCoefficient', 'thicknessNm',
]);
const SPECTRAL_CURVE_KEYS = exhaustiveKnownKeys<SpectralCurve>()([
  'wavelengthStart', 'wavelengthEnd', 'values',
]);

function typedArrayBrand(value: unknown): string | undefined {
  if (!ArrayBuffer.isView(value) || TYPED_ARRAY_BRAND_DESCRIPTOR?.get === undefined) {
    return undefined;
  }
  try {
    return TYPED_ARRAY_BRAND_DESCRIPTOR.get.call(value) as string | undefined;
  } catch {
    return undefined;
  }
}

function typedArrayLength(value: unknown): number | undefined {
  if (
    typedArrayBrand(value) === undefined ||
    TYPED_ARRAY_LENGTH_DESCRIPTOR?.get === undefined
  ) return undefined;
  try {
    const length = TYPED_ARRAY_LENGTH_DESCRIPTOR.get.call(value) as unknown;
    return Number.isSafeInteger(length) && (length as number) >= 0
      ? length as number
      : undefined;
  } catch {
    return undefined;
  }
}

function failType(path: string, message: string): never {
  throw new TypeError(`validateScene: ${path} ${message}`);
}

function failRange(path: string, message: string): never {
  throw new RangeError(`validateScene: ${path} ${message}`);
}

function assertRecord(value: unknown, path: string): asserts value is Record<string, unknown> {
  if (value == null || typeof value !== 'object' || Array.isArray(value) || ArrayBuffer.isView(value)) {
    failType(path, 'must be a non-array object');
  }
}

function assertKnownKeys(
  value: Record<string, unknown>,
  path: string,
  knownKeys: ReadonlySet<string>,
): void {
  // Contract fields are enumerable strings. Non-enumerable symbols are the one
  // reserved metadata lane (used by adapters for provenance); all other own
  // keys participate in strict schema validation.
  for (const key of Reflect.ownKeys(value)) {
    const enumerable = Object.prototype.propertyIsEnumerable.call(value, key);
    if (typeof key === 'symbol') {
      if (enumerable) {
        failRange(`${path}[${String(key)}]`, 'enumerable symbol fields are not allowed');
      }
      continue;
    }
    if (!enumerable) {
      failRange(`${path}.${key}`, 'contract fields must be enumerable');
    }
    if (!knownKeys.has(key)) failRange(`${path}.${key}`, 'is not a known contract field');
  }
}

function assertBoolean(value: unknown, path: string): asserts value is boolean {
  if (typeof value !== 'boolean') failType(path, `must be a boolean (got ${String(value)})`);
}

function assertFinite(value: unknown, path: string): asserts value is number {
  if (typeof value !== 'number') failType(path, `must be a number (got ${String(value)})`);
  const float32 = Math.fround(value);
  if (!Number.isFinite(value) || !Number.isFinite(float32)) {
    failRange(path, `must be finite and representable as float32 (got ${String(value)})`);
  }
  if (value !== 0 && float32 === 0) {
    failRange(path, `must not underflow to zero as float32 (got ${String(value)})`);
  }
}

function assertRange(value: unknown, path: string, min: number, max: number): asserts value is number {
  assertFinite(value, path);
  if (value < min || value > max) {
    failRange(path, `must be in [${min}, ${max}] (got ${value})`);
  }
}

function assertNonNegative(value: unknown, path: string): asserts value is number {
  assertFinite(value, path);
  if (value < 0) failRange(path, `must be >= 0 (got ${value})`);
}

function assertPositive(value: unknown, path: string): asserts value is number {
  assertFinite(value, path);
  if (!(value > 0)) failRange(path, `must be > 0 (got ${value})`);
}

function assertComputedPositiveFloat32(value: number, path: string): void {
  const float32 = Math.fround(value);
  if (!Number.isFinite(value) || !Number.isFinite(float32) || !(float32 > 0)) {
    failRange(
      path,
      `canonical measure must be > 0 and representable as float32 (got ${String(value)})`,
    );
  }
}

function arrayIndexFromOwnKey(key: string): number | undefined {
  const index = Number(key);
  if (
    !Number.isSafeInteger(index) ||
    index < 0 ||
    String(index) !== key
  ) return undefined;
  return index;
}

/**
 * Validate an authored ordinary-array container without consulting its
 * prototype. Dense arrays require every indexed slot to be an own enumerable
 * data property. Sparse set arrays retain optional holes, but inherited values
 * can never fill those holes.
 */
function assertArrayContainer(
  value: unknown,
  path: string,
  options: {
    readonly allowHoles?: boolean;
    readonly exactLength?: number;
    readonly nonEmpty?: boolean;
  } = {},
): asserts value is unknown[] {
  if (!Array.isArray(value)) failType(path, 'must be an array');
  const lengthDescriptor = Object.getOwnPropertyDescriptor(value, 'length');
  if (
    lengthDescriptor === undefined ||
    !('value' in lengthDescriptor) ||
    !Number.isSafeInteger(lengthDescriptor.value) ||
    lengthDescriptor.value < 0
  ) {
    failType(`${path}.length`, 'must be the built-in array length data property');
  }
  const length = lengthDescriptor.value as number;
  if (options.exactLength !== undefined && length !== options.exactLength) {
    failType(path, `must be an exact array of length ${options.exactLength}`);
  }

  let ownIndexCount = 0;
  for (const key of Reflect.ownKeys(value)) {
    if (key === 'length') continue;
    if (typeof key === 'symbol') {
      failRange(`${path}[${String(key)}]`, 'array containers may only own numeric indices and built-in length');
    }
    const index = arrayIndexFromOwnKey(key);
    if (
      index === undefined ||
      (options.allowHoles !== true && (index >= 0xffff_ffff || index >= length))
    ) {
      failRange(`${path}[${JSON.stringify(key)}]`, 'array containers may only own numeric indices and built-in length');
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (!('value' in descriptor) || descriptor.enumerable !== true) {
      failRange(
        `${path}[${index}]`,
        'array elements must be own enumerable data properties',
      );
    }
    ownIndexCount += 1;
  }
  if (
    options.nonEmpty === true &&
    (options.allowHoles === true ? ownIndexCount === 0 : length === 0)
  ) {
    failRange(path, 'must contain at least one element');
  }
  if (options.allowHoles !== true && ownIndexCount !== length) {
    failRange(path, 'must be dense with every indexed element supplied as an own data property');
  }
}

function ownArrayElement<T>(value: readonly T[], index: number): T | undefined {
  if (!Object.prototype.hasOwnProperty.call(value, index)) return undefined;
  return value[index];
}

function assertFiniteArray(
  value: unknown,
  length: number,
  path: string,
): asserts value is readonly number[] {
  assertArrayContainer(value, path, { exactLength: length });
  const array = value as readonly number[];
  for (let index = 0; index < length; index += 1) {
    assertFinite(array[index], `${path}[${index}]`);
  }
}

function assertVecInRange(
  value: unknown,
  length: number,
  path: string,
  min: number,
  max: number,
): void {
  assertFiniteArray(value, length, path);
  for (let index = 0; index < length; index += 1) {
    const component = value[index]!;
    if (component < min || component > max) {
      failRange(`${path}[${index}]`, `must be in [${min}, ${max}] (got ${component})`);
    }
  }
}

function assertVecNonNegative(value: unknown, length: number, path: string): void {
  assertFiniteArray(value, length, path);
  for (let index = 0; index < length; index += 1) {
    if (value[index]! < 0) {
      failRange(`${path}[${index}]`, `must be >= 0 (got ${value[index]})`);
    }
  }
}

function assertFloat32Array(value: unknown, path: string, allowEmpty = false): asserts value is Float32Array {
  const length = typedArrayLength(value);
  if (typedArrayBrand(value) !== FLOAT32_BRAND || length === undefined) {
    failType(path, 'must be a Float32Array');
  }
  const array = value as Float32Array;
  if (!allowEmpty && length === 0) failRange(path, 'must not be empty');
  for (let index = 0; index < length; index += 1) {
    if (!Number.isFinite(array[index])) {
      failRange(`${path}[${index}]`, `must be finite (got ${String(array[index])})`);
    }
  }
}

function assertEnum(value: unknown, path: string, values: readonly string[]): void {
  if (typeof value !== 'string' || !values.includes(value)) {
    failRange(path, `must be one of ${values.map((entry) => `"${entry}"`).join(', ')} (got ${String(value)})`);
  }
}

function assertUnitVector(value: unknown, path: string): void {
  assertFiniteArray(value, 3, path);
  const length = Math.hypot(value[0]!, value[1]!, value[2]!);
  if (!(length > 1e-8)) failRange(path, 'must be non-zero');
  if (Math.abs(length - 1) > 1e-3) {
    failRange(path, `must be unit length within 1e-3 (got length ${length})`);
  }
}

function assertMat4(value: unknown, path: string): asserts value is Mat4 {
  if (
    typedArrayBrand(value) !== FLOAT32_BRAND ||
    typedArrayLength(value) !== 16
  ) {
    failType(path, 'must be a 16-element Float32Array');
  }
  const matrix = value as Float32Array;
  for (let index = 0; index < 16; index += 1) {
    if (!Number.isFinite(matrix[index])) {
      failRange(`${path}[${index}]`, `must be finite (got ${String(matrix[index])})`);
    }
  }
  if (
    matrix[3] !== 0 ||
    matrix[7] !== 0 ||
    matrix[11] !== 0 ||
    matrix[15] !== 1
  ) {
    failRange(path, 'must be an affine column-major matrix with bottom row [0, 0, 0, 1]');
  }
  const m00 = matrix[0]!, m01 = matrix[4]!, m02 = matrix[8]!;
  const m10 = matrix[1]!, m11 = matrix[5]!, m12 = matrix[9]!;
  const m20 = matrix[2]!, m21 = matrix[6]!, m22 = matrix[10]!;
  const determinant =
    m00 * (m11 * m22 - m12 * m21) -
    m01 * (m10 * m22 - m12 * m20) +
    m02 * (m10 * m21 - m11 * m20);
  if (!Number.isFinite(determinant) || determinant === 0) {
    failRange(path, 'must have an invertible linear transform');
  }
  const inverseLinear = [
    (m11 * m22 - m12 * m21) / determinant,
    (m02 * m21 - m01 * m22) / determinant,
    (m01 * m12 - m02 * m11) / determinant,
    (m12 * m20 - m10 * m22) / determinant,
    (m00 * m22 - m02 * m20) / determinant,
    (m02 * m10 - m00 * m12) / determinant,
    (m10 * m21 - m11 * m20) / determinant,
    (m01 * m20 - m00 * m21) / determinant,
    (m00 * m11 - m01 * m10) / determinant,
  ];
  const tx = matrix[12]!, ty = matrix[13]!, tz = matrix[14]!;
  const inverseTranslation = [
    -(inverseLinear[0]! * tx + inverseLinear[1]! * ty + inverseLinear[2]! * tz),
    -(inverseLinear[3]! * tx + inverseLinear[4]! * ty + inverseLinear[5]! * tz),
    -(inverseLinear[6]! * tx + inverseLinear[7]! * ty + inverseLinear[8]! * tz),
  ];
  for (const component of [...inverseLinear, ...inverseTranslation]) {
    if (!Number.isFinite(component) || !Number.isFinite(Math.fround(component))) {
      failRange(path, 'must have an inverse representable as float32');
    }
  }
  const inverse = new Float32Array([
    inverseLinear[0]!, inverseLinear[3]!, inverseLinear[6]!, 0,
    inverseLinear[1]!, inverseLinear[4]!, inverseLinear[7]!, 0,
    inverseLinear[2]!, inverseLinear[5]!, inverseLinear[8]!, 0,
    inverseTranslation[0]!, inverseTranslation[1]!, inverseTranslation[2]!, 1,
  ]) as Mat4;
  assertReciprocalMat4(matrix as Mat4, inverse, path);
}

function assertReciprocalMat4(left: Mat4, right: Mat4, path: string): void {
  for (const [a, b, order] of [
    [left, right, 'forward'],
    [right, left, 'reverse'],
  ] as const) {
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        let product = 0;
        let absoluteTermSum = 0;
        for (let inner = 0; inner < 4; inner += 1) {
          const term = a[inner * 4 + row]! * b[column * 4 + inner]!;
          product += term;
          absoluteTermSum += Math.abs(term);
        }
        const expected = row === column ? 1 : 0;
        const tolerance = 1e-5 * Math.max(1, absoluteTermSum);
        if (!Number.isFinite(product) || Math.abs(product - expected) > tolerance) {
          failRange(
            path,
            `must be reciprocal (${order} product row ${row}, column ${column} is ${product}; expected ${expected} ± ${tolerance})`,
          );
        }
      }
    }
  }
}

type MaterialTextureKey = {
  [Key in keyof MaterialSpec]-?: NonNullable<MaterialSpec[Key]> extends TextureRef ? Key : never;
}[keyof MaterialSpec];

const MATERIAL_TEXTURE_KEYS = exhaustiveValues<MaterialTextureKey>()([
  'baseColorMap', 'normalMap', 'roughnessMap', 'metallicMap', 'transmissionMap',
  'thicknessMap', 'emissiveMap', 'alphaMap', 'aoMap', 'clearcoatMap',
  'clearcoatRoughnessMap', 'clearcoatNormalMap', 'sheenColorMap',
  'sheenRoughnessMap', 'iridescenceMap', 'iridescenceThicknessMap',
  'anisotropyMap', 'specularColorMap', 'specularIntensityMap', 'bumpMap',
  'displacementMap', 'lightMap',
]);

function validateTextureRef(texture: TextureRef, path: string): void {
  assertRecord(texture, path);
  assertKnownKeys(texture, path, TEXTURE_REF_KEYS);
  if (texture.handle == null) failType(`${path}.handle`, 'must be a non-null opaque handle');
  if (texture.texCoord !== undefined) {
    if (!Number.isSafeInteger(texture.texCoord) || texture.texCoord < 0) {
      failRange(`${path}.texCoord`, `must be a non-negative safe integer (got ${String(texture.texCoord)})`);
    }
  }
  if (texture.transform !== undefined) {
    assertRecord(texture.transform, `${path}.transform`);
    assertKnownKeys(texture.transform, `${path}.transform`, UV_TRANSFORM_KEYS);
    if (texture.transform.offset !== undefined) {
      assertFiniteArray(texture.transform.offset, 2, `${path}.transform.offset`);
    }
    if (texture.transform.scale !== undefined) {
      assertFiniteArray(texture.transform.scale, 2, `${path}.transform.scale`);
    }
    if (texture.transform.rotation !== undefined) {
      assertFinite(texture.transform.rotation, `${path}.transform.rotation`);
    }
  }
  if (texture.wrapS !== undefined) {
    assertEnum(texture.wrapS, `${path}.wrapS`, ['repeat', 'clamp-to-edge', 'mirrored-repeat']);
  }
  if (texture.wrapT !== undefined) {
    assertEnum(texture.wrapT, `${path}.wrapT`, ['repeat', 'clamp-to-edge', 'mirrored-repeat']);
  }
  if (texture.magFilter !== undefined) {
    assertEnum(texture.magFilter, `${path}.magFilter`, ['nearest', 'linear']);
  }
  if (texture.minFilter !== undefined) {
    assertEnum(texture.minFilter, `${path}.minFilter`, ['nearest', 'linear']);
  }
  if (texture.mipFilter !== undefined) {
    assertEnum(texture.mipFilter, `${path}.mipFilter`, ['none', 'nearest', 'linear']);
  }
}

function validateSurfaceLayer(layer: SurfaceAbsorptionLayer, path: string): void {
  assertRecord(layer, path);
  assertKnownKeys(layer, path, SURFACE_LAYER_KEYS);
  assertVecInRange(layer.transmission, 3, `${path}.transmission`, 0, 1);
  if (layer.roughness !== undefined) assertRange(layer.roughness, `${path}.roughness`, 0, 1);
  if (layer.normalMap !== undefined) validateTextureRef(layer.normalMap, `${path}.normalMap`);
  if (layer.normalScale !== undefined) assertFinite(layer.normalScale, `${path}.normalScale`);
}

function validateThinFilmStack(stack: ThinFilmStack, path: string): void {
  assertRecord(stack, path);
  assertKnownKeys(stack, path, THIN_FILM_STACK_KEYS);
  assertArrayContainer(stack.layers, `${path}.layers`, { nonEmpty: true });
  for (let index = 0; index < stack.layers.length; index += 1) {
    const layer = stack.layers[index]!;
    assertRecord(layer, `${path}.layers[${index}]`);
    assertKnownKeys(layer, `${path}.layers[${index}]`, THIN_FILM_LAYER_KEYS);
    assertPositive(layer.ior, `${path}.layers[${index}].ior`);
    if (layer.extinctionCoefficient !== undefined) {
      assertNonNegative(
        layer.extinctionCoefficient,
        `${path}.layers[${index}].extinctionCoefficient`,
      );
    }
    assertPositive(layer.thicknessNm, `${path}.layers[${index}].thicknessNm`);
  }
  if (stack.incidentIor !== undefined) assertPositive(stack.incidentIor, `${path}.incidentIor`);
  if (stack.angleDependent !== undefined) assertBoolean(stack.angleDependent, `${path}.angleDependent`);
}

export function validateMaterialSpec(material: MaterialSpec, path = 'material'): void {
  assertRecord(material, path);
  assertKnownKeys(material, path, MATERIAL_KEYS);
  assertVecInRange(material.baseColor, 3, `${path}.baseColor`, 0, 1);
  assertRange(material.roughness, `${path}.roughness`, 0, 1);
  assertRange(material.metallic, `${path}.metallic`, 0, 1);
  if (material.emissive !== undefined) assertVecNonNegative(material.emissive, 3, `${path}.emissive`);
  if (material.emissiveIntensity !== undefined) {
    assertNonNegative(material.emissiveIntensity, `${path}.emissiveIntensity`);
  }
  if (material.shadingModel !== undefined) {
    assertEnum(material.shadingModel, `${path}.shadingModel`, ['pbr', 'unlit']);
  }
  if (material.alphaMode !== undefined) {
    assertEnum(material.alphaMode, `${path}.alphaMode`, ['opaque', 'mask', 'blend']);
  }
  if (material.alphaCutoff !== undefined) assertRange(material.alphaCutoff, `${path}.alphaCutoff`, 0, 1);
  if (material.opacity !== undefined) assertRange(material.opacity, `${path}.opacity`, 0, 1);
  if (material.doubleSided !== undefined) assertBoolean(material.doubleSided, `${path}.doubleSided`);
  if (material.transmission !== undefined) assertRange(material.transmission, `${path}.transmission`, 0, 1);
  if (material.ior !== undefined) assertPositive(material.ior, `${path}.ior`);
  if (material.attenuationColor !== undefined) {
    assertVecInRange(material.attenuationColor, 3, `${path}.attenuationColor`, 0, 1);
  }
  if (material.attenuationDistance !== undefined) {
    if (material.attenuationDistance !== Number.POSITIVE_INFINITY) {
      assertPositive(material.attenuationDistance, `${path}.attenuationDistance`);
    }
  }
  if (material.thickness !== undefined) assertNonNegative(material.thickness, `${path}.thickness`);
  for (const key of MATERIAL_TEXTURE_KEYS) {
    const texture = material[key];
    if (texture !== undefined) validateTextureRef(texture, `${path}.${key}`);
  }

  const range01Keys = [
    'aoMapIntensity', 'sheen', 'sheenRoughness', 'clearcoat', 'clearcoatRoughness',
    'iridescence', 'specularIntensity', 'anisotropy',
  ] as const satisfies readonly (keyof MaterialSpec)[];
  for (const key of range01Keys) {
    const value = material[key];
    if (value !== undefined) assertRange(value, `${path}.${key}`, 0, 1);
  }
  const finiteKeys = [
    'normalScale', 'clearcoatNormalScale', 'bumpScale', 'displacementScale',
    'displacementBias', 'anisotropyRotation',
  ] as const satisfies readonly (keyof MaterialSpec)[];
  for (const key of finiteKeys) {
    const value = material[key];
    if (value !== undefined) assertFinite(value, `${path}.${key}`);
  }
  const nonNegativeKeys = [
    'lightMapIntensity', 'envMapIntensity',
  ] as const satisfies readonly (keyof MaterialSpec)[];
  for (const key of nonNegativeKeys) {
    const value = material[key];
    if (value !== undefined) assertNonNegative(value, `${path}.${key}`);
  }
  if (material.displacementSubdivisions !== undefined) {
    if (!Number.isSafeInteger(material.displacementSubdivisions) || material.displacementSubdivisions < 0) {
      failRange(
        `${path}.displacementSubdivisions`,
        `must be a non-negative safe integer (got ${String(material.displacementSubdivisions)})`,
      );
    }
  }
  if (material.sheenColor !== undefined) {
    assertVecInRange(material.sheenColor, 3, `${path}.sheenColor`, 0, 1);
  }
  if (material.iridescenceIor !== undefined) assertPositive(material.iridescenceIor, `${path}.iridescenceIor`);
  if (material.iridescenceThicknessRange !== undefined) {
    assertFiniteArray(material.iridescenceThicknessRange, 2, `${path}.iridescenceThicknessRange`);
    const [minimum, maximum] = material.iridescenceThicknessRange;
    if (minimum < 0 || maximum < minimum) {
      failRange(`${path}.iridescenceThicknessRange`, 'must satisfy 0 <= minimum <= maximum');
    }
  }
  if (material.specularColor !== undefined) {
    assertVecInRange(material.specularColor, 3, `${path}.specularColor`, 0, 1);
  }
  if (material.spectralAttenuation !== undefined) {
    const curve = material.spectralAttenuation;
    assertRecord(curve, `${path}.spectralAttenuation`);
    assertKnownKeys(curve, `${path}.spectralAttenuation`, SPECTRAL_CURVE_KEYS);
    assertFinite(curve.wavelengthStart, `${path}.spectralAttenuation.wavelengthStart`);
    assertFinite(curve.wavelengthEnd, `${path}.spectralAttenuation.wavelengthEnd`);
    if (!(curve.wavelengthEnd > curve.wavelengthStart)) {
      failRange(`${path}.spectralAttenuation.wavelengthEnd`, 'must be greater than wavelengthStart');
    }
    assertFloat32Array(curve.values, `${path}.spectralAttenuation.values`);
    const curveLength = typedArrayLength(curve.values)!;
    if (curveLength < 3) failRange(`${path}.spectralAttenuation.values`, 'must contain at least 3 samples');
    for (let index = 0; index < curveLength; index += 1) {
      if (curve.values[index]! < 0) {
        failRange(`${path}.spectralAttenuation.values[${index}]`, 'must be >= 0');
      }
    }
  }
  if (material.dispersionAbbeNumber !== undefined) {
    assertPositive(material.dispersionAbbeNumber, `${path}.dispersionAbbeNumber`);
  }
  if (material.scatteringCoefficient !== undefined) {
    assertNonNegative(material.scatteringCoefficient, `${path}.scatteringCoefficient`);
  }
  if (material.scatteringAnisotropy !== undefined) {
    assertFinite(material.scatteringAnisotropy, `${path}.scatteringAnisotropy`);
    if (!(material.scatteringAnisotropy > -1 && material.scatteringAnisotropy < 1)) {
      failRange(`${path}.scatteringAnisotropy`, 'must be strictly between -1 and 1');
    }
  }
  if (material.scatteringCoefficientRGB !== undefined) {
    assertVecNonNegative(material.scatteringCoefficientRGB, 3, `${path}.scatteringCoefficientRGB`);
  }
  if (material.frontLayer !== undefined) validateSurfaceLayer(material.frontLayer, `${path}.frontLayer`);
  if (material.backLayer !== undefined) validateSurfaceLayer(material.backLayer, `${path}.backLayer`);
  if (material.thinFilmStack !== undefined) validateThinFilmStack(material.thinFilmStack, `${path}.thinFilmStack`);
  if (material.extensions !== undefined) assertRecord(material.extensions, `${path}.extensions`);
}

type MeshStreams = Pick<
  MeshPrimitive,
  'positions' | 'normals' | 'uvs' | 'uv1' | 'uvSets' | 'tangents' | 'colors' | 'colorSets' | 'indices' | 'castShadow'
>;

function materialTextureEntries(
  material: MaterialSpec,
  materialPath: string,
): ReadonlyArray<readonly [TextureRef, string]> {
  const entries: Array<readonly [TextureRef, string]> = [];
  for (const key of MATERIAL_TEXTURE_KEYS) {
    const texture = material[key];
    if (texture !== undefined) entries.push([texture, `${materialPath}.${key}`]);
  }
  if (material.frontLayer?.normalMap !== undefined) {
    entries.push([material.frontLayer.normalMap, `${materialPath}.frontLayer.normalMap`]);
  }
  if (material.backLayer?.normalMap !== undefined) {
    entries.push([material.backLayer.normalMap, `${materialPath}.backLayer.normalMap`]);
  }
  return entries;
}

function validateMaterialTexCoords(
  material: MaterialSpec,
  streams: MeshStreams | undefined,
  materialPath: string,
  streamsPath: string,
): void {
  for (const [texture, texturePath] of materialTextureEntries(material, materialPath)) {
    const texCoord = texture.texCoord ?? 0;
    const stream = streams?.uvSets === undefined
      ? undefined
      : ownArrayElement(streams.uvSets, texCoord);
    const resolvedStream = stream ??
      (texCoord === 0 ? streams?.uvs : texCoord === 1 ? streams?.uv1 : undefined);
    if (resolvedStream === undefined) {
      failRange(
        `${texturePath}.texCoord`,
        `references TEXCOORD_${texCoord}, but ${streamsPath} does not provide that UV stream`,
      );
    }
  }
}

function assertNodeId(value: unknown, path: string): asserts value is string {
  if (typeof value !== 'string') failType(path, `must be a string (got ${String(value)})`);
  if (value.length === 0) failRange(path, 'must not be empty');
}

function validateMeshStreams(streams: MeshStreams, path: string): number {
  assertFloat32Array(streams.positions, `${path}.positions`, true);
  const positionsLength = typedArrayLength(streams.positions)!;
  if (positionsLength % 3 !== 0) {
    failRange(`${path}.positions`, `length must be divisible by 3 (got ${positionsLength})`);
  }
  const vertexCount = positionsLength / 3;

  assertFloat32Array(streams.normals, `${path}.normals`, true);
  const normalsLength = typedArrayLength(streams.normals)!;
  if (normalsLength !== positionsLength) {
    failRange(
      `${path}.normals`,
      `length must equal positions.length (${positionsLength}; got ${normalsLength})`,
    );
  }
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const offset = vertex * 3;
    const normalLength = Math.hypot(
      streams.normals[offset]!,
      streams.normals[offset + 1]!,
      streams.normals[offset + 2]!,
    );
    if (!(normalLength > 0) || !Number.isFinite(Math.fround(1 / normalLength))) {
      failRange(`${path}.normals[${vertex}]`, 'must define a non-degenerate normal direction');
    }
  }

  const validateOptionalStream = (
    value: unknown,
    components: number,
    field: string,
  ): void => {
    if (value === undefined) return;
    assertFloat32Array(value, `${path}.${field}`, true);
    const expected = vertexCount * components;
    const actualLength = typedArrayLength(value)!;
    if (actualLength !== expected) {
      failRange(`${path}.${field}`, `length must be ${expected} (got ${actualLength})`);
    }
  };
  validateOptionalStream(streams.uvs, 2, 'uvs');
  validateOptionalStream(streams.uv1, 2, 'uv1');
  if (streams.uvSets !== undefined) {
    assertArrayContainer(streams.uvSets, `${path}.uvSets`, { allowHoles: true });
    for (const texCoord of sparseArrayOwnIndices(streams.uvSets)) {
      const stream = ownArrayElement(
        streams.uvSets as readonly (Float32Array | undefined)[],
        texCoord,
      );
      if (stream === undefined) continue;
      validateOptionalStream(stream, 2, `uvSets[${texCoord}]`);
      const legacy = texCoord === 0 ? streams.uvs : texCoord === 1 ? streams.uv1 : undefined;
      if (legacy !== undefined && !floatStreamsEqual(stream, legacy)) {
        failRange(
          `${path}.uvSets[${texCoord}]`,
          `must match the legacy ${texCoord === 0 ? 'uvs' : 'uv1'} alias when both are present`,
        );
      }
    }
  }
  validateOptionalStream(streams.tangents, 4, 'tangents');
  if (streams.tangents !== undefined) {
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      const offset = vertex * 4;
      const tangentLength = Math.hypot(
        streams.tangents[offset]!,
        streams.tangents[offset + 1]!,
        streams.tangents[offset + 2]!,
      );
      if (!(tangentLength > 0) || !Number.isFinite(Math.fround(1 / tangentLength))) {
        failRange(`${path}.tangents[${vertex}]`, 'must define a non-degenerate tangent direction');
      }
      const handedness = streams.tangents[offset + 3]!;
      if (handedness !== -1 && handedness !== 1) {
        failRange(`${path}.tangents[${vertex}].w`, `must be -1 or 1 (got ${handedness})`);
      }
    }
  }
  const validateColorStream = (value: unknown, field: string): void => {
    if (value === undefined) return;
    assertFloat32Array(value, `${path}.${field}`, true);
    const rgbLength = vertexCount * 3;
    const rgbaLength = vertexCount * 4;
    const actualLength = typedArrayLength(value)!;
    if (actualLength !== rgbLength && actualLength !== rgbaLength) {
      failRange(
        `${path}.${field}`,
        `length must be ${rgbLength} (RGB) or ${rgbaLength} (RGBA; got ${actualLength})`,
      );
    }
  };
  validateColorStream(streams.colors, 'colors');
  if (streams.colorSets !== undefined) {
    assertArrayContainer(streams.colorSets, `${path}.colorSets`, { allowHoles: true });
    for (const colorSet of sparseArrayOwnIndices(streams.colorSets)) {
      const stream = ownArrayElement(
        streams.colorSets as readonly (Float32Array | undefined)[],
        colorSet,
      );
      if (stream === undefined) continue;
      validateColorStream(stream, `colorSets[${colorSet}]`);
      if (colorSet === 0 && streams.colors !== undefined && !floatStreamsEqual(stream, streams.colors)) {
        failRange(
          `${path}.colorSets[0]`,
          'must match the legacy colors alias when both are present',
        );
      }
    }
  }

  if (streams.indices === undefined) {
    if (vertexCount % 3 !== 0) {
      failRange(`${path}.positions`, `non-indexed vertex count must be divisible by 3 (got ${vertexCount})`);
    }
  } else {
    const indicesBrand = typedArrayBrand(streams.indices);
    if (
      typedArrayLength(streams.indices) === undefined ||
      (indicesBrand !== UINT16_BRAND && indicesBrand !== UINT32_BRAND)
    ) {
      failType(`${path}.indices`, 'must be a Uint16Array or Uint32Array');
    }
    const indicesLength = typedArrayLength(streams.indices)!;
    if (indicesLength % 3 !== 0) {
      failRange(`${path}.indices`, `length must be divisible by 3 (got ${indicesLength})`);
    }
    for (let index = 0; index < indicesLength; index += 1) {
      const vertexIndex = streams.indices[index]!;
      if (vertexIndex >= vertexCount) {
        failRange(
          `${path}.indices[${index}]`,
          `must reference a vertex in [0, ${Math.max(0, vertexCount - 1)}] (got ${vertexIndex})`,
        );
      }
    }
  }
  if (streams.castShadow !== undefined) assertBoolean(streams.castShadow, `${path}.castShadow`);
  return vertexCount;
}

function floatStreamsEqual(a: Float32Array, b: Float32Array): boolean {
  if (a === b) return true;
  const aLength = typedArrayLength(a)!;
  const bLength = typedArrayLength(b)!;
  if (aLength !== bLength) return false;
  for (let index = 0; index < aLength; index += 1) {
    if (a[index] !== b[index]) return false;
  }
  return true;
}

function validateMeshPrimitive(primitive: MeshPrimitive, path: string): void {
  assertRecord(primitive, path);
  assertKnownKeys(primitive, path, MESH_KEYS);
  assertNodeId(primitive.id, `${path}.id`);
  validateMeshStreams(primitive, path);
  validateMaterialSpec(primitive.material, `${path}.material`);
  validateMaterialTexCoords(primitive.material, primitive, `${path}.material`, path);
  if (primitive.transform !== undefined) assertMat4(primitive.transform, `${path}.transform`);
}

function validateInstancedMeshPrimitive(primitive: InstancedMeshPrimitive, path: string): void {
  assertRecord(primitive, path);
  assertKnownKeys(primitive, path, INSTANCED_MESH_KEYS);
  assertNodeId(primitive.id, `${path}.id`);
  validateMeshStreams(primitive, path);
  validateMaterialSpec(primitive.material, `${path}.material`);
  validateMaterialTexCoords(primitive.material, primitive, `${path}.material`, path);
  assertArrayContainer(primitive.instances, `${path}.instances`);
  for (let index = 0; index < primitive.instances.length; index += 1) {
    assertMat4(primitive.instances[index], `${path}.instances[${index}]`);
  }
}

function validateMorphTargets(
  primitive: SkinnedMeshPrimitive,
  vertexCount: number,
  path: string,
): void {
  const positions = primitive.morphTargets;
  const related = [
    ['morphTargetNormals', primitive.morphTargetNormals, vertexCount * 3],
    ['morphTargetTangents', primitive.morphTargetTangents, vertexCount * 3],
    ['morphTargetUvs', primitive.morphTargetUvs, vertexCount * 2],
    ['morphTargetUv1s', primitive.morphTargetUv1s, vertexCount * 2],
  ] as const;

  for (const [field, base, baseField] of [
    ['morphTargetTangents', primitive.tangents, 'tangents'],
    ['morphTargetUvs', primitive.uvs, 'uvs'],
    ['morphTargetUv1s', primitive.uv1, 'uv1'],
  ] as const) {
    if (primitive[field] !== undefined && base === undefined) {
      failRange(`${path}.${field}`, `requires the base ${baseField} stream`);
    }
  }

  if (positions === undefined) {
    if (
      primitive.morphWeights !== undefined ||
      primitive.morphTargetUvSets !== undefined ||
      related.some(([, value]) => value !== undefined)
    ) {
      failRange(path, 'morphTargets is required when morph weights or auxiliary morph streams are present');
    }
    return;
  }
  assertArrayContainer(positions, `${path}.morphTargets`, { nonEmpty: true });
  for (let index = 0; index < positions.length; index += 1) {
    const target = positions[index];
    assertFloat32Array(target, `${path}.morphTargets[${index}]`, true);
    const targetLength = typedArrayLength(target)!;
    if (targetLength !== vertexCount * 3) {
      failRange(
        `${path}.morphTargets[${index}]`,
        `length must be ${vertexCount * 3} (got ${targetLength})`,
      );
    }
  }
  for (const [field, targets, expectedLength] of related) {
    if (targets === undefined) continue;
    assertArrayContainer(targets, `${path}.${field}`, { exactLength: positions.length });
    for (let index = 0; index < targets.length; index += 1) {
      const target = targets[index];
      assertFloat32Array(target, `${path}.${field}[${index}]`, true);
      const targetLength = typedArrayLength(target)!;
      if (targetLength !== expectedLength) {
        failRange(`${path}.${field}[${index}]`, `length must be ${expectedLength} (got ${targetLength})`);
      }
    }
  }
  const morphUvSets = primitive.morphTargetUvSets;
  if (morphUvSets !== undefined) {
    assertArrayContainer(morphUvSets, `${path}.morphTargetUvSets`, { allowHoles: true });
    for (const texCoord of sparseArrayOwnIndices(morphUvSets)) {
      const targets = ownArrayElement(morphUvSets, texCoord);
      if (targets === undefined) continue;
      const base = (primitive.uvSets === undefined
        ? undefined
        : ownArrayElement(primitive.uvSets, texCoord)) ??
        (texCoord === 0 ? primitive.uvs : texCoord === 1 ? primitive.uv1 : undefined);
      if (base === undefined) {
        failRange(
          `${path}.morphTargetUvSets[${texCoord}]`,
          `requires a matching uvSets[${texCoord}] base stream`,
        );
      }
      assertArrayContainer(targets, `${path}.morphTargetUvSets[${texCoord}]`, {
        exactLength: positions.length,
      });
      for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
        const target = targets[targetIndex];
        assertFloat32Array(
          target,
          `${path}.morphTargetUvSets[${texCoord}][${targetIndex}]`,
          true,
        );
        const targetLength = typedArrayLength(target)!;
        if (targetLength !== vertexCount * 2) {
          failRange(
            `${path}.morphTargetUvSets[${texCoord}][${targetIndex}]`,
            `length must be ${vertexCount * 2} (got ${targetLength})`,
          );
        }
        const legacyTargets = texCoord === 0
          ? primitive.morphTargetUvs
          : texCoord === 1
            ? primitive.morphTargetUv1s
            : undefined;
        const legacyTarget = legacyTargets?.[targetIndex];
        if (legacyTarget !== undefined && !floatStreamsEqual(target, legacyTarget)) {
          failRange(
            `${path}.morphTargetUvSets[${texCoord}][${targetIndex}]`,
            `must match the legacy ${texCoord === 0 ? 'morphTargetUvs' : 'morphTargetUv1s'} alias when both are present`,
          );
        }
      }
    }
  }
  // Omitted weights are the public contract's all-zero/rest-pose state.
  // When supplied, they remain an exact, finite per-target vector.
  if (primitive.morphWeights !== undefined) {
    assertFloat32Array(primitive.morphWeights, `${path}.morphWeights`, true);
    const morphWeightLength = typedArrayLength(primitive.morphWeights)!;
    if (morphWeightLength !== positions.length) {
      failRange(
        `${path}.morphWeights`,
        `length must equal morphTargets.length (${positions.length}; got ${morphWeightLength})`,
      );
    }
  }
}

function validateSkinnedMeshPrimitive(primitive: SkinnedMeshPrimitive, path: string): void {
  assertRecord(primitive, path);
  assertKnownKeys(primitive, path, SKINNED_MESH_KEYS);
  assertNodeId(primitive.id, `${path}.id`);
  const vertexCount = validateMeshStreams(primitive, path);
  validateMaterialSpec(primitive.material, `${path}.material`);
  validateMaterialTexCoords(primitive.material, primitive, `${path}.material`, path);
  if (primitive.transform !== undefined) assertMat4(primitive.transform, `${path}.transform`);

  if (
    typedArrayBrand(primitive.skinIndices) !== UINT32_BRAND ||
    typedArrayLength(primitive.skinIndices) === undefined
  ) {
    failType(`${path}.skinIndices`, 'must be a Uint32Array');
  }
  const influencesPerVertex = primitive.skinInfluencesPerVertex ?? 4;
  if (!Number.isSafeInteger(influencesPerVertex) || influencesPerVertex <= 0) {
    failRange(
      `${path}.skinInfluencesPerVertex`,
      `must be a positive safe integer (got ${String(influencesPerVertex)})`,
    );
  }
  const expectedInfluenceCount = vertexCount * influencesPerVertex;
  const skinIndicesLength = typedArrayLength(primitive.skinIndices)!;
  if (skinIndicesLength !== expectedInfluenceCount) {
    failRange(`${path}.skinIndices`, `length must be ${expectedInfluenceCount} (got ${skinIndicesLength})`);
  }
  assertFloat32Array(primitive.skinWeights, `${path}.skinWeights`, true);
  const skinWeightsLength = typedArrayLength(primitive.skinWeights)!;
  if (skinWeightsLength !== expectedInfluenceCount) {
    failRange(`${path}.skinWeights`, `length must be ${expectedInfluenceCount} (got ${skinWeightsLength})`);
  }
  assertFloat32Array(primitive.bones, `${path}.bones`);
  const bonesLength = typedArrayLength(primitive.bones)!;
  if (bonesLength % 16 !== 0) {
    failRange(`${path}.bones`, `length must be divisible by 16 (got ${bonesLength})`);
  }
  assertFloat32Array(primitive.boneInverses, `${path}.boneInverses`);
  const boneInversesLength = typedArrayLength(primitive.boneInverses)!;
  if (boneInversesLength !== bonesLength) {
    failRange(
      `${path}.boneInverses`,
      `length must equal bones.length (${bonesLength}; got ${boneInversesLength})`,
    );
  }
  const boneCount = bonesLength / 16;
  for (let bone = 0; bone < boneCount; bone += 1) {
    assertMat4(primitive.bones.subarray(bone * 16, bone * 16 + 16), `${path}.bones[${bone}]`);
    assertMat4(
      primitive.boneInverses.subarray(bone * 16, bone * 16 + 16),
      `${path}.boneInverses[${bone}]`,
    );
  }
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    let weightSum = 0;
    for (let influence = 0; influence < influencesPerVertex; influence += 1) {
      const offset = vertex * influencesPerVertex + influence;
      const weight = primitive.skinWeights[offset]!;
      if (weight < 0) failRange(`${path}.skinWeights[${offset}]`, `must be >= 0 (got ${weight})`);
      weightSum += weight;
      const boneIndex = primitive.skinIndices[offset]!;
      if (boneIndex >= boneCount) {
        failRange(`${path}.skinIndices[${offset}]`, `must be less than bone count ${boneCount} (got ${boneIndex})`);
      }
    }
    if (Math.abs(weightSum - 1) > 1e-4) {
      failRange(`${path}.skinWeights`, `weights for vertex ${vertex} must sum to 1 within 1e-4 (got ${weightSum})`);
    }
  }
  if ((primitive.bindMatrix === undefined) !== (primitive.bindMatrixInverse === undefined)) {
    failRange(path, 'bindMatrix and bindMatrixInverse must be supplied together');
  }
  if (primitive.bindMatrix !== undefined) assertMat4(primitive.bindMatrix, `${path}.bindMatrix`);
  if (primitive.bindMatrixInverse !== undefined) {
    assertMat4(primitive.bindMatrixInverse, `${path}.bindMatrixInverse`);
  }
  if (primitive.bindMatrix !== undefined && primitive.bindMatrixInverse !== undefined) {
    assertReciprocalMat4(
      primitive.bindMatrix,
      primitive.bindMatrixInverse,
      `${path}.bindMatrix/bindMatrixInverse`,
    );
  }
  validateMorphTargets(primitive, vertexCount, path);
}

function validateAnalyticPrimitive(primitive: AnalyticPrimitive, path: string): void {
  assertRecord(primitive, path);
  assertKnownKeys(primitive, path, ANALYTIC_KEYS);
  assertNodeId(primitive.id, `${path}.id`);
  validateAnalyticParams(primitive.shape, primitive.params);
  validateMaterialSpec(primitive.material, `${path}.material`);
  if (primitive.transform !== undefined) assertMat4(primitive.transform, `${path}.transform`);
  if (primitive.castShadow !== undefined) assertBoolean(primitive.castShadow, `${path}.castShadow`);
  if (primitive.fallbackMesh !== undefined) {
    assertRecord(primitive.fallbackMesh, `${path}.fallbackMesh`);
    assertKnownKeys(
      primitive.fallbackMesh,
      `${path}.fallbackMesh`,
      ANALYTIC_FALLBACK_MESH_KEYS,
    );
    validateMeshStreams(primitive.fallbackMesh, `${path}.fallbackMesh`);
    // A native analytic has shape-local parameterisation in the backend and
    // therefore does not need mesh UV streams merely because its material
    // references a texture. When a generated fallback is supplied, however,
    // every referenced texCoord must be representable by that fallback.
    validateMaterialTexCoords(
      primitive.material,
      primitive.fallbackMesh,
      `${path}.material`,
      `${path}.fallbackMesh`,
    );
  }
}

function validatePrimitive(primitive: ScenePrimitive, path: string): void {
  assertRecord(primitive, path);
  switch ((primitive as { readonly kind?: unknown }).kind) {
    case 'mesh':
      validateMeshPrimitive(primitive as MeshPrimitive, path);
      return;
    case 'instanced-mesh':
      validateInstancedMeshPrimitive(primitive as InstancedMeshPrimitive, path);
      return;
    case 'skinned-mesh':
      validateSkinnedMeshPrimitive(primitive as SkinnedMeshPrimitive, path);
      return;
    case 'analytic':
      validateAnalyticPrimitive(primitive as AnalyticPrimitive, path);
      return;
    default:
      failRange(`${path}.kind`, `is not a supported primitive kind (got ${String((primitive as { kind?: unknown }).kind)})`);
  }
}

function validateEmitter(emitter: SceneEmitter, path: string): void {
  assertRecord(emitter, path);
  assertNodeId(emitter.id, `${path}.id`);
  assertVecNonNegative(emitter.color, 3, `${path}.color`);
  assertNonNegative(emitter.intensity, `${path}.intensity`);
  if (emitter.castShadow !== undefined) assertBoolean(emitter.castShadow, `${path}.castShadow`);

  switch ((emitter as { readonly kind?: unknown }).kind) {
    case 'directional':
      assertKnownKeys(emitter, path, DIRECTIONAL_EMITTER_KEYS);
      assertUnitVector(emitter.direction, `${path}.direction`);
      if (emitter.angularDiameter !== undefined) {
        assertRange(emitter.angularDiameter, `${path}.angularDiameter`, 0, Math.PI);
      }
      return;
    case 'disc-area':
      assertKnownKeys(emitter, path, DISC_AREA_EMITTER_KEYS);
      assertFiniteArray(emitter.position, 3, `${path}.position`);
      assertUnitVector(emitter.normal, `${path}.normal`);
      assertPositive(emitter.radius, `${path}.radius`);
      assertComputedPositiveFloat32(emitter.radius, `${path}.radius`);
      return;
    case 'rect-area': {
      assertKnownKeys(emitter, path, RECT_AREA_EMITTER_KEYS);
      assertFiniteArray(emitter.position, 3, `${path}.position`);
      assertFiniteArray(emitter.uAxis, 3, `${path}.uAxis`);
      assertFiniteArray(emitter.vAxis, 3, `${path}.vAxis`);
      const ux = emitter.uAxis[0]!, uy = emitter.uAxis[1]!, uz = emitter.uAxis[2]!;
      const vx = emitter.vAxis[0]!, vy = emitter.vAxis[1]!, vz = emitter.vAxis[2]!;
      const uLength = Math.hypot(ux, uy, uz);
      const vLength = Math.hypot(vx, vy, vz);
      const crossLength = Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
      assertComputedPositiveFloat32(uLength, `${path}.uAxis`);
      assertComputedPositiveFloat32(vLength, `${path}.vAxis`);
      assertComputedPositiveFloat32(crossLength, `${path}.uAxis×vAxis`);
      return;
    }
    case 'point':
      assertKnownKeys(emitter, path, POINT_EMITTER_KEYS);
      assertFiniteArray(emitter.position, 3, `${path}.position`);
      if (emitter.distance !== undefined) assertNonNegative(emitter.distance, `${path}.distance`);
      if (emitter.decay !== undefined) assertNonNegative(emitter.decay, `${path}.decay`);
      return;
    case 'spot':
      assertKnownKeys(emitter, path, SPOT_EMITTER_KEYS);
      assertFiniteArray(emitter.position, 3, `${path}.position`);
      assertUnitVector(emitter.direction, `${path}.direction`);
      assertPositive(emitter.angle, `${path}.angle`);
      if (emitter.angle > Math.PI) failRange(`${path}.angle`, `must be <= PI (got ${emitter.angle})`);
      if (emitter.penumbra !== undefined) assertRange(emitter.penumbra, `${path}.penumbra`, 0, 1);
      if (emitter.distance !== undefined) assertNonNegative(emitter.distance, `${path}.distance`);
      if (emitter.decay !== undefined) assertNonNegative(emitter.decay, `${path}.decay`);
      return;
    case 'mesh-area':
      assertKnownKeys(emitter, path, MESH_AREA_EMITTER_KEYS);
      assertNodeId(emitter.meshId, `${path}.meshId`);
      return;
    default:
      failRange(`${path}.kind`, `is not a supported emitter kind (got ${String((emitter as { kind?: unknown }).kind)})`);
  }
}

function validateEnvironment(environment: SceneEnvironment, path: string): void {
  assertRecord(environment, path);
  switch ((environment as { readonly kind?: unknown }).kind) {
    case 'none':
      assertKnownKeys(environment, path, NONE_ENVIRONMENT_KEYS);
      return;
    case 'hdri':
      assertKnownKeys(environment, path, HDRI_ENVIRONMENT_KEYS);
      if (environment.hdri == null) failType(`${path}.hdri`, 'must be a non-null opaque handle');
      if (environment.intensity !== undefined) assertNonNegative(environment.intensity, `${path}.intensity`);
      if (environment.rotationY !== undefined) assertFinite(environment.rotationY, `${path}.rotationY`);
      return;
    case 'procedural-sky':
      assertKnownKeys(environment, path, PROCEDURAL_SKY_ENVIRONMENT_KEYS);
      assertUnitVector(environment.sunDirection, `${path}.sunDirection`);
      assertNonNegative(environment.turbidity, `${path}.turbidity`);
      assertNonNegative(environment.rayleigh, `${path}.rayleigh`);
      assertNonNegative(environment.mieCoefficient, `${path}.mieCoefficient`);
      assertFinite(environment.mieDirectionalG, `${path}.mieDirectionalG`);
      if (!(environment.mieDirectionalG > -1 && environment.mieDirectionalG < 1)) {
        failRange(`${path}.mieDirectionalG`, 'must be strictly between -1 and 1');
      }
      if (environment.intensity !== undefined) assertNonNegative(environment.intensity, `${path}.intensity`);
      return;
    default:
      failRange(`${path}.kind`, `is not a supported environment kind (got ${String((environment as { kind?: unknown }).kind)})`);
  }
}

/**
 * Validate an authored Scene snapshot before any backend allocates or mutates
 * GPU resources. The validator is deliberately backend-independent: every
 * shipping backend receives the same finite-number, stream-shape, identity,
 * transform, skin/morph, material, emitter, and reference guarantees.
 *
 * Valid empty triangle streams and zero-instance instanced meshes are retained
 * because they are useful dynamic-scene states. Unsafe partial streams,
 * out-of-range indices, non-invertible transforms, and dangling mesh emitters
 * throw synchronously.
 */
export function validateScene(scene: Scene): void {
  assertRecord(scene, 'scene');
  assertKnownKeys(scene, 'scene', SCENE_KEYS);
  assertArrayContainer(scene.primitives, 'scene.primitives');
  assertArrayContainer(scene.emitters, 'scene.emitters');

  const primitiveById = new Map<string, ScenePrimitive>();
  const allIds = new Set<string>();
  const meshAreaOwnerPathByMeshId = new Map<string, string>();
  for (let index = 0; index < scene.primitives.length; index += 1) {
    const primitive = scene.primitives[index]!;
    const path = `scene.primitives[${index}]`;
    validatePrimitive(primitive, path);
    if (allIds.has(primitive.id)) failRange(`${path}.id`, `duplicates scene node id "${primitive.id}"`);
    allIds.add(primitive.id);
    primitiveById.set(primitive.id, primitive);
  }
  for (let index = 0; index < scene.emitters.length; index += 1) {
    const emitter = scene.emitters[index]!;
    const path = `scene.emitters[${index}]`;
    validateEmitter(emitter, path);
    if (allIds.has(emitter.id)) failRange(`${path}.id`, `duplicates scene node id "${emitter.id}"`);
    allIds.add(emitter.id);
    if (emitter.kind === 'mesh-area') {
      const target = primitiveById.get(emitter.meshId);
      if (target === undefined) {
        failRange(`${path}.meshId`, `references missing primitive "${emitter.meshId}"`);
      }
      if (target.kind === 'analytic') {
        failRange(`${path}.meshId`, `must reference a mesh-like primitive (got analytic "${emitter.meshId}")`);
      }
      const existingOwnerPath = meshAreaOwnerPathByMeshId.get(emitter.meshId);
      if (existingOwnerPath !== undefined) {
        failRange(
          `${path}.meshId`,
          `duplicates mesh-area ownership of primitive "${emitter.meshId}" already claimed by ${existingOwnerPath}.meshId`,
        );
      }
      meshAreaOwnerPathByMeshId.set(emitter.meshId, path);
    }
  }
  validateEnvironment(scene.environment, 'scene.environment');
}

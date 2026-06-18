import type { Vec3 } from '@vitrum/core';

interface ColorLike {
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

type ColorSource = ColorLike | Vec3;

interface StandardMaterialScalars {
  readonly baseColor?: Vec3;
  readonly color?: ColorSource;
  readonly emissive?: ColorSource;
  readonly emissiveIntensity?: number;
  readonly roughness?: number;
  readonly metallic?: number;
  readonly metalness?: number;
  readonly userData?: unknown;
}

interface PhysicalMaterialScalars extends StandardMaterialScalars {
  readonly transmission?: number;
  readonly ior?: number;
  readonly attenuationColor?: ColorSource;
  readonly attenuationDistance?: number;
  readonly thickness?: number;
}

export type PbrScalarSource = PhysicalMaterialScalars;

export interface PbrScalars {
  readonly baseColor: Vec3;
  readonly emissive: Vec3;
  readonly emissiveIntensity: number;
  readonly roughness: number;
  readonly metallic: number;
  readonly transmission: number;
  readonly ior: number;
  readonly attenuationColor: Vec3;
  readonly attenuationDistance: number;
  readonly thickness: number;
}

export interface PbrDefaults {
  readonly baseColor: Vec3;
  readonly emissive: Vec3;
  readonly emissiveIntensity: number;
  readonly roughness: number;
  readonly metallic: number;
  readonly transmission: number;
  readonly ior: number;
  readonly attenuationColor: Vec3;
  readonly attenuationDistance: number;
  readonly thickness: number;
}

const PHYSICAL_MATERIAL_DEFAULT_IOR = 1.5;

const PBR_DEFAULTS: PbrDefaults = {
  baseColor: [1, 1, 1],
  emissive: [0, 0, 0],
  emissiveIntensity: 1,
  roughness: 0.5,
  metallic: 0,
  transmission: 0,
  ior: PHYSICAL_MATERIAL_DEFAULT_IOR,
  attenuationColor: [1, 1, 1],
  attenuationDistance: Infinity,
  thickness: 0,
};

function colorToVec3(c: ColorSource): Vec3 {
  if ('r' in c) return [c.r, c.g, c.b];
  return [c[0], c[1], c[2]];
}

function scalarColorToVec3(mat: StandardMaterialScalars, fallback: Vec3): Vec3 {
  if (mat.baseColor != null) return mat.baseColor;
  return mat.color ? colorToVec3(mat.color) : fallback;
}

/**
 * Structural PBR scalar extraction for material-like host objects. Accept both
 * THREE-style fields (`color`, `metalness`) and core `MaterialSpec` spelling
 * (`baseColor`, `metallic`) so fallback DDGI/RC material packing cannot silently
 * drop core-scene material data when no ReSTIR snapshot is available.
 */
export function extractPbrScalars(
  mat: PbrScalarSource,
  defaults: Partial<PbrDefaults> = {},
): PbrScalars {
  const d = { ...PBR_DEFAULTS, ...defaults };
  const stdMat = mat as StandardMaterialScalars;
  const physMat = mat;
  const emissive = stdMat.emissive;
  return {
    baseColor: scalarColorToVec3(stdMat, d.baseColor),
    emissive: emissive ? colorToVec3(emissive) : d.emissive,
    emissiveIntensity: stdMat.emissiveIntensity ?? d.emissiveIntensity,
    roughness: stdMat.roughness ?? d.roughness,
    metallic: stdMat.metallic ?? stdMat.metalness ?? d.metallic,
    transmission: physMat.transmission ?? d.transmission,
    ior: physMat.ior ?? d.ior,
    attenuationColor: physMat.attenuationColor
      ? colorToVec3(physMat.attenuationColor)
      : d.attenuationColor,
    attenuationDistance: physMat.attenuationDistance ?? d.attenuationDistance,
    thickness: physMat.thickness ?? d.thickness,
  };
}

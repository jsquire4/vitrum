import type { Vec3 } from '@vitrum/core';

interface ColorLike {
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

interface StandardMaterialScalars {
  readonly color?: ColorLike;
  readonly emissive?: ColorLike;
  readonly emissiveIntensity?: number;
  readonly roughness?: number;
  readonly metalness?: number;
  readonly userData?: unknown;
}

interface PhysicalMaterialScalars extends StandardMaterialScalars {
  readonly transmission?: number;
  readonly ior?: number;
  readonly attenuationColor?: ColorLike;
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

function colorToVec3(c: ColorLike): Vec3 {
  return [c.r, c.g, c.b];
}

/**
 * Structural PBR scalar extraction for material-like host objects. Keep the
 * same `?? default` semantics as the core Material conversion path so legacy
 * material-pack bytes stay stable while callers move through core Scene data.
 */
export function extractPbrScalars(
  mat: PbrScalarSource,
  defaults: Partial<PbrDefaults> = {},
): PbrScalars {
  const d = { ...PBR_DEFAULTS, ...defaults };
  const stdMat = mat as StandardMaterialScalars;
  const physMat = mat;
  const color = stdMat.color;
  const emissive = stdMat.emissive;
  return {
    baseColor: color ? colorToVec3(color) : d.baseColor,
    emissive: emissive ? colorToVec3(emissive) : d.emissive,
    emissiveIntensity: stdMat.emissiveIntensity ?? d.emissiveIntensity,
    roughness: stdMat.roughness ?? d.roughness,
    metallic: stdMat.metalness ?? d.metallic,
    transmission: physMat.transmission ?? d.transmission,
    ior: physMat.ior ?? d.ior,
    attenuationColor: physMat.attenuationColor
      ? colorToVec3(physMat.attenuationColor)
      : d.attenuationColor,
    attenuationDistance: physMat.attenuationDistance ?? d.attenuationDistance,
    thickness: physMat.thickness ?? d.thickness,
  };
}

import type { Vec3 } from '@vitrum/core';

interface ColorLike {
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

interface ThreeStandardMaterialScalars {
  readonly color?: ColorLike;
  readonly emissive?: ColorLike;
  readonly emissiveIntensity?: number;
  readonly roughness?: number;
  readonly metalness?: number;
  readonly userData?: unknown;
}

interface ThreePhysicalMaterialScalars extends ThreeStandardMaterialScalars {
  readonly transmission?: number;
  readonly ior?: number;
  readonly attenuationColor?: ColorLike;
  readonly attenuationDistance?: number;
  readonly thickness?: number;
}

export type ThreePbrScalarSource = ThreePhysicalMaterialScalars;

export interface ThreePbrScalars {
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

export interface ThreePbrDefaults {
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

const THREE_PHYSICAL_DEFAULT_IOR = 1.5;

const THREE_PBR_DEFAULTS: ThreePbrDefaults = {
  baseColor: [1, 1, 1],
  emissive: [0, 0, 0],
  emissiveIntensity: 1,
  roughness: 0.5,
  metallic: 0,
  transmission: 0,
  ior: THREE_PHYSICAL_DEFAULT_IOR,
  attenuationColor: [1, 1, 1],
  attenuationDistance: Infinity,
  thickness: 0,
};

function colorToVec3(c: ColorLike): Vec3 {
  return [c.r, c.g, c.b];
}

/**
 * Walkaround-local copy of the legacy `@vitrum/three-bindings`
 * PBR scalar extraction. Keep the same `?? default` semantics so the remaining
 * THREE ingestion paths preserve their material bytes while the package
 * decouples from the bindings adapter.
 */
export function extractThreePbrScalars(
  mat: ThreePbrScalarSource,
  defaults: Partial<ThreePbrDefaults> = {},
): ThreePbrScalars {
  const d = { ...THREE_PBR_DEFAULTS, ...defaults };
  const stdMat = mat as ThreeStandardMaterialScalars;
  const physMat = mat as ThreePhysicalMaterialScalars;
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

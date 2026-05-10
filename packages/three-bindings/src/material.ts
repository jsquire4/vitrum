/**
 * material.ts — THREE.js material → @vitrum/core Material converter.
 *
 * Handles MeshStandardMaterial and MeshPhysicalMaterial, including all
 * physical-material extensions (transmission, clearcoat, sheen, iridescence).
 * Other material types (ShaderMaterial, etc.) are rejected at the mesh level
 * in mesh.ts before this converter is called.
 */

import type * as THREE from 'three';
import type { Material, Vec3 } from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// Internal type aliases
// ────────────────────────────────────────────────────────────────────────────

export type ThreeStdMat = THREE.MeshStandardMaterial;
export type ThreePhysMat = THREE.MeshPhysicalMaterial;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

export function colorToVec3(c: THREE.Color): Vec3 {
  return [c.r, c.g, c.b];
}

function isPhysical(m: ThreeStdMat): m is ThreePhysMat {
  return (m as ThreePhysMat).isMeshPhysicalMaterial === true;
}

// ────────────────────────────────────────────────────────────────────────────
// Material converter
// ────────────────────────────────────────────────────────────────────────────

export function convertMaterial(m: ThreeStdMat): Material {
  const base: Material = {
    baseColor: colorToVec3(m.color),
    roughness: m.roughness,
    metallic: m.metalness,
  };

  const emR = m.emissive.r * m.emissiveIntensity;
  const emG = m.emissive.g * m.emissiveIntensity;
  const emB = m.emissive.b * m.emissiveIntensity;
  if (emR !== 0 || emG !== 0 || emB !== 0) {
    base.emissive = [emR, emG, emB];
    base.emissiveIntensity = m.emissiveIntensity;
  }

  if (m.map != null) base.baseColorMap = m.map;
  if (m.normalMap != null) {
    base.normalMap = m.normalMap;
    base.normalScale = m.normalScale.x;
  }
  if (m.roughnessMap != null) base.roughnessMap = m.roughnessMap;
  if (m.metalnessMap != null) base.metallicMap = m.metalnessMap;
  if (m.emissiveMap != null) base.emissiveMap = m.emissiveMap;
  if (m.alphaMap != null) base.alphaMap = m.alphaMap;

  if (!isPhysical(m)) return base;

  const p = m;
  if (p.transmission !== 0) base.transmission = p.transmission;
  if (p.ior !== 1.5) base.ior = p.ior;

  if (p.attenuationDistance !== Infinity) {
    base.attenuationColor = colorToVec3(p.attenuationColor);
    base.attenuationDistance = p.attenuationDistance;
  }

  if (p.thickness !== 0) base.thickness = p.thickness;
  if (p.transmissionMap != null) base.transmissionMap = p.transmissionMap;

  if (p.sheen !== 0) {
    base.sheen = p.sheen;
    base.sheenColor = colorToVec3(p.sheenColor);
    base.sheenRoughness = p.sheenRoughness;
  }

  if (p.clearcoat !== 0) {
    base.clearcoat = p.clearcoat;
    base.clearcoatRoughness = p.clearcoatRoughness;
  }

  if (p.iridescence !== 0) {
    base.iridescence = p.iridescence;
    // THREE uses iridescenceIOR (caps); core uses iridescenceIor (camelCase).
    base.iridescenceIor = p.iridescenceIOR;
    base.iridescenceThicknessRange = p.iridescenceThicknessRange;
  }

  return base;
}

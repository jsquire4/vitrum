import type { Scene } from '@vitrum/core';
import { ROUGH_DIELECTRIC_SMOOTH_THRESHOLD } from './math/roughDielectric.js';
import { THIN_FILM_LAYER_LIMIT } from './scene/materialPacking.js';

/**
 * KHR_materials_iridescence is currently evaluated as an integrated RGB lobe.
 * Spectral transport may use the wavelength-resolved TMM stack, but must reject
 * the RGB-only single-layer model instead of leaking three-channel energy into a
 * scalar hero path.
 */
export function assertSpectralSceneSupported(scene: Scene): void {
  for (const primitive of scene.primitives) {
    const material = primitive.material;
    if ((material.iridescence ?? 0) > 0) {
      throw new Error(
        'pt-webgpu spectral scene validation: primitive ' +
          `"${primitive.id}" uses KHR_materials_iridescence, whose current BSDF is RGB-integrated. ` +
          'Use thinFilmStack for wavelength-resolved TMM iridescence or disable spectral rendering.',
      );
    }
  }
}

/** Validate homogeneous participating-medium inputs before any GPU mutation. */
export function assertVolumeSceneSupported(scene: Scene): void {
  for (const primitive of scene.primitives) {
    const material = primitive.material;
    const fail = (field: string): never => {
      throw new Error(
        `pt-webgpu volume scene validation: primitive "${primitive.id}" has invalid ${field}.`,
      );
    };
    const finiteNonnegative = (value: number | undefined, field: string): void => {
      if (value != null && (!Number.isFinite(value) || value < 0)) fail(field);
    };
    finiteNonnegative(material.scatteringCoefficient, "scatteringCoefficient");
    finiteNonnegative(material.thickness, "thickness");
    if (material.attenuationDistance != null &&
        (!Number.isFinite(material.attenuationDistance) || material.attenuationDistance <= 0)) {
      fail("attenuationDistance");
    }
    if (material.scatteringAnisotropy != null &&
        (!Number.isFinite(material.scatteringAnisotropy) || Math.abs(material.scatteringAnisotropy) >= 1)) {
      fail("scatteringAnisotropy (requires |g| < 1)");
    }
    for (const [field, values] of [
      ["scatteringCoefficientRGB", material.scatteringCoefficientRGB],
      ["attenuationColor", material.attenuationColor],
    ] as const) {
      if (values != null && values.some((value) => !Number.isFinite(value) || value < 0)) fail(field);
    }
    const curve = material.spectralAttenuation;
    if (curve != null) {
      if (!Number.isFinite(curve.wavelengthStart) || !Number.isFinite(curve.wavelengthEnd) ||
          curve.wavelengthEnd <= curve.wavelengthStart || curve.values.length < 3) {
        fail("spectralAttenuation domain");
      }
      for (const value of curve.values) {
        if (!Number.isFinite(value) || value < 0) fail("spectralAttenuation coefficient");
      }
    }
  }
}

/** Validate the deliberately narrow coherent dielectric-interface domain. */
export function assertThinFilmSceneSupported(scene: Scene): void {
  for (const primitive of scene.primitives) {
    const material = primitive.material;
    const stack = material.thinFilmStack;
    if (stack == null || stack.layers.length === 0) continue;
    const fail = (reason: string): never => {
      throw new Error(
        'pt-webgpu thin-film scene validation: primitive ' +
          `"${primitive.id}" ${reason}. ` +
          'thinFilmStack currently supports a smooth, fully transmissive, ' +
          'non-metal dielectric interface only.',
      );
    };
    if (stack.layers.length > THIN_FILM_LAYER_LIMIT) {
      fail(`has more than ${THIN_FILM_LAYER_LIMIT} coherent layers`);
    }
    const substrateIor = material.ior ?? 1.5;
    if (!Number.isFinite(substrateIor) || substrateIor < 1) {
      fail('has a non-finite or sub-unity substrate IOR');
    }
    if ((material.metallic ?? 0) !== 0) fail('is metallic');
    // The authored scalar must select the coherent dielectric interface. A
    // LINEAR transmissionMap is supported as a hit-local gate on the TMM
    // transmitted branch: reflection remains coherent, rejected T becomes
    // absorption, and both T and p(T) receive the same map factor.
    if ((material.transmission ?? 0) !== 1) {
      fail('does not have scalar transmission=1');
    }
    if (
      !Number.isFinite(material.roughness) ||
      material.roughness > ROUGH_DIELECTRIC_SMOOTH_THRESHOLD
    ) {
      fail(`has roughness above ${ROUGH_DIELECTRIC_SMOOTH_THRESHOLD}`);
    }
    if (material.roughnessMap != null || material.metallicMap != null) {
      fail('uses a roughnessMap or metallicMap that can change the interface classifier');
    }
    if (material.anisotropy != null && material.anisotropy !== 0) {
      fail('combines the smooth coherent interface with anisotropy');
    }
    if ((material.clearcoat ?? 0) !== 0 || (material.sheen ?? 0) !== 0) {
      fail('combines the coherent interface with clearcoat or sheen');
    }
    if ((material.iridescence ?? 0) !== 0) {
      fail('combines thinFilmStack with KHR_materials_iridescence');
    }
    if (material.frontLayer != null || material.backLayer != null) {
      fail('combines the coherent interface with a surface-absorption layer');
    }
    const incidentIor = stack.incidentIor ?? 1;
    if (!Number.isFinite(incidentIor) || incidentIor < 1) {
      fail('has a non-finite or sub-unity incident IOR');
    }
    for (let layerIndex = 0; layerIndex < stack.layers.length; layerIndex += 1) {
      const layer = stack.layers[layerIndex]!;
      if (!Number.isFinite(layer.ior) || layer.ior < 1) {
        fail(`has invalid layer ${layerIndex} IOR`);
      }
      if (!Number.isFinite(layer.thicknessNm) || layer.thicknessNm < 0) {
        fail(`has invalid layer ${layerIndex} thickness`);
      }
      const extinction = layer.extinctionCoefficient ?? 0;
      if (!Number.isFinite(extinction) || extinction < 0) {
        fail(`has invalid layer ${layerIndex} extinction coefficient`);
      }
    }
  }
}

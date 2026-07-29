import type { Scene } from '@vitrum/core';
import { THIN_FILM_LAYER_LIMIT } from './scene/materialPacking.js';

/**
 * KHR_materials_iridescence is currently evaluated as an integrated RGB lobe.
 * Spectral transport may use the wavelength-resolved TMM stack, but must reject
 * the RGB-only single-layer model instead of leaking three-channel energy into a
 * scalar hero path. When both contracts are authored, thinFilmStack explicitly
 * overrides KHR_materials_iridescence in the production BSDF and is therefore
 * the supported wavelength-resolved path.
 */
export function assertSpectralSceneSupported(scene: Scene): void {
  for (const primitive of scene.primitives) {
    const material = primitive.material;
    const hasCoherentStack = (material.thinFilmStack?.layers.length ?? 0) > 0;
    if ((material.iridescence ?? 0) > 0 && !hasCoherentStack) {
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

/** Validate the coherent-stack numeric and fixed-capacity GPU domain. */
export function assertThinFilmSceneSupported(scene: Scene): void {
  for (const primitive of scene.primitives) {
    const material = primitive.material;
    const stack = material.thinFilmStack;
    if (stack == null || stack.layers.length === 0) continue;
    const fail = (reason: string): never => {
      throw new Error(
        'pt-webgpu thin-film scene validation: primitive ' +
          `"${primitive.id}" ${reason}.`,
      );
    };
    if (stack.layers.length > THIN_FILM_LAYER_LIMIT) {
      fail(`has more than ${THIN_FILM_LAYER_LIMIT} coherent layers`);
    }
    const substrateIor = material.ior ?? 1.5;
    if (!Number.isFinite(substrateIor) || substrateIor < 1) {
      fail('has a non-finite or sub-unity substrate IOR');
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

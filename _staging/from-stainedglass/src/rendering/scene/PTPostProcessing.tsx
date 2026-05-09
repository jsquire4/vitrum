// Photorealism Phase 2 — PT post-processing pipeline.
//
// Mounts EffectComposer with the active camera-look preset's effect
// chain after the path tracer renders. This is the camera-response
// layer that takes the raw radiance buffer and produces a result that
// reads as "photographed" instead of "rendered" — bloom around bright
// caustics, DoF defocus on out-of-plane objects, chromatic aberration
// at edges, vignette, and grain.
//
// Effect order: Bloom → DepthOfField → ChromaticAberration → Vignette
// → Noise. Bloom MUST precede DoF so bloom haloes also defocus when
// they fall outside the focal plane (otherwise the bloom reads as a
// CG overlay, not a lens response).
//
// Mounted only inside <PathTracingLayer> so the post chain runs on
// PT output exclusively. Raster mode keeps the editor's clean
// presentation.

import { useMemo } from 'react';
import { useSelector } from 'react-redux';
import {
  EffectComposer,
  Bloom,
  DepthOfField,
  ChromaticAberration,
  Vignette,
  Noise,
} from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import { Vector2 } from 'three';
import { selectCameraLook } from '@/store/selectors';
import { CAMERA_LOOK_PRESETS } from './cameraLookPresets';

export function PTPostProcessing() {
  const cameraLook = useSelector(selectCameraLook);
  const preset = CAMERA_LOOK_PRESETS[cameraLook];

  // ChromaticAberration takes a Vector2 for offset (postprocessing
  // expects an actual instance, not a tuple). Memoize per preset
  // change so the EffectComposer doesn't see a fresh ref every render.
  const caOffset = useMemo(
    () => new Vector2(preset.chromaticAberration.offset[0], preset.chromaticAberration.offset[1]),
    [preset.chromaticAberration.offset],
  );

  return (
    <EffectComposer
      // Inputs are the PT-accumulated radiance buffer; ACES tone mapping
      // is applied AFTER the post chain via three.js's outputColorSpace
      // path, so the effects operate in physical-radiance space.
      multisampling={0}
    >
      <Bloom
        intensity={preset.bloom.intensity}
        luminanceThreshold={preset.bloom.luminanceThreshold}
        luminanceSmoothing={preset.bloom.luminanceSmoothing}
        mipmapBlur={preset.bloom.mipmapBlur}
      />
      <DepthOfField
        focusDistance={preset.depthOfField.focusDistance}
        bokehScale={preset.depthOfField.bokehScale}
        focusRange={preset.depthOfField.focusRange}
      />
      <ChromaticAberration
        blendFunction={BlendFunction.NORMAL}
        offset={caOffset}
        radialModulation={false}
        modulationOffset={0}
      />
      <Vignette
        darkness={preset.vignette.darkness}
        offset={preset.vignette.offset}
      />
      <Noise
        opacity={preset.noise.opacity}
        blendFunction={BlendFunction.OVERLAY}
      />
    </EffectComposer>
  );
}

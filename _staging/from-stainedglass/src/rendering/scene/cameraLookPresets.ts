// Photorealism Phase 2 — camera-look post-processing presets.
//
// Each preset describes the EffectComposer chain run after PT
// accumulates. The chain is fixed in order — Bloom → DepthOfField →
// ChromaticAberration → Vignette → Noise — but each effect's params
// vary per preset to deliver three distinct aesthetics:
//
//   documentary    — f/8, subtle bloom, ISO 400 grain. Closest to
//                    "indistinguishable from a DSLR photo." Default.
//   cinematic      — f/2.8, dramatic bloom, ISO 800 grain. Movie-still.
//   architectural  — f/16 (everything sharp), minimal bloom, no grain.
//                    Magazine interior-design rendering.
//
// Effect order rationale: Bloom must come BEFORE DoF (bloom is computed
// on the focused image, then defocused along with everything else).
// Otherwise the bloom doesn't soften when out of focus, breaking the
// "this is one continuous photographic image" illusion.

import type { CameraLook } from '@/store/viewportSlice';

export interface CameraLookPreset {
  bloom: {
    intensity: number;
    luminanceThreshold: number;
    luminanceSmoothing: number;
    /** Larger kernel = softer, more buttery glow. */
    mipmapBlur: boolean;
  };
  depthOfField: {
    /** World-space focus distance — typically the panel's z position
     *  from the camera. PTStage computes this from camera + frameLayout. */
    focusDistance: number;
    /** Aperture diameter (f-stop derived → physical aperture). Smaller
     *  bokehScale = more in-focus, larger = stronger background blur. */
    bokehScale: number;
    /** Width of the in-focus band around focusDistance. */
    focusRange: number;
  };
  chromaticAberration: {
    /** Pixel offset on the red and blue channels at frame edges. */
    offset: [number, number];
  };
  vignette: {
    /** 0 = no darkening, 1 = full corner darkening. */
    darkness: number;
    /** Smoothing of the vignette mask edge. */
    offset: number;
  };
  noise: {
    /** 0 = no grain, 1 = full grain. ISO 400 ≈ 0.06; ISO 800 ≈ 0.12. */
    opacity: number;
  };
}

export const CAMERA_LOOK_PRESETS: Record<CameraLook, CameraLookPreset> = {
  documentary: {
    bloom: {
      intensity: 0.4,
      luminanceThreshold: 1.0,
      luminanceSmoothing: 0.2,
      mipmapBlur: true,
    },
    depthOfField: {
      // f/8 at panel viewing distance. focusDistance is a normalized
      // (0..1) value in postprocessing — we set this dynamically per-
      // frame in PTStage based on camera distance to panel. Default
      // 0.05 corresponds to roughly 100" focus distance in our scale.
      focusDistance: 0.05,
      bokehScale: 1.0,
      focusRange: 0.05,
    },
    chromaticAberration: {
      offset: [0.0008, 0.0008],
    },
    vignette: {
      darkness: 0.4,
      offset: 0.5,
    },
    noise: {
      opacity: 0.06,  // ISO 400 grain
    },
  },
  cinematic: {
    bloom: {
      intensity: 0.9,
      luminanceThreshold: 0.85,
      luminanceSmoothing: 0.3,
      mipmapBlur: true,
    },
    depthOfField: {
      // f/2.8 — shallow DoF, panel sharp, floor caustic visibly
      // defocused.
      focusDistance: 0.05,
      bokehScale: 4.0,
      focusRange: 0.02,
    },
    chromaticAberration: {
      offset: [0.0016, 0.0016],
    },
    vignette: {
      darkness: 0.55,
      offset: 0.4,
    },
    noise: {
      opacity: 0.12,  // ISO 800 grain
    },
  },
  architectural: {
    bloom: {
      intensity: 0.2,
      luminanceThreshold: 1.2,
      luminanceSmoothing: 0.2,
      mipmapBlur: true,
    },
    depthOfField: {
      // f/16 — everything tack-sharp. bokehScale 0.5 effectively
      // disables visible bokeh.
      focusDistance: 0.05,
      bokehScale: 0.5,
      focusRange: 0.20,
    },
    chromaticAberration: {
      offset: [0, 0],
    },
    vignette: {
      darkness: 0.2,
      offset: 0.6,
    },
    noise: {
      opacity: 0,
    },
  },
};

/** UI label per preset. */
export const CAMERA_LOOK_LABEL: Record<CameraLook, string> = {
  documentary:   'Documentary (f/8)',
  cinematic:     'Cinematic (f/2.8)',
  architectural: 'Architectural (f/16)',
};

export const CAMERA_LOOK_ORDER: readonly CameraLook[] = [
  'documentary',
  'cinematic',
  'architectural',
] as const;

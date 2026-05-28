interface UniformRef<T> {
  value: T;
}

export interface PathTracerMaterialLike {
  uniforms?: Record<string, UniformRef<unknown>>;
  setDefine?(name: string, value: number): void;
}

/**
 * Typed surface of the fork we depend on. `WebGLPathTracer`'s published types
 * use Three.js Scene/Camera types that diverge slightly between three.js
 * `@types` versions; this wrapper interface lets the engine cast once at
 * construction and call methods on the wrapper using local types. Add methods
 * here as we depend on them — keep the surface minimal.
 *
 * NOTE: this surface intentionally covers only the fork's PUBLIC API (methods +
 * public fields). Fork-PRIVATE internals (`_generator`, `_pathTracer`, …) are
 * never spelled here — they go through the named {@link ForkAccess} methods.
 */
export interface WebGLPathTracerCompat {
  setScene(scene: unknown, camera: unknown): void;
  setCamera(camera: unknown): void;
  setSize(width: number, height: number): void;
  reset(): void;
  renderSample(): void;
  /** Re-reads `scene.environment` / `scene.environmentIntensity` /
   *  `scene.environmentRotation` (and the matching background fields) into the
   *  fork's IBL uniforms WITHOUT touching geometry, materials, or the BVH.
   *  Internally calls `reset()` (one accumulator-clear) — no BVH rebuild,
   *  no geometry re-upload. Used by `PTEngineWebGL2.updateEnvironment()` to
   *  service host-driven timeOfDay scrubs cheaply. */
  updateEnvironment?(): void;
  /** Re-pack MaterialsTexture from the cached scene without BVH rebuild (PR-8). */
  updateMaterials?(): void;
  /** Re-pack light buffers from the cached scene without BVH rebuild (PR-8). */
  updateLights?(): void;
  configureAdditiveAccumulation?(enabled: boolean, blendFrames: boolean): void;
  renderBdptLightSubpathPass?(
    lightPathTarget: import('three').WebGLRenderTarget,
    maxLightBounces: number,
    frameSeed: number,
  ): void;
  /** Optional fork field — the wrapper stores a reference to the THREE scene
   *  most recently passed to `setScene()`. updateEnvironment() reads
   *  `scene.environment*` off this reference, so the host MUST mutate the
   *  same scene object the wrapper has cached, not pass in a new one. */
  scene?: unknown;
  dispose?(): void;
  samples: number;
  tileRepeatFactors?: Uint8Array | null;
  tiles: { setScalar: (n: number) => void; set(x: number, y: number): void; x: number; y: number };
  bounces: number;
  filterGlossyFactor: number;
  fastUpdate: boolean;
  domElement?: HTMLCanvasElement;
}

/** Result of the fork generator's `generate()` — the merged-geometry + BVH
 *  payload `ForkAccess.regenerateSceneGeometry` consumes. Shapes are the
 *  intersection of fields the engine reads; fork-private, never exported. */
export interface PathTracerGenerateResult {
  bvhChanged?: boolean;
  bvh?: unknown;
  needsMaterialIndexUpdate?: boolean;
  geometry?: {
    attributes: {
      normal?: { array: ArrayLike<number> };
      tangent?: { array: ArrayLike<number> };
      uv?: { array: ArrayLike<number> };
      color?: { array: ArrayLike<number> };
      materialIndex?: unknown;
    };
  };
}

interface ForkGeneratorLike {
  initialized?: boolean;
  generate: () => PathTracerGenerateResult;
}

interface ForkMaterialGeometryLike {
  bvh: { updateFrom: (b: unknown) => void };
  attributesArray: {
    updateFrom: (
      normal: { array: ArrayLike<number> } | undefined,
      tangent: { array: ArrayLike<number> } | undefined,
      uv: { array: ArrayLike<number> } | undefined,
      color: { array: ArrayLike<number> } | undefined,
    ) => void;
  };
  materialIndexAttribute: { updateFrom: (attr: unknown) => void };
}

interface ForkTracerLike {
  _generator?: ForkGeneratorLike;
  _pathTracer?: {
    material?: PathTracerMaterialLike & Partial<ForkMaterialGeometryLike>;
    target?: { texture?: unknown };
  };
  target?: { texture?: unknown };
}

/** Centralized compatibility access for private fork fields.
 *
 * This is the SINGLE seam through which `PTEngineWebGL2` reaches into
 * `three-gpu-pathtracer` (`WebGLPathTracer`) internals. The engine never
 * spells a fork-private field (`_generator`, `_pathTracer.material.bvh`, …)
 * directly — every such access is a named method here so the cast surface
 * stays auditable and the fork dependency stays swappable. */
export class ForkAccess {
  static getMaterial(pathTracer: unknown): PathTracerMaterialLike | null {
    const tracer = pathTracer as ForkTracerLike;
    return tracer._pathTracer?.material ?? null;
  }

  static getRenderTexture(pathTracer: unknown): unknown | null {
    const tracer = pathTracer as ForkTracerLike;
    return tracer.target?.texture ?? tracer._pathTracer?.target?.texture ?? null;
  }

  /**
   * Regenerate merged geometry + BVH via the fork's incremental generator —
   * no full `setScene`. Returns `false` when the generator hasn't been
   * initialized yet (the caller must fall back to a full `setScene`), `true`
   * when the refit completed and the accumulator was reset.
   *
   * The caller is responsible for having mutated the THREE scene root
   * (transforms / positions) BEFORE invoking this; this method calls
   * `updateMatrixWorld(true)` then re-runs the generator.
   */
  static regenerateSceneGeometry(
    pathTracer: unknown,
    threeRoot: { updateMatrixWorld: (force: boolean) => void },
    reset: () => void,
  ): boolean {
    const tracer = pathTracer as ForkTracerLike;
    const gen = tracer._generator;
    if (gen?.initialized !== true) {
      return false;
    }
    threeRoot.updateMatrixWorld(true);
    const result = gen.generate();
    const mat = tracer._pathTracer?.material;
    if (result.bvhChanged === true && result.bvh != null && mat?.bvh != null) {
      mat.bvh.updateFrom(result.bvh);
      const attrs = result.geometry?.attributes;
      if (attrs != null && mat.attributesArray != null) {
        mat.attributesArray.updateFrom(attrs.normal, attrs.tangent, attrs.uv, attrs.color);
      }
      if (
        result.needsMaterialIndexUpdate === true &&
        attrs?.materialIndex != null &&
        mat.materialIndexAttribute != null
      ) {
        mat.materialIndexAttribute.updateFrom(attrs.materialIndex);
      }
    }
    reset();
    return true;
  }
}

# Manifold Next-Event Estimation for Specular Caustics

**Status:** Applied (runtime-unverified)
**Originating need:** Scenes containing refractive objects between light sources and diffuse receivers (e.g., glass panels casting colored caustics onto a room floor) produce near-zero contribution from standard NEE because the refraction geometry must be solved before a valid shadow ray can be constructed.
**Priority:** Medium (biased photon mapping is a viable fallback)

## 1. Motivation

Standard unidirectional path tracing with Next-Event Estimation (NEE) efficiently handles direct illumination on diffuse and specular surfaces. It fails on **specular-chain paths**: a path segment Light → ⟨one or more refractive/reflective surfaces⟩ → diffuse receiver → eye. When NEE attempts to connect a diffuse shading point to the light, the direct shadow ray must pass through refractive surfaces to reach the light. Standard NEE can account for transmission (it multiplies the shadow ray throughput by the material's transmission BSDF), but it cannot account for refraction — the shadow ray is a straight line, so it misses the light unless the geometry happens to align. The result is near-zero contribution from refracted caustics in unidirectional PT, producing slow variance reduction and firefly-dominated caustic regions.

Two illustrative cases:

- **Sunlight through a glass facade onto a lobby floor.** A modern building with a large glazed facade casts complex refracted caustic patterns on interior floors. The caustics are architecturally significant and are the primary differentiator between a physically accurate rendering and a raster approximation. Without MNEE or photon mapping, a unidirectional PT render converges extremely slowly at the caustic regions.
- **Gemstone brilliance.** A faceted gemstone refracts light through many internal total-internal-reflection events before exiting. Standard NEE cannot efficiently connect to the light source through the specular chain; MNEE walks the manifold defined by the gem's facets to find the valid connection.

The refractive window / diffuse floor caustic is a canonical benchmark for MNEE. Any consumer where colored or attenuating panels filter exterior light onto interior diffuse surfaces depends on the same path topology — without MNEE the colored caustic is missed by NEE and arrives only via slow brute-force diffuse bounces.

## 2. Proposed API Surface

MNEE is an engine-level integration strategy, not a material-level parameter. The consumer does not need to configure it per-material. The proposed surface is a creation-time engine option and a capabilities flag:

```typescript
// Addition to EngineOptions in engine.ts

interface EngineOptions {
  // ... existing fields unchanged ...

  // ── Specular caustics strategy ────────────────────────────────────────
  /**
   * Strategy for handling specular-chain caustic paths (LS+E, LSS+E, …).
   *
   * 'none':    No special caustic handling. Standard NEE only. Caustics
   *            accumulate slowly via BSDF-sampled paths (may require many
   *            thousands of samples to converge).
   *
   * 'manifold-nee': Manifold Next-Event Estimation (Hanika et al. 2015).
   *            At each diffuse vertex, launch a manifold walk to find valid
   *            specular connections to sampled light positions. Unbiased.
   *            Adds per-shading-event cost proportional to the number of
   *            specular interfaces in the scene (typically 2–5 Newton steps
   *            per walk attempt). May fail to converge for highly curved or
   *            rough specular surfaces.
   *
   * 'photon-map': Biased photon mapping for caustics. Trace forward photons
   *            from lights; store caustic photons in a spatial data structure;
   *            use density estimation at diffuse shading points to reconstruct
   *            caustic radiance. Biased (photon density estimation introduces
   *            blur), but memory-efficient per photon and robust to geometry
   *            complexity.
   *
   * Default: 'none'.
   */
  readonly causticStrategy?: 'none' | 'manifold-nee' | 'photon-map';

  /**
   * MNEE: maximum number of Newton iterations per manifold walk attempt.
   * Higher values improve convergence for curved surfaces at greater per-vertex
   * cost. Typical range: 4–12. Default: 8.
   * Ignored when causticStrategy !== 'manifold-nee'.
   */
  readonly mneeMaxIterations?: number;

  /**
   * MNEE: maximum number of specular vertices in a chain that MNEE will
   * attempt to connect. Longer chains are expensive; capping at 3–4 covers
   * most architectural cases (one or two refractive surfaces).
   * Default: 3. Ignored when causticStrategy !== 'manifold-nee'.
   */
  readonly mneeMaxChainLength?: number;
}

// Addition to EngineCapabilities in engine.ts

interface EngineCapabilities {
  // ... existing fields unchanged ...

  /**
   * Whether this engine instance was created with a caustic strategy.
   * 'none' means standard NEE only; consumers should not expect fast
   * caustic convergence. Other values indicate the active strategy.
   */
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
}
```

No changes to `Material`, `Scene`, or `FrameInput`.

## 3. Algorithmic Notes

**Manifold Next-Event Estimation** — Hanika, Droske, Fascione (Computer Graphics Forum 34(4), 2015) [1]:

Given a diffuse shading point x_d and a sampled light position x_L, MNEE seeks intermediate specular vertices x_1 … x_k such that the path x_d → x_1 → … → x_k → x_L satisfies Fermat's principle at each specular vertex (the half-vector manifold constraint: H_i = 0). The algorithm uses Newton's method on the manifold constraint function:

```
C(x_1, …, x_k) = [H_1, H_2, …, H_k]^T = 0
where H_i = the specular constraint at vertex i (zero when half-vector bisects correctly)
```

Each Newton step updates all specular vertices simultaneously. Convergence is typically achieved in 4–10 iterations for planar glass surfaces; curved surfaces may diverge or require more steps.

**MIS integration:** MNEE connections are importance-sampled and combined with standard BSDF-sampled paths using Multiple Importance Sampling (MIS), exactly as standard NEE connections are. This ensures unbiasedness and avoids double-counting.

**Known limitations:**
- Fails to converge for highly rough specular surfaces (roughness > ~0.1). For rough glass, BSDF sampling is more efficient and MNEE attempts should be skipped.
- Very long specular chains (> 4 vertices) are expensive and rarely significant in architectural scenes.
- Requires that specular surfaces have analytically differentiable IOR/roughness fields at vertices — i.e., rough glass with a normal map at the manifold vertex needs the normal map's derivatives, which adds complexity.

**Photon mapping alternative** — Jensen 1996 [2]: Forward trace photons from lights, store them in a KD-tree. During rendering, density estimate at diffuse shading points yields caustic radiance. Biased (blur from kernel radius) but robust to all specular path topologies including rough glass and curved geometry. Memory cost: O(N_photons) per scene; N_photons = 100k–1M is typical.

## 4. Backend Implementation Guidance

**`@vitrum/pt-webgl` (forked three-gpu-pathtracer):**
MNEE requires per-vertex manifold-walk state — a small buffer of specular-vertex positions and normals for the Newton iteration. In three-gpu-pathtracer's shader architecture, this means extending the path-bounce loop to maintain a specular-chain accumulator and launching the Newton solve when a diffuse vertex is reached (checking if any specular geometry lies between shading point and light). Implementation complexity is moderate (est. 300–500 lines of GLSL). GPU-side Newton solves are feasible; the iteration count is fixed per launch.

`photon-map` is not currently supported in three-gpu-pathtracer. It would require a separate photon-tracing pass writing to a sorted buffer, plus a density estimation step at shade time. A GPU KD-tree or hash grid is the standard WebGL approach (see Dachsbacher & Stamminger 2005 for the hash grid variant). Significant implementation work.

**`@vitrum/pt-webgpu` (future):**
WebGPU compute shaders are well-suited to MNEE: the manifold walk is a small iterative compute kernel, and the per-path specular-chain state fits naturally in path-state storage buffers. Photon mapping maps to a two-pass compute: (1) photon trace pass storing into a storage buffer, (2) shade pass reading the photon buffer via a spatial query.

**`@vitrum/walkaround-hybrid` (real-time GI):**
Neither MNEE nor photon mapping is compatible with real-time frame cadence at production quality. For the walkaround engine, specular caustics are best approximated by: (a) increasing the light-path sampling probability for refractive surfaces (importance-sample the glass geometry from the light), or (b) a screen-space caustic approximation (blur the specular refraction buffer onto the diffuse floor). Neither is physically accurate; document in capabilities.

## 5. Validation

**Reference scene — refractive plane caustic:** A flat glass plane (ior 1.52, roughness 0) 3 m above a diffuse floor, lit by a small area light above the glass. Standard NEE: caustic converges only after > 10,000 SPP. MNEE: caustic pattern should be clearly visible at < 500 SPP. Compare shape and intensity against Mitsuba 3 `path` integrator with `manifold_nee = true` [3].

**Colored caustic test:** Add Beer-Lambert attenuation (cobalt blue, attenuationDistance = 3 mm, slab thickness = 5 mm) to the glass plane. The floor caustic should show a saturated blue patch. Validate RGB hue against an analytic Beer-Lambert calculation through a 5 mm path.

**Published reference:** Hanika et al. 2015 [1] Figure 3 — the glass-sphere caustic scene. The MNEE result converges the caustic ring in < 1 min; PT without MNEE shows only fireflies at the same sample budget.

## 6. Consumer-Side Integration

A consumer enabling MNEE at engine creation:

```typescript
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';

const engine = await createPTEngine_WebGL2({
  device: renderer,
  maxBounces: 8,
  maxSamplesPerPixel: 4096,
  causticStrategy: 'manifold-nee',
  mneeMaxIterations: 8,
  mneeMaxChainLength: 3,
});

// Query at runtime to communicate capability to the UI:
if (engine.capabilities.causticStrategy === 'manifold-nee') {
  console.log('Refractive caustics: MNEE active');
}
```

No per-material or per-scene configuration is needed. MNEE operates automatically on any refractive surface in the scene (transmission > 0, roughness < threshold).

## 7. Open Questions

- **Roughness threshold for skipping MNEE:** At what material roughness should MNEE attempts be suppressed in favor of BSDF sampling only? The Hanika 2015 paper suggests roughness > 0.05 degrades convergence. An adaptive per-vertex decision (check Jacobian condition number) is more robust but adds complexity.
- **Multiple refractive surfaces per chain:** When multiple colored panels are stacked, the manifold walk must solve through all of them simultaneously. Chain length scales cost linearly; testing with 4–6 panels in series is recommended before settling the `mneeMaxChainLength` default.

## 8. References

[1] J. Hanika, M. Droske, L. Fascione. "Manifold Next Event Estimation." *Computer Graphics Forum*, vol. 34, no. 4, pp. 87–97, 2015. DOI: 10.1111/cgf.12681.

[2] H.W. Jensen. "Global Illumination using Photon Maps." In *Proceedings of the Eurographics Workshop on Rendering Techniques 1996*, pp. 21–30, 1996. Available at graphics.ucsd.edu/~henrik/papers/photon_map/.

[3] Mitsuba 3 documentation. "Path Integrator." Available at mitsuba-renderer.org/docs/. (manifold_nee flag in the `path` integrator.)

[4] C. Dachsbacher, M. Stamminger. "Photon Splatting using a View-Sample Cluster Hierarchy." *ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games*, 2005. (GPU hash-grid photon map — reference for WebGL photon mapping implementation.)

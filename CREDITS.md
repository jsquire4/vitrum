# Credits

vitrum builds on decades of foundational work in physically-based rendering, real-time global illumination, and web graphics infrastructure. This document attributes prior work that vitrum depends on — directly (as a software dependency) or indirectly (as the basis of an algorithm reimplemented in vitrum's WGSL/GLSL).

## Software dependencies

- **three.js** — MIT — Mr.doob et al. — `https://github.com/mrdoob/three.js`
- **three-gpu-pathtracer** — MIT — Garrett Johnson (gkjohnson) — `https://github.com/gkjohnson/three-gpu-pathtracer`
- **three-mesh-bvh** — MIT — Garrett Johnson — `https://github.com/gkjohnson/three-mesh-bvh`

## Ecosystem / prior art (not direct dependencies)

The following projects are credited as prior art and ecosystem context. They are not imported by vitrum (vitrum is host-agnostic and carries no React dependency).

- **@react-three/fiber** — MIT — poimandres
- **@react-three/drei** — MIT — poimandres
- **@react-three/gpu-pathtracer** — MIT — poimandres
- **@react-three/postprocessing** — MIT — poimandres
- **postprocessing** — Zlib — Raoul van Rüschen

## Foundational techniques

Each technique is cited at its implementation site in the source code. This list is the canonical record.

### Path tracing

- **Disney BSDF** — Brent Burley, "Physically-Based Shading at Disney," SIGGRAPH 2012 Course Notes
- **Beer-Lambert attenuation** — Pierre Bouguer (1729), August Beer (1852), Johann Heinrich Lambert (1760)
- **Cauchy IOR formula** — Augustin-Louis Cauchy (1830)
- **Schlick Fresnel approximation** — Christophe Schlick, "An Inexpensive BRDF Model for Physically-based Rendering," Eurographics 1994
- **GGX visible normal distribution function (VNDF) sampling** — Eric Heitz, "Sampling the GGX Distribution of Visible Normals," JCGT 7(4):1–13, 2018. <https://jcgt.org/published/0007/04/01/paper.pdf>
- **Multiple importance sampling (MIS) — power heuristic** — Eric Veach, "Robust Monte Carlo Methods for Light Transport Simulation," PhD thesis, Stanford 1997
- **RIS (Resampled Importance Sampling) estimator** — Justin Talbot, David Cline, Parris Egbert, "Importance Resampling for Global Illumination," EGSR 2005
- **Sobol low-discrepancy sequence** — Ilya Sobol (1967), as adapted in the three-gpu-pathtracer library

### Geometry & acceleration

- **Möller-Trumbore ray-triangle intersection** — Tomas Möller, Ben Trumbore, "Fast, Minimum Storage Ray-Triangle Intersection," Journal of Graphics Tools 2(1):21–28, 1997
- **Binned SAH BVH construction** — Ingo Wald, "On fast Construction of SAH-based Bounding Volume Hierarchies," IEEE Symposium on Interactive Ray Tracing, 2007 (K=16 bins per axis)
- **Robust ray-AABB slab traversal** — Amy Williams, Steve Barrus, R. Keith Morley, Peter Shirley, "An Efficient and Robust Ray-Box Intersection Algorithm," Journal of Graphics Tools 10(1), 2005 (safe-inverse-direction handling of `dir.* == 0`)

### Real-time global illumination

- **DDGI (Dynamic Diffuse Global Illumination)** — Zander Majercik, Jean-Philippe Guertin, Derek Nowrouzezahrai, Morgan McGuire, "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields," JCGT 2019
- **Radiance Cascades** — Alexander Sannikov, "Radiance Cascades: A Novel Approach to Calculating Global Illumination," 2023
- **ReSTIR DI (Reservoir-based Spatiotemporal Importance Resampling)** — Benedikt Bitterli, Chris Wyman, Matt Pharr, Peter Shirley, Aaron Lefohn, Wojciech Jarosz, "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting," SIGGRAPH 2020
- **ReSTIR GI (diffuse-indirect resampling)** — Zander Majercik, Adam Marrs, Josef Spjut, Morgan McGuire, "Dynamic Diffuse Global Illumination Resampling," SIGGRAPH 2021 (§4.2 initial-sample RIS, §4.5 temporal/spatial reuse)

### Ambient occlusion

- **GTAO (Ground-Truth Ambient Occlusion)** — Jorge Jiménez, Xian-Chun Wu, Angelo Pesce, Adrian Jarabo, "Practical Realtime Strategies for Accurate Indirect Occlusion," SIGGRAPH 2016 Course Notes (§4.2 Eq. 11). Reference implementation cross-checked against Intel XeGTAO (<https://github.com/GameTechDev/XeGTAO>).

### Volumetric & participating media

- **Henyey-Greenstein phase function** — Louis Henyey, Jesse Greenstein, "Diffuse radiation in the Galaxy," Astrophysical Journal 1941
- **Equi-angular volume scatter PDF** — Christopher Kulla, Marcos Fajardo, "Importance Sampling Techniques for Path Tracing in Participating Media," Eurographics 2012

### Spectral rendering

- **Jakob+Hanika spectral upsampling** — Wenzel Jakob, Johannes Hanika, "A Low-Dimensional Function Space for Efficient Spectral Upsampling," Computer Graphics Forum 2019
- **Hero-wavelength spectral path tracing** — Alexander Wilkie, Sehera Nawaz, Marc Droske, Andrea Weidlich, Johannes Hanika, "Hero Wavelength Spectral Sampling," Computer Graphics Forum 2014

### Denoising

- **À-trous wavelet GI denoiser** — Holger Dammertz, Daniel Sewtz, Johannes Hanika, Hendrik P. A. Lensch, "Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering," HPG 2010
- **SVGF (Spatiotemporal Variance-Guided Filtering)** — Christoph Schied et al., "Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination," HPG 2017
- **Variance-clamped temporal accumulation** — Schied et al. SVGF (above)
- **Intel Open Image Denoise (OIDN)** — Intel Corporation, Apache-2.0 (model weights), executed in-browser via ONNX Runtime Web

### Tone mapping & post-processing

- **ACES filmic tone mapping** — Academy of Motion Picture Arts and Sciences (AMPAS), Krzysztof Narkowicz approximation, Stephen Hill exposure adjustment

### Path guiding

- **PPG (Practical Path Guiding)** — Thomas Müller, Markus Gross, Jan Novák, "Practical Path Guiding for Efficient Light-Transport Simulation," EGSR 2017

### Caustic methods

- **Bidirectional path tracing (BDPT)** — Eric P. Lafortune, Yves D. Willems, "Bi-directional Path Tracing," CompuGraphics 1993 (partial: vertex tables + connection-strategy MIS math under `shared-samplers/src/bdpt*` — (Sprint 10c applied 2026-05-12; GPU visual A/B remains follow-up))

### Textbook references

Foundational textbooks cited from JSDoc comments across the codebase:

- **PBR4e** — Matt Pharr, Wenzel Jakob, Greg Humphreys, "Physically Based Rendering: From Theory to Implementation," 4th edition, MIT Press, 2023. Cited from `shared-samplers/src/{bdptVertex,bdptMIS,hgPhase}.ts`, `shared-samplers/__tests__/pdfNormalization.test.ts`, `pt-webgpu/src/wgsl/pathTrace/*.wgsl.ts`. Referenced for: HG phase normalisation (§11.4), equi-angular volume PDF (§14.1.2), BDPT MIS recursive ratio (§16.3.5 Eq. 16.16), hero-wavelength MIS reconstruction (§4.6.2), barycentric reconstruction (§6.8).
- **Veach 1997** — Eric Veach, "Robust Monte Carlo Methods for Light Transport Simulation," PhD thesis, Stanford University. Cited from `shared-samplers/src/bdptMIS.ts` (§10.3 BDPT MIS connection formulae). The canonical reference for multiple-importance-sampling theory.

## Candidate techniques (not yet implemented)

The following techniques are tracked as roadmap candidates. They are documented here for transparency about what vitrum is *not* yet shipping, and to credit the prior art that future implementations would build on. See `plan/` for current status.

- **BMFR (Blockwise Multi-Order Feature Regression)** — Matias Koskela, Kalle Immonen, Markku Mäkitalo, Alessandro Foi, Timo Viitanen, Pekka Jääskeläinen, "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing Reconstruction," ACM TOG 2019. `shared-denoisers/src/index.ts`: "BMFR remains a roadmap candidate; no BMFR module is exported."
- **ReSTIR BDPT** — Hedstrom et al., "Bidirectional ReSTIR Path Tracing with Caustics," ACM TOG 2025. Not implemented as of 2026-05-17.
- **Vertex Connection and Merging (VCM)** — Iliyan Georgiev, Jaroslav Křivánek, Tomáš Davidovič, Philipp Slusallek, "Light Transport Simulation with Vertex Connection and Merging," ACM TOG 2012. Not implemented as of 2026-05-17.

## Asset attribution

vitrum's example scenes use:

- **HDRIs** — [Polyhaven](https://polyhaven.com), CC0
- **Wood textures** — [Polyhaven](https://polyhaven.com), CC0

## Citing vitrum

If vitrum's specific contributions are useful in your work — particularly the WebGPU layered DDGI + RC + ReSTIR DI composition, the WebGL2 normalMap-perturbed NEE shadow rays, or the hybrid analytic-CSG + BVH-mesh intersection scheme — please cite this repository in your acknowledgments.

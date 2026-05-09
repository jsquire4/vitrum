# Credits

vitrum builds on decades of foundational work in physically-based rendering, real-time global illumination, and web graphics infrastructure. This document attributes prior work that vitrum depends on — directly (as a software dependency) or indirectly (as the basis of an algorithm reimplemented in vitrum's WGSL/GLSL).

## Software dependencies

- **three.js** — MIT — Mr.doob et al. — `https://github.com/mrdoob/three.js`
- **three-gpu-pathtracer** — MIT — Garrett Johnson (gkjohnson) — `https://github.com/gkjohnson/three-gpu-pathtracer`
- **three-mesh-bvh** — MIT — Garrett Johnson — `https://github.com/gkjohnson/three-mesh-bvh`
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
- **Multiple importance sampling (MIS) — power heuristic** — Eric Veach, "Robust Monte Carlo Methods for Light Transport Simulation," PhD thesis, Stanford 1997
- **Sobol low-discrepancy sequence** — Ilya Sobol (1967), as adapted in the three-gpu-pathtracer library

### Real-time global illumination

- **DDGI (Dynamic Diffuse Global Illumination)** — Zander Majercik, Jean-Philippe Guertin, Derek Nowrouzezahrai, Morgan McGuire, "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields," JCGT 2019
- **Radiance Cascades** — Alexander Sannikov, "Radiance Cascades: A Novel High-Resolution Formal Solution for Multidimensional Non-LTE Radiative Transfer," 2024
- **ReSTIR DI (Reservoir-based Spatiotemporal Importance Resampling)** — Benedikt Bitterli, Chris Wyman, Matt Pharr, Peter Shirley, Aaron Lefohn, Wojciech Jarosz, "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting," SIGGRAPH 2020

### Volumetric & participating media

- **Henyey-Greenstein phase function** — Louis Henyey, Jesse Greenstein, "Diffuse radiation in the Galaxy," Astrophysical Journal 1941
- **Equi-angular volume scatter PDF** — Christopher Kulla, Marcos Fajardo, "Importance Sampling Techniques for Path Tracing in Participating Media," Eurographics 2012

### Spectral rendering

- **Jakob+Hanika spectral upsampling** — Wenzel Jakob, Johannes Hanika, "A Low-Dimensional Function Space for Efficient Spectral Upsampling," Computer Graphics Forum 2019
- **Hero-wavelength spectral path tracing** — Alexander Wilkie, Sehera Nawaz, Marc Droske, Andrea Weidlich, Johannes Hanika, "Hero Wavelength Spectral Sampling," Computer Graphics Forum 2014

### Denoising

- **À-trous wavelet GI denoiser** — Holger Dammertz, Daniel Sewtz, Johannes Hanika, Hendrik P. A. Lensch, "Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering," HPG 2010
- **SVGF (Spatiotemporal Variance-Guided Filtering)** — Christoph Schied et al., "Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination," HPG 2017
- **BMFR (Blockwise Multi-Order Feature Regression)** — Matias Koskela, Kalle Immonen, Markku Mäkitalo, Alessandro Foi, Timo Viitanen, Pekka Jääskeläinen, "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing Reconstruction," ACM TOG 2019
- **Variance-clamped temporal accumulation** — Schied et al. SVGF (above)

### Tone mapping & post-processing

- **ACES filmic tone mapping** — Academy of Motion Picture Arts and Sciences (AMPAS), Krzysztof Narkowicz approximation, Stephen Hill exposure adjustment

### Path guiding

- **PPG (Practical Path Guiding)** — Thomas Müller, Markus Gross, Jan Novák, "Practical Path Guiding for Efficient Light-Transport Simulation," EGSR 2017

### Caustic methods

- **Bidirectional path tracing (BDPT)** — Eric P. Lafortune, Yves D. Willems, "Bi-directional Path Tracing," CompuGraphics 1993
- **ReSTIR BDPT** — Hedstrom et al., "Bidirectional ReSTIR Path Tracing with Caustics," ACM TOG 2025
- **Vertex Connection and Merging (VCM)** — Iliyan Georgiev, Jaroslav Křivánek, Tomáš Davidovič, Philipp Slusallek, "Light Transport Simulation with Vertex Connection and Merging," ACM TOG 2012

## Asset attribution

vitrum's example scenes use:

- **HDRIs** — [Polyhaven](https://polyhaven.com), CC0
- **Wood textures** — [Polyhaven](https://polyhaven.com), CC0

## Citing vitrum

If vitrum's specific contributions are useful in your work — particularly the WebGPU layered DDGI + RC + ReSTIR DI composition, the WebGL2 normalMap-perturbed NEE shadow rays, or the hybrid analytic-CSG + BVH-mesh intersection scheme — please cite this repository in your acknowledgments.

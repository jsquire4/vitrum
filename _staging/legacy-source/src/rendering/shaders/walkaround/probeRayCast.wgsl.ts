/**
 * Probe ray-cast compute kernel (§6).
 *
 * Each thread traces one ray for one probe in one cascade.
 * Writes radiance (rgb) + escaped-flag (a) into cascadeOut.
 * alpha = 0 means ray escaped the interval (for merge pass).
 * alpha = 1 means ray hit something (local contribution only).
 *
 * Glass transmission shortcut (§6.3): on glass hit, continues
 * one step through the slab to capture transmitted env/sun light.
 *
 * See §5.2 for the octahedron direction encoding.
 *
 * Structure note: wgslFn() expects the code string to start with `fn`.
 * Struct definitions are exported as wgsl() nodes and passed as includes.
 */

import { wgslFn, wgsl } from 'three/tsl';
import {
  bvhIntersectFirstHit,
  rayStruct,
  bvhNodeStruct,
  bvhNodeBoundsStruct,
  intersectionResultStruct,
  intersectsTriangle,
  intersectTriangles,
  intersectsBounds,
} from 'three-mesh-bvh/src/webgpu/index.js';
import { OCTAHEDRAL_WGSL } from '../../scene/walkaround/wgsl/octahedral.wgsl';

// Canonical octahedral helpers shared with DDGI. Strip the leading
// header comments so wgslFn's entry-detection regex
// (/^[fn]*\s*([a-z_0-9]+)?\s*\(/) sees `fn octEncode(...)` as the first
// declaration — it can't skip a leading `//` comment.
const OCTAHEDRAL_INCLUDE = wgsl(OCTAHEDRAL_WGSL.replace(/^[\s\S]*?(?=fn\s+oct)/, ''));

/** CascadeUniforms struct (must match layout in cascadeDispatch.ts). */
export const cascadeUniformsStruct = wgsl(/* wgsl */`
  struct CascadeUniforms {
    probeOriginWorld : vec3f,
    _pad0            : f32,
    roomSize         : vec3f,
    _pad1            : f32,
    probeCount       : vec3u,
    raysPerProbe     : u32,
    rayGridSize      : u32,
    intervalNear     : f32,
    intervalFar      : f32,
    cascadeIndex     : u32,
    sunDirection     : vec3f,
    _pad2            : f32,
    sunColor         : vec3f,
    envIntensity     : f32,
    frameSeed        : u32,
    lastCascade      : u32,
    _pad4            : vec2u,
  };
`);

/**
 * MaterialEntry struct — flat f32 fields only (no vec3f members).
 *
 * WGSL vec3f has AlignOf=16 bytes, which introduces implicit 8-byte padding
 * gaps after scalar fields (e.g. after ior, after metalness). The CPU packs
 * materialTable as 16 contiguous floats with no gaps, so the struct must use
 * individual f32 members to guarantee the CPU and GPU layouts are identical.
 *
 * Layout (16 × f32 = 64 bytes per entry, matches itemSize=16 in bvhCompute.ts):
 *   [0]  colorR         [1]  colorG         [2]  colorB         [3]  colorA
 *   [4]  transmission   [5]  ior            [6]  attenColorR    [7]  attenColorG
 *   [8]  attenColorB    [9]  attenDist      [10] roughness      [11] metalness
 *   [12] emissiveR      [13] emissiveG      [14] emissiveB      [15] thickness
 */
export const materialEntryStruct = wgsl(/* wgsl */`
  struct MaterialEntry {
    colorR      : f32,
    colorG      : f32,
    colorB      : f32,
    colorA      : f32,
    transmission: f32,
    ior         : f32,
    attenColorR : f32,
    attenColorG : f32,
    attenColorB : f32,
    attenDist   : f32,
    roughness   : f32,
    metalness   : f32,
    emissiveR   : f32,
    emissiveG   : f32,
    emissiveB   : f32,
    thickness   : f32,
  };
`);

/** PCG hash + equirect UV helpers. octDecode now comes from the
 *  canonical OCTAHEDRAL_WGSL include above (signed-input convention,
 *  call sites do `octDecode(uv * 2.0 - 1.0)` to remap from [0,1]). */
const probeRayHelpers = wgslFn(/* wgsl */`
  fn pcgHash(seed: u32) -> f32 {
    var s = seed * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967295.0;
  }

  fn dirToEquirectUV(d: vec3f) -> vec2f {
    let phi   = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    return vec2f(phi / (2.0 * 3.14159265) + 0.5, 1.0 - theta / 3.14159265);
  }
`);

/**
 * Glass-aware sun visibility helper (caustic enabler).
 *
 * Traces a shadow ray from the receiver toward the sun and returns a
 * per-channel visibility multiplier:
 *   - Unobstructed     → vec3f(1, 1, 1)
 *   - Hit opaque       → vec3f(0)
 *   - Hit glass        → glass.color × Beer-Lambert(atten, thickness, attenDist),
 *                        then continue past the slab and recurse (bounded to 2 crossings).
 *
 * Why this matters: without this, sun-shadow tests treat glass as opaque, so
 * receivers behind the panel never see direct sun in the cascade radiance —
 * the ONLY signature of a caustic is glass-tinted direct light landing on
 * the floor / wall. The pre-existing `transContrib` block fires only when
 * the probe ray itself hits a glass tri, which is a small fraction of probe
 * hits; the dominant case (probe hits floor → asks "am I in sun?") was
 * always returning "no" because the panel blocked the shadow ray.
 *
 * Cost: 1-2 extra BVH casts per probe hit when the shadow ray traverses
 * glass. Probes whose sun direction misses the panel pay only the original
 * single cast. Total compute increase is bounded to ~2× per cell that sees
 * the panel along the sun direction.
 *
 * Bounded loop (≤ 2 glass crossings) keeps WGSL simple — a stained-glass
 * panel is at most one slab between any receiver and the sun for our scenes;
 * the second crossing handles obscure corner cases (panel-on-panel) without
 * runaway cost.
 */
const sunVisibilityHelper = wgslFn(/* wgsl */`
  fn traceSunVisibility(
    bvh:           ptr<storage, array<BVHNode>,        read>,
    geom_index:    ptr<storage, array<vec3u>,          read>,
    geom_position: ptr<storage, array<vec3f>,          read>,
    materials:     ptr<storage, array<MaterialEntry>,  read>,
    triMatId:      ptr<storage, array<u32>,            read>,
    origin:        vec3f,
    sunDir:        vec3f,
  ) -> vec3f {
    var visibility = vec3f(1.0);
    var rayOrigin  = origin;
    // Bounded glass-crossing loop. Upper bound 2 handles single-slab and
    // edge-case double-slab paths; ≥3 stained-glass panels in series is not
    // a configuration the walkaround pipeline produces.
    for (var iter: u32 = 0u; iter < 3u; iter = iter + 1u) {
      var sRay = Ray();
      sRay.origin    = rayOrigin;
      sRay.direction = sunDir;
      let sHit = bvhIntersectFirstHit(geom_index, geom_position, bvh, sRay);
      if (!sHit.didHit) {
        // Reached sky — sun is unobstructed (modulo accumulated glass tint).
        return visibility;
      }
      let sMatId = (*triMatId)[sHit.indices.w];
      let sMat   = (*materials)[sMatId];
      if (sMat.transmission <= 0.5) {
        // Opaque occluder — sun is fully blocked.
        return vec3f(0.0);
      }
      // Glass slab — apply Beer-Lambert × tint, then continue past the slab.
      let gThick    = max(0.001, sMat.thickness);
      let gAttenCol = vec3f(sMat.attenColorR, sMat.attenColorG, sMat.attenColorB);
      let gColor    = vec3f(sMat.colorR,      sMat.colorG,      sMat.colorB);
      let beerAtten = exp(-gAttenCol * (gThick / max(0.001, sMat.attenDist)));
      visibility = visibility * gColor * beerAtten;
      // Step past the slab. The 0.5 inch step matches the existing
      // glass-shortcut convention in the probe ray and is comfortably larger
      // than any glass slab thickness in current scenes (~0.25-0.4 inches).
      let hitPos = rayOrigin + sunDir * sHit.dist;
      rayOrigin  = hitPos + sunDir * 0.5;
    }
    // Loop exhausted (more than 2 glass crossings) — treat as fully attenuated.
    return vec3f(0.0);
  }
`, [
  rayStruct,
  bvhNodeStruct,
  bvhNodeBoundsStruct,
  intersectionResultStruct,
  intersectsTriangle,
  intersectTriangles,
  intersectsBounds,
  bvhIntersectFirstHit,
  materialEntryStruct,
]);

/** The probe ray-cast compute kernel, exported as a wgslFn.
 *  Code string must start with `fn` — structs are passed as wgsl() includes.
 *
 *  CascadeUniforms is passed as a storage buffer (ptr<storage, array<CascadeUniforms>, read>)
 *  instead of a plain uniform, because three.js TSL does not support custom struct uniform
 *  types. A storage buffer with READ_ONLY access is semantically equivalent. */
export const probeRayCastKernel = wgslFn(
  /* wgsl */`
  fn probeRayCastKernel(
    bvh:           ptr<storage, array<BVHNode>,              read>,
    geom_index:    ptr<storage, array<vec3u>,                 read>,
    geom_position: ptr<storage, array<vec3f>,                 read>,
    materials:     ptr<storage, array<MaterialEntry>,         read>,
    triMatId:      ptr<storage, array<u32>,                   read>,
    cascadeOut:    ptr<storage, array<vec4f>,                 read_write>,
    envMap:        texture_2d<f32>,
    envSampler:    sampler,
    u_arr:         ptr<storage, array<CascadeUniforms>,       read>,
    index:         u32,
  ) -> void {
    let u = (*u_arr)[0];
    let totalProbes  = u.probeCount.x * u.probeCount.y * u.probeCount.z;
    let totalThreads = totalProbes * u.raysPerProbe;
    if (index >= totalThreads) { return; }

    let probeIdx = index / u.raysPerProbe;
    let rayIdx   = index % u.raysPerProbe;

    // Probe world position from 3D grid index.
    let pz = probeIdx / (u.probeCount.x * u.probeCount.y);
    let py = (probeIdx / u.probeCount.x) % u.probeCount.y;
    let px = probeIdx % u.probeCount.x;
    let probeUV  = (vec3f(f32(px), f32(py), f32(pz)) + 0.5) / vec3f(u.probeCount);
    let probePos = u.probeOriginWorld + probeUV * u.roomSize;

    // Ray direction via octahedron grid + jitter.
    let gx = f32(rayIdx % u.rayGridSize);
    let gy = f32(rayIdx / u.rayGridSize);
    let jSeed = (probeIdx * 0x9E3779B9u + rayIdx) ^ u.frameSeed;
    let jitter = vec2f(pcgHash(jSeed), pcgHash(jSeed * 7919u + 1u));
    let rayUV   = (vec2f(gx, gy) + jitter) / f32(u.rayGridSize);
    let rayDir  = octDecode(rayUV * 2.0 - 1.0);

    // Build ray for this cascade's interval.
    var ray = Ray();
    ray.origin    = probePos + rayDir * u.intervalNear;
    ray.direction = rayDir;
    let maxT = u.intervalFar - u.intervalNear;

    var radiance     = vec3f(0.0);
    var escaped      = true;

    let hit = bvhIntersectFirstHit(geom_index, geom_position, bvh, ray);

    if (!hit.didHit || hit.dist > maxT) {
      // Ray escaped this cascade's interval.
      // Only the last cascade samples the env (others defer to cascade merge).
      if (u.cascadeIndex == u.lastCascade) {
        let envUV  = dirToEquirectUV(rayDir);
        let sample = textureSampleLevel(envMap, envSampler, envUV, 0.0);
        radiance   = sample.rgb * u.envIntensity;
        escaped = false;  // env was sampled — don't merge from upper
      }
      // else: leave escaped=true, merge will fill from above
    } else {
      escaped = false;  // hit something within interval — local contribution

      let triIdx = hit.indices.w;
      let matId  = (*triMatId)[triIdx];
      let mat    = (*materials)[matId];

      let hitPos = ray.origin + ray.direction * hit.dist;
      let n      = hit.normal;

      // Reconstruct material color vectors from flat f32 fields (struct uses f32 not vec3f).
      let matColor    = vec3f(mat.colorR,     mat.colorG,     mat.colorB);
      let matAtten    = vec3f(mat.attenColorR, mat.attenColorG, mat.attenColorB);
      let matEmissive = vec3f(mat.emissiveR,  mat.emissiveG,  mat.emissiveB);

      // Direct sun term — Lambertian: Lo = L_sun × albedo × cos(N,L) / π
      // 0.31831 = 1/π (correct for Lambertian diffuse BRDF with cosine factor).
      //
      // Glass-aware sun visibility (caustic enabler): traceSunVisibility
      // returns a per-channel multiplier instead of a binary in-shadow test.
      // When the shadow ray passes through stained glass, the multiplier
      // encodes the glass colour × Beer-Lambert attenuation; opaque hits
      // return 0; clear sky returns 1. This is what makes a glass-tinted
      // caustic appear on the floor / walls under the panel — without it,
      // the panel reads as a fully opaque sun-blocker for every probe ray
      // whose receiver sits behind it.
      let sunVis = traceSunVisibility(
        bvh, geom_index, geom_position, materials, triMatId,
        hitPos + n * 0.01,   // 0.01" self-intersect bias
        u.sunDirection,
      );
      let nDotL  = max(0.0, dot(n, u.sunDirection));
      let directSun = u.sunColor * matColor * nDotL * 0.31831 * sunVis;

      // Emissive term.
      let emissive = matEmissive;

      // Glass transmission shortcut.
      // Beer-Lambert: transmittance = exp(-attenuationColor × thickness / attenuationDistance)
      // Uses mat.thickness (actual slab thickness in scene units, stored in slot [15]).
      var transContrib = vec3f(0.0);
      if (mat.transmission > 0.5) {
        let glassThickness = max(0.001, mat.thickness);
        let beerAttenColor = exp(-matAtten * (glassThickness / max(0.001, mat.attenDist)));
        var refRay = Ray();
        refRay.origin    = hitPos + ray.direction * 0.5;
        refRay.direction = ray.direction;
        let secondHit = bvhIntersectFirstHit(geom_index, geom_position, bvh, refRay);
        if (!secondHit.didHit) {
          let envUV = dirToEquirectUV(refRay.direction);
          let envS  = textureSampleLevel(envMap, envSampler, envUV, 0.0);
          transContrib = envS.rgb * u.envIntensity * beerAttenColor * matColor;
        } else {
          let secondPos   = refRay.origin + refRay.direction * secondHit.dist;
          let secondMatId = (*triMatId)[secondHit.indices.w];
          let secondMat   = (*materials)[secondMatId];
          let secondColor = vec3f(secondMat.colorR, secondMat.colorG, secondMat.colorB);
          // Use the glass-aware shadow trace so a sun ray re-entering the
          // glass on its way back out is attenuated correctly.
          let sunVis2 = traceSunVisibility(
            bvh, geom_index, geom_position, materials, triMatId,
            secondPos + secondHit.normal * 0.01,  // 0.01" self-intersect bias
            u.sunDirection,
          );
          let nDotL2 = max(0.0, dot(secondHit.normal, u.sunDirection));
          transContrib = u.sunColor * secondColor * nDotL2 * 0.31831
                         * beerAttenColor * matColor * sunVis2;
        }
      }

      radiance = directSun + emissive + transContrib;
    }

    // Write output: alpha=0 means escaped (merge from upper cascade), 1 means hit.
    let outIdx = probeIdx * u.raysPerProbe + rayIdx;
    let escapedF = select(1.0, 0.0, escaped);
    (*cascadeOut)[outIdx] = vec4f(radiance, escapedF);
  }
  `,
  [
    cascadeUniformsStruct,
    materialEntryStruct,
    OCTAHEDRAL_INCLUDE,
    probeRayHelpers,
    sunVisibilityHelper,
    bvhIntersectFirstHit,
    rayStruct,
    bvhNodeStruct,
    bvhNodeBoundsStruct,
    intersectionResultStruct,
    intersectsTriangle,
    intersectTriangles,
    intersectsBounds,
  ],
);

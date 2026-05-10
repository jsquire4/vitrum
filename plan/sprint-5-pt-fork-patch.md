# Sprint 5 — PT Fork Patch Plan

**Sprint goal**: Analytic H-channel came geometry + MRT G-buffer scaffold.

**Fork repo**: `~/projects/three-gpu-pathtracer/` branch `phase4-normalmap-shadow-rays`.
**Do NOT modify the fork without explicit user instruction.** This document
specifies what must be patched in the fork once Sprint 5 kicks off.

---

## 1. Analytic came intersection

### 1.1 `src/shader/shaders/pathtracing/shape_intersection_functions.glsl.js`

Add two new analytic intersection routines.  Both are inserted **before** the
BVH-traversal helpers so they can be called from `traceScene`.

#### `intersectCameSegment`

Performs ray–H-channel intersection.  The H-channel cross-section is
approximated as a capsule for primary intersection, then refined with a box
clip for the web recess.

```glsl
// CameSegment UBO layout (std140, 16 floats per entry):
//   [0..2]  startWorld.xyz
//   [3]     railWidth
//   [4..6]  endWorld.xyz
//   [7]     blockHeight
//   [8]     webThickness
//   [9..15] padding

struct CameSegment {
  vec3  startWorld;
  float railWidth;
  vec3  endWorld;
  float blockHeight;
  float webThickness;
  float _pad[7];       // std140 16-float boundary
};

// Returns hit distance in `tHit`; normal in `nHit`.
// Returns false if no intersection in (tMin, tMax).
bool intersectCameSegment(
  vec3 ro, vec3 rd,
  CameSegment seg,
  float tMin, float tMax,
  out float tHit, out vec3 nHit
);
```

Implementation strategy:
1. Transform ray into the segment's local frame (align +Y with the rail axis).
2. Test ray against a capsule of radius `max(railWidth, blockHeight) / 2` to
   cull quickly.
3. For rays that pass the capsule test, perform an oriented box intersection:
   half-extents `(railWidth/2, length/2, blockHeight/2)` in local space.
4. For the web recess (H-profile web), subtract an inner box of half-extents
   `(webThickness/2, length/2, blockHeight/2)`.  Use CSG difference: the
   fragment hit is only valid if it's inside the outer box **but outside the
   inner box** (the rail flanges) OR inside the inner box at flange height.
   This is simplified to: hit the outer box, compute local intersection point,
   and return the flange normal if the point falls within the flange region.
5. Normal is the outward-pointing face normal in world space.

**Fallback**: if CSG logic is too expensive per-ray, use the capsule
approximation only for Sprint 5.  Exact H-profile is a Sprint 5b refinement.

#### `intersectCameNode`

Sphere intersection for solder joints.

```glsl
// CameNode UBO layout (std140, 4 floats per entry / vec4):
//   [0..2] position.xyz
//   [3]    radius

struct CameNode {
  vec3  position;
  float radius;
};

// Returns hit distance in `tHit`; normal in `nHit`.
// Returns false if no intersection in (tMin, tMax).
bool intersectCameNode(
  vec3 ro, vec3 rd,
  CameNode node,
  float tMin, float tMax,
  out float tHit, out vec3 nHit
);
```

Implementation: standard ray–sphere intersection (quadratic formula).

---

### 1.2 `src/shader/shaders/pathtracing/trace_scene_function.glsl.js`

Modify `traceScene` to run analytic-came intersection in parallel with BVH
traversal, then pick the closest hit.

```glsl
// After the BVH traversal resolves `bvhHit` (surface record from BVH):
float tCame = INF;
SurfaceHit cameHit;
bool hasCameHit = false;

if (u_analyticCameEnabled) {
  for (int i = 0; i < u_cameSegmentCount; i++) {
    float t; vec3 n;
    if (intersectCameSegment(ro, rd, u_cameSegments[i], RAY_MIN, tCame, t, n)) {
      tCame = t;
      cameHit = makeCameSurfaceHit(ro + rd * t, n, u_cameMaterial);
      hasCameHit = true;
    }
  }
  for (int j = 0; j < u_cameNodeCount; j++) {
    float t; vec3 n;
    if (intersectCameNode(ro, rd, u_cameNodes[j], RAY_MIN, tCame, t, n)) {
      tCame = t;
      cameHit = makeNodeSurfaceHit(ro + rd * t, n, u_cameMaterial);
      hasCameHit = true;
    }
  }
}

// Pick closest hit between BVH and analytic came.
SurfaceRecord finalRecord = bvhHit;
if (hasCameHit && tCame < bvhHit.dist) {
  finalRecord = cameHitToSurfaceRecord(cameHit);
}
```

`makeCameSurfaceHit` fills a synthetic `SurfaceRecord`:
- position: world-space hit point
- normal: world-space outward normal from intersect routine
- material: the came material (lead grey, roughness 0.6, metallic 0.7)
- UV: `vec2(0)` (came has no texture mapping)
- `isMesh = false`, `isAnalytic = true`

---

### 1.3 `src/materials/PhysicalPathTracingMaterial.js`

Add uniform declarations for the came UBO:

```javascript
// In the uniform declarations object:
analyticCameEnabled: { value: false },
cameSegmentCount:    { value: 0 },
cameNodeCount:       { value: 0 },
// CameSegment array — bound as Uniform Buffer Object
// Binding handled via WebGLRenderer.uniformsLib extension (Sprint 5 host side)
```

**UBO binding strategy**: use `gl.uniformBlockBinding` + `gl.bindBufferBase`
to attach the packed Float32Arrays from `packCameUBO` (vitrum-side,
`@vitrum/pt-webgl/src/cameUniformUploader.ts`) to the named uniform blocks.

Host-side pseudo-code (in the host application's PT pipeline layer):

```typescript
const { segments, nodes, segmentCount, nodeCount } = packCameUBO(
  cameSegments, cameNodes, { maxSegments: 500, maxNodes: 200 }
);

const segBuffer = gl.createBuffer();
gl.bindBuffer(gl.UNIFORM_BUFFER, segBuffer);
gl.bufferData(gl.UNIFORM_BUFFER, segments, gl.DYNAMIC_DRAW);

const nodeBuffer = gl.createBuffer();
gl.bindBuffer(gl.UNIFORM_BUFFER, nodeBuffer);
gl.bufferData(gl.UNIFORM_BUFFER, nodes, gl.DYNAMIC_DRAW);

// Bind to the binding points declared in the shader
gl.bindBufferBase(gl.UNIFORM_BUFFER, CAME_SEG_BINDING, segBuffer);
gl.bindBufferBase(gl.UNIFORM_BUFFER, CAME_NODE_BINDING, nodeBuffer);
pathTracer.material.uniforms.cameSegmentCount.value = segmentCount;
pathTracer.material.uniforms.cameNodeCount.value = nodeCount;
pathTracer.material.uniforms.analyticCameEnabled.value = true;
```

---

## 2. MRT G-buffer rider (Decision 12)

### 2.1 Allocation in the PT pipeline layer (host side)

```typescript
// Host-side (e.g., PathTracingLayer.tsx) — Sprint 5 MRT rider
import { WebGLMultipleRenderTargets } from 'three';

const mrtTarget = new WebGLMultipleRenderTargets(
  width, height,
  3,  // gColor (0), gNormalDepth (1), gAlbedo (2)
  {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  }
);

// gColor (location 0): RGBA16F — accumulated radiance
mrtTarget.texture[0].format = THREE.RGBAFormat;
mrtTarget.texture[0].type   = THREE.HalfFloatType;
mrtTarget.texture[0].name   = 'gColor';

// gNormalDepth (location 1): RGBA16F — world-space normal (xyz) + linear depth (w)
mrtTarget.texture[1].format = THREE.RGBAFormat;
mrtTarget.texture[1].type   = THREE.HalfFloatType;
mrtTarget.texture[1].name   = 'gNormalDepth';

// gAlbedo (location 2): RGBA8 — base color × occlusion, no lighting
mrtTarget.texture[2].format = THREE.RGBAFormat;
mrtTarget.texture[2].type   = THREE.UnsignedByteType;
mrtTarget.texture[2].name   = 'gAlbedo';
```

### 2.2 Fragment outputs in `PhysicalPathTracingMaterial.js`

The primary-hit surface record populates all three output locations in the
fragment shader.  Downstream sprints (Sprint 6 spatial filter, Sprint 10a
SVGF, Sprint 10b OIDN) read from these fixed locations without modification.

```glsl
layout(location = 0) out vec4 gColor;        // accumulated radiance
layout(location = 1) out vec4 gNormalDepth;  // world normal + linear depth
layout(location = 2) out vec4 gAlbedo;       // demodulated base color

// In the fragment shader, at primary-hit resolution:
gColor       = vec4(radiance, 1.0);
gNormalDepth = vec4(worldNormal, linearDepth);   // linearDepth = -dot(hitPos, viewDir_cam)
gAlbedo      = vec4(surfaceRecord.albedo, 1.0);  // baseColor × ao, no lighting
```

**Note on `gAlbedo`**: "demodulated" means the base color without any direct
or indirect lighting contribution.  This matches OIDN's convention (the
denoiser subtracts lighting, denoises, then re-multiplies).

Full channel spec: see `plan/sprint-5-mrt-gbuffer-spec.md`.

---

## 3. Definition of done (vitrum-side check)

The following vitrum-side items are already complete after Sprint 5 implementation:

- [x] `packCameUBO` in `@vitrum/pt-webgl/src/cameUniformUploader.ts` — packs
  500-segment came UBO in std140 layout matching the fork's CameSegment struct.
- [x] Device-tier fallback: `capabilities.supportedAnalyticShapes` includes
  `'h-channel-came'` iff `MAX_FRAGMENT_UNIFORM_VECTORS >= 256`.
- [x] MRT G-buffer types documented in `@vitrum/core/src/frame.ts`
  (`normalDepth`, `albedo` on `FrameOutput`).
- [x] Fork patch specification documented here for implementation by the fork
  maintainer.

The following items are **fork-side** and must be verified against the
`~/projects/three-gpu-pathtracer/` working copy:

- [ ] `shape_intersection_functions.glsl.js` — `intersectCameSegment` +
  `intersectCameNode` analytic functions.
- [ ] `trace_scene_function.glsl.js` — dual BVH + analytic came traversal,
  pick closest hit.
- [ ] `PhysicalPathTracingMaterial.js` — uniform binding for came UBO +
  `analyticCameEnabled` flag.
- [ ] Profile baseline: BVH-walk node-visits per ray ≥30% reduction on a
  500-segment scene.
- [ ] Host-side MRT allocation (`WebGLMultipleRenderTargets`) and fragment
  output population in `PhysicalPathTracingMaterial.js`.

---

## 4. Risk notes

- **UBO size**: 500 segments × 64 bytes + 200 nodes × 16 bytes = 35.2 KB.
  WebGL2 guarantees `MAX_UNIFORM_BLOCK_SIZE >= 16384` bytes (16 KB).  The
  full 500-segment came UBO exceeds the minimum guarantee.  Mitigation: query
  `gl.MAX_UNIFORM_BLOCK_SIZE` at engine creation; if less than 35.2 KB, reduce
  segment cap accordingly.  Typical desktop GL reports 65536 bytes (64 KB) or
  higher; 35.2 KB fits comfortably.

- **Normal reconstruction**: the synthetic `SurfaceRecord` filled from came
  hits must produce correct tangent/bitangent frames for the BSDF to evaluate
  reflection directions.  The simplest approach: use the came surface normal
  and compute an arbitrary tangent via `cross(normal, vec3(0,1,0))`.

- **UV**: came geometry has no texture coordinates.  Set `uv = vec2(0.0)` and
  ensure the material profile used for came has no texture maps bound.

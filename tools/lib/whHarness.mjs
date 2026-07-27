// @ts-nocheck
/**
 * tools/lib/whHarness.mjs
 *
 * Shared walkaround-hybrid (and pt-webgpu) GPU-harness scaffolding for the
 * native (lavapipe/wgpu-native) tooling harnesses. These camera-matrix,
 * device-acquisition and readback helpers were previously
 * RE-DECLARED — behaviorally identical modulo whitespace — across
 * tools/behavioral-gate/gate.mjs and the tools/radiometric-ab/* harnesses.
 * Promoting them here keeps one source of truth (D17-5 / T8).
 *
 * NOTE (verified 2026-07-20): the *scene builders* (`makeQuad`,
 * `makeCornellScene`/`buildCornellScene`) have genuinely DIVERGED across the
 * harnesses (different makeQuad signatures; the radiometric-ab glass-pane scene
 * carries ior/attenuation/thickness that the diagnostic scratch scenes do not).
 * Those are intentionally NOT promoted here — unifying them would change render
 * output. Only the behavior-identical camera / device / readback helpers live in
 * this module.
 */

// ── Camera ────────────────────────────────────────────────────────────────────

export function makePerspectiveMatrix(fovDeg, aspect, near, far) {
  const f  = 1.0 / Math.tan((fovDeg * Math.PI) / 180 / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

export function makeLookAtMatrix(eye, center, up) {
  const fx = center[0]-eye[0], fy = center[1]-eye[1], fz = center[2]-eye[2];
  const fL  = Math.hypot(fx, fy, fz);
  const fnx = fx/fL, fny = fy/fL, fnz = fz/fL;
  const sx = fny*up[2]-fnz*up[1], sy = fnz*up[0]-fnx*up[2], sz = fnx*up[1]-fny*up[0];
  const sL  = Math.hypot(sx, sy, sz);
  const snx = sx/sL, sny = sy/sL, snz = sz/sL;
  const ux  = sny*fnz-snz*fny, uy = snz*fnx-snx*fnz, uz = snx*fny-sny*fnx;
  return new Float32Array([
    snx, ux, -fnx, 0,
    sny, uy, -fny, 0,
    snz, uz, -fnz, 0,
    -(snx*eye[0]+sny*eye[1]+snz*eye[2]),
    -(ux *eye[0]+uy *eye[1]+uz *eye[2]),
     fnx*eye[0]+fny*eye[1]+fnz*eye[2],
    1,
  ]);
}

// Device acquisition

export async function acquireWhDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const limits = {};
  const sb = adapter.limits.maxStorageBuffersPerShaderStage ?? 8;
  const st = adapter.limits.maxStorageTexturesPerShaderStage ?? 4;
  if (sb >= 16) limits.maxStorageBuffersPerShaderStage = sb;
  if (st >= 8)  limits.maxStorageTexturesPerShaderStage = st;
  const bg = adapter.limits.maxBindGroups ?? 4;
  if (bg > 4) limits.maxBindGroups = bg;
  return adapter.requestDevice(Object.keys(limits).length ? { requiredLimits: limits } : {});
}

// ── Readback ──────────────────────────────────────────────────────────────────

/** Readback bgra8unorm texture (walkaround swap chain). Returns RGBA pixels. */
export async function readbackBgra8(device, tex, texW, texH) {
  const bpr  = Math.ceil(texW * 4 / 256) * 256;
  const buf  = device.createBuffer({ size: bpr * texH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc  = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: texH }, { width: texW, height: texH, depthOrArrayLayers: 1 });
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buf.getMappedRange());
  const pixels = new Uint8Array(texW * texH * 4);
  for (let row = 0; row < texH; row++) {
    pixels.set(mapped.subarray(row * bpr, row * bpr + texW * 4), row * texW * 4);
  }
  buf.unmap(); buf.destroy();
  // swap B↔R (bgra → rgba)
  for (let i = 0; i < pixels.length; i += 4) {
    const b = pixels[i]; pixels[i] = pixels[i+2]; pixels[i+2] = b;
  }
  return pixels;
}

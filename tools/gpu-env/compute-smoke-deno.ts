// End-to-end WebGPU compute smoke test on the native (Deno/lavapipe) device.
// Proves the adapter doesn't just *report* limits but can compile WGSL, bind a
// storage buffer, dispatch a compute pass, and read back correct results.
// Run: deno run --unstable-webgpu -A tools/gpu-env/compute-smoke-deno.ts

const gpu = (navigator as unknown as { gpu: GPU }).gpu;
const adapter = await gpu.requestAdapter();
if (!adapter) throw new Error("no adapter");
const device = await adapter.requestDevice();

const N = 256;
const data = new Uint32Array(N);
for (let i = 0; i < N; i++) data[i] = i;

const storage = device.createBuffer({
  size: data.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(storage, 0, data);

const module = device.createShaderModule({
  code: `
    @group(0) @binding(0) var<storage, read_write> buf: array<u32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i < arrayLength(&buf)) { buf[i] = buf[i] * 2u + 1u; }
    }
  `,
});
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});
const bind = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: storage } }],
});

const readback = device.createBuffer({
  size: data.byteLength,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

const enc = device.createCommandEncoder();
const pass = enc.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bind);
pass.dispatchWorkgroups(Math.ceil(N / 64));
pass.end();
enc.copyBufferToBuffer(storage, 0, readback, 0, data.byteLength);
device.queue.submit([enc.finish()]);

await readback.mapAsync(GPUMapMode.READ);
const got = new Uint32Array(readback.getMappedRange().slice(0));
readback.unmap();

let ok = true;
for (let i = 0; i < N; i++) {
  if (got[i] !== i * 2 + 1) { ok = false; break; }
}
console.log(JSON.stringify({
  computeSmoke: ok ? "PASS" : "FAIL",
  sample: { in0: 0, out0: got[0], in255: 255, out255: got[255], expected255: 255 * 2 + 1 },
}, null, 2));
device.destroy();
Deno.exit(ok ? 0 : 1);

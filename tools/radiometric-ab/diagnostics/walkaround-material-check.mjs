#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
const { packBVHRoughMetalFromCore, packBVHIndexWFromCore } = await import("../../../packages/walkaround-hybrid/src/restir/packingHelpers.ts");

const triMaterialId = new Uint32Array([0,0, 1,1, 2,2, 3,3, 4,4]);
const indices3 = new Uint32Array(30);

const diffuseMats = [
  { baseColor: [0.8,0.8,0.8], roughness: 1.0, metallic: 0.0 },
  { baseColor: [0.8,0.8,0.8], roughness: 1.0, metallic: 0.0 },
  { baseColor: [0.8,0.8,0.8], roughness: 1.0, metallic: 0.0 },
  { baseColor: [0.75,0.1,0.1], roughness: 1.0, metallic: 0.0 },
  { baseColor: [0.1,0.6,0.1], roughness: 1.0, metallic: 0.0 },
];
const metalMats = [
  { baseColor: [0.9,0.9,0.9], roughness: 0.05, metallic: 1.0 },
  { baseColor: [0.8,0.8,0.8], roughness: 1.0, metallic: 0.0 },
  { baseColor: [0.8,0.8,0.8], roughness: 1.0, metallic: 0.0 },
  { baseColor: [0.75,0.1,0.1], roughness: 1.0, metallic: 0.0 },
  { baseColor: [0.1,0.6,0.1], roughness: 1.0, metallic: 0.0 },
];

const dRm = packBVHRoughMetalFromCore(triMaterialId, diffuseMats, 10);
const mRm = packBVHRoughMetalFromCore(triMaterialId, metalMats,   10);

console.log("Diffuse scene roughMetal texture (10 tris):");
for(let i=0;i<10;i++){
  const packed = dRm[i] || 0;
  const rough=((packed>>>24)&0xFF)/255;
  const metal=((packed>>>16)&0xFF)/255;
  console.log("  tri "+i+": rough="+rough.toFixed(3)+" metal="+metal.toFixed(3)+" (0x"+packed.toString(16).padStart(8,'0')+")");
}

console.log("\nMetal scene roughMetal texture (10 tris):");
for(let i=0;i<10;i++){
  const packed = mRm[i] || 0;
  const rough=((packed>>>24)&0xFF)/255;
  const metal=((packed>>>16)&0xFF)/255;
  console.log("  tri "+i+": rough="+rough.toFixed(3)+" metal="+metal.toFixed(3)+" (0x"+packed.toString(16).padStart(8,'0')+")");
}

const dIdx = packBVHIndexWFromCore(indices3, triMaterialId, diffuseMats, 10);
const mIdx = packBVHIndexWFromCore(indices3, triMaterialId, metalMats,   10);

console.log("\nDiffuse isMetal bits:");
for(let i=0;i<10;i++){
  const w = dIdx[i*4+3] || 0;
  const isMetal = (w>>>3)&1;
  console.log("  tri "+i+": isMetal="+isMetal);
}
console.log("\nMetal isMetal bits:");
for(let i=0;i<10;i++){
  const w = mIdx[i*4+3] || 0;
  const isMetal = (w>>>3)&1;
  console.log("  tri "+i+": isMetal="+isMetal);
}

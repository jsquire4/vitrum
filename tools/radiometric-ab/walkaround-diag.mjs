#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
import { createWalkaroundEngine_Hybrid } from "@vitrum/walkaround-hybrid";
import { asMat4 } from "@vitrum/core";
import { applyNagaFix } from "../shader-gate/nagaFix.mjs";

const W = 128, H = 128, SPP = 16;

function makePerspectiveMatrix(fovDeg, aspect, near, far) {
  const f  = 1.0 / Math.tan((fovDeg * Math.PI) / 180 / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function makeLookAtMatrix(eye, center, up) {
  const fx=center[0]-eye[0],fy=center[1]-eye[1],fz=center[2]-eye[2];
  const fL=Math.hypot(fx,fy,fz),fnx=fx/fL,fny=fy/fL,fnz=fz/fL;
  const sx=fny*up[2]-fnz*up[1],sy=fnz*up[0]-fnx*up[2],sz=fnx*up[1]-fny*up[0];
  const sL=Math.hypot(sx,sy,sz),snx=sx/sL,sny=sy/sL,snz=sz/sL;
  const ux=sny*fnz-snz*fny,uy=snz*fnx-snx*fnz,uz=snx*fny-sny*fnx;
  return new Float32Array([snx,ux,-fnx,0,sny,uy,-fny,0,snz,uz,-fnz,0,
    -(snx*eye[0]+sny*eye[1]+snz*eye[2]),-(ux*eye[0]+uy*eye[1]+uz*eye[2]),fnx*eye[0]+fny*eye[1]+fnz*eye[2],1]);
}

const EYE=[0,0,2.5],CENTER=[0,0,0];
const proj=asMat4(makePerspectiveMatrix(60,W/H,0.1,50));
const view=asMat4(makeLookAtMatrix(EYE,CENTER,[0,1,0]));

function patchDeviceForWh(device) {
  const orig=device.createShaderModule.bind(device);
  device.createShaderModule=(desc)=>{
    if(typeof desc.code==="string"){try{return orig({...desc,code:applyNagaFix(desc.code)});}catch{return orig(desc);}}
    return orig(desc);
  };
}

async function acquireWhDevice() {
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)throw new Error("No adapter");
  const limits={};
  const sb=adapter.limits.maxStorageBuffersPerShaderStage??8;
  const st=adapter.limits.maxStorageTexturesPerShaderStage??4;
  if(sb>=16)limits.maxStorageBuffersPerShaderStage=sb;
  if(st>=8)limits.maxStorageTexturesPerShaderStage=st;
  const bg=adapter.limits.maxBindGroups??4;
  if(bg>4)limits.maxBindGroups=bg;
  return adapter.requestDevice(Object.keys(limits).length?{requiredLimits:limits}:{});
}

async function readbackBgra8(device,tex,texW,texH) {
  const bpr=Math.ceil(texW*4/256)*256;
  const buf=device.createBuffer({size:bpr*texH,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const enc=device.createCommandEncoder();
  enc.copyTextureToBuffer({texture:tex},{buffer:buf,bytesPerRow:bpr,rowsPerImage:texH},{width:texW,height:texH,depthOrArrayLayers:1});
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const mapped=new Uint8Array(buf.getMappedRange());
  const pixels=new Uint8Array(texW*texH*4);
  for(let row=0;row<texH;row++)pixels.set(mapped.subarray(row*bpr,row*bpr+texW*4),row*texW*4);
  buf.unmap();buf.destroy();
  for(let i=0;i<pixels.length;i+=4){const b=pixels[i];pixels[i]=pixels[i+2];pixels[i+2]=b;}
  return pixels;
}

function makeQuad(id,verts,normal,color,roughness=1.0,metallic=0.0) {
  return {kind:"mesh",id,positions:new Float32Array(verts.flat()),normals:new Float32Array([...normal,...normal,...normal,...normal]),uvs:new Float32Array(8),indices:new Uint32Array([0,2,1,2,0,3]),material:{baseColor:color,roughness,metallic}};
}

function makeCornellScene(opts={}) {
  const primitives=[
    makeQuad("floor",[[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]],[0,1,0],[0.8,0.8,0.8],opts.floorRoughness??1.0,opts.floorMetallic??0.0),
    makeQuad("ceiling",[[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]],[0,-1,0],[0.8,0.8,0.8]),
    makeQuad("back-wall",[[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],[0,0,-1],[0.8,0.8,0.8]),
    makeQuad("left-wall",[[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]],[1,0,0],[0.75,0.1,0.1]),
    makeQuad("right-wall",[[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]],[-1,0,0],[0.1,0.6,0.1]),
  ];
  if(opts.glass){
    primitives.push({kind:"mesh",id:"glass-pane",positions:new Float32Array([-0.5,-0.5,0.5,0.5,-0.5,0.5,0.5,0.5,0.5,-0.5,0.5,0.5]),normals:new Float32Array([0,0,-1,0,0,-1,0,0,-1,0,0,-1]),uvs:new Float32Array(8),indices:new Uint32Array([0,2,1,2,0,3]),material:{baseColor:[1,1,1],roughness:0.05,metallic:0.0,transmission:1.0}});
  }
  const emitters=opts.noEmitter?[]:[{kind:"rect-area",id:"ceiling-light",position:[0,0.95,0],uAxis:[0,0,0.2],vAxis:[0.2,0,0],color:[1,1,1],intensity:12.0}];
  return {primitives,emitters,environment:{kind:"none"}};
}

async function run(label,engineOpts,sceneOpts) {
  const device=await acquireWhDevice();
  patchDeviceForWh(device);
  const engine=await createWalkaroundEngine_Hybrid({device,width:W,height:H,primaryLightDir:[0.3,-0.8,0.5],primaryLightIntensity:0.6,skyTint:[0.5,0.7,1.0],skyIrradiance:0.15,verbose:false,ppgEnabled:false,rcEnabled:false,denoiser:"atrous-variance",...engineOpts});
  engine.setScene(makeCornellScene(sceneOpts));
  const deadline=Date.now()+90000;
  while(engine.state!=="ready"&&engine.state!=="error"){await new Promise(r=>setTimeout(r,50));if(Date.now()>deadline)throw new Error("timeout");}
  if(engine.state==="error")throw new Error("engine error");
  const swapTex=device.createTexture({label:`swap-${label}`,size:[W,H,1],format:"bgra8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC|GPUTextureUsage.TEXTURE_BINDING});
  const swapView=swapTex.createView();
  for(let fi=0;fi<SPP;fi++){engine.renderFrame({viewMatrix:view,projMatrix:proj,cameraPosition:EYE,viewport:{width:W,height:H,devicePixelRatio:1},frameIndex:fi,frameSeed:fi*1664525+1013904223,swapChainView:swapView,swapChainFormat:"bgra8unorm"});await device.queue.onSubmittedWorkDone();}
  const pixels=await readbackBgra8(device,swapTex,W,H);
  swapTex.destroy();engine.dispose();device.destroy();
  return pixels;
}

function regionLum(pixels,texW,x0,y0,x1,y1) {
  let sum=0,count=0;
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){const i=(y*texW+x)*4;sum+=0.2126*(pixels[i]/255)+0.7152*(pixels[i+1]/255)+0.0722*(pixels[i+2]/255);count++;}
  return count>0?sum/count:0;
}

// Glass vs no-glass: scan centre pixels
console.log("=== Glass vs no-glass diagnostic ===");
const pGlass   = await run("glass",   {},{glass:true});
const pNoGlass = await run("no-glass",{},{glass:false});

console.log("Centre diagonal pixels (glass/no-glass):");
for(let y=40;y<90;y+=5){
  const x=y; // diagonal
  const i=(y*W+x)*4;
  const lG=0.2126*(pGlass[i]/255)+0.7152*(pGlass[i+1]/255)+0.0722*(pGlass[i+2]/255);
  const lN=0.2126*(pNoGlass[i]/255)+0.7152*(pNoGlass[i+1]/255)+0.0722*(pNoGlass[i+2]/255);
  console.log(`  (${x},${y}): glass=${lG.toFixed(4)} noGlass=${lN.toFixed(4)} diff=${(lG-lN).toFixed(4)}`);
}
console.log("Centre row y=64:");
for(let x=44;x<84;x+=4){
  const i=(64*W+x)*4;
  const lG=0.2126*(pGlass[i]/255)+0.7152*(pGlass[i+1]/255)+0.0722*(pGlass[i+2]/255);
  const lN=0.2126*(pNoGlass[i]/255)+0.7152*(pNoGlass[i+1]/255)+0.0722*(pNoGlass[i+2]/255);
  console.log(`  (${x},64): glass=${lG.toFixed(4)} noGlass=${lN.toFixed(4)} diff=${(lG-lN).toFixed(4)}`);
}

// Overall mean luminance
let sumG=0,sumN=0,cnt=0;
for(let i=0;i<pGlass.length;i+=4){
  sumG+=0.2126*(pGlass[i]/255)+0.7152*(pGlass[i+1]/255)+0.0722*(pGlass[i+2]/255);
  sumN+=0.2126*(pNoGlass[i]/255)+0.7152*(pNoGlass[i+1]/255)+0.0722*(pNoGlass[i+2]/255);
  cnt++;
}
console.log(`Overall mean: glass=${(sumG/cnt).toFixed(4)} noGlass=${(sumN/cnt).toFixed(4)}`);

// Metallic vs diffuse diagnostic
console.log("\n=== Metallic vs diffuse diagnostic ===");
const pMetal   = await run("metal",  {},{floorRoughness:0.05,floorMetallic:1.0,floorColor:[0.9,0.9,0.9]});
const pDiffuse = await run("diffuse",{},{floorRoughness:1.0,floorMetallic:0.0,floorColor:[0.8,0.8,0.8]});

console.log("Floor pixels (bottom strip):");
for(let y=90;y<128;y+=5){
  for(let x=30;x<100;x+=10){
    const i=(y*W+x)*4;
    const lM=0.2126*(pMetal[i]/255)+0.7152*(pMetal[i+1]/255)+0.0722*(pMetal[i+2]/255);
    const lD=0.2126*(pDiffuse[i]/255)+0.7152*(pDiffuse[i+1]/255)+0.0722*(pDiffuse[i+2]/255);
    console.log(`  (${x},${y}): metal=${lM.toFixed(4)} diffuse=${lD.toFixed(4)} diff=${(lM-lD).toFixed(4)}`);
  }
}

let sumM=0,sumD=0; cnt=0;
for(let i=0;i<pMetal.length;i+=4){
  sumM+=0.2126*(pMetal[i]/255)+0.7152*(pMetal[i+1]/255)+0.0722*(pMetal[i+2]/255);
  sumD+=0.2126*(pDiffuse[i]/255)+0.7152*(pDiffuse[i+1]/255)+0.0722*(pDiffuse[i+2]/255);
  cnt++;
}
console.log(`Overall mean: metal=${(sumM/cnt).toFixed(4)} diffuse=${(sumD/cnt).toFixed(4)}`);

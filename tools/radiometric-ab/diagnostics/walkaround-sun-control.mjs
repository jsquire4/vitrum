#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write
// @ts-nocheck
// Control: sun at intensity=0 should give near-black (no area emitter, no sun).
// Also: stronger sun (I=0.9) should be proportionally brighter than I=0.3.
import { createWalkaroundEngine_Hybrid } from "@vitrum/walkaround-hybrid";
import { asMat4 } from "@vitrum/core";

const W = 128, H = 128, SPP = 16;

function makePerspectiveMatrix(fovDeg, aspect, near, far) {
  const f=1.0/Math.tan((fovDeg*Math.PI)/180/2),nf=1/(near-far);
  return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]);
}
function makeLookAtMatrix(eye,center,up) {
  const fx=center[0]-eye[0],fy=center[1]-eye[1],fz=center[2]-eye[2],fL=Math.hypot(fx,fy,fz),fnx=fx/fL,fny=fy/fL,fnz=fz/fL;
  const sx=fny*up[2]-fnz*up[1],sy=fnz*up[0]-fnx*up[2],sz=fnx*up[1]-fny*up[0],sL=Math.hypot(sx,sy,sz),snx=sx/sL,sny=sy/sL,snz=sz/sL;
  const ux=sny*fnz-snz*fny,uy=snz*fnx-snx*fnz,uz=snx*fny-sny*fnx;
  return new Float32Array([snx,ux,-fnx,0,sny,uy,-fny,0,snz,uz,-fnz,0,-(snx*eye[0]+sny*eye[1]+snz*eye[2]),-(ux*eye[0]+uy*eye[1]+uz*eye[2]),fnx*eye[0]+fny*eye[1]+fnz*eye[2],1]);
}
const EYE=[0,0,2.5],CENTER=[0,0,0];
const proj=asMat4(makePerspectiveMatrix(60,W/H,0.1,50));
const view=asMat4(makeLookAtMatrix(EYE,CENTER,[0,1,0]));
const SUN_TRAVEL_DIRECTION = [0, 0, -1];
const SUN_TO_LIGHT_DIRECTION = [0, 0, 1];

async function acquireWhDevice() {
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)throw new Error("No adapter");
  const limits={};const sb=adapter.limits.maxStorageBuffersPerShaderStage??8;const st=adapter.limits.maxStorageTexturesPerShaderStage??4;
  if(sb>=16)limits.maxStorageBuffersPerShaderStage=sb;if(st>=8)limits.maxStorageTexturesPerShaderStage=st;
  const bg=adapter.limits.maxBindGroups??4;if(bg>4)limits.maxBindGroups=bg;
  return adapter.requestDevice(Object.keys(limits).length?{requiredLimits:limits}:{});
}

async function readbackBgra8(device,tex,texW,texH) {
  const bpr=Math.ceil(texW*4/256)*256;
  const buf=device.createBuffer({size:bpr*texH,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const enc=device.createCommandEncoder();
  enc.copyTextureToBuffer({texture:tex},{buffer:buf,bytesPerRow:bpr,rowsPerImage:texH},{width:texW,height:texH,depthOrArrayLayers:1});
  device.queue.submit([enc.finish()]);await buf.mapAsync(GPUMapMode.READ);
  const mapped=new Uint8Array(buf.getMappedRange()),pixels=new Uint8Array(texW*texH*4);
  for(let row=0;row<texH;row++)pixels.set(mapped.subarray(row*bpr,row*bpr+texW*4),row*texW*4);
  buf.unmap();buf.destroy();
  for(let i=0;i<pixels.length;i+=4){const b=pixels[i];pixels[i]=pixels[i+2];pixels[i+2]=b;}
  return pixels;
}

function makeQuad(id,verts,normal,color) {
  return {kind:"mesh",id,positions:new Float32Array(verts.flat()),normals:new Float32Array([...normal,...normal,...normal,...normal]),uvs:new Float32Array(8),indices:new Uint32Array([0,2,1,2,0,3]),material:{baseColor:color,roughness:1.0,metallic:0.0}};
}

function makeDirOnlyScene() {
  return {primitives:[
    makeQuad("floor",[[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]],[0,1,0],[0.8,0.8,0.8]),
    makeQuad("back-wall",[[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],[0,0,-1],[0.8,0.8,0.8]),
    makeQuad("left-wall",[[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]],[1,0,0],[0.75,0.1,0.1]),
    makeQuad("right-wall",[[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]],[-1,0,0],[0.1,0.6,0.1]),
  ],emitters:[{kind:"directional",id:"sun-control-light",direction:SUN_TO_LIGHT_DIRECTION,color:[1,1,1],intensity:0.3,castShadow:false}],environment:{kind:"none"}};
}

async function run(label, sunIntensity) {
  const device=await acquireWhDevice();
  const engine=await createWalkaroundEngine_Hybrid({device,width:W,height:H,primaryLightDir:SUN_TO_LIGHT_DIRECTION,primaryLightIntensity:sunIntensity,skyTint:[0,0,0],skyIrradiance:0.0,verbose:false,ppgEnabled:false,rcEnabled:false,denoiser:"atrous-variance"});
  engine.setScene(makeDirOnlyScene());
  const deadline=Date.now()+90000;
  while(engine.state!=="ready"&&engine.state!=="error"){await new Promise(r=>setTimeout(r,50));if(Date.now()>deadline)throw new Error("timeout");}
  if(engine.state==="error")throw new Error("engine error");
  const swapTex=device.createTexture({label:"swap",size:[W,H,1],format:"bgra8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC|GPUTextureUsage.TEXTURE_BINDING});
  const swapView=swapTex.createView();
  for(let fi=0;fi<SPP;fi++){engine.renderFrame({viewMatrix:view,projMatrix:proj,cameraPosition:EYE,viewport:{width:W,height:H,devicePixelRatio:1},frameIndex:fi,frameSeed:fi*1664525+1013904223,swapChainView:swapView,swapChainFormat:"bgra8unorm"});await device.queue.onSubmittedWorkDone();}
  const pixels=await readbackBgra8(device,swapTex,W,H);
  swapTex.destroy();engine.dispose();device.destroy();
  return pixels;
}

function regionLum(pixels,texW,x0,y0,x1,y1) {
  let sum=0,cnt=0;
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){const i=(y*texW+x)*4;sum+=0.2126*(pixels[i]/255)+0.7152*(pixels[i+1]/255)+0.0722*(pixels[i+2]/255);cnt++;}
  return cnt>0?sum/cnt:0;
}

const pCtrl  = await run("ctrl-zero-sun",  0.0);   // sun off → should be near-black
const pLow   = await run("low-sun",        0.3);   // the test value
const pHigh  = await run("high-sun",       0.9);   // 3× low → floor should be ~3× brighter

const receiverCtrl = regionLum(pCtrl,  W, 30, 42, 98, 86);
const receiverLow  = regionLum(pLow,   W, 30, 42, 98, 86);
const receiverHigh = regionLum(pHigh,  W, 30, 42, 98, 86);

// Linear analytic for low: I=0.3 × cosθ=1 × albedo=0.8 / π = 0.0764.
// This control reads the BGRA swap target, so use the value as a directionality
// and intensity-linearity smoke rather than a strict linear-HDR proof.
const analytic = 0.3 * 1.0 * 0.8 / Math.PI;

console.log("Sun intensity control runs:");
console.log("  travel direction:", SUN_TRAVEL_DIRECTION, "to-light direction:", SUN_TO_LIGHT_DIRECTION);
console.log("  I=0.0 (zero sun, zero sky): receiver lum =", receiverCtrl.toFixed(4), " (expected ≈0)");
console.log("  I=0.3:                       receiver lum =", receiverLow.toFixed(4), " (linear analytic="+analytic.toFixed(4)+", ratio="+( receiverLow/analytic).toFixed(3)+")");
console.log("  I=0.9 (3× I=0.3):           receiver lum =", receiverHigh.toFixed(4), " (expected ≈3× I=0.3 receiver = "+(receiverLow*3).toFixed(4)+", ratio="+(receiverHigh/receiverLow).toFixed(3)+")");
console.log("  Linearity check (I=0.9 / I=0.3):", (receiverHigh/receiverLow).toFixed(3), " (expected ≈3.0)");

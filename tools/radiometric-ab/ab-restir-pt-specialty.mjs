#!/usr/bin/env node
// @ts-check
/**
 * Static CPU A/B reference for the ReSTIR-PT specialty-lobe path.
 *
 * This is intentionally not a GPU recapture. It models the load-bearing
 * producer/finalize/resolve identity for a single ReSTIR-PT reconnection sample:
 *
 *   producer candidate: w = pHat / pdfSrc
 *   GRIS finalize:      W = w_sum / pHat
 *   resolve:            L = f_bsdf * cosTheta * Lo * W
 *
 * For a one-sample producer reservoir, W cancels pHat and the resolved result
 * must match the base path estimator f_bsdf * cosTheta * Lo / pdfSrc. The cases
 * below keep clearcoat, sheen, iridescence, anisotropy, and KHR specular nonzero
 * in bounded scalar and map-backed-effective-value fixtures so package tests can
 * pin coverage without requiring WebGPU.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PI = Math.PI;
const INV_PI = 1 / PI;
const FIXTURE_URL = new URL('./results-restir-pt-specialty.json', import.meta.url);
const FIXTURE_PATH = fileURLToPath(FIXTURE_URL);

/** @typedef {readonly [number, number, number]} Vec3 */

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const round = (v) => {
  if (Math.abs(v) < 5e-15) return 0;
  return Number(v.toFixed(12));
};

/** @param {Vec3} a @param {Vec3} b */
function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

/** @param {Vec3} a @param {Vec3} b */
function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** @param {Vec3} a @param {number} s */
function scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

/** @param {Vec3} a @param {Vec3} b */
function mul(a, b) {
  return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

/** @param {Vec3} a @param {Vec3} b */
function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** @param {Vec3} a @param {Vec3} b */
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/** @param {Vec3} v */
function normalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

/** @param {Vec3} c */
function luminance(c) {
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

/** @param {Vec3} a @param {Vec3} b @param {number} t */
function mix3(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

/** @param {Vec3} a */
function clamp3(a) {
  return [clamp(a[0], 0, 1), clamp(a[1], 0, 1), clamp(a[2], 0, 1)];
}

/** @param {Vec3} n */
function buildOnb(n) {
  const up = Math.abs(n[2]) < 0.999 ? [0, 0, 1] : [0, 1, 0];
  const t = normalize(cross(up, n));
  const b = cross(n, t);
  return [t, b];
}

/** @param {Vec3} v @param {Vec3} t @param {Vec3} b @param {Vec3} n */
function toLocal(v, t, b, n) {
  return [dot(v, t), dot(v, b), dot(v, n)];
}

/** @param {number} roughness @param {number} anisotropy */
function axAy(roughness, anisotropy) {
  const aspect = Math.sqrt(Math.max(0.1, 1 - 0.9 * clamp(Math.abs(anisotropy), 0, 1)));
  return [Math.max(roughness / aspect, 0.001), Math.max(roughness * aspect, 0.001)];
}

/** @param {number} cosTheta @param {Vec3} f0 */
function fresnelSchlick(cosTheta, f0) {
  const m = clamp(1 - cosTheta, 0, 1);
  const m5 = m * m * m * m * m;
  return [
    f0[0] + (1 - f0[0]) * m5,
    f0[1] + (1 - f0[1]) * m5,
    f0[2] + (1 - f0[2]) * m5,
  ];
}

/** @param {number} nDotH @param {number} alpha */
function ggxD(nDotH, alpha) {
  const a2 = alpha * alpha;
  const d = nDotH * nDotH * (a2 - 1) + 1;
  return a2 / Math.max(PI * d * d, 1e-6);
}

/** @param {number} nDotV @param {number} roughness */
function smithG1(nDotV, roughness) {
  const r = roughness + 1;
  const k = r * r * 0.125;
  return nDotV / Math.max(nDotV * (1 - k) + k, 1e-6);
}

/** @param {Vec3} h @param {number} ax @param {number} ay */
function ggxDAnisotropic(h, ax, ay) {
  const den = (h[0] / ax) * (h[0] / ax) + (h[1] / ay) * (h[1] / ay) + h[2] * h[2];
  return 1 / Math.max(PI * ax * ay * den * den, 1e-30);
}

/** @param {Vec3} v @param {number} ax @param {number} ay */
function smithG1Anisotropic(v, ax, ay) {
  const vN = Math.max(v[2], 1e-6);
  const rad = vN * vN + (v[0] * ax) * (v[0] * ax) + (v[1] * ay) * (v[1] * ay);
  return (2 * vN) / Math.max(vN + Math.sqrt(rad), 1e-12);
}

/** @param {number} nDotH @param {number} alpha */
function charlieD(nDotH, alpha) {
  const invAlpha = 1 / Math.max(alpha, 1e-4);
  const sinThetaH = Math.sqrt(Math.max(0, 1 - nDotH * nDotH));
  return ((2 + invAlpha) * Math.pow(sinThetaH, invAlpha)) / (2 * PI);
}

/** @param {number} nDotL @param {number} nDotV */
function sheenVisibility(nDotL, nDotV) {
  return 1 / Math.max(4 * (nDotL + nDotV - nDotL * nDotV), 1e-6);
}

/** @param {Record<string, any>} material */
function materialF0(material) {
  const dielectric = scale(material.specularColor, 0.04 * material.specularIntensity);
  const base = mix3(dielectric, material.baseColor, material.metallic);
  if (material.iridescence <= 1e-4) return base;

  const thickness = 0.5 * (material.iridescenceThicknessMin + material.iridescenceThicknessMax);
  const filmPhase = thickness * (material.iridescenceIor - 1);
  const filmTint = [
    0.5 + 0.5 * Math.cos(filmPhase / 48 + 0.17),
    0.5 + 0.5 * Math.cos(filmPhase / 42 + 2.09),
    0.5 + 0.5 * Math.cos(filmPhase / 37 + 4.21),
  ];
  const filmF0 = clamp3(add(scale(base, 0.72), scale(filmTint, 0.28)));
  return mix3(base, filmF0, material.iridescence);
}

/**
 * @param {Record<string, any>} material
 * @param {Vec3} normal
 * @param {Vec3} wo
 * @param {Vec3} wi
 */
function baseBrdfAndPdf(material, normal, wo, wi) {
  const nDotL = Math.max(dot(normal, wi), 0);
  const nDotV = Math.max(dot(normal, wo), 0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) {
    return { brdf: [0, 0, 0], pdf: 0 };
  }

  const h = normalize(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const vDotH = Math.max(dot(wo, h), 1e-6);
  const f0 = materialF0(material);
  const fres = fresnelSchlick(vDotH, f0);
  const diffuse = scale(material.baseColor, (1 - material.metallic) * INV_PI);

  let d;
  let gWo;
  let gWi;
  if (Math.abs(material.anisotropy) > 1e-4) {
    let [tanT, tanB] = buildOnb(normal);
    const c = Math.cos(material.anisotropyRotation);
    const s = Math.sin(material.anisotropyRotation);
    const rotatedT = add(scale(tanT, c), scale(tanB, s));
    const rotatedB = add(scale(tanT, -s), scale(tanB, c));
    tanT = rotatedT;
    tanB = rotatedB;
    const [ax, ay] = axAy(material.roughness, material.anisotropy);
    d = ggxDAnisotropic(toLocal(h, tanT, tanB, normal), ax, ay);
    gWo = smithG1Anisotropic(toLocal(wo, tanT, tanB, normal), ax, ay);
    gWi = smithG1Anisotropic(toLocal(wi, tanT, tanB, normal), ax, ay);
  } else {
    const alpha = Math.max(material.roughness * material.roughness, 1e-3);
    d = ggxD(nDotH, alpha);
    gWo = smithG1(nDotV, material.roughness);
    gWi = smithG1(nDotL, material.roughness);
  }

  const spec = scale(fres, (d * gWo * gWi) / Math.max(4 * nDotV * nDotL, 1e-6));
  const baseSpecProb = clamp(0.04 + (0.96 - 0.04) * Math.max(luminance(fres), material.metallic), 0.04, 0.96);
  const baseDiffProb = Math.max(0, 1 - material.metallic);
  const sumProb = Math.max(baseSpecProb + baseDiffProb, 1e-4);
  const specProb = baseSpecProb / sumProb;
  const diffProb = baseDiffProb / sumProb;
  const pdfSpec = (d * gWo) / Math.max(4 * nDotV, 1e-6);
  const pdfDiff = nDotL * INV_PI;

  return {
    brdf: add(diffuse, spec),
    pdf: diffProb * pdfDiff + specProb * pdfSpec,
  };
}

/**
 * @param {Record<string, any>} material
 * @param {Vec3} normal
 * @param {Vec3} wo
 * @param {Vec3} wi
 */
function clearcoatBrdfAndPdf(material, normal, wo, wi) {
  if (material.clearcoat <= 1e-4) return { brdf: [0, 0, 0], pdf: 0 };
  const nDotL = Math.max(dot(normal, wi), 0);
  const nDotV = Math.max(dot(normal, wo), 0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return { brdf: [0, 0, 0], pdf: 0 };

  const h = normalize(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const vDotH = Math.max(dot(wo, h), 0);
  const alpha = Math.max(material.clearcoatRoughness * material.clearcoatRoughness, 1e-3);
  const d = ggxD(nDotH, alpha);
  const gWo = smithG1(nDotV, material.clearcoatRoughness);
  const gWi = smithG1(nDotL, material.clearcoatRoughness);
  const f = fresnelSchlick(vDotH, [0.04, 0.04, 0.04]);
  return {
    brdf: scale(f, (material.clearcoat * d * gWo * gWi) / Math.max(4 * nDotV * nDotL, 1e-6)),
    pdf: (d * gWo) / Math.max(4 * nDotV, 1e-6),
  };
}

/**
 * @param {Record<string, any>} material
 * @param {Vec3} normal
 * @param {Vec3} wo
 * @param {Vec3} wi
 */
function sheenBrdf(material, normal, wo, wi) {
  if (material.sheen <= 1e-4) return [0, 0, 0];
  const nDotL = Math.max(dot(normal, wi), 0);
  const nDotV = Math.max(dot(normal, wo), 0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return [0, 0, 0];
  const h = normalize(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const alpha = Math.max(material.sheenRoughness * material.sheenRoughness, 1e-3);
  return scale(
    material.sheenColor,
    material.sheen * charlieD(nDotH, alpha) * sheenVisibility(nDotL, nDotV),
  );
}

/**
 * @param {Record<string, any>} material
 * @param {Vec3} normal
 * @param {Vec3} wo
 * @param {Vec3} wi
 */
function sheenPdf(material, normal, wo, wi) {
  if (material.sheen <= 1e-4) return 0;
  const nDotL = Math.max(dot(normal, wi), 0);
  const nDotV = Math.max(dot(normal, wo), 0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) return 0;
  const h = normalize(add(wi, wo));
  const nDotH = Math.max(dot(normal, h), 0);
  const vDotH = Math.max(dot(wo, h), 1e-6);
  const alpha = Math.max(material.sheenRoughness * material.sheenRoughness, 1e-3);
  return (charlieD(nDotH, alpha) * nDotH) / Math.max(4 * vDotH, 1e-6);
}

/**
 * @param {Record<string, any>} material
 * @param {Vec3} normal
 * @param {Vec3} clearcoatNormal
 * @param {Vec3} wo
 * @param {Vec3} wi
 */
function evaluateSpecialtyBrdf(material, normal, clearcoatNormal, wo, wi) {
  const base = baseBrdfAndPdf(material, normal, wo, wi).brdf;
  const clearcoat = clearcoatBrdfAndPdf(material, clearcoatNormal, wo, wi).brdf;
  const sheen = sheenBrdf(material, normal, wo, wi);
  return add(add(base, clearcoat), sheen);
}

/**
 * @param {Record<string, any>} material
 * @param {Vec3} normal
 * @param {Vec3} clearcoatNormal
 * @param {Vec3} wo
 * @param {Vec3} wi
 */
function sourcePdfFull(material, normal, clearcoatNormal, wo, wi) {
  const base = baseBrdfAndPdf(material, normal, wo, wi).pdf;
  const clearcoat = clearcoatBrdfAndPdf(material, clearcoatNormal, wo, wi).pdf;
  const sheen = sheenPdf(material, normal, wo, wi);
  const lobeWeightSum = Math.max(1 + Math.max(material.clearcoat, 0) + Math.max(material.sheen, 0), 1e-4);
  return (base + material.clearcoat * clearcoat + material.sheen * sheen) / lobeWeightSum;
}

function neutralMaterial() {
  return {
    baseColor: [0.63, 0.49, 0.31],
    roughness: 0.42,
    metallic: 0,
    clearcoat: 0,
    clearcoatRoughness: 0.35,
    sheen: 0,
    sheenRoughness: 0.7,
    sheenColor: [0, 0, 0],
    iridescence: 0,
    iridescenceIor: 1.3,
    iridescenceThicknessMin: 100,
    iridescenceThicknessMax: 400,
    specularColor: [1, 1, 1],
    specularIntensity: 1,
    anisotropy: 0,
    anisotropyRotation: 0,
  };
}

function specialtyCases() {
  const common = neutralMaterial();
  return [
    {
      id: 'clearcoat-sheen',
      activeLobes: ['clearcoat', 'sheen'],
      material: {
        ...common,
        clearcoat: 0.72,
        clearcoatRoughness: 0.24,
        sheen: 0.48,
        sheenRoughness: 0.62,
        sheenColor: [0.95, 0.42, 0.18],
      },
      wo: normalize([0.32, -0.18, 0.93]),
      wi: normalize([-0.24, 0.41, 0.88]),
      Lo: [1.35, 0.82, 0.46],
    },
    {
      id: 'iridescent-anisotropic',
      activeLobes: ['iridescence', 'anisotropy'],
      material: {
        ...common,
        baseColor: [0.34, 0.51, 0.86],
        roughness: 0.36,
        iridescence: 0.66,
        iridescenceIor: 1.42,
        iridescenceThicknessMin: 260,
        iridescenceThicknessMax: 520,
        anisotropy: 0.68,
        anisotropyRotation: 0.73,
        specularColor: [0.92, 0.96, 1],
        specularIntensity: 0.88,
      },
      wo: normalize([-0.29, 0.23, 0.91]),
      wi: normalize([0.51, 0.08, 0.86]),
      Lo: [0.42, 1.18, 1.65],
    },
    {
      id: 'all-specialty-lobes',
      materialSource: 'scalar',
      activeLobes: ['clearcoat', 'sheen', 'iridescence', 'anisotropy'],
      material: {
        ...common,
        baseColor: [0.72, 0.28, 0.18],
        roughness: 0.31,
        clearcoat: 0.55,
        clearcoatRoughness: 0.19,
        sheen: 0.38,
        sheenRoughness: 0.58,
        sheenColor: [0.28, 0.72, 0.96],
        iridescence: 0.47,
        iridescenceIor: 1.36,
        iridescenceThicknessMin: 190,
        iridescenceThicknessMax: 470,
        specularColor: [1, 0.91, 0.82],
        specularIntensity: 0.94,
        anisotropy: 0.52,
        anisotropyRotation: 1.11,
      },
      wo: normalize([0.44, 0.16, 0.89]),
      wi: normalize([-0.39, -0.31, 0.84]),
      Lo: [1.8, 1.1, 0.74],
    },
    {
      id: 'map-backed-effective-lobes',
      materialSource: 'map-backed-effective-values',
      activeLobes: ['clearcoat', 'sheen', 'iridescence', 'anisotropy', 'specular'],
      material: {
        ...common,
        baseColor: [0.52, 0.43, 0.33],
        roughness: 0.28,
        clearcoat: 0.84,
        clearcoatRoughness: 0.16,
        sheen: 0.57,
        sheenRoughness: 0.41,
        sheenColor: [0.22, 0.81, 0.64],
        iridescence: 0.71,
        iridescenceIor: 1.48,
        iridescenceThicknessMin: 340,
        iridescenceThicknessMax: 340,
        specularColor: [0.62, 0.88, 1.0],
        specularIntensity: 0.73,
        anisotropy: 0.46,
        anisotropyRotation: 0.39,
      },
      texturePayload: {
        clearcoatMap: 0.84,
        clearcoatRoughnessMap: 0.64,
        sheenColorMap: [0.22, 0.81, 0.64],
        sheenRoughnessMap: 0.41,
        iridescenceMap: 0.71,
        iridescenceThicknessMap: 0.5,
        anisotropyMap: [0.46, 0.39],
        specularColorMap: [0.62, 0.88, 1.0],
        specularIntensityMap: 0.73,
      },
      wo: normalize([-0.18, 0.37, 0.91]),
      wi: normalize([0.44, -0.22, 0.87]),
      Lo: [0.74, 1.46, 1.12],
    },
  ];
}

/** @param {unknown} value */
function roundDeep(value) {
  if (typeof value === 'number') return round(value);
  if (Array.isArray(value)) return value.map(roundDeep);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, roundDeep(v)]));
  }
  return value;
}

/** @param {Record<string, any>} cfg */
function evaluateCase(cfg) {
  const normal = /** @type {Vec3} */ ([0, 0, 1]);
  const clearcoatNormal = normalize([0.08, -0.04, 0.996]);
  const xv = /** @type {Vec3} */ ([0, 0, 0]);
  const xs = scale(cfg.wi, 1.75);
  const cosTheta = Math.max(dot(normal, cfg.wi), 0);
  const fBsdf = evaluateSpecialtyBrdf(cfg.material, normal, clearcoatNormal, cfg.wo, cfg.wi);
  const pdfSrc = sourcePdfFull(cfg.material, normal, clearcoatNormal, cfg.wo, cfg.wi);
  const pHat = luminance(mul(scale(fBsdf, cosTheta), cfg.Lo));
  const wCandidate = pdfSrc > 1e-8 ? pHat / pdfSrc : 0;
  const W = pHat > 1e-9 ? wCandidate / pHat : 0;
  const restirIndirect = mul(scale(fBsdf, cosTheta * W), cfg.Lo);
  const baseIndirect = mul(scale(fBsdf, cosTheta / pdfSrc), cfg.Lo);
  const diff = sub(restirIndirect, baseIndirect);
  const absDiff = Math.max(Math.abs(diff[0]), Math.abs(diff[1]), Math.abs(diff[2]));
  const baseLum = luminance(baseIndirect);
  const relDiff = baseLum > 0 ? Math.abs(luminance(diff)) / baseLum : 0;

  const neutral = {
    ...cfg,
    material: neutralMaterial(),
    activeLobes: [],
  };
  const neutralBrdf = evaluateSpecialtyBrdf(neutral.material, normal, clearcoatNormal, cfg.wo, cfg.wi);
  const neutralPdf = sourcePdfFull(neutral.material, normal, clearcoatNormal, cfg.wo, cfg.wi);
  const neutralIndirect = mul(scale(neutralBrdf, cosTheta / neutralPdf), cfg.Lo);

  return roundDeep({
    id: cfg.id,
    materialSource: cfg.materialSource ?? 'scalar',
    activeLobes: cfg.activeLobes,
    ...(cfg.texturePayload ? { texturePayload: cfg.texturePayload } : {}),
    geometry: { xv, xs, normal, clearcoatNormal, wo: cfg.wo, wi: cfg.wi, cosTheta },
    material: cfg.material,
    reference: {
      fBsdf,
      Lo: cfg.Lo,
      pdfSrc,
      pHat,
      wCandidate,
      W,
    },
    basePath: {
      indirect: baseIndirect,
      luminance: baseLum,
    },
    restirPt: {
      indirect: restirIndirect,
      luminance: luminance(restirIndirect),
    },
    ab: {
      absDiff,
      relativeError: relDiff,
      lobeDeltaFromNeutral: luminance(restirIndirect) - luminance(neutralIndirect),
    },
  });
}

export function runRestirPtSpecialtyReference() {
  const cases = specialtyCases().map(evaluateCase);
  const coveredLobes = [...new Set(cases.flatMap((c) => c.activeLobes))].sort();
  const materialSources = [...new Set(cases.map((c) => c.materialSource))].sort();
  const maxAbsoluteError = Math.max(...cases.map((c) => c.ab.absDiff));
  const maxRelativeError = Math.max(...cases.map((c) => c.ab.relativeError));
  const luminanceChecksum = cases.reduce((acc, c, i) => acc + (i + 1) * c.restirPt.luminance, 0);
  const pdfChecksum = cases.reduce((acc, c, i) => acc + (i + 1) * c.reference.pdfSrc, 0);

  return roundDeep({
    schema: 'vitrum.restir-pt.specialty-reference.v1',
    mode: 'cpu-static',
    invariant: 'single-sample ReSTIR-PT resolves to f_bsdf*cos*Lo/pdfSrc for specialty visible lobes',
    coverage: {
      specialtyLobes: coveredLobes,
      materialSources,
      requiresGpuRecapture: false,
    },
    summary: {
      caseCount: cases.length,
      maxAbsoluteError,
      maxRelativeError,
      luminanceChecksum,
      pdfChecksum,
    },
    cases,
  });
}

function stableJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function main() {
  const result = runRestirPtSpecialtyReference();
  const json = stableJson(result);
  const args = new Set(process.argv.slice(2));

  if (args.has('--write')) {
    writeFileSync(FIXTURE_PATH, json);
    console.log(`Wrote ${FIXTURE_PATH}`);
    return;
  }

  if (args.has('--check')) {
    const expected = readFileSync(FIXTURE_PATH, 'utf8');
    if (expected !== json) {
      console.error(`${FIXTURE_PATH} is stale; run tools/radiometric-ab/ab-restir-pt-specialty.mjs --write`);
      process.exitCode = 1;
    }
    return;
  }

  process.stdout.write(json);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  main();
}

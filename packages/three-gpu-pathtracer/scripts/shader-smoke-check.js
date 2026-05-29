import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

function read(relPath) {
	return readFileSync(resolve(process.cwd(), relPath), 'utf8');
}

function expectMatch(text, pattern, message) {
	if (!pattern.test(text)) {
		throw new Error(message);
	}
}

function expectNoMatch(text, pattern, message) {
	if (pattern.test(text)) {
		throw new Error(message);
	}
}

const renderStructs = read('./src/materials/pathtracing/glsl/render_structs.glsl.js');
const directLight = read('./src/materials/pathtracing/glsl/direct_light_contribution_function.glsl.js');
const bsdf = read('./src/shader/bsdf/bsdf_functions.glsl.js');
const spectral = read('./src/shader/bsdf/spectral_accumulator.glsl.js');
const util = read('./src/shader/common/util_functions.glsl.js');
const materialMain = read('./src/materials/pathtracing/PhysicalPathTracingMaterial.js');
const lightSampling = read('./src/shader/sampling/light_sampling_functions.glsl.js');

expectMatch(renderStructs, /float wavelength;/, 'RenderState missing wavelength field');
expectMatch(renderStructs, /float wavelengthPdf;/, 'RenderState missing wavelengthPdf field');
expectMatch(renderStructs, /float throughput;/, 'RenderState missing scalar throughput field');
expectNoMatch(renderStructs, /throughputColor/, 'RenderState still contains legacy throughputColor');

expectMatch(
	directLight,
	/wavelengthToRGB\s*\(\s*state\.wavelength\s*,\s*state\.throughput\s*,\s*state\.wavelengthPdf\s*\)/,
	'direct_light_contribution is not using hero-wavelength throughput conversion',
);
expectMatch(
	directLight,
	/bsdfResult\s*\(\s*worldWo,\s*lightRec\.direction,\s*surf,\s*state\.wavelength,/,
	'direct light path must pass state.wavelength into bsdfResult',
);
expectMatch(
	directLight,
	/bsdfResult\s*\(\s*worldWo,\s*envDirection,\s*surf,\s*state\.wavelength,/,
	'environment light path must pass state.wavelength into bsdfResult',
);
expectNoMatch(directLight, /throughputColor/, 'direct_light_contribution still references throughputColor');
expectMatch(materialMain, /uniform float uRadianceClamp;/, 'PhysicalPathTracingMaterial missing radiance clamp uniform');
// Sprint 5: output color names have moved across fork eras:
// gl_FragColor (legacy), gColor (MRT location 0), pc_fragColor (current three.js GLSL3 output).
expectMatch(
	materialMain,
	/sampleLuminance\s*=\s*dot\s*\(\s*(?:gl_FragColor|gColor|pc_fragColor)\.rgb,\s*vec3\s*\(\s*0\.2126,\s*0\.7152,\s*0\.0722\s*\)\s*\)/,
	'PhysicalPathTracingMaterial must clamp final sample luminance for firefly control',
);
expectMatch(spectral, /uniform int uSpectralRendering;/, 'spectral accumulator missing preview-mode gate uniform');
expectMatch(
	spectral,
	/if\s*\(\s*uSpectralRendering\s*==\s*0\s*\)\s*return vec3\s*\(\s*throughput\s*\)/,
	'spectral accumulator must keep low-SPP preview path RGB-stable by default',
);

expectMatch(
	bsdf,
	/ScatterRecord bsdfSample\(\s*vec3 worldWo,\s*SurfaceRecord surf,\s*float heroWavelength\s*\)/,
	'bsdfSample signature missing hero wavelength parameter',
);
expectMatch(
	bsdf,
	/ScatterRecord sssSample\(\s*vec3 worldWo,\s*SurfaceRecord surf,\s*float heroWavelength\s*\)/,
	'sssSample signature missing hero wavelength parameter',
);
expectNoMatch(bsdf, /TODO\(sprint-7-flags\)/, 'Stale sprint-7 flags TODO still present in bsdf_functions');
const evalSpectrumIndex = bsdf.indexOf('float evalSpectrum( vec3 coeffs, float lambda )');
const evalSpectrumAtHeroIndex = bsdf.indexOf('float evalSpectrumAtHero( float lambdaNm )');
if (evalSpectrumIndex === -1 || evalSpectrumAtHeroIndex === -1 || evalSpectrumIndex > evalSpectrumAtHeroIndex) {
	throw new Error('evalSpectrum must be declared before evalSpectrumAtHero for GLSL compile order');
}
expectMatch(
	bsdf,
	/float bsdfResult\s*\(\s*vec3 worldWo,\s*vec3 worldWi,\s*SurfaceRecord surf,\s*float heroWavelength,\s*inout vec3 color\s*\)/,
	'bsdfResult must thread hero wavelength for NEE consistency',
);
expectMatch(
	bsdf,
	/return ggxPDF\s*\(\s*wo,\s*wh,\s*filteredRoughness\s*\)\s*\/\s*denom/,
	'transmissionEval must use GGX BTDF Jacobian PDF (Walter et al.)',
);
expectMatch(
	bsdf,
	/vec3 transmissionDirection[\s\S]*?ggxDirection\s*\(\s*wo,\s*vec2\s*\(\s*filteredRoughness\s*\),\s*rand2\s*\(\s*13\s*\)\s*\)/,
	'transmissionDirection must sample GGX half-vectors to match transmissionEval PDF',
);
expectMatch(
	bsdf,
	/vec3 dispersionTransmissionDirection[\s\S]*?ggxDirection\s*\(\s*wo,\s*vec2\s*\(\s*filteredRoughness\s*\),\s*rand2\s*\(\s*13\s*\)\s*\)/,
	'dispersionTransmissionDirection must sample GGX half-vectors to match transmissionEval PDF',
);
expectNoMatch(bsdf, /incorrect PDF/, 'transmissionEval must not retain incorrect-PDF TODO');
expectMatch(util, /float heroWeightFromRgb\(/, 'heroWeightFromRgb helper missing');

// ── Jakob & Hanika 2019 RGB→spectrum upsampling is actually consumed ─────────
// The host upsamples a representative medium albedo into u_jakobCoeffs; without
// these assertions the coefficients would die at the uniform (their previous
// state). The integrator must (a) define a gated mediumAlbedoHero helper that
// routes through evalSpectrumAtHero only when upsampling is active, and (b) call
// it at the volume-scatter and SSS single-scatter albedo sites.
expectMatch(
	bsdf,
	/float mediumAlbedoHero\(\s*vec3 rgb,\s*float heroWavelength\s*\)\s*\{[\s\S]*?if\s*\(\s*spectralUpsamplingActive\(\)\s*\)\s*\{[\s\S]*?return evalSpectrumAtHero\(\s*heroWavelength\s*\)/,
	'mediumAlbedoHero must route the hero albedo through evalSpectrumAtHero under the spectral gate',
);
expectMatch(
	bsdf,
	/float mediumAlbedoHero\(\s*vec3 rgb,\s*float heroWavelength\s*\)\s*\{[\s\S]*?return heroScalarFromRgb\(\s*rgb,\s*heroWavelength\s*\)/,
	'mediumAlbedoHero must fall back to the legacy heroScalarFromRgb projection when the gate is off',
);
// The gate must require BOTH spectral rendering on AND a non-flat coefficient set.
// The flat (0,0,0) default evaluates to S ≡ ½ and would wash colour, so it must
// NOT route through evalSpectrum — the default RGB path stays bit-identical.
expectMatch(
	bsdf,
	/bool spectralUpsamplingActive\(\)\s*\{\s*return uSpectralRendering == 1 &&\s*\(\s*u_jakobCoeffs\.x != 0\.0 \|\| u_jakobCoeffs\.y != 0\.0 \|\| u_jakobCoeffs\.z != 0\.0\s*\)/,
	'spectralUpsamplingActive must gate on spectral rendering AND a non-default u_jakobCoeffs so the flat default never routes through evalSpectrum',
);
// evalSpectrumAtHero must be declared before mediumAlbedoHero (GLSL compile order),
// and mediumAlbedoHero before its call sites further down bsdf_functions.
const evalSpectrumAtHeroIdx = bsdf.indexOf('float evalSpectrumAtHero( float lambdaNm )');
const mediumAlbedoHeroIdx = bsdf.indexOf('float mediumAlbedoHero( vec3 rgb, float heroWavelength )');
if (evalSpectrumAtHeroIdx === -1 || mediumAlbedoHeroIdx === -1 || evalSpectrumAtHeroIdx > mediumAlbedoHeroIdx) {
	throw new Error('mediumAlbedoHero must be declared after evalSpectrumAtHero for GLSL compile order');
}
expectMatch(
	bsdf,
	/sssRec\.throughput = mediumAlbedoHero\(\s*u_sssAlbedo,\s*heroWavelength\s*\)\s*\*\s*beerLambert/,
	'SSS single-scatter albedo must flow through mediumAlbedoHero (gated Jakob–Hanika reflectance)',
);
expectMatch(
	materialMain,
	/state\.throughput \*= mediumAlbedoHero\(\s*u_scatterAlbedo,\s*state\.wavelength\s*\)\s*\*\s*transmittance/,
	'volume single-scatter albedo must flow through mediumAlbedoHero (gated Jakob–Hanika reflectance)',
);
// The legacy preview-stable RGB projection must still exist as the default branch.
expectMatch(util, /dot\(\s*rgb,\s*vec3\(\s*tR,\s*tG,\s*tB\s*\)\s*\)/, 'heroScalarFromRgb smoothstep projection must remain as the default-path fallback');
expectMatch(
	util,
	/readSpectralAttenuationMu\s*\(\s*sampler2D materialsTex/,
	'util_functions must sample packed spectral μ from materials texture',
);
expectMatch(
	util,
	/exp\s*\(\s*-\s*muLambda\s*\*\s*dist\s*\)/,
	'transmissionAttenuationHero must apply Beer-Lambert with interpolated spectral μ',
);
expectMatch(
	util,
	/transmissionAttenuationHero\s*\(\s*sampler2D materialsTex/,
	'transmissionAttenuationHero must accept materials sampler for spectral lookup',
);
expectNoMatch(
	lightSampling,
	/denominator is potentially zero/,
	'light sampling must not retain zero-denominator TODOs in area-light PDF code',
);
expectMatch(
	lightSampling,
	/lightRec\.pdf\s*=\s*max\s*\(/,
	'area-light sampling must clamp PDFs to finite positive values',
);

expectMatch(
	materialMain,
	/bsdfSample\s*\(\s*-\s*ray\.direction,\s*surf,\s*state\.wavelength\s*\)/,
	'PhysicalPathTracingMaterial must call bsdfSample with hero wavelength',
);
expectMatch(
	materialMain,
	/sssSample\s*\(\s*-\s*ray\.direction,\s*surf,\s*state\.wavelength\s*\)/,
	'PhysicalPathTracingMaterial must call sssSample with hero wavelength',
);

console.log('Shader smoke checks passed.');

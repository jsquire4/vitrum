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

/**
 * BMFR blockwise feature regression.
 *
 * Each fit invocation factors the augmented feature matrix directly with
 * Householder QR. Rows are consumed in bounded chunks by repeatedly factoring
 * [R; A_chunk], a streaming QR reduction that never forms A-transpose-A.
 *
 * Overlapping blocks write unique coefficient records. A second pixel-grid
 * entry point evaluates every block covering a pixel in a stable order and
 * averages the reconstructions before temporal accumulation. There are no
 * cross-workgroup texture-write races.
 *
 * Reference: Koskela et al. Blockwise Multi-Order Feature Regression for
 * Real-Time Path-Tracing Reconstruction. ACM TOG 38(5), 2019.
 */

import { FLOAT16_MAX_FINITE } from '../halfFloat.js';

/** Cooperative TSQR lanes used by one deterministic block fit. */
export const BMFR_WORKGROUP_SIZE = 32 as const;

/** Pixel-grid workgroup edge for the overlap resolve pass. */
export const BMFR_RESOLVE_WORKGROUP_SIZE = 8 as const;

/** Storage stride of one block fit: ten vec4f words. */
export const BMFR_BLOCK_FIT_SIZE_BYTES = 160 as const;

/** Feature columns; must equal BMFR_FEATURE_COUNT in bmfrRegression.ts. */
export const BMFR_WGSL_FEATURE_COUNT = 10 as const;

export const BMFR_WGSL = /* wgsl */ `
const F: u32 = ${BMFR_WGSL_FEATURE_COUNT}u;
const QR_CHUNK_ROWS: u32 = 16u;
const QR_ROWS: u32 = F + QR_CHUNK_ROWS;
const FIT_LANES: u32 = ${BMFR_WORKGROUP_SIZE}u;
const UPPER_R_FLOATS: u32 = 55u;

struct BmfrUBO {
  blockSize:      u32,
  blockStride:    u32,
  positionScale:  f32,
  temporalAlpha:  f32,
  regularisation: f32,
  hasHistory:     f32,
  // 0 = world-position xyz with explicit positive validity in w.
  // 1 = screen-space xy plus ABSOLUTE signed depth from w. Negative depth is
  //     the transmissive-surface discriminator, not a miss.
  positionMode:   u32,
};

struct BmfrBlockFit {
  // word 0 = mean position xyz + valid count.
  // words 1..3, 4..6, 7..9 = R, G, B coefficients respectively.
  words: array<vec4f, 10>,
};

@group(0) @binding(0) var bmfr_color:    texture_2d<f32>;
@group(0) @binding(1) var bmfr_normal:   texture_2d<f32>;
@group(0) @binding(2) var bmfr_worldPos: texture_2d<f32>;
@group(0) @binding(3) var bmfr_history:  texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> bmfr_blockFits: array<BmfrBlockFit>;
@group(0) @binding(5) var<uniform> bmfr_ubo: BmfrUBO;
@group(0) @binding(6) var bmfr_out: texture_storage_2d<rgba16float, write>;

// One block is factored cooperatively with a tree-structured TSQR reduction.
// Each lane first reduces its disjoint pixel rows to a private 10x10 R and
// transformed RGB target. Upper triangles and targets then merge pairwise in
// workgroup memory. At 32 lanes this reduces the serial chunk depth from 64 to
// two local chunks plus five logarithmic merge levels for a full 32x32 block.
var<workgroup> bmfr_positionPartials: array<vec3f, ${BMFR_WORKGROUP_SIZE}>;
var<workgroup> bmfr_countPartials: array<u32, ${BMFR_WORKGROUP_SIZE}>;
var<workgroup> bmfr_sharedR: array<f32, ${BMFR_WORKGROUP_SIZE * 55}>;
var<workgroup> bmfr_sharedZR: array<f32, ${BMFR_WORKGROUP_SIZE * BMFR_WGSL_FEATURE_COUNT}>;
var<workgroup> bmfr_sharedZG: array<f32, ${BMFR_WORKGROUP_SIZE * BMFR_WGSL_FEATURE_COUNT}>;
var<workgroup> bmfr_sharedZB: array<f32, ${BMFR_WORKGROUP_SIZE * BMFR_WGSL_FEATURE_COUNT}>;

fn ceilDiv(value: u32, divisor: u32) -> u32 {
  return (value + divisor - 1u) / divisor;
}

fn upperRIndex(row: u32, col: u32) -> u32 {
  // Number of upper-triangular entries preceding the row, plus its column offset.
  return row * (2u * F - row + 1u) / 2u + (col - row);
}

fn loadPosition(coord: vec2u) -> vec4f {
  let raw = textureLoad(bmfr_worldPos, coord, 0);
  if (bmfr_ubo.positionMode == 1u) {
    let depth = abs(raw.w);
    return vec4f(f32(coord.x), f32(coord.y), depth, depth);
  }
  return raw;
}

fn featureRow(
  pLocal: vec3f,
  normal: vec3f,
  row: ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>,
) {
  (*row)[0] = 1.0;
  (*row)[1] = pLocal.x;
  (*row)[2] = pLocal.y;
  (*row)[3] = pLocal.z;
  (*row)[4] = normal.x;
  (*row)[5] = normal.y;
  (*row)[6] = normal.z;
  (*row)[7] = pLocal.x * pLocal.x;
  (*row)[8] = pLocal.y * pLocal.y;
  (*row)[9] = pLocal.z * pLocal.z;
}

// Direct Householder QR of [R_previous; A_chunk]. The same reflectors are
// applied to all three target channels. Retaining the leading R and Q-transformed
// targets is an exact streaming QR reduction, subject only to f32 rounding.
fn qrReduceChunk(
  matrix: ptr<function, array<f32, 260>>,
  rhsR: ptr<function, array<f32, 26>>,
  rhsG: ptr<function, array<f32, 26>>,
  rhsB: ptr<function, array<f32, 26>>,
  rState: ptr<function, array<f32, 100>>,
  zRState: ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>,
  zGState: ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>,
  zBState: ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>,
) {
  for (var col = 0u; col < F; col = col + 1u) {
    var normSq = 0.0;
    for (var row = col; row < QR_ROWS; row = row + 1u) {
      let value = (*matrix)[row * F + col];
      normSq = normSq + value * value;
    }
    var norm = sqrt(normSq);
    if (norm < 1e-20) {
      continue;
    }

    let x0 = (*matrix)[col * F + col];
    norm = norm * select(-1.0, 1.0, x0 >= 0.0);
    var reflector: array<f32, 26>;
    for (var row = 0u; row < QR_ROWS; row = row + 1u) {
      reflector[row] = 0.0;
    }
    reflector[col] = x0 + norm;
    for (var row = col + 1u; row < QR_ROWS; row = row + 1u) {
      reflector[row] = (*matrix)[row * F + col];
    }

    var reflectorNormSq = 0.0;
    for (var row = col; row < QR_ROWS; row = row + 1u) {
      reflectorNormSq =
        reflectorNormSq + reflector[row] * reflector[row];
    }
    if (reflectorNormSq < 1e-30) {
      continue;
    }

    for (var matrixCol = col; matrixCol < F; matrixCol = matrixCol + 1u) {
      var dotValue = 0.0;
      for (var row = col; row < QR_ROWS; row = row + 1u) {
        dotValue =
          dotValue + reflector[row] * (*matrix)[row * F + matrixCol];
      }
      let factor = 2.0 * dotValue / reflectorNormSq;
      for (var row = col; row < QR_ROWS; row = row + 1u) {
        let index = row * F + matrixCol;
        (*matrix)[index] = (*matrix)[index] - factor * reflector[row];
      }
    }

    var dotR = 0.0;
    var dotG = 0.0;
    var dotB = 0.0;
    for (var row = col; row < QR_ROWS; row = row + 1u) {
      dotR = dotR + reflector[row] * (*rhsR)[row];
      dotG = dotG + reflector[row] * (*rhsG)[row];
      dotB = dotB + reflector[row] * (*rhsB)[row];
    }
    let factorR = 2.0 * dotR / reflectorNormSq;
    let factorG = 2.0 * dotG / reflectorNormSq;
    let factorB = 2.0 * dotB / reflectorNormSq;
    for (var row = col; row < QR_ROWS; row = row + 1u) {
      (*rhsR)[row] = (*rhsR)[row] - factorR * reflector[row];
      (*rhsG)[row] = (*rhsG)[row] - factorG * reflector[row];
      (*rhsB)[row] = (*rhsB)[row] - factorB * reflector[row];
    }
  }

  for (var row = 0u; row < F; row = row + 1u) {
    (*zRState)[row] = (*rhsR)[row];
    (*zGState)[row] = (*rhsG)[row];
    (*zBState)[row] = (*rhsB)[row];
    for (var col = 0u; col < F; col = col + 1u) {
      (*rState)[row * F + col] = (*matrix)[row * F + col];
    }
  }
}

fn backSubstitute(
  rState: ptr<function, array<f32, 100>>,
  zState: ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>,
  output: ptr<function, array<f32, ${BMFR_WGSL_FEATURE_COUNT}>>,
) {
  for (var i = 0u; i < F; i = i + 1u) {
    (*output)[i] = 0.0;
  }
  for (var reverse = 0u; reverse < F; reverse = reverse + 1u) {
    let row = F - 1u - reverse;
    var value = (*zState)[row];
    for (var col = row + 1u; col < F; col = col + 1u) {
      value = value - (*rState)[row * F + col] * (*output)[col];
    }
    let diagonal = (*rState)[row * F + row];
    if (abs(diagonal) > 1e-20) {
      (*output)[row] = value / diagonal;
    }
  }
}

@compute @workgroup_size(${BMFR_WORKGROUP_SIZE}, 1, 1)
fn bmfrMain(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) lane: u32,
) {
  let dimensions = textureDimensions(bmfr_color);
  let blockSize = bmfr_ubo.blockSize;
  let stride = bmfr_ubo.blockStride;
  let blockCount = vec2u(
    ceilDiv(dimensions.x, stride),
    ceilDiv(dimensions.y, stride),
  );
  if (any(workgroupId.xy >= blockCount)) {
    return;
  }

  let blockOrigin = workgroupId.xy * stride;
  let pixelCapacity = blockSize * blockSize;
  var localPositionSum = vec3f(0.0);
  var localValidCount = 0u;
  for (
    var linearIndex = lane;
    linearIndex < pixelCapacity;
    linearIndex = linearIndex + FIT_LANES
  ) {
    let localCoord = vec2u(
      linearIndex % blockSize,
      linearIndex / blockSize,
    );
    let coord = blockOrigin + localCoord;
    if (any(coord >= dimensions)) {
      continue;
    }
    let position = loadPosition(coord);
    if (position.w > 0.0) {
      localPositionSum = localPositionSum + position.xyz;
      localValidCount = localValidCount + 1u;
    }
  }
  bmfr_positionPartials[lane] = localPositionSum;
  bmfr_countPartials[lane] = localValidCount;
  workgroupBarrier();
  if (lane == 0u) {
    var sum = vec3f(0.0);
    var count = 0u;
    for (var sourceLane = 0u; sourceLane < FIT_LANES; sourceLane = sourceLane + 1u) {
      sum = sum + bmfr_positionPartials[sourceLane];
      count = count + bmfr_countPartials[sourceLane];
    }
    bmfr_positionPartials[0] = sum;
    bmfr_countPartials[0] = count;
  }
  workgroupBarrier();
  let validCount = bmfr_countPartials[0];
  var meanPosition = vec3f(0.0);
  if (validCount > 0u) {
    meanPosition = bmfr_positionPartials[0] / f32(validCount);
  }

  // Only lane zero seeds sqrt(lambda) I. The TSQR tree therefore contains one,
  // and only one, copy of the augmented regularisation rows.
  var rState: array<f32, 100>;
  var zRState: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
  var zGState: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
  var zBState: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
  for (var index = 0u; index < F * F; index = index + 1u) {
    rState[index] = 0.0;
  }
  let regularisationRoot = sqrt(max(0.0, bmfr_ubo.regularisation));
  for (var index = 0u; index < F; index = index + 1u) {
    if (lane == 0u) {
      rState[index * F + index] = regularisationRoot;
    }
    zRState[index] = 0.0;
    zGState[index] = 0.0;
    zBState[index] = 0.0;
  }

  let inverseScale = 1.0 / bmfr_ubo.positionScale;
  for (
    var chunkStart = lane * QR_CHUNK_ROWS;
    chunkStart < pixelCapacity;
    chunkStart = chunkStart + QR_CHUNK_ROWS * FIT_LANES
  ) {
    var matrix: array<f32, 260>;
    var rhsR: array<f32, 26>;
    var rhsG: array<f32, 26>;
    var rhsB: array<f32, 26>;
    for (var index = 0u; index < QR_ROWS * F; index = index + 1u) {
      matrix[index] = 0.0;
    }
    for (var row = 0u; row < QR_ROWS; row = row + 1u) {
      rhsR[row] = 0.0;
      rhsG[row] = 0.0;
      rhsB[row] = 0.0;
    }
    for (var row = 0u; row < F; row = row + 1u) {
      rhsR[row] = zRState[row];
      rhsG[row] = zGState[row];
      rhsB[row] = zBState[row];
      for (var col = 0u; col < F; col = col + 1u) {
        matrix[row * F + col] = rState[row * F + col];
      }
    }

    for (var chunkRow = 0u; chunkRow < QR_CHUNK_ROWS; chunkRow = chunkRow + 1u) {
      let linearIndex = chunkStart + chunkRow;
      if (linearIndex >= pixelCapacity) {
        continue;
      }
      let localCoord = vec2u(
        linearIndex % blockSize,
        linearIndex / blockSize,
      );
      let coord = blockOrigin + localCoord;
      if (any(coord >= dimensions)) {
        continue;
      }
      let position = loadPosition(coord);
      if (position.w <= 0.0) {
        continue;
      }

      let localPosition = (position.xyz - meanPosition) * inverseScale;
      let normal = textureLoad(bmfr_normal, coord, 0).xyz * 2.0 - 1.0;
      let color = textureLoad(bmfr_color, coord, 0).rgb;
      var features: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
      featureRow(localPosition, normal, &features);
      let targetRow = F + chunkRow;
      for (var col = 0u; col < F; col = col + 1u) {
        matrix[targetRow * F + col] = features[col];
      }
      rhsR[targetRow] = color.r;
      rhsG[targetRow] = color.g;
      rhsB[targetRow] = color.b;
    }

    qrReduceChunk(
      &matrix,
      &rhsR,
      &rhsG,
      &rhsB,
      &rState,
      &zRState,
      &zGState,
      &zBState,
    );
  }

  // Publish each lane's compact upper R plus transformed RGB targets.
  let laneRBase = lane * UPPER_R_FLOATS;
  let laneZBase = lane * F;
  for (var row = 0u; row < F; row = row + 1u) {
    for (var col = row; col < F; col = col + 1u) {
      bmfr_sharedR[laneRBase + upperRIndex(row, col)] =
        rState[row * F + col];
    }
    bmfr_sharedZR[laneZBase + row] = zRState[row];
    bmfr_sharedZG[laneZBase + row] = zGState[row];
    bmfr_sharedZB[laneZBase + row] = zBState[row];
  }
  workgroupBarrier();

  // Pairwise TSQR: factor [R_left; R_right] at each logarithmic tree level.
  for (var mergeStep = 1u; mergeStep < FIT_LANES; mergeStep = mergeStep * 2u) {
    let partner = lane + mergeStep;
    if (lane % (mergeStep * 2u) == 0u && partner < FIT_LANES) {
      var matrix: array<f32, 260>;
      var rhsR: array<f32, 26>;
      var rhsG: array<f32, 26>;
      var rhsB: array<f32, 26>;
      for (var index = 0u; index < QR_ROWS * F; index = index + 1u) {
        matrix[index] = 0.0;
      }
      for (var row = 0u; row < QR_ROWS; row = row + 1u) {
        rhsR[row] = 0.0;
        rhsG[row] = 0.0;
        rhsB[row] = 0.0;
      }
      let ownRBase = lane * UPPER_R_FLOATS;
      let partnerRBase = partner * UPPER_R_FLOATS;
      let ownZBase = lane * F;
      let partnerZBase = partner * F;
      for (var row = 0u; row < F; row = row + 1u) {
        rhsR[row] = bmfr_sharedZR[ownZBase + row];
        rhsG[row] = bmfr_sharedZG[ownZBase + row];
        rhsB[row] = bmfr_sharedZB[ownZBase + row];
        rhsR[F + row] = bmfr_sharedZR[partnerZBase + row];
        rhsG[F + row] = bmfr_sharedZG[partnerZBase + row];
        rhsB[F + row] = bmfr_sharedZB[partnerZBase + row];
        for (var col = row; col < F; col = col + 1u) {
          matrix[row * F + col] =
            bmfr_sharedR[ownRBase + upperRIndex(row, col)];
          matrix[(F + row) * F + col] =
            bmfr_sharedR[partnerRBase + upperRIndex(row, col)];
        }
      }
      qrReduceChunk(
        &matrix,
        &rhsR,
        &rhsG,
        &rhsB,
        &rState,
        &zRState,
        &zGState,
        &zBState,
      );
      for (var row = 0u; row < F; row = row + 1u) {
        for (var col = row; col < F; col = col + 1u) {
          bmfr_sharedR[ownRBase + upperRIndex(row, col)] =
            rState[row * F + col];
        }
        bmfr_sharedZR[ownZBase + row] = zRState[row];
        bmfr_sharedZG[ownZBase + row] = zGState[row];
        bmfr_sharedZB[ownZBase + row] = zBState[row];
      }
    }
    workgroupBarrier();
  }

  if (lane != 0u) {
    return;
  }

  var alphaR: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
  var alphaG: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
  var alphaB: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
  backSubstitute(&rState, &zRState, &alphaR);
  backSubstitute(&rState, &zGState, &alphaG);
  backSubstitute(&rState, &zBState, &alphaB);

  var fit: BmfrBlockFit;
  for (var word = 0u; word < 10u; word = word + 1u) {
    fit.words[word] = vec4f(0.0);
  }
  fit.words[0] = vec4f(meanPosition, f32(validCount));
  fit.words[1] = vec4f(alphaR[0], alphaR[1], alphaR[2], alphaR[3]);
  fit.words[2] = vec4f(alphaR[4], alphaR[5], alphaR[6], alphaR[7]);
  fit.words[3] = vec4f(alphaR[8], alphaR[9], 0.0, 0.0);
  fit.words[4] = vec4f(alphaG[0], alphaG[1], alphaG[2], alphaG[3]);
  fit.words[5] = vec4f(alphaG[4], alphaG[5], alphaG[6], alphaG[7]);
  fit.words[6] = vec4f(alphaG[8], alphaG[9], 0.0, 0.0);
  fit.words[7] = vec4f(alphaB[0], alphaB[1], alphaB[2], alphaB[3]);
  fit.words[8] = vec4f(alphaB[4], alphaB[5], alphaB[6], alphaB[7]);
  fit.words[9] = vec4f(alphaB[8], alphaB[9], 0.0, 0.0);
  let blockIndex = workgroupId.y * blockCount.x + workgroupId.x;
  bmfr_blockFits[blockIndex] = fit;
}

fn fitAlpha(fit: BmfrBlockFit, channel: u32, feature: u32) -> f32 {
  let word = 1u + channel * 3u + feature / 4u;
  return fit.words[word][feature % 4u];
}

fn firstCoveringBlock(pixel: u32, blockSize: u32, stride: u32) -> u32 {
  if (pixel + 1u <= blockSize) {
    return 0u;
  }
  return ceilDiv(pixel + 1u - blockSize, stride);
}

fn finiteHalfChannel(value: f32) -> f32 {
  if (value != value) {
    return 0.0;
  }
  return clamp(value, -${FLOAT16_MAX_FINITE}.0, ${FLOAT16_MAX_FINITE}.0);
}

fn finiteHalfRgb(value: vec3f) -> vec3f {
  return vec3f(
    finiteHalfChannel(value.r),
    finiteHalfChannel(value.g),
    finiteHalfChannel(value.b),
  );
}

@compute @workgroup_size(${BMFR_RESOLVE_WORKGROUP_SIZE}, ${BMFR_RESOLVE_WORKGROUP_SIZE}, 1)
fn bmfrResolve(@builtin(global_invocation_id) globalId: vec3u) {
  let dimensions = textureDimensions(bmfr_color);
  let coord = globalId.xy;
  if (any(coord >= dimensions)) {
    return;
  }

  let rawColor = textureLoad(bmfr_color, coord, 0).rgb;
  let position = loadPosition(coord);
  if (position.w <= 0.0) {
    textureStore(bmfr_out, coord, vec4f(finiteHalfRgb(rawColor), 1.0));
    return;
  }

  let blockSize = bmfr_ubo.blockSize;
  let stride = bmfr_ubo.blockStride;
  let blockCount = vec2u(
    ceilDiv(dimensions.x, stride),
    ceilDiv(dimensions.y, stride),
  );
  let firstBlock = vec2u(
    firstCoveringBlock(coord.x, blockSize, stride),
    firstCoveringBlock(coord.y, blockSize, stride),
  );
  let lastBlock = min(coord / stride, blockCount - vec2u(1u));
  let normal = textureLoad(bmfr_normal, coord, 0).xyz * 2.0 - 1.0;
  let inverseScale = 1.0 / bmfr_ubo.positionScale;

  var reconstructionSum = vec3f(0.0);
  var contributionCount = 0u;
  for (
    var blockY = firstBlock.y;
    blockY <= lastBlock.y;
    blockY = blockY + 1u
  ) {
    for (
      var blockX = firstBlock.x;
      blockX <= lastBlock.x;
      blockX = blockX + 1u
    ) {
      let fitIndex = blockY * blockCount.x + blockX;
      let fit = bmfr_blockFits[fitIndex];
      if (fit.words[0].w <= 0.0) {
        continue;
      }
      let localPosition = (position.xyz - fit.words[0].xyz) * inverseScale;
      var features: array<f32, ${BMFR_WGSL_FEATURE_COUNT}>;
      featureRow(localPosition, normal, &features);
      var reconstruction = vec3f(0.0);
      for (var feature = 0u; feature < F; feature = feature + 1u) {
        reconstruction = reconstruction + features[feature] * vec3f(
          fitAlpha(fit, 0u, feature),
          fitAlpha(fit, 1u, feature),
          fitAlpha(fit, 2u, feature),
        );
      }
      reconstructionSum =
        reconstructionSum + max(reconstruction, vec3f(0.0));
      contributionCount = contributionCount + 1u;
    }
  }

  var outputColor = rawColor;
  if (contributionCount > 0u) {
    outputColor = reconstructionSum / f32(contributionCount);
  }
  if (bmfr_ubo.hasHistory > 0.5) {
    let history = textureLoad(bmfr_history, coord, 0).rgb;
    outputColor = mix(history, outputColor, bmfr_ubo.temporalAlpha);
  }
  textureStore(bmfr_out, coord, vec4f(finiteHalfRgb(outputColor), 1.0));
}
`;

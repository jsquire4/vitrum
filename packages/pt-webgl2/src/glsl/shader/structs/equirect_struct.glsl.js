export const equirect_struct = /* glsl */`

	struct EquirectHdrInfo {

		// .r = per-row conditional inverse CDF; .g = marginal inverse
		// CDF repeated across each row.
		sampler2D distributionWeights;
		sampler2D map;

		float totalSum;

	};

`;

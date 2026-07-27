/**
 * Cross-family MIS for infinite emitters.
 *
 * The WebGL2 estimator evaluates the pure-eye escape (s=0), replayed distant
 * NEE (s=1), and bounded explicit light-subpath connections (s>=2) in separate
 * draws.  This block gives the replay and forward draws the same Veach
 * power-heuristic denominator used by the explicit connection draw.
 */
export const BDPT_INFINITE_MIS_GLSL = /* glsl */ `

        #ifndef BDPT_MAX_EYE_DEPTH
        #define BDPT_MAX_EYE_DEPTH 8
        #endif

        #define BDPT_MAX_INFINITE_STRATEGIES 10

        vec2 bdptOctEncodeDirection( vec3 direction ) {

                vec3 n = normalize( direction );
                n /= abs( n.x ) + abs( n.y ) + abs( n.z );
                vec2 encoded = n.xy;
                if ( n.z < 0.0 ) {

                        vec2 signNotZero = vec2(
                                encoded.x >= 0.0 ? 1.0 : -1.0,
                                encoded.y >= 0.0 ? 1.0 : -1.0
                        );
                        encoded = ( 1.0 - abs( encoded.yx ) ) * signNotZero;

                }
                return encoded;

        }

        vec3 bdptOctDecodeDirection( vec2 encoded ) {

                vec3 n = vec3(
                        encoded,
                        1.0 - abs( encoded.x ) - abs( encoded.y )
                );
                if ( n.z < 0.0 ) {

                        vec2 signNotZero = vec2(
                                n.x >= 0.0 ? 1.0 : -1.0,
                                n.y >= 0.0 ? 1.0 : -1.0
                        );
                        n.xy = ( 1.0 - abs( n.yx ) ) * signNotZero;

                }
                return normalize( n );

        }

        float bdptInfiniteDensityToArea(
                float pdf,
                vec3 fromPosition,
                vec3 destinationPosition,
                vec3 destinationNormal,
                bool destinationIsMedium
        ) {

                vec3 delta = destinationPosition - fromPosition;
                float distanceSquared = dot( delta, delta );
                if ( pdf <= 0.0 || distanceSquared <= 0.0 ) return 0.0;
                float destinationCosine = destinationIsMedium
                        ? 1.0
                        : abs( dot(
                                destinationNormal,
                                delta * inversesqrt( distanceSquared )
                        ) );
                return pdf * destinationCosine / distanceSquared;

        }

        float bdptInfiniteLaunchDensityToArea(
                float launchPdf,
                vec3 receiverNormal,
                vec3 receiverToSource,
                bool receiverIsMedium
        ) {

                if ( launchPdf <= 0.0 ) return 0.0;
                // An infinite emitter launches parallel rays from the scene
                // bounding disk.  Disk area maps to receiver area by orthogonal
                // projection; a finite-point 1/r^2 Jacobian is not applicable.
                float projection = receiverIsMedium
                        ? 1.0
                        : abs( dot(
                                normalize( receiverNormal ),
                                normalize( receiverToSource )
                        ) );
                return launchPdf * projection;

        }

        float bdptInfiniteEyeFamilyWeight(
                int selectedS,
                bool pureEyeImplemented,
                bool pureEyeSampledDelta,
                float neePdf,
                float launchPdf,
                vec3 receiverToSource,
                SurfaceRecord currentSurface,
                vec3 currentPosition,
                vec3 currentWo,
                float heroWavelength,
                int eyeDepth,
                vec3 eyePosition[ BDPT_MAX_EYE_DEPTH ],
                vec3 eyeNormal[ BDPT_MAX_EYE_DEPTH ],
                float eyePdfForward[ BDPT_MAX_EYE_DEPTH ],
                float eyePdfReverse[ BDPT_MAX_EYE_DEPTH ],
                bool eyeSpecular[ BDPT_MAX_EYE_DEPTH ],
                bool eyeMedium[ BDPT_MAX_EYE_DEPTH ]
        ) {

                if ( selectedS < 0 || selectedS > 1 ) return 0.0;
                if ( eyeDepth < 0 || eyeDepth >= BDPT_MAX_EYE_DEPTH ) return 0.0;
                if ( selectedS == 0 && pureEyeSampledDelta ) return 1.0;
                if ( neePdf <= 0.0 ) return selectedS == 0 ? 1.0 : 0.0;

                float logPdfs[ BDPT_MAX_INFINITE_STRATEGIES ];
                bool validPdfs[ BDPT_MAX_INFINITE_STRATEGIES ];
                for ( int strategy = 0; strategy < BDPT_MAX_INFINITE_STRATEGIES; strategy ++ ) {

                        logPdfs[ strategy ] = 0.0;
                        validPdfs[ strategy ] = false;

                }
                validPdfs[ 1 ] = true;

                bool terminalDelta = false;
                float terminalPdf = bsdfPdfResult(
                        currentWo,
                        receiverToSource,
                        currentSurface,
                        heroWavelength,
                        terminalDelta
                );
                // An explicitly connected NEE edge cannot be sampled through a
                // delta BSDF.  The pure-eye draw owns an actually sampled delta
                // escape (the early return above); replay never treats its
                // discrete mass as a solid-angle density.
                if ( terminalDelta ) return 0.0;
                if ( pureEyeImplemented && terminalPdf > 0.0 ) {

                        logPdfs[ 0 ] = log( terminalPdf ) - log( neePdf );
                        validPdfs[ 0 ] = true;

                }

                int pathVertexCount = eyeDepth + 3;
                if (
                        eyeDepth >= 1 &&
                        uBdptMaxLightBounces >= 2 &&
                        launchPdf > 0.0
                ) {

                        vec3 previousPosition = eyePosition[ eyeDepth - 1 ];
                        vec3 toPrevious = normalize( previousPosition - currentPosition );
                        bool currentConnectionDelta = false;
                        float currentSwappedPdf = bsdfPdfResult(
                                receiverToSource,
                                toPrevious,
                                currentSurface,
                                heroWavelength,
                                currentConnectionDelta
                        );
                        bool transitionBlocked =
                                currentConnectionDelta ||
                                eyeSpecular[ eyeDepth - 1 ];
                        float launchAreaPdf = bdptInfiniteLaunchDensityToArea(
                                launchPdf,
                                eyeNormal[ eyeDepth ],
                                receiverToSource,
                                eyeMedium[ eyeDepth ]
                        );
                        float incomingEyeAreaPdf = bdptInfiniteDensityToArea(
                                eyePdfReverse[ eyeDepth ],
                                previousPosition,
                                currentPosition,
                                eyeNormal[ eyeDepth ],
                                eyeMedium[ eyeDepth ]
                        );
                        if (
                                ! transitionBlocked &&
                                launchAreaPdf > 0.0 &&
                                incomingEyeAreaPdf > 0.0
                        ) {

                                // Both launchPdf and neePdf contain the same
                                // source-direction measure.  It is a continuous
                                // solid-angle density for environment/soft-sun
                                // sources and a unit discrete mass for a hard
                                // directional source.  Cancel it in this ratio
                                // before applying the receiver-area Jacobian.
                                float launchToNeeRatio = launchAreaPdf / neePdf;
                                float logRatio =
                                        log( launchToNeeRatio ) -
                                        log( incomingEyeAreaPdf );
                                logPdfs[ 2 ] = logRatio;
                                validPdfs[ 2 ] = true;

                                for ( int strategy = 3; strategy < BDPT_MAX_INFINITE_STRATEGIES; strategy ++ ) {

                                        if (
                                                strategy > uBdptMaxLightBounces ||
                                                strategy > pathVertexCount - 2
                                        ) break;
                                        int destinationDepth = eyeDepth - strategy + 2;
                                        if ( destinationDepth < 1 ) break;
                                        if (
                                                eyeSpecular[ destinationDepth ] ||
                                                eyeSpecular[ destinationDepth - 1 ]
                                        ) break;

                                        vec3 lightwardPosition = strategy == 3
                                                ? currentPosition
                                                : eyePosition[ destinationDepth + 1 ];
                                        float forwardPdf = strategy == 3
                                                ? currentSwappedPdf
                                                : eyePdfForward[ destinationDepth ];
                                        float forwardAreaPdf = bdptInfiniteDensityToArea(
                                                forwardPdf,
                                                lightwardPosition,
                                                eyePosition[ destinationDepth ],
                                                eyeNormal[ destinationDepth ],
                                                eyeMedium[ destinationDepth ]
                                        );
                                        float reverseAreaPdf = bdptInfiniteDensityToArea(
                                                eyePdfReverse[ destinationDepth ],
                                                eyePosition[ destinationDepth - 1 ],
                                                eyePosition[ destinationDepth ],
                                                eyeNormal[ destinationDepth ],
                                                eyeMedium[ destinationDepth ]
                                        );
                                        if ( forwardAreaPdf <= 0.0 || reverseAreaPdf <= 0.0 ) break;
                                        logRatio += log( forwardAreaPdf ) - log( reverseAreaPdf );
                                        logPdfs[ strategy ] = logRatio;
                                        validPdfs[ strategy ] = true;

                                }

                        }

                }

                if ( ! validPdfs[ selectedS ] ) return 0.0;
                float maxPowerLog = 2.0 * logPdfs[ selectedS ];
                for ( int strategy = 0; strategy < BDPT_MAX_INFINITE_STRATEGIES; strategy ++ ) {

                        if ( validPdfs[ strategy ] ) {

                                maxPowerLog = max(
                                        maxPowerLog,
                                        2.0 * logPdfs[ strategy ]
                                );

                        }

                }
                float denominator = 0.0;
                for ( int strategy = 0; strategy < BDPT_MAX_INFINITE_STRATEGIES; strategy ++ ) {

                        if ( validPdfs[ strategy ] ) {

                                denominator += exp(
                                        2.0 * logPdfs[ strategy ] - maxPowerLog
                                );

                        }

                }
                if ( denominator <= 0.0 ) return 0.0;
                return exp(
                        2.0 * logPdfs[ selectedS ] - maxPowerLog
                ) / denominator;

        }
`;

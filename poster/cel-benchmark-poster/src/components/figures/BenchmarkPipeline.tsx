import { posterData, resolveClaim } from '@/data/posterData'

export function BenchmarkPipeline() {
  const protocol = resolveClaim('scope.protocol')
  const methods = resolveClaim('scope.methods')
  const colors = posterData.visualSpec.colors

  return (
    <figure className="pipeline-figure" data-claim-id={protocol.id}>
      <svg viewBox="0 0 760 330" role="img" aria-labelledby="pipeline-title pipeline-description">
        <title id="pipeline-title">CEL controlled benchmark pipeline</title>
        <desc id="pipeline-description">
          Data and constraints flow through predictive models, three counterfactual explanation paradigms,
          shared metrics, and comparable reports under one controlled protocol.
        </desc>
        <defs>
          <marker id="pipeline-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={colors.navyMuted} />
          </marker>
        </defs>
        <rect x="12" y="18" width="736" height="294" fill="none" stroke={colors.teal} strokeWidth="3" />
        <text x="34" y="49" fill={colors.teal} fontSize="16" fontWeight="800">CONTROLLED PROTOCOL</text>
        <text x="714" y="49" textAnchor="end" fill={colors.navyMuted} fontSize="13">fixed comparison conditions</text>

        <g className="pipeline-node">
          <rect x="28" y="92" width="112" height="120" fill={colors.white} stroke={colors.navy} strokeWidth="2" />
          <text x="43" y="124" fontSize="17" fontWeight="800">Data</text>
          <text x="43" y="151" fontSize="13">splits</text>
          <text x="43" y="174" fontSize="13">preprocessing</text>
          <text x="43" y="197" fontSize="13">constraints</text>
        </g>
        <path d="M140 152 H169" stroke={colors.navyMuted} strokeWidth="3" markerEnd="url(#pipeline-arrow)" />
        <g className="pipeline-node">
          <rect x="177" y="105" width="92" height="94" fill={colors.white} stroke={colors.navy} strokeWidth="2" />
          <text x="192" y="139" fontSize="17" fontWeight="800">Model</text>
          <text x="192" y="166" fontSize="13">predictor</text>
        </g>
        <path d="M269 152 H298" stroke={colors.navyMuted} strokeWidth="3" markerEnd="url(#pipeline-arrow)" />
        <g className="pipeline-node" data-claim-id={methods.id}>
          <rect x="306" y="76" width="154" height="152" fill={colors.tealLight} stroke={colors.teal} strokeWidth="3" />
          <text x="321" y="108" fontSize="16" fontWeight="800">Explanation engine</text>
          <text x="321" y="140" fontSize="14">Local</text>
          <text x="321" y="169" fontSize="14">Global</text>
          <text x="321" y="198" fontSize="14">Group-wise</text>
        </g>
        <path d="M460 152 H489" stroke={colors.navyMuted} strokeWidth="3" markerEnd="url(#pipeline-arrow)" />
        <g className="pipeline-node">
          <rect x="497" y="92" width="92" height="120" fill={colors.orangeLight} stroke={colors.orange} strokeWidth="2" />
          <text x="512" y="124" fontSize="17" fontWeight="800">Metrics</text>
          <text x="512" y="155" fontSize="13">evaluate</text>
          <text x="512" y="180" fontSize="13">score</text>
        </g>
        <path d="M589 152 H618" stroke={colors.navyMuted} strokeWidth="3" markerEnd="url(#pipeline-arrow)" />
        <g className="pipeline-node">
          <rect x="626" y="105" width="100" height="94" fill={colors.white} stroke={colors.navy} strokeWidth="2" />
          <text x="641" y="139" fontSize="17" fontWeight="800">Reports</text>
          <text x="641" y="166" fontSize="13">compare</text>
        </g>
        <path d="M676 199 V267 H84 V220" fill="none" stroke={colors.navyMuted} strokeWidth="2" strokeDasharray="6 5" markerEnd="url(#pipeline-arrow)" />
        <text x="385" y="291" textAnchor="middle" fill={colors.navyMuted} fontSize="13">repeatable evidence loop</text>
      </svg>
      <figcaption>{protocol.posterWording}</figcaption>
    </figure>
  )
}

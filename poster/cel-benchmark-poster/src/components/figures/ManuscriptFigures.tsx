import { ClaimWording } from '@/components/poster'
import { posterData } from '@/data/posterData'

const globalGraphic = new URL('../../../../plots/generated/manuscript-global.svg', import.meta.url).href
const groupGraphic = new URL('../../../../plots/generated/manuscript-group.svg', import.meta.url).href
const localGraphic = new URL('../../../../plots/generated/manuscript-local.svg', import.meta.url).href
const regressionGraphic = new URL('../../../../plots/generated/manuscript-regression.svg', import.meta.url).href

type ResultFigureProps = {
  alt: string
  claimId: string
  className?: string
  finding: 'local' | 'global' | 'group' | 'regression'
  image: string
  source: string
  sourceLabel: string
}

function ResultFigure({ alt, claimId, className = '', finding, image, source, sourceLabel }: ResultFigureProps) {
  return (
    <figure
      className={`manuscript-figure result-manuscript-figure ${className}`.trim()}
      data-claim-id={claimId}
      data-finding={finding}
      data-manuscript-source={source}
      data-typography-asset={finding}
      data-result-surface={finding}
      aria-label={sourceLabel}
      data-dataset={finding === 'regression' ? 'Concrete' : 'Adult Census'}
    >
      <div className="manuscript-image-window">
        <img src={image} alt={`${alt}. ${finding === 'regression' ? 'Concrete' : 'Adult Census'}; original plot marks with enlarged Arial labels.`} />
      </div>
    </figure>
  )
}

const arch = {
  ...posterData.visualSpec.colors,
  tintNavy: '#eef1f6',
  metricInk: '#a53a0e',
}

type ChipProps = {
  x: number
  y: number
  w: number
  h: number
  title: string
  sub?: string
  stroke?: string
  accent?: string
  titleSize?: number
  subSize?: number
}

function ArchChip({ x, y, w, h, title, sub, stroke = arch.navy, accent, titleSize = 18, subSize = 14 }: ChipProps) {
  const cx = x + w / 2
  const titleY = y + (sub ? h * 0.42 : h * 0.58) + 4
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={arch.white} stroke={stroke} strokeWidth={1.5} />
      {accent ? <rect x={x} y={y} width={6} height={h} rx={3} fill={accent} /> : null}
      <text x={cx} y={titleY} textAnchor="middle" fill={arch.navy} fontSize={titleSize} fontWeight={700}>{title}</text>
      {sub ? <text x={cx} y={y + h * 0.72 + 6} textAnchor="middle" fill={arch.navyMuted} fontSize={subSize}>{sub}</text> : null}
    </g>
  )
}

/**
 * Poster-native redraw of the manuscript architecture (Fig. 1): five stages of the
 * controlled protocol flowing left to right. Recreated as vector shapes rather than a
 * shrunk PDF export so the labels stay legible at A1.
 */
export function ArchitectureFigure() {
  return (
    <figure
      className="manuscript-figure architecture-figure"
      data-claim-id="scope.protocol"
      data-manuscript-source="manuscript/figures/teaser.pdf"
      data-typography-asset="architecture"
      data-manuscript-presentation="native"
    >
      <div className="architecture-image-window">
        <svg
          viewBox="0 0 1210 470"
          role="img"
          aria-labelledby="architecture-title architecture-desc"
        >
          <title id="architecture-title">CEL controlled benchmark architecture</title>
          <desc id="architecture-desc">
            One controlled protocol: a data module and predictive or probabilistic models feed local,
            global, and group-wise counterfactual methods, which a shared metrics suite scores into
            comparable reports.
          </desc>
          <defs>
            <marker id="arch-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
              <path d="M0,0 L9,4.5 L0,9 z" fill={arch.navyMuted} />
            </marker>
          </defs>

          {/* Protocol frame */}
          <rect x={3} y={6} width={1204} height={458} rx={16} fill="none" stroke={arch.teal} strokeWidth={3} />
          <text x={24} y={34} fill={arch.teal} fontSize={18} fontWeight={800} letterSpacing="0.04em">CONTROLLED PROTOCOL</text>
          <text x={1186} y={34} textAnchor="end" fill={arch.navyMuted} fontSize={14}>fixed splits · shared models · one metric suite</text>

          {/* Spine arrows */}
          <path d="M182 254 H222" stroke={arch.navyMuted} strokeWidth={3} markerEnd="url(#arch-arrow)" />
          <path d="M472 254 H512" stroke={arch.navyMuted} strokeWidth={3} markerEnd="url(#arch-arrow)" />
          <path d="M780 254 H820" stroke={arch.navyMuted} strokeWidth={3} markerEnd="url(#arch-arrow)" />
          <path d="M1030 254 H1070" stroke={arch.navyMuted} strokeWidth={3} markerEnd="url(#arch-arrow)" />

          {/* Stage captions */}
          <text x={94} y={64} textAnchor="middle" fill={arch.navy} fontSize={17} fontWeight={800} letterSpacing="0.04em">DATA MODULE</text>
          <text x={347} y={64} textAnchor="middle" fill={arch.navy} fontSize={17} fontWeight={800} letterSpacing="0.04em">MODEL MODULE</text>
          <text x={646} y={64} textAnchor="middle" fill={arch.teal} fontSize={17} fontWeight={800} letterSpacing="0.04em">EXPLANATION ENGINE</text>
          <text x={925} y={64} textAnchor="middle" fill={arch.metricInk} fontSize={16} fontWeight={800} letterSpacing="0.03em">METRICS ORCHESTRATOR</text>
          <text x={1137} y={64} textAnchor="middle" fill={arch.navy} fontSize={17} fontWeight={800} letterSpacing="0.04em">OUTPUT</text>

          {/* Data module */}
          <rect x={6} y={78} width={176} height={352} rx={12} fill={arch.tintNavy} stroke={arch.navyMuted} strokeWidth={1.5} />
          <ArchChip x={18} y={92} w={152} h={68} title="Datasets" sub="18 tabular sets" />
          <ArchChip x={18} y={174} w={152} h={68} title="Preprocessing" sub="scaling · encoding" />
          <ArchChip x={18} y={256} w={152} h={68} title="Actionability" sub="mutable features" />
          <ArchChip x={18} y={338} w={152} h={68} title="Feature bounds" sub="valid domains" />

          {/* Model module */}
          <rect x={222} y={78} width={250} height={352} rx={12} fill={arch.tintNavy} stroke={arch.navyMuted} strokeWidth={1.5} />
          <line x1={236} y1={254} x2={458} y2={254} stroke={arch.rule} strokeWidth={1.5} />
          <text x={236} y={106} fill={arch.navyMuted} fontSize={14} fontWeight={700}>Predictive</text>
          <ArchChip x={236} y={116} w={222} h={54} title="Classifiers" sub="LR · MLP" titleSize={16} subSize={13} />
          <ArchChip x={236} y={178} w={222} h={54} title="Regressors" sub="Linear · MLP" titleSize={16} subSize={13} />
          <text x={236} y={282} fill={arch.navyMuted} fontSize={14} fontWeight={700}>Probabilistic</text>
          <ArchChip x={236} y={292} w={222} h={54} title="Density" sub="KDE · GMM" titleSize={16} subSize={13} />
          <ArchChip x={236} y={354} w={222} h={54} title="Normalizing flows" sub="MAF · RealNVP · NICE" titleSize={16} subSize={13} />

          {/* Explanation engine — the hero stage */}
          <rect x={512} y={78} width={268} height={352} rx={12} fill={arch.tealLight} stroke={arch.teal} strokeWidth={2.5} />
          <g data-claim-id="scope.methods">
            <ArchChip x={528} y={94} w={236} h={96} title="Local" sub="CCHVAE · DiCE · PPCEF · …" stroke={arch.teal} accent={arch.teal} titleSize={22} subSize={15} />
            <ArchChip x={528} y={204} w={236} h={96} title="Global" sub="AReS · GLOBE-CE" stroke={arch.teal} accent={arch.teal} titleSize={22} subSize={15} />
            <ArchChip x={528} y={314} w={236} h={96} title="Group-wise" sub="GLANCE · T-CREx" stroke={arch.teal} accent={arch.teal} titleSize={22} subSize={15} />
          </g>

          {/* Metrics orchestrator */}
          <rect x={820} y={78} width={210} height={352} rx={12} fill={arch.orangeLight} stroke={arch.orange} strokeWidth={2} />
          <ArchChip x={832} y={92} w={186} h={68} title="Coverage & validity" stroke={arch.orange} titleSize={16} />
          <ArchChip x={832} y={174} w={186} h={68} title="Proximity" sub="L1 · L2 · MAD" stroke={arch.orange} />
          <ArchChip x={832} y={256} w={186} h={68} title="Sparsity" stroke={arch.orange} />
          <ArchChip x={832} y={338} w={186} h={68} title="Plausibility" sub="density · LOF · IsoForest" stroke={arch.orange} subSize={13} />

          {/* Output */}
          <rect x={1070} y={194} width={134} height={120} rx={12} fill={arch.navy} />
          <text x={1137} y={248} textAnchor="middle" fill={arch.white} fontSize={16} fontWeight={700}>Reports &amp;</text>
          <text x={1137} y={274} textAnchor="middle" fill={arch.white} fontSize={15} fontWeight={700}>visualisations</text>
        </svg>
      </div>
    </figure>
  )
}

export function LocalBenchmarkFigure() {
  return (
    <div className="local-figure-stack">
      <ResultFigure
        alt="Manuscript boxplots comparing local counterfactual methods across validity, distance, sparsity, density, and runtime"
        claimId="result.local.overview"
        finding="local"
        image={localGraphic}
        source="manuscript/figures/metrics_boxplot_local.png"
        sourceLabel="Adult Census · local methods"
      />
      <span hidden data-claim-id="caveat.sparsity-direction"><ClaimWording claimId="caveat.sparsity-direction" /></span>
    </div>
  )
}

export function GlobalBenchmarkFigure() {
  return (
    <ResultFigure
      alt="Manuscript boxplots comparing global counterfactual methods in two rows; shared method key: 1 AReS, 2 GLOBE-CE, 3 GlobalGLANCE"
      claimId="result.global.overview"
      className="result-manuscript-figure--strip"
      finding="global"
      image={globalGraphic}
      source="manuscript/figures/metrics_boxplot_global.png"
      sourceLabel="Adult Census · global methods"
    />
  )
}

export function GroupBenchmarkFigure() {
  return (
    <ResultFigure
      alt="Manuscript boxplots comparing GLANCE and T-CREx group-wise counterfactual methods"
      claimId="result.group.overview"
      className="result-manuscript-figure--strip"
      finding="group"
      image={groupGraphic}
      source="manuscript/figures/metrics_boxplot_group_wise.png"
      sourceLabel="Adult Census · group-wise methods"
    />
  )
}

export function RegressionBenchmarkFigure() {
  return (
    <ResultFigure
      alt="Manuscript Concrete regression boxplots comparing CEARM and Wachter across target MAE, L2 distance, sparsity, log-density, and runtime"
      claimId="result.regression.overview"
      className="result-manuscript-figure--regression"
      finding="regression"
      image={regressionGraphic}
      source="manuscript/figures/regression_metrics_boxplot.png"
      sourceLabel="Concrete regression methods"
    />
  )
}

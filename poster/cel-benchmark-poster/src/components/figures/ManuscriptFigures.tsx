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
  scopeInk: '#10384c',
  tintNavy: '#eef1f6',
}

type ChipProps = {
  x: number
  y: number
  w: number
  h: number
  title: string
  sub?: string
  stroke?: string
  titleSize?: number
  subSize?: number
}

function ArchChip({ x, y, w, h, title, sub, stroke = arch.scopeInk, titleSize = 18, subSize = 14 }: ChipProps) {
  const cx = x + w / 2
  const cy = y + h / 2
  const lineGap = 2
  const titleY = sub ? cy - (subSize + lineGap) / 2 : cy
  const subY = cy + (titleSize + lineGap) / 2
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={arch.white} stroke={stroke} strokeWidth={1.5} />
      <text x={cx} y={titleY} textAnchor="middle" dominantBaseline="middle" fill={arch.scopeInk} fontSize={titleSize} fontWeight={700}>{title}</text>
      {sub ? <text x={cx} y={subY} textAnchor="middle" dominantBaseline="middle" fill={arch.scopeInk} fontSize={subSize}>{sub}</text> : null}
    </g>
  )
}

/**
 * Poster-native redraw of the manuscript architecture (Fig. 1), keeping its hub
 * arrangement: data and model modules on top, the explanation engine in the middle,
 * metrics and reports below. Recreated as vector shapes rather than a shrunk PDF
 * export so the labels stay legible at A1.
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
          viewBox="0 0 620 464"
          role="img"
          aria-labelledby="architecture-title architecture-desc"
        >
          <title id="architecture-title">CEL benchmark architecture</title>
          <desc id="architecture-desc">
            A data module and predictive or probabilistic models feed local, global, and group-wise
            counterfactual methods, which a metrics suite scores into comparable reports.
          </desc>
          <defs>
            <marker id="arch-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">
              <path d="M0,0 L8,4 L0,8 z" fill={arch.scopeInk} />
            </marker>
          </defs>

          {/* Row 1: data and model modules */}
          <rect x={14} y={4} width={290} height={152} rx={10} fill={arch.tintNavy} stroke={arch.scopeInk} strokeWidth={1.5} />
          <text x={159} y={18} textAnchor="middle" dominantBaseline="middle" fill={arch.scopeInk} fontSize={12} fontWeight={800} letterSpacing="0.04em">DATA MODULE</text>
          <ArchChip x={26} y={30} w={129} h={50} title="Datasets" sub="18 tabular sets" titleSize={13} subSize={10} />
          <ArchChip x={163} y={30} w={129} h={50} title="Preprocessing" sub="scaling · encoding" titleSize={13} subSize={10} />
          <ArchChip x={26} y={90} w={129} h={50} title="Actionability" sub="mutable features" titleSize={13} subSize={10} />
          <ArchChip x={163} y={90} w={129} h={50} title="Feature bounds" sub="valid domains" titleSize={13} subSize={10} />

          <rect x={316} y={4} width={290} height={152} rx={10} fill={arch.tintNavy} stroke={arch.scopeInk} strokeWidth={1.5} />
          <text x={461} y={18} textAnchor="middle" dominantBaseline="middle" fill={arch.scopeInk} fontSize={12} fontWeight={800} letterSpacing="0.04em">MODEL MODULE</text>
          <text x={328} y={33} dominantBaseline="middle" fill={arch.scopeInk} fontSize={10} fontWeight={700}>Predictive</text>
          <ArchChip x={328} y={40} w={129} h={42} title="Classifiers" sub="LR · MLP" titleSize={12} subSize={9.5} />
          <ArchChip x={328} y={94} w={129} h={42} title="Regressors" sub="Linear · MLP" titleSize={12} subSize={9.5} />
          <text x={465} y={33} dominantBaseline="middle" fill={arch.scopeInk} fontSize={10} fontWeight={700}>Probabilistic</text>
          <ArchChip x={465} y={40} w={129} h={42} title="Density" sub="KDE · GMM" titleSize={12} subSize={9.5} />
          <ArchChip x={465} y={94} w={129} h={42} title="Normalizing flows" sub="MAF · RealNVP · NICE" titleSize={12} subSize={9.5} />

          {/* Flow arrows: data and models feed the engine, then metrics and reports. */}
          <path className="architecture-arrow" d="M159 156 V178" stroke={arch.scopeInk} strokeWidth={2.5} markerEnd="url(#arch-arrow)" />
          <path className="architecture-arrow" d="M461 156 V178" stroke={arch.scopeInk} strokeWidth={2.5} markerEnd="url(#arch-arrow)" />

          {/* Row 2: explanation engine */}
          <rect x={14} y={180} width={592} height={110} rx={10} fill={arch.tealLight} stroke={arch.scopeInk} strokeWidth={2} />
          <text x={310} y={196} textAnchor="middle" dominantBaseline="middle" fill={arch.scopeInk} fontSize={13} fontWeight={800} letterSpacing="0.04em">EXPLANATION ENGINE</text>
          <g data-claim-id="scope.methods">
            <ArchChip x={26} y={210} w={184} h={66} title="Local" sub="CCHVAE · DiCE · PPCEF · …" titleSize={16} subSize={11} />
            <ArchChip x={218} y={210} w={184} h={66} title="Global" sub="AReS · GLOBE-CE" titleSize={16} subSize={11} />
            <ArchChip x={410} y={210} w={184} h={66} title="Group-wise" sub="GLANCE · T-CREx" titleSize={16} subSize={11} />
          </g>

          <path className="architecture-arrow" d="M310 290 V312" stroke={arch.scopeInk} strokeWidth={2.5} markerEnd="url(#arch-arrow)" />

          {/* Row 3: metrics orchestrator */}
          <rect x={14} y={314} width={592} height={96} rx={10} fill={arch.orangeLight} stroke={arch.scopeInk} strokeWidth={2} />
          <text x={310} y={328} textAnchor="middle" dominantBaseline="middle" fill={arch.scopeInk} fontSize={12} fontWeight={800} letterSpacing="0.04em">METRICS ORCHESTRATOR</text>
          <ArchChip x={26} y={340} w={136} h={56} title="Validity" sub="coverage" titleSize={12} subSize={9.5} />
          <ArchChip x={170} y={340} w={136} h={56} title="Proximity" sub="L1 · L2 · MAD" titleSize={12} subSize={9.5} />
          <ArchChip x={314} y={340} w={136} h={56} title="Sparsity" sub="changed features" titleSize={12} subSize={9.5} />
          <ArchChip x={458} y={340} w={136} h={56} title="Plausibility" sub="density · LOF · IsoForest" titleSize={12} subSize={9.5} />

          <path className="architecture-arrow" d="M310 410 V426" stroke={arch.scopeInk} strokeWidth={2.5} markerEnd="url(#arch-arrow)" />

          {/* Row 4: output */}
          <rect x={210} y={428} width={200} height={32} rx={8} fill={arch.white} stroke={arch.scopeInk} strokeWidth={1.5} />
          <text x={310} y={444} textAnchor="middle" dominantBaseline="middle" fill={arch.scopeInk} fontSize={12} fontWeight={700}>Reports &amp; visualisations</text>
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

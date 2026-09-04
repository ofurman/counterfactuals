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
          viewBox="0 0 620 460"
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
            <marker id="arch-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill={arch.navyMuted} />
            </marker>
          </defs>

          {/* Protocol frame */}
          <rect x={3} y={3} width={614} height={454} rx={14} fill="none" stroke={arch.teal} strokeWidth={2.5} />
          <text x={18} y={27} fill={arch.teal} fontSize={14} fontWeight={800} letterSpacing="0.04em">CONTROLLED PROTOCOL</text>
          <text x={602} y={27} textAnchor="end" fill={arch.navyMuted} fontSize={10.5}>fixed splits · shared models · one metric suite</text>

          {/* Flow arrows: data + models feed the engine; engine feeds metrics; metrics feed reports */}
          <path d="M159 172 V194" stroke={arch.navyMuted} strokeWidth={2.5} markerEnd="url(#arch-arrow)" />
          <path d="M461 172 V194" stroke={arch.navyMuted} strokeWidth={2.5} markerEnd="url(#arch-arrow)" />
          <path d="M310 298 V320" stroke={arch.navyMuted} strokeWidth={2.5} markerEnd="url(#arch-arrow)" />
          <path d="M310 408 V420" stroke={arch.navyMuted} strokeWidth={2.5} markerEnd="url(#arch-arrow)" />
          {/* Probabilistic backbones also score plausibility */}
          <path d="M606 106 H611 V366 H609" fill="none" stroke={arch.navyMuted} strokeWidth={1.5} strokeDasharray="4 3" markerEnd="url(#arch-arrow)" />

          {/* Row 1: data + model modules */}
          <rect x={14} y={40} width={290} height={132} rx={10} fill={arch.tintNavy} stroke={arch.navyMuted} strokeWidth={1.5} />
          <text x={159} y={58} textAnchor="middle" fill={arch.navy} fontSize={12} fontWeight={800} letterSpacing="0.04em">DATA MODULE</text>
          <ArchChip x={26} y={66} w={130} h={40} title="Datasets" sub="18 tabular sets" titleSize={13} subSize={10} />
          <ArchChip x={164} y={66} w={130} h={40} title="Preprocessing" sub="scaling · encoding" titleSize={13} subSize={10} />
          <ArchChip x={26} y={114} w={130} h={40} title="Actionability" sub="mutable features" titleSize={13} subSize={10} />
          <ArchChip x={164} y={114} w={130} h={40} title="Feature bounds" sub="valid domains" titleSize={13} subSize={10} />

          <rect x={316} y={40} width={290} height={132} rx={10} fill={arch.tintNavy} stroke={arch.navyMuted} strokeWidth={1.5} />
          <text x={461} y={58} textAnchor="middle" fill={arch.navy} fontSize={12} fontWeight={800} letterSpacing="0.04em">MODEL MODULE</text>
          <text x={328} y={75} fill={arch.navyMuted} fontSize={10} fontWeight={700}>Predictive</text>
          <ArchChip x={328} y={80} w={130} h={34} title="Classifiers" sub="LR · MLP" titleSize={12} subSize={9.5} />
          <ArchChip x={328} y={120} w={130} h={34} title="Regressors" sub="Linear · MLP" titleSize={12} subSize={9.5} />
          <text x={466} y={75} fill={arch.navyMuted} fontSize={10} fontWeight={700}>Probabilistic</text>
          <ArchChip x={466} y={80} w={130} h={34} title="Density" sub="KDE · GMM" titleSize={12} subSize={9.5} />
          <ArchChip x={466} y={120} w={130} h={34} title="Normalizing flows" sub="MAF · RealNVP · NICE" titleSize={12} subSize={9.5} />

          {/* Row 2: explanation engine — the hero stage */}
          <rect x={14} y={198} width={592} height={100} rx={10} fill={arch.tealLight} stroke={arch.teal} strokeWidth={2.5} />
          <text x={310} y={216} textAnchor="middle" fill={arch.teal} fontSize={13} fontWeight={800} letterSpacing="0.04em">EXPLANATION ENGINE</text>
          <g data-claim-id="scope.methods">
            <ArchChip x={26} y={230} w={184} h={54} title="Local" sub="CCHVAE · DiCE · PPCEF · …" stroke={arch.teal} accent={arch.teal} titleSize={16} subSize={11} />
            <ArchChip x={218} y={230} w={184} h={54} title="Global" sub="AReS · GLOBE-CE" stroke={arch.teal} accent={arch.teal} titleSize={16} subSize={11} />
            <ArchChip x={410} y={230} w={184} h={54} title="Group-wise" sub="GLANCE · T-CREx" stroke={arch.teal} accent={arch.teal} titleSize={16} subSize={11} />
          </g>

          {/* Row 3: metrics orchestrator */}
          <rect x={14} y={324} width={592} height={84} rx={10} fill={arch.orangeLight} stroke={arch.orange} strokeWidth={2} />
          <text x={310} y={342} textAnchor="middle" fill={arch.metricInk} fontSize={12} fontWeight={800} letterSpacing="0.04em">METRICS ORCHESTRATOR</text>
          <ArchChip x={26} y={352} w={138} h={44} title="Validity" sub="coverage" stroke={arch.orange} titleSize={12} subSize={9.5} />
          <ArchChip x={172} y={352} w={138} h={44} title="Proximity" sub="L1 · L2 · MAD" stroke={arch.orange} titleSize={12} subSize={9.5} />
          <ArchChip x={318} y={352} w={138} h={44} title="Sparsity" sub="changed features" stroke={arch.orange} titleSize={12} subSize={9.5} />
          <ArchChip x={464} y={352} w={138} h={44} title="Plausibility" sub="density · LOF · IsoForest" stroke={arch.orange} titleSize={12} subSize={9.5} />

          {/* Row 4: output */}
          <rect x={210} y={424} width={200} height={28} rx={8} fill={arch.navy} />
          <text x={310} y={443} textAnchor="middle" fill={arch.white} fontSize={12} fontWeight={700}>Reports &amp; visualisations</text>
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

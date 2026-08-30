import { ClaimWording } from '@/components/poster'

const architectureGraphic = new URL('../../../../plots/generated/manuscript-architecture.svg', import.meta.url).href
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

export function ArchitectureFigure() {
  return (
    <figure
      className="manuscript-figure architecture-figure"
      data-claim-id="scope.protocol"
      data-manuscript-source="manuscript/figures/teaser.pdf"
      data-typography-asset="architecture"
    >
      <div className="architecture-image-window">
        <img
          src={architectureGraphic}
          alt="CEL manuscript architecture: data and model modules feed local, global, and group-wise counterfactual methods, then shared metrics and reports"
          data-claim-id="scope.methods"
        />
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
      alt="Manuscript boxplots comparing global counterfactual methods"
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

import { ClaimQualifier } from '@/components/poster'

const architectureGraphic = new URL('../../assets/manuscript/architecture.png', import.meta.url).href
const globalGraphic = new URL('../../assets/manuscript/global-adult.jpg', import.meta.url).href
const groupGraphic = new URL('../../assets/manuscript/group-adult.jpg', import.meta.url).href
const localGraphic = new URL('../../assets/manuscript/local-mixed.jpg', import.meta.url).href
const regressionGraphic = new URL('../../assets/manuscript/regression.jpg', import.meta.url).href

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
      data-result-surface={finding}
    >
      <div className="manuscript-image-window">
        <img src={image} alt={alt} />
      </div>
      <figcaption>
        <span>{sourceLabel}</span>
      </figcaption>
    </figure>
  )
}

export function ArchitectureFigure() {
  return (
    <figure
      className="manuscript-figure architecture-figure"
      data-claim-id="scope.protocol"
      data-manuscript-source="manuscript/figures/teaser.pdf"
    >
      <img
        src={architectureGraphic}
        alt="CEL manuscript architecture: data and model modules feed local, global, and group-wise counterfactual methods, then shared metrics and reports"
        data-claim-id="scope.methods"
      />
      <figcaption>
        <span>Manuscript architecture</span>
        Data, models, CE methods, and metrics share one evaluation system.
      </figcaption>
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
        sourceLabel="Manuscript local-method figure - Adult Census row"
      />
      <p className="figure-caveat" data-claim-id="caveat.sparsity-direction"><ClaimQualifier claimId="caveat.sparsity-direction" /></p>
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
      sourceLabel="Manuscript global-method figure - Adult Census row"
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
      sourceLabel="Manuscript group-wise figure - Adult Census row"
    />
  )
}

export function RegressionBenchmarkFigure() {
  return (
    <ResultFigure
      alt="Manuscript boxplots comparing CEARM and Wachter on regression counterfactual tasks"
      claimId="result.regression.overview"
      className="result-manuscript-figure--regression"
      finding="regression"
      image={regressionGraphic}
      source="manuscript/figures/regression_metrics_boxplot.png"
      sourceLabel="Manuscript regression-method figure"
    />
  )
}

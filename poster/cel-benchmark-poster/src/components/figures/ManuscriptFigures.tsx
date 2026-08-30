import { ClaimWording } from '@/components/poster'
import type { CSSProperties } from 'react'

const architectureGraphic = new URL('../../assets/manuscript/architecture.png', import.meta.url).href
const globalGraphic = new URL('../../assets/manuscript/global-adult.jpg', import.meta.url).href
const groupGraphic = new URL('../../assets/manuscript/group-adult.jpg', import.meta.url).href
const localGraphic = new URL('../../assets/manuscript/local-mixed.jpg', import.meta.url).href

// The local manuscript uses much wider axes than the global/group-wise figures.
// Reflow its complete Adult Census metric panels without stretching the raster.
const localMetricCrops = [
  { metric: 'Validity', x: 0, width: 290 },
  { metric: 'L2-Hamming', x: 290, width: 282 },
  { metric: 'Sparsity', x: 572, width: 279 },
  { metric: 'Log-density', x: 851, width: 282 },
  { metric: 'Runtime', x: 1133, width: 267 },
] as const

type ResultFigureProps = {
  alt: string
  claimId: string
  className?: string
  finding: 'local' | 'global' | 'group'
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
      aria-label={sourceLabel}
      data-dataset="Adult Census"
    >
      {finding === 'local' ? (
        <div className="local-metric-grid">
          {localMetricCrops.map(({ metric, x, width }) => (
            <div
              className="local-metric-window"
              key={metric}
              data-metric={metric}
              data-crop={`${x} 0 ${width} 168`}
              style={{ '--crop-x': x, '--crop-width': width } as CSSProperties}
            >
              <img src={image} alt={`${alt}: Adult Census, ${metric}`} />
            </div>
          ))}
        </div>
      ) : (
        <div className="manuscript-image-window">
          <img src={image} alt={alt} />
        </div>
      )}
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
      <div className="architecture-image-window" data-crop="90 28 1220 622">
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

import type { CSSProperties } from 'react'
import { ClaimQualifier, ClaimValue, ClaimVerdict } from '@/components/poster'
import { claimMean, resultDescriptor } from '@/data/posterData'

type BarStyle = CSSProperties & { '--bar-size': string }
const barStyle = (value: number | null, max: number): BarStyle => ({
  '--bar-size': `${Math.max(2, Math.min(100, ((value ?? 0) / max) * 100))}%`,
})

function MetricBar({ claimId, max, direction }: { claimId: string; max: number; direction: string }) {
  const descriptor = resultDescriptor(claimId)
  return (
    <div className="metric-bar" data-claim-id={claimId}>
      <div className="metric-bar__label">
        <span>{descriptor.label}</span>
        <small>{direction}</small>
      </div>
      <div className="metric-bar__track" aria-hidden="true">
        <span style={barStyle(claimMean(claimId), max)} />
      </div>
      <ClaimValue claimId={claimId} className="metric-bar__value" />
    </div>
  )
}

export function LocalTradeoffFigure() {
  return (
    <figure className="evidence-figure" data-finding="local" data-result-surface="local">
      <figcaption>Blobs · local · MLP</figcaption>
      <MetricBar claimId="local.blobs.ppcef.pp" max={1} direction="higher plausibility ↑" />
      <MetricBar claimId="local.blobs.ppcef.l2" max={0.6} direction="lower change ↓" />
      <p className="figure-note"><ClaimVerdict claimId="local.blobs.ppcef.l2" /></p>
    </figure>
  )
}

export function GroupTradeoffFigure() {
  const rows = [
    { method: resultDescriptor('group.adult.tcrex.validity').label.replace(' validity', ''), validity: 'group.adult.tcrex.validity', distance: 'group.adult.tcrex.distance' },
    { method: resultDescriptor('group.adult.glance.validity').label.replace(' validity', ''), validity: 'group.adult.glance.validity', distance: 'group.adult.glance.distance' },
  ]
  return (
    <figure className="evidence-figure" data-finding="group" data-result-surface="group">
      <figcaption>Adult Census · group-wise · MLP</figcaption>
      <div className="group-result group-result--head"><span>method</span><span>validity ↑</span><span>change ↓</span></div>
      {rows.map((row) => (
        <div className="group-result" key={row.method}>
          <strong>{row.method}</strong>
          <div className="group-metric" data-claim-id={row.validity}>
            <span className="mini-bar" style={barStyle(claimMean(row.validity), 1)} aria-hidden="true" />
            <ClaimValue claimId={row.validity} />
          </div>
          <div className="group-metric group-metric--change" data-claim-id={row.distance}>
            <span className="mini-bar" style={barStyle(claimMean(row.distance), 0.5)} aria-hidden="true" />
            <ClaimValue claimId={row.distance} />
          </div>
        </div>
      ))}
      <p className="figure-note rounded-note"><ClaimQualifier claimId="group.adult.tcrex.distance" /></p>
    </figure>
  )
}

export function RegressionTradeoffFigure() {
  const rows = [
    { label: 'Target MAE ↓', left: 'regression.concrete.cearm.mae', right: 'regression.concrete.wachter.mae', max: 0.05 },
    { label: 'L2 change ↓', left: 'regression.concrete.cearm.l2', right: 'regression.concrete.wachter.l2', max: 1.2 },
  ]
  return (
    <figure className="evidence-figure regression-figure" data-finding="regression" data-result-surface="regression">
      <figcaption>Concrete · regression · MLP</figcaption>
      <div className="regression-head"><span>metric</span><strong>CEARM</strong><strong>Wachter</strong></div>
      {rows.map((row) => (
        <div className="regression-row" key={row.label}>
          <strong data-claim-id={row.left}>{row.label}</strong>
          <div data-claim-id={row.left} className="mini-result">
            <span className="mini-bar" style={barStyle(claimMean(row.left), row.max)} aria-hidden="true" />
            <ClaimValue claimId={row.left} />
          </div>
          <div data-claim-id={row.right} className="mini-result">
            <span className="mini-bar" style={barStyle(claimMean(row.right), row.max)} aria-hidden="true" />
            <ClaimValue claimId={row.right} />
          </div>
        </div>
      ))}
      <p className="plausibility-note"><strong>Plausibility:</strong> <ClaimValue claimId="regression.concrete.wachter.pp" /> <ClaimVerdict claimId="regression.concrete.wachter.pp" /></p>
    </figure>
  )
}

export function ApplicabilityFigure() {
  return (
    <figure className="applicability-figure" data-finding="applicability" data-result-surface="applicability">
      <figcaption>Blobs · global · MLP</figcaption>
      <div className="missing-mark">
        <ClaimValue claimId="global.blobs.ares.missing" />
        <strong>AReS output</strong>
      </div>
      <p><ClaimQualifier claimId="global.blobs.ares.missing" /></p>
    </figure>
  )
}

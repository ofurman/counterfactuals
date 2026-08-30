import {
  counterfactualExample as example,
  examplePlotBounds,
  examplePlotPoint,
  examplePrediction,
  exampleTransitions,
  type ExampleParadigm,
} from '@/data/counterfactualExample'

export function RecoursePlot({ paradigm }: { paradigm: ExampleParadigm }) {
  const transitions = exampleTransitions(paradigm)
  const { left, right, top, bottom } = examplePlotBounds
  const boundaryBottom = examplePlotPoint({ monthlyIncome: example.model.minimumIncomeAfterDebt, monthlyDebt: 0 })
  const boundaryTop = examplePlotPoint({ monthlyIncome: example.model.minimumIncomeAfterDebt + example.plot.maximumDebt, monthlyDebt: example.plot.maximumDebt })
  return (
    <svg className="recourse-plot" viewBox="0 0 385 160" role="img" aria-labelledby={`${paradigm}-plot-title`} data-example-plot={paradigm}>
      <title id={`${paradigm}-plot-title`}>{paradigm}: {transitions.map(({ id, original, counterfactual }) => `Applicant ${id}: declined at income ${original.monthlyIncome} and monthly debt payments ${original.monthlyDebt}; approved at income ${counterfactual.monthlyIncome} and monthly debt payments ${counterfactual.monthlyDebt}`).join('. ')}. Employment stays full-time.</title>
      <defs>
        {['shared', 'higher-debt', 'lower-income', 'individual'].map((group) => (
          <marker key={group} id={`${paradigm}-arrow-${group}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse" markerUnits="userSpaceOnUse">
            <path className={`recourse-arrowhead recourse-group--${group}`} d="M 0 0 L 8 4 L 0 8 Z" />
          </marker>
        ))}
      </defs>
      <path className="recourse-region" d={`M ${boundaryBottom.x} ${bottom} L ${boundaryTop.x} ${top} H ${right} V ${bottom} Z`} />
      <path className="recourse-boundary" d={`M ${boundaryBottom.x} ${bottom} L ${boundaryTop.x} ${top}`} />
      <path className="recourse-axis" d={`M ${left} ${top} V ${bottom} H ${right}`} />
      {[2000, 3000, 4000].map((income) => {
        const { x } = examplePlotPoint({ monthlyIncome: income, monthlyDebt: 0 })
        return <g key={income}><path className="recourse-axis" d={`M ${x} ${bottom} v 4`} /><text x={x} y={138} textAnchor="middle">{income.toLocaleString('en-US')}</text></g>
      })}
      {[0, 800, 1600].map((debt) => {
        const { y } = examplePlotPoint({ monthlyIncome: 2000, monthlyDebt: debt })
        return <g key={debt}><path className="recourse-axis" d={`M ${left - 4} ${y} h 4`} /><text x={left - 7} y={y + 4} textAnchor="end">{debt.toLocaleString('en-US')}</text></g>
      })}
      <text className="recourse-axis-label" x={240} y={156} textAnchor="middle">Monthly income (€)</text>
      <text className="recourse-axis-label" transform="translate(12 75) rotate(-90)" textAnchor="middle">Debt payments (€)</text>
      <text className="recourse-region-label recourse-region-label--declined" x={64} y={31}>Declined</text>
      <text className="recourse-region-label recourse-region-label--approved" x={right - 6} y={bottom - 10} textAnchor="end">Approved</text>
      {transitions.map(({ id, group, original, counterfactual }) => {
        const start = examplePlotPoint(original)
        const end = examplePlotPoint(counterfactual)
        const length = Math.hypot(end.x - start.x, end.y - start.y)
        // Stop the arrow at the endpoint's rim so its arrowhead remains visible.
        const tip = { x: end.x - (end.x - start.x) / length * 5, y: end.y - (end.y - start.y) / length * 5 }
        return (
          <g key={id} data-example-transition={id} data-group={group} data-from={examplePrediction(original)} data-to={examplePrediction(counterfactual)} data-original={`${original.monthlyIncome},${original.monthlyDebt}`} data-counterfactual={`${counterfactual.monthlyIncome},${counterfactual.monthlyDebt}`}>
            <line className={`recourse-arrow recourse-group--${group}`} x1={start.x} y1={start.y} x2={tip.x} y2={tip.y} markerEnd={`url(#${paradigm}-arrow-${group})`} />
            <circle className="recourse-point recourse-point--original" cx={start.x} cy={start.y} r={4} />
            <circle className="recourse-point recourse-point--counterfactual" cx={end.x} cy={end.y} r={4} />
            <text className="recourse-applicant" x={start.x - 5} y={start.y - 8} textAnchor="end">{id}</text>
          </g>
        )
      })}
    </svg>
  )
}

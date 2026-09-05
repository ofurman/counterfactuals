import { exampleTransitions, type ExampleParadigm } from '@/data/counterfactualExample'

const images = {
  local: new URL('../../../../plots/generated/ce-example-local.svg', import.meta.url).href,
  global: new URL('../../../../plots/generated/ce-example-global.svg', import.meta.url).href,
  'group-wise': new URL('../../../../plots/generated/ce-example-group-wise.svg', import.meta.url).href,
}

export function RecoursePlot({ paradigm }: { paradigm: ExampleParadigm }) {
  const description = exampleTransitions(paradigm).map(({ id, original, counterfactual }) =>
    `${id}: Declined at income €${original.monthlyIncome} and debt payments €${original.monthlyDebt}; Approved at income €${counterfactual.monthlyIncome} and debt payments €${counterfactual.monthlyDebt}`,
  ).join('. ')
  return (
    <img
      className="recourse-plot"
      src={images[paradigm]}
      data-example-plot={paradigm}
      data-example-asset={`poster/plots/generated/ce-example-${paradigm}.svg`}
      data-example-renderer="matplotlib"
      width="640"
      height={paradigm === 'global' ? '298' : '265'}
      alt={`${paradigm} loan example. ${description}. All amounts are monthly; employment stays full-time.${paradigm === 'global' ? ' Legend: blue original points; orange counterfactual points; decision boundary line.' : ''}`}
    />
  )
}

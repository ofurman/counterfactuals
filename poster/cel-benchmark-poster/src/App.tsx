import {
  PosterCanvas,
  PosterFooter,
  PosterHeader,
  PosterStage,
  PrintToolbar,
  ScopeStrip,
  SectionBlock,
} from '@/components/poster'
import {
  ApplicabilityFigure,
  BenchmarkPipeline,
  ConfoundDiagram,
  GroupTradeoffFigure,
  LocalTradeoffFigure,
  RegressionTradeoffFigure,
  SelectionGuide,
} from '@/components/figures'
import { posterData, resolveSection } from '@/data/posterData'

export default function App() {
  return (
    <>
      <PrintToolbar />
      <PosterStage>
        <PosterCanvas>
          <PosterHeader section={resolveSection('header')} />
          <ScopeStrip section={resolveSection('scope')} />
          <div className="poster-grid">
            <SectionBlock section={resolveSection('problem')} className="problem-block" showClaims={false}>
              <ConfoundDiagram />
            </SectionBlock>
            <SectionBlock section={resolveSection('protocol')} className="protocol-block" showClaims={false}>
              <BenchmarkPipeline />
              <p className="argument-band">{posterData.argument}</p>
            </SectionBlock>
            <SectionBlock section={resolveSection('local-tradeoff')} showClaims={false}>
              <LocalTradeoffFigure />
            </SectionBlock>
          </div>
          <div className="poster-bottom-grid">
            <SectionBlock section={resolveSection('group-tradeoff')} showClaims={false}>
              <GroupTradeoffFigure />
            </SectionBlock>
            <SectionBlock section={resolveSection('regression-tradeoff')} showClaims={false}>
              <RegressionTradeoffFigure />
            </SectionBlock>
            <SectionBlock section={resolveSection('applicability')} showClaims={false}>
              <ApplicabilityFigure />
            </SectionBlock>
            <SectionBlock section={resolveSection('guidance-limitations')} showClaims={false}>
              <SelectionGuide />
            </SectionBlock>
          </div>
          <PosterFooter section={resolveSection('reproducibility')} />
        </PosterCanvas>
      </PosterStage>
    </>
  )
}

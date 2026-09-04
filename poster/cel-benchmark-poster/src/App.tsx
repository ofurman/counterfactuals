import {
  PosterCanvas,
  PosterHeader,
  PosterStage,
  PrintToolbar,
  ScopeStrip,
  SectionBlock,
} from '@/components/poster'
import {
  ArchitectureFigure,
  CounterfactualExample,
  ContributionStack,
  GlobalBenchmarkFigure,
  GroupBenchmarkFigure,
  LocalBenchmarkFigure,
  RegressionBenchmarkFigure,
} from '@/components/figures'
import { resolveSection } from '@/data/posterData'

export default function App() {
  return (
    <>
      <PrintToolbar />
      <PosterStage>
        <PosterCanvas>
          <PosterHeader section={resolveSection('header')} />
          <div className="poster-grid benchmark-grid">
            <div className="poster-column poster-column--top">
              <ScopeStrip section={resolveSection('scope')}>
                <SectionBlock section={resolveSection('protocol')} className="protocol-block" showClaims={false}>
                  <ArchitectureFigure />
                </SectionBlock>
              </ScopeStrip>
            </div>
            <div className="poster-column poster-column--examples">
              <SectionBlock section={resolveSection('problem')} className="problem-block" showClaims={false}>
                <CounterfactualExample />
              </SectionBlock>
            </div>
            <div className="poster-column poster-column--right">
              <SectionBlock section={resolveSection('results')} className="unified-results-block" showClaims={false}>
                <div className="result-panels">
                  <SectionBlock as="article" section={resolveSection('applicability')} className="result-panel" showClaims={false}>
                    <GlobalBenchmarkFigure />
                  </SectionBlock>
                  <SectionBlock as="article" section={resolveSection('local-tradeoff')} className="result-panel result-panel--local" showClaims={false}>
                    <LocalBenchmarkFigure />
                  </SectionBlock>
                  <SectionBlock as="article" section={resolveSection('group-tradeoff')} className="result-panel" showClaims={false}>
                    <GroupBenchmarkFigure />
                  </SectionBlock>
                  <SectionBlock as="article" section={resolveSection('regression-tradeoff')} className="result-panel" showClaims={false}>
                    <RegressionBenchmarkFigure />
                  </SectionBlock>
                </div>
              </SectionBlock>
            </div>
            <div className="poster-column poster-column--bottom">
              <SectionBlock section={resolveSection('guidance-limitations')} className="contributions-block" showClaims={false}>
                <ContributionStack />
              </SectionBlock>
            </div>
          </div>
        </PosterCanvas>
      </PosterStage>
    </>
  )
}

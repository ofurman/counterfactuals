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
  ArchitectureFigure,
  BenchmarkMotivation,
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
          <ScopeStrip section={resolveSection('scope')} />
          <div className="poster-grid benchmark-grid">
            <div className="poster-column poster-column--left">
              <SectionBlock section={resolveSection('problem')} className="problem-block" showClaims={false}>
                <BenchmarkMotivation />
              </SectionBlock>
              <SectionBlock section={resolveSection('guidance-limitations')} className="contributions-block" showClaims={false}>
                <ContributionStack />
              </SectionBlock>
            </div>
            <div className="poster-column poster-column--center">
              <SectionBlock section={resolveSection('protocol')} className="protocol-block" showClaims={false}>
                <ArchitectureFigure />
              </SectionBlock>
              <SectionBlock section={resolveSection('local-tradeoff')} className="local-results-block" showClaims={false}>
                <LocalBenchmarkFigure />
              </SectionBlock>
            </div>
            <div className="poster-column poster-column--right">
              <SectionBlock section={resolveSection('applicability')} className="compact-results-block" showClaims={false}>
                <GlobalBenchmarkFigure />
              </SectionBlock>
              <SectionBlock section={resolveSection('group-tradeoff')} className="compact-results-block" showClaims={false}>
                <GroupBenchmarkFigure />
              </SectionBlock>
              <SectionBlock section={resolveSection('regression-tradeoff')} className="compact-results-block regression-results-block" showClaims={false}>
                <RegressionBenchmarkFigure />
              </SectionBlock>
            </div>
          </div>
          <PosterFooter section={resolveSection('reproducibility')} />
        </PosterCanvas>
      </PosterStage>
    </>
  )
}

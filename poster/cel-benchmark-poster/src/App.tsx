import {
  PosterCanvas,
  PosterFooter,
  PosterHeader,
  PosterStage,
  PrintToolbar,
  ScopeStrip,
  SectionBlock,
} from '@/components/poster'
import { posterData, resolveSection } from '@/data/posterData'

function ProtocolRail() {
  return (
    <figure className="protocol-figure">
      <figcaption>{posterData.protocol.caption}</figcaption>
      <ol className="protocol-rail" aria-label={posterData.protocol.caption}>
        {posterData.protocol.controls.map((control, index) => (
          <li key={control}>
            <span>{String(index + 1).padStart(2, '0')}</span>
            {control}
          </li>
        ))}
      </ol>
    </figure>
  )
}

export default function App() {
  return (
    <>
      <PrintToolbar />
      <PosterStage>
        <PosterCanvas>
          <PosterHeader section={resolveSection('header')} />
          <ScopeStrip section={resolveSection('scope')} />
          <div className="poster-grid">
            <SectionBlock section={resolveSection('problem')} className="problem-block" />
            <SectionBlock section={resolveSection('protocol')} className="protocol-block" showClaims={false}>
              <ProtocolRail />
              <p className="argument-band">{posterData.argument}</p>
            </SectionBlock>
            <div className="tradeoff-stack">
              <SectionBlock section={resolveSection('local-tradeoff')} />
              <SectionBlock section={resolveSection('group-tradeoff')} />
            </div>
          </div>
          <div className="poster-bottom-grid">
            <SectionBlock section={resolveSection('regression-tradeoff')} />
            <SectionBlock section={resolveSection('applicability')} />
            <SectionBlock section={resolveSection('guidance-limitations')} />
          </div>
          <PosterFooter section={resolveSection('reproducibility')} />
        </PosterCanvas>
      </PosterStage>
    </>
  )
}

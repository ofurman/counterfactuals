import { posterData, type ResolvedSection } from '@/data/posterData'
import { QRCodeSVG } from 'qrcode.react'

export function PosterHeader({ section }: { section: ResolvedSection }) {
  const { identity } = posterData

  return (
    <header className="poster-header" data-section={section.id}>
      <div className="header-copy">
        <p className="eyebrow">{identity.venue}</p>
        <h1>{section.heading}</h1>
        <p className="poster-thesis">{section.copy[0]}</p>
        <p className="authors">{identity.authors.map((author) => author.name).join(' · ')}</p>
        <p className="affiliation">{identity.affiliation}</p>
      </div>
      <a className="project-mark" href={identity.links.repository} data-qr-destination={identity.qr.url} aria-label="Open the CEL project repository">
        <span className="project-mark__qr" aria-hidden="true">
          <QRCodeSVG
            value={identity.qr.url}
            size={138}
            level="M"
            marginSize={2}
            bgColor={posterData.visualSpec.colors.white}
            fgColor={posterData.visualSpec.colors.navy}
          />
        </span>
        <span>{identity.qr.label}</span>
      </a>
    </header>
  )
}

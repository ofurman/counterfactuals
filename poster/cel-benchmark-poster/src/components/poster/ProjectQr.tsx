import { QRCodeSVG } from 'qrcode.react'
import { posterData } from '@/data/posterData'

const githubMark = new URL('../../assets/qr/GitHub_Invertocat_Black_Clearspace.svg', import.meta.url).href

export function ProjectQr() {
  const { identity, visualSpec } = posterData
  return (
    <a className="project-mark" href={identity.links.repository} data-qr-destination={identity.qr.url} aria-label="Open the CEL project repository">
      <span className="project-mark__qr" aria-hidden="true">
        <QRCodeSVG
          value={identity.qr.url}
          size={96}
          level="H"
          marginSize={4}
          bgColor={visualSpec.colors.white}
          fgColor={visualSpec.colors.navy}
          imageSettings={{ src: githubMark, width: 22, height: 22, excavate: true }}
          data-qr-logo="github"
          data-qr-error-level="H"
          data-qr-margin="4"
        />
      </span>
    </a>
  )
}

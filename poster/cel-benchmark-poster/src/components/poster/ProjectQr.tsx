import { QRCodeSVG } from 'qrcode.react'
import { posterData } from '@/data/posterData'

export function ProjectQr() {
  const { identity, visualSpec } = posterData
  return (
    <a className="project-mark" href={identity.links.repository} data-qr-destination={identity.qr.url} aria-label="Open the CEL project repository">
      <span className="project-mark__qr" aria-hidden="true">
        <QRCodeSVG
          value={identity.qr.url}
          size={96}
          level="M"
          marginSize={4}
          bgColor={visualSpec.colors.white}
          fgColor={visualSpec.colors.navy}
        />
      </span>
      <span>{identity.qr.label}</span>
    </a>
  )
}

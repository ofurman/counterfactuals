import { Printer } from 'lucide-react'
import { posterData } from '@/data/posterData'

export function PrintToolbar() {
  const { widthPx, heightPx } = posterData.visualSpec.canvas
  return (
    <div className="print-toolbar screen-only" role="toolbar" aria-label="Poster controls">
      <span>{posterData.visualSpec.page.format} {posterData.visualSpec.page.orientation} · {posterData.visualSpec.page.widthMm} × {posterData.visualSpec.page.heightMm} mm · {widthPx} × {heightPx} preview</span>
      <button type="button" onClick={() => window.print()}>
        <Printer aria-hidden="true" size={18} />
        Print poster
      </button>
    </div>
  )
}

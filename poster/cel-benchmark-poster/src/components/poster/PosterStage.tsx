import { useEffect, useRef, useState, type ReactNode } from 'react'
import { posterData } from '@/data/posterData'

const { widthPx: CANVAS_WIDTH, heightPx: CANVAS_HEIGHT } = posterData.visualSpec.canvas

export function PosterStage({ children }: { children: ReactNode }) {
  const stageRef = useRef<HTMLDivElement>(null)
  const [scale, setScale] = useState(1)

  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return

    const updateScale = () => setScale(Math.min(1, stage.clientWidth / CANVAS_WIDTH))
    updateScale()
    const observer = new ResizeObserver(updateScale)
    observer.observe(stage)
    return () => observer.disconnect()
  }, [])

  return (
    <div className="poster-stage" ref={stageRef}>
      <div
        className="poster-frame"
        style={{ width: CANVAS_WIDTH * scale, height: CANVAS_HEIGHT * scale }}
      >
        <div className="poster-transform" style={{ transform: `scale(${scale})` }}>
          {children}
        </div>
      </div>
    </div>
  )
}

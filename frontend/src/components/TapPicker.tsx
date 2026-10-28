import { useEffect, useRef, useState } from 'react'

export interface Tap {
  time: number
  x: number // video pixels
  y: number
}

// Shows the chosen clip paused so the user can tap the moving object. SAM 3.1 tracks from that point,
// which works for small or plain objects that a text description misses.
export function TapPicker({ file, tap, onTap }: { file: File; tap: Tap | null; onTap: (t: Tap | null) => void }) {
  const video = useRef<HTMLVideoElement>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [duration, setDuration] = useState(0)
  const [time, setTime] = useState(0)

  useEffect(() => {
    const u = URL.createObjectURL(file)
    setUrl(u)
    onTap(null)
    return () => URL.revokeObjectURL(u)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file])

  function seek(t: number) {
    setTime(t)
    if (video.current) video.current.currentTime = t
  }

  function pick(e: React.MouseEvent<HTMLDivElement>) {
    const v = video.current
    if (!v || !v.videoWidth) return
    const r = v.getBoundingClientRect()
    onTap({ time, x: ((e.clientX - r.left) / r.width) * v.videoWidth, y: ((e.clientY - r.top) / r.height) * v.videoHeight })
  }

  const v = video.current
  return (
    <div className="tap-picker">
      <div className="tap-frame" onClick={pick}>
        {url && (
          <video
            ref={video}
            src={url}
            muted
            playsInline
            preload="auto"
            onLoadedMetadata={(e) => {
              setDuration(e.currentTarget.duration)
              e.currentTarget.currentTime = 0
            }}
          />
        )}
        {tap && v && v.videoWidth > 0 && (
          <span className="tap-mark" style={{ left: `${(tap.x / v.videoWidth) * 100}%`, top: `${(tap.y / v.videoHeight) * 100}%` }} />
        )}
      </div>
      {duration > 0 && (
        <input type="range" min={0} max={duration} step={0.01} value={time} onChange={(e) => seek(Number(e.target.value))} aria-label="frame" />
      )}
      <p className="muted small">{tap ? `Tracking from your tap at ${tap.time.toFixed(2)} s. Tap again to move it.` : 'Tap the moving object. Slide to a frame where it is clearly visible.'}</p>
    </div>
  )
}

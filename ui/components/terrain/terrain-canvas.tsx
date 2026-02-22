"use client"

import { useRef, useEffect, useCallback, useState } from "react"
import type { TerrainAsset } from "@/lib/types"
import type { OverlayState, OverlayOpacity } from "./overlay-controls"
import { RoverOverlay } from "./rover-overlay"

interface TerrainCanvasProps {
  asset: TerrainAsset
  overlays: OverlayState
  opacity: OverlayOpacity
  roverPosition?: [number, number] | null
  roverHeading?: number
  roverProgress?: number
  pathPoints?: [number, number][]
  startPos?: [number, number] | null
  goalPos?: [number, number] | null
  onCanvasClick?: (row: number, col: number) => void
  onStartDrag?: (pos: [number, number]) => void
  onGoalDrag?: (pos: [number, number]) => void
  className?: string
  canvasClassName?: string
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

const MARKER_RADIUS = 12
const DRAG_THRESHOLD = 20

export function TerrainCanvas({
  asset,
  overlays,
  opacity,
  roverPosition,
  roverHeading = 0,
  roverProgress = 0,
  pathPoints,
  startPos,
  goalPos,
  onCanvasClick,
  onStartDrag,
  onGoalDrag,
  className = "",
  canvasClassName = "",
}: TerrainCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imagesRef = useRef<Record<string, HTMLImageElement>>({})
  const [dimensions, setDimensions] = useState({ w: 960, h: 540 })
  const dragRef = useRef<{ type: "start" | "goal"; active: boolean } | null>(null)
  const pulseRef = useRef(0)

  const loadAll = useCallback(async () => {
    try {
      const [terrain, mask, pathOvl, conf] = await Promise.all([
        loadImage(asset.terrain),
        loadImage(asset.maskOverlay),
        loadImage(asset.pathOverlay),
        loadImage(asset.confidence),
      ])
      imagesRef.current = { terrain, mask, pathOverlay: pathOvl, confidence: conf }
      setDimensions({ w: terrain.naturalWidth, h: terrain.naturalHeight })
    } catch {
      // partial load is ok
    }
  }, [asset])

  useEffect(() => {
    loadAll()
  }, [loadAll])

  const toCanvas = useCallback((e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return [
      Math.round((e.clientY - rect.top) * scaleY),
      Math.round((e.clientX - rect.left) * scaleX),
    ]
  }, [])

  const isNearMarker = useCallback((pos: [number, number], marker: [number, number] | null): boolean => {
    if (!marker) return false
    const dy = pos[0] - marker[0]
    const dx = pos[1] - marker[1]
    return Math.sqrt(dx * dx + dy * dy) < DRAG_THRESHOLD
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = toCanvas(e)
    if (onStartDrag && isNearMarker(pos, startPos ?? null)) {
      dragRef.current = { type: "start", active: true }
      e.preventDefault()
    } else if (onGoalDrag && isNearMarker(pos, goalPos ?? null)) {
      dragRef.current = { type: "goal", active: true }
      e.preventDefault()
    }
  }, [toCanvas, isNearMarker, startPos, goalPos, onStartDrag, onGoalDrag])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragRef.current?.active) return
    const pos = toCanvas(e)
    if (dragRef.current.type === "start" && onStartDrag) {
      onStartDrag(pos)
    } else if (dragRef.current.type === "goal" && onGoalDrag) {
      onGoalDrag(pos)
    }
  }, [toCanvas, onStartDrag, onGoalDrag])

  const handleMouseUp = useCallback(() => {
    dragRef.current = null
  }, [])

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onCanvasClick || !canvasRef.current) return
    if (dragRef.current) return
    const pos = toCanvas(e)
    onCanvasClick(pos[0], pos[1])
  }, [onCanvasClick, toCanvas])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const imgs = imagesRef.current
    const W = canvas.width
    const H = canvas.height

    ctx.clearRect(0, 0, W, H)

    if (imgs.terrain) {
      ctx.drawImage(imgs.terrain, 0, 0, W, H)
    } else {
      ctx.fillStyle = "#0a0f1a"
      ctx.fillRect(0, 0, W, H)
    }

    if (overlays.costmap && imgs.pathOverlay) {
      ctx.globalAlpha = opacity.costmap
      ctx.drawImage(imgs.pathOverlay, 0, 0, W, H)
      ctx.globalAlpha = 1
    }

    if (overlays.segmentation && imgs.mask) {
      ctx.globalAlpha = opacity.segmentation
      ctx.drawImage(imgs.mask, 0, 0, W, H)
      ctx.globalAlpha = 1
    }

    if (overlays.confidence && imgs.confidence) {
      ctx.globalAlpha = opacity.confidence
      ctx.drawImage(imgs.confidence, 0, 0, W, H)
      ctx.globalAlpha = 1
    }

    if (overlays.grid) {
      ctx.strokeStyle = "rgba(6,182,212,0.10)"
      ctx.lineWidth = 0.5
      const step = 40
      for (let x = 0; x < W; x += step) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
      }
      for (let y = 0; y < H; y += step) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
      }
    }

    const progressIdx = Math.floor(roverProgress)

    if (overlays.path && pathPoints && pathPoints.length > 1) {
      // Remaining path (dashed, dim)
      if (progressIdx < pathPoints.length - 1) {
        ctx.save()
        ctx.strokeStyle = "rgba(34,197,94,0.25)"
        ctx.lineWidth = 2
        ctx.setLineDash([6, 4])
        ctx.beginPath()
        ctx.moveTo(pathPoints[progressIdx][1], pathPoints[progressIdx][0])
        for (let i = progressIdx + 1; i < pathPoints.length; i++) {
          ctx.lineTo(pathPoints[i][1], pathPoints[i][0])
        }
        ctx.stroke()
        ctx.restore()
      }

      // Traversed path (solid, bright, glowing)
      if (progressIdx > 0) {
        ctx.save()
        ctx.strokeStyle = "#22C55E"
        ctx.lineWidth = 3
        ctx.shadowColor = "#22C55E"
        ctx.shadowBlur = 8
        ctx.beginPath()
        ctx.moveTo(pathPoints[0][1], pathPoints[0][0])
        for (let i = 1; i <= Math.min(progressIdx, pathPoints.length - 1); i++) {
          ctx.lineTo(pathPoints[i][1], pathPoints[i][0])
        }
        ctx.stroke()
        ctx.shadowBlur = 0
        ctx.restore()
      }
    }

    const pulse = (Math.sin(pulseRef.current) + 1) / 2

    // Start marker
    if (startPos) {
      const [sy, sx] = startPos
      // Pulsing outer ring
      ctx.save()
      ctx.globalAlpha = 0.2 + pulse * 0.3
      ctx.strokeStyle = "#22C55E"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(sx, sy, MARKER_RADIUS + 4 + pulse * 3, 0, Math.PI * 2)
      ctx.stroke()
      ctx.restore()
      // Filled circle
      ctx.fillStyle = "#22C55E"
      ctx.shadowColor = "#22C55E"
      ctx.shadowBlur = 12
      ctx.beginPath()
      ctx.arc(sx, sy, MARKER_RADIUS, 0, Math.PI * 2)
      ctx.fill()
      ctx.shadowBlur = 0
      ctx.strokeStyle = "rgba(0,0,0,0.5)"
      ctx.lineWidth = 2
      ctx.stroke()
      // Label
      ctx.fillStyle = "#fff"
      ctx.font = "bold 11px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("START", sx, sy - MARKER_RADIUS - 8)
    }

    // Goal marker
    if (goalPos) {
      const [gy, gx] = goalPos
      ctx.save()
      ctx.globalAlpha = 0.2 + pulse * 0.3
      ctx.strokeStyle = "#EF4444"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(gx, gy, MARKER_RADIUS + 4 + pulse * 3, 0, Math.PI * 2)
      ctx.stroke()
      ctx.restore()
      ctx.fillStyle = "#EF4444"
      ctx.shadowColor = "#EF4444"
      ctx.shadowBlur = 12
      ctx.beginPath()
      ctx.arc(gx, gy, MARKER_RADIUS, 0, Math.PI * 2)
      ctx.fill()
      ctx.shadowBlur = 0
      ctx.strokeStyle = "rgba(0,0,0,0.5)"
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.fillStyle = "#fff"
      ctx.font = "bold 11px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("GOAL", gx, gy - MARKER_RADIUS - 8)
    }

  }, [overlays, opacity, roverProgress, pathPoints, startPos, goalPos])

  // Animation loop for pulsing markers
  useEffect(() => {
    let running = true
    const animate = () => {
      if (!running) return
      pulseRef.current += 0.05
      draw()
      requestAnimationFrame(animate)
    }
    animate()
    return () => { running = false }
  }, [draw])

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`}
    >
      <canvas
        ref={canvasRef}
        width={dimensions.w}
        height={dimensions.h}
        onClick={handleClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`w-full h-auto cursor-crosshair ${canvasClassName}`}
      />
      <RoverOverlay
        position={roverPosition ?? null}
        heading={roverHeading}
        width={dimensions.w}
        height={dimensions.h}
      />
    </div>
  )
}

"use client"

import { useState, useEffect } from "react"
import { cn } from "@/lib/utils"

interface StatusStripProps {
  modelName?: string
  terrainLabel?: string
  safetyScore?: number
  pathFound?: boolean
}

export function StatusStrip({ modelName, terrainLabel, safetyScore, pathFound }: StatusStripProps) {
  const [ts, setTs] = useState("--:--:--")

  useEffect(() => {
    const update = () =>
      setTs(new Date().toLocaleTimeString("en-US", { hour12: false }))
    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="hud-panel flex items-center justify-between gap-4 px-5 py-2.5 rounded-xl text-sm flex-wrap">
      <div className="flex items-center gap-4 flex-wrap">
        {modelName && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-xs uppercase tracking-wide">Model</span>
              <span className="font-mono text-foreground font-medium">{modelName}</span>
            </div>
            <div className="w-px h-4 bg-border hidden sm:block" />
          </>
        )}
        {terrainLabel && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-xs uppercase tracking-wide">Terrain</span>
              <span className="font-mono text-foreground text-sm">{terrainLabel}</span>
            </div>
            <div className="w-px h-4 bg-border hidden sm:block" />
          </>
        )}
        {pathFound !== undefined && (
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground text-xs uppercase tracking-wide">Path</span>
            <span
              className={cn(
                "font-mono text-xs font-medium",
                pathFound ? "text-emerald-400" : "text-amber-400"
              )}
            >
              {pathFound ? "Found" : "None"}
            </span>
          </div>
        )}
        {safetyScore !== undefined && (
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground text-xs uppercase tracking-wide">Safety</span>
            <span
              className={cn(
                "font-mono text-sm font-semibold tabular-nums",
                safetyScore >= 0.7 && "text-emerald-400",
                safetyScore >= 0.5 && safetyScore < 0.7 && "text-amber-400",
                safetyScore < 0.5 && "text-red-400"
              )}
            >
              {(safetyScore * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 ml-auto">
        <div className="flex items-center gap-2">
          <div className="status-dot status-dot-green" />
          <span className="text-emerald-400 text-xs font-medium uppercase tracking-wide">Live</span>
        </div>
        <span className="font-mono text-sm text-muted-foreground tabular-nums">{ts}</span>
      </div>
    </div>
  )
}

"use client"

import { Layers } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"

export interface OverlayState {
  segmentation: boolean
  costmap: boolean
  confidence: boolean
  path: boolean
  grid: boolean
}

export interface OverlayOpacity {
  segmentation: number
  costmap: number
  confidence: number
}

interface OverlayControlsProps {
  overlays: OverlayState
  opacity: OverlayOpacity
  onToggle: (layer: keyof OverlayState) => void
  onOpacity: (layer: keyof OverlayOpacity, val: number) => void
}

const LAYERS: { key: keyof OverlayState; label: string; desc: string; color: string; hasOpacity?: boolean }[] = [
  { key: "segmentation", label: "Segmentation", desc: "Class mask overlay", color: "#10B981", hasOpacity: true },
  { key: "costmap", label: "Cost Map", desc: "Traversability costs", color: "#EF4444", hasOpacity: true },
  { key: "confidence", label: "Confidence", desc: "Model certainty heatmap", color: "#8B5CF6", hasOpacity: true },
  { key: "path", label: "Path", desc: "Planned route", color: "#22C55E" },
  { key: "grid", label: "Grid", desc: "Coordinate overlay", color: "#06B6D4" },
]

function LayerToggle({
  label,
  desc,
  active,
  color,
  onToggle,
  opacity,
  onOpacity,
}: {
  label: string
  desc: string
  active: boolean
  color: string
  onToggle: () => void
  opacity?: number
  onOpacity?: (v: number) => void
}) {
  return (
    <div className="space-y-2">
      <button
        onClick={onToggle}
        className={cn(
          "flex items-center gap-3 w-full text-left px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
          active
            ? "text-foreground bg-secondary/50"
            : "text-muted-foreground hover:text-foreground hover:bg-secondary/20"
        )}
        aria-pressed={active}
        aria-label={`${label}: ${active ? "on" : "off"}`}
      >
        <div
          className="w-3 h-3 rounded-sm shrink-0"
          style={{ backgroundColor: active ? color : "oklch(0.30 0.01 250)" }}
          aria-hidden
        />
        <div className="flex-1 min-w-0">
          <span className="block">{label}</span>
          <span className="text-xs text-muted-foreground truncate block">{desc}</span>
        </div>
      </button>
      {active && opacity !== undefined && onOpacity && (
        <div className="flex items-center gap-3 pl-6">
          <Slider
            value={[opacity]}
            onValueChange={([v]) => onOpacity(v)}
            min={0}
            max={1}
            step={0.05}
            className="flex-1 min-w-0"
            aria-label={`${label} opacity: ${Math.round(opacity * 100)}%`}
          />
          <span className="text-sm font-mono text-muted-foreground w-10 text-right shrink-0 tabular-nums">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}
    </div>
  )
}

export function OverlayControls({
  overlays,
  opacity,
  onToggle,
  onOpacity,
}: OverlayControlsProps) {
  return (
    <section
      className="hud-panel rounded-xl p-4 flex-1 min-w-0"
      aria-labelledby="overlay-heading"
    >
      <header className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center shrink-0" aria-hidden>
          <Layers className="w-5 h-5 text-cyan-500" />
        </div>
        <div>
          <h2 id="overlay-heading" className="text-sm font-semibold text-foreground">
            Overlay Layers
          </h2>
          <p className="text-sm text-muted-foreground">Toggle layers and adjust opacity</p>
        </div>
      </header>

      <nav className="space-y-2" aria-label="Overlay controls">
        {LAYERS.map(({ key, label, desc, color, hasOpacity }) => (
          <LayerToggle
            key={key}
            label={label}
            desc={desc}
            active={overlays[key]}
            color={color}
            onToggle={() => onToggle(key)}
            opacity={hasOpacity ? opacity[key as keyof OverlayOpacity] : undefined}
            onOpacity={hasOpacity ? (v) => onOpacity(key as keyof OverlayOpacity, v) : undefined}
          />
        ))}
      </nav>
    </section>
  )
}

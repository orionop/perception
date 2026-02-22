"use client"

import { Bot, Compass, MapPin, Gauge } from "lucide-react"

export interface RoverState {
  position: [number, number]
  heading: number
  progress: number
  totalSteps: number
  terrainClass: string
  terrainColor: string
  confidence: number
  cost: number
}

function BarGauge({ value, max, color, label, status }: { value: number; max: number; color: string; label: string; status?: string }) {
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="space-y-1.5" role="group" aria-label={`${label}${status ? `: ${status}` : ""} ${(value / max * 100).toFixed(0)}%`}>
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className="text-sm font-mono font-semibold tabular-nums" style={{ color }}>
          {max <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)}
          {status && <span className="text-xs font-normal text-muted-foreground ml-1">({status})</span>}
        </span>
      </div>
      <div className="h-2 bg-secondary/40 rounded-full overflow-hidden" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

export function RoverTelemetry({ state }: { state: RoverState }) {
  const confColor = state.confidence >= 0.7 ? "#10B981" : state.confidence >= 0.4 ? "#F59E0B" : "#EF4444"
  const costColor = state.cost <= 1.0 ? "#10B981" : state.cost <= 3.0 ? "#F59E0B" : "#EF4444"
  const confStatus = state.confidence >= 0.7 ? "Good" : state.confidence >= 0.4 ? "Moderate" : "Low"
  const costStatus = state.cost <= 1.0 ? "Traversable" : state.cost <= 3.0 ? "Soft terrain" : "Obstacle"

  return (
    <section
      className="hud-panel rounded-xl p-4 flex-1 min-w-0"
      aria-labelledby="rover-status-heading"
    >
      <header className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center shrink-0" aria-hidden>
          <Bot className="w-5 h-5 text-amber-500" />
        </div>
        <div>
          <h2 id="rover-status-heading" className="text-sm font-semibold text-foreground">
            Rover Status
          </h2>
          <p className="text-sm text-muted-foreground">Live telemetry from current position</p>
        </div>
      </header>

      <nav className="space-y-4" aria-label="Rover metrics">
        <div className="grid grid-cols-2 gap-3">
          <div className="flex items-center gap-2 p-2 rounded-lg bg-secondary/20" title={`Heading: ${state.heading.toFixed(0)}°`}>
            <Compass className="w-4 h-4 text-amber-500 shrink-0" aria-hidden />
            <div>
              <span className="text-xs text-muted-foreground block">Heading</span>
              <span className="text-sm font-mono font-semibold">{state.heading.toFixed(0)}°</span>
            </div>
          </div>
          <div className="flex items-center gap-2 p-2 rounded-lg bg-secondary/20" title={`Position: (${state.position[0]}, ${state.position[1]})`}>
            <MapPin className="w-4 h-4 text-emerald-500 shrink-0" aria-hidden />
            <div className="min-w-0">
              <span className="text-xs text-muted-foreground block">Position</span>
              <span className="text-sm font-mono font-semibold truncate block">
                ({state.position[0].toFixed(0)}, {state.position[1].toFixed(0)})
              </span>
            </div>
          </div>
        </div>

        <div className="p-2 rounded-lg bg-secondary/20" title={`Current terrain: ${state.terrainClass}`}>
          <span className="text-xs text-muted-foreground block mb-0.5">Terrain Class</span>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: state.terrainColor }} aria-hidden />
            <span className="text-sm font-medium text-foreground">{state.terrainClass}</span>
          </div>
        </div>

        <div className="space-y-3" role="group" aria-label="Performance metrics">
          <div className="flex items-center gap-2" aria-hidden>
            <Gauge className="w-4 h-4 text-cyan-500 shrink-0" />
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Performance</span>
          </div>
          <BarGauge value={state.confidence} max={1} color={confColor} label="Confidence" status={confStatus} />
          <BarGauge value={state.cost} max={5} color={costColor} label="Traversal Cost" status={costStatus} />
        </div>

        <div className="flex items-center justify-between pt-3 border-t border-border/50">
          <span className="text-sm text-muted-foreground">Path Progress</span>
          <span className="text-sm font-mono font-semibold text-amber-400">
            {state.progress} / {state.totalSteps}
          </span>
        </div>
      </nav>
    </section>
  )
}

"use client"

import { useState, useCallback } from "react"
import {
  Activity,
  Map,
  Shield,
  Gauge,
  Layers,
  Cpu,
  Route,
  AlertTriangle,
} from "lucide-react"
import { StatusStrip } from "@/components/mission/status-strip"
import { TerrainCanvas } from "@/components/terrain/terrain-canvas"
import { OverlayControls } from "@/components/terrain/overlay-controls"
import { PerClassChart } from "@/components/dashboard/per-class-chart"
import { RobustnessChart } from "@/components/dashboard/robustness-chart"
import {
  getPrimaryReport,
  getAllReports,
  getTerrainAsset,
  terrainAssets,
} from "@/lib/data"
import { cn } from "@/lib/utils"
import type { OverlayState, OverlayOpacity } from "@/components/terrain/overlay-controls"

function MetricCard({
  label,
  value,
  unit,
  status,
  icon: Icon,
}: {
  label: string
  value: string
  unit?: string
  status?: "good" | "warn" | "critical" | "neutral"
  icon?: React.ComponentType<{ className?: string }>
}) {
  return (
    <div className="hud-panel rounded-xl p-3 flex items-start gap-2.5 min-h-[72px]">
      <div
        className={cn(
          "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
          status === "good" && "bg-emerald-500/10",
          status === "warn" && "bg-amber-500/10",
          status === "critical" && "bg-red-500/10",
          (!status || status === "neutral") && "bg-secondary"
        )}
      >
        {Icon && (
          <Icon
            className={cn(
              "w-4 h-4 shrink-0",
              status === "good" && "text-emerald-400",
              status === "warn" && "text-amber-400",
              status === "critical" && "text-red-400",
              (!status || status === "neutral") && "text-muted-foreground"
            )}
          />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[11px] text-muted-foreground uppercase tracking-wide">
          {label}
        </div>
        <div className="flex items-baseline gap-1 mt-0.5">
          <span
            className={cn(
              "text-base font-mono font-bold tabular-nums",
              status === "good" && "text-emerald-400",
              status === "warn" && "text-amber-400",
              status === "critical" && "text-red-400",
              (!status || status === "neutral") && "text-foreground"
            )}
          >
            {value}
          </span>
          {unit && (
            <span className="text-xs text-muted-foreground font-normal">
              {unit}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

function getStatus(v: number, thresholds: { good: number; warn: number }): "good" | "warn" | "critical" {
  if (v >= thresholds.good) return "good"
  if (v >= thresholds.warn) return "warn"
  return "critical"
}

export default function CommandCenterPage() {
  const reports = getAllReports()
  const [selectedModelIdx, setSelectedModelIdx] = useState(0)
  const report = reports[selectedModelIdx] ?? getPrimaryReport()
  const [selectedTerrain, setSelectedTerrain] = useState("terrain_01")
  const asset = getTerrainAsset(selectedTerrain)

  const [overlays, setOverlays] = useState<OverlayState>({
    segmentation: true,
    costmap: false,
    confidence: false,
    path: true,
    grid: false,
  })
  const [opacity, setOpacity] = useState<OverlayOpacity>({
    segmentation: 0.45,
    costmap: 0.5,
    confidence: 0.5,
  })

  const toggleOverlay = useCallback((layer: keyof OverlayState) => {
    setOverlays((prev) => ({ ...prev, [layer]: !prev[layer] }))
  }, [])

  const setLayerOpacity = useCallback((layer: keyof OverlayOpacity, val: number) => {
    setOpacity((prev) => ({ ...prev, [layer]: val }))
  }, [])

  const pathPoints: [number, number][] = []
  if (report.navigation.path_found) {
    const h = 540
    for (let i = 0; i < report.navigation.path_length; i++) {
      pathPoints.push([h - 2, i])
    }
  }

  const safetyStatus = getStatus(report.safety.safety_score, { good: 0.7, warn: 0.5 })
  const mIouStatus = getStatus(report.metrics.mean_iou, { good: 0.5, warn: 0.15 })

  return (
    <div className="space-y-6 w-full max-w-7xl mx-auto min-w-0">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground tracking-tight">
          Command Center
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Mission overview, perception metrics, and navigation status
        </p>
      </div>

      {/* Status bar: context + Live + timestamp */}
      <StatusStrip
        modelName={report.model_name}
        terrainLabel={asset.label}
        safetyScore={report.safety.safety_score}
        pathFound={report.navigation.path_found}
      />

      {/* Mission Metrics - at top */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-foreground tracking-tight flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-500" />
          Mission Metrics
        </h2>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          <MetricCard
            label="Safety Score"
            value={(report.safety.safety_score * 100).toFixed(1)}
            unit="%"
            status={safetyStatus}
            icon={Shield}
          />
          <MetricCard
            label="Mean IoU"
            value={(report.metrics.mean_iou * 100).toFixed(1)}
            unit="%"
            status={mIouStatus}
            icon={Gauge}
          />
          <MetricCard
            label="Pixel Accuracy"
            value={(report.metrics.pixel_accuracy * 100).toFixed(1)}
            unit="%"
            status={getStatus(report.metrics.pixel_accuracy, { good: 0.5, warn: 0.3 })}
            icon={Activity}
          />
          <MetricCard
            label="Inference"
            value={report.inference_time_ms.toFixed(0)}
            unit="ms"
            status={report.inference_time_ms < 200 ? "good" : "warn"}
            icon={Cpu}
          />
          <MetricCard
            label="Path Found"
            value={report.navigation.path_found ? "Yes" : "No"}
            status={report.navigation.path_found ? "good" : "critical"}
            icon={Route}
          />
          <MetricCard
            label="Path Cost"
            value={report.navigation.path_cost.toFixed(0)}
            status="neutral"
            icon={Route}
          />
          <MetricCard
            label="Confidence"
            value={(report.segmentation.confidence_mean * 100).toFixed(1)}
            unit="%"
            status={report.segmentation.confidence_mean >= 0.7 ? "good" : "warn"}
            icon={Gauge}
          />
          <MetricCard
            label="Obstacle Overlap"
            value={(report.safety.obstacle_overlap_pct * 100).toFixed(1)}
            unit="%"
            status={report.safety.obstacle_overlap_pct === 0 ? "good" : "critical"}
            icon={AlertTriangle}
          />
        </div>
      </div>

      {/* Main layout: Map + Sidebar + Charts */}
      <div className="flex flex-col lg:flex-row gap-5">
        {/* Map section - compact */}
        <div className="flex-1 min-w-0 max-w-2xl">
          <div className="hud-panel rounded-xl overflow-hidden">
            <div className="px-4 py-2.5 border-b border-border flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Map className="w-4 h-4 text-cyan-500" />
                <span className="text-sm font-semibold text-foreground">
                  Terrain View
                </span>
              </div>
              <select
                value={selectedTerrain}
                onChange={(e) => setSelectedTerrain(e.target.value)}
                className="bg-input border border-border rounded-lg px-2.5 py-1 text-sm font-mono text-foreground h-7"
              >
                {terrainAssets.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.label}
                  </option>
                ))}
              </select>
            </div>
            <TerrainCanvas
              asset={asset}
              canvasClassName="max-h-[28vh] w-full"
              overlays={overlays}
              opacity={opacity}
              pathPoints={pathPoints}
              roverPosition={pathPoints.length > 0 ? pathPoints[0] : null}
              roverHeading={90}
            />
          </div>
        </div>

        {/* Right: Overlay + Model + Per-Class + Robustness + Cost Map */}
        <div className="flex-1 min-w-0 flex flex-col gap-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="w-full sm:w-[260px] shrink-0 space-y-4">
              <div className="hud-panel rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Cpu className="w-4 h-4 text-cyan-500" />
                  <span className="text-sm font-semibold text-foreground">Model</span>
                </div>
                <select
                  value={selectedModelIdx}
                  onChange={(e) => setSelectedModelIdx(Number(e.target.value))}
                  className="w-full bg-input border border-border rounded-lg px-3 py-2 text-sm font-mono text-foreground"
                >
                  {reports.map((r, i) => (
                    <option key={r.model_name} value={i}>
                      {r.model_name}
                    </option>
                  ))}
                </select>
              </div>
              <OverlayControls
                overlays={overlays}
                opacity={opacity}
                onToggle={toggleOverlay}
                onOpacity={setLayerOpacity}
              />
            </div>
            <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4">
              <PerClassChart compact report={report} />
              <RobustnessChart compact report={report} />
            </div>
          </div>
          <MiniCostMap terrainId={selectedTerrain} />
        </div>
      </div>
    </div>
  )
}

function MiniCostMap({ terrainId = "terrain_01" }: { terrainId?: string }) {
  const asset = getTerrainAsset(terrainId)
  return (
    <div className="hud-panel rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Layers className="w-4 h-4 text-cyan-500" />
        <span className="text-sm font-semibold text-foreground">Cost Map</span>
      </div>
      <div className="rounded-lg overflow-hidden border border-border max-h-[140px]">
        <img
          src={asset.pathOverlay}
          alt="Cost map"
          className="w-full h-full object-cover object-top"
        />
      </div>
    </div>
  )
}

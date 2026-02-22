"use client"

import { useState, useCallback, useRef, useEffect, useMemo } from "react"
import { Play, Pause, RotateCcw, MapPin, Flag, ShieldCheck, ShieldAlert, TrendingDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { TerrainCanvas } from "@/components/terrain/terrain-canvas"
import { OverlayControls } from "@/components/terrain/overlay-controls"
import { RoverTelemetry } from "@/components/terrain/rover-telemetry"
import { terrainAssets, getTerrainAsset, getPrimaryReport, getTerrainClasses } from "@/lib/data"
import { cn } from "@/lib/utils"
import type { OverlayState, OverlayOpacity } from "@/components/terrain/overlay-controls"
import type { RoverState } from "@/components/terrain/rover-telemetry"

type PlacingMode = "start" | "goal" | null
type PlannerType = "astar" | "potential" | "rrt"

function generatePath(
  start: [number, number],
  goal: [number, number],
  _planner: PlannerType
): [number, number][] {
  const path: [number, number][] = []
  const [sr, sc] = start
  const [gr, gc] = goal
  const maxR = 540
  const maxC = 960

  // Clamp coords to canvas
  const clamp = (r: number, c: number): [number, number] => [
    Math.max(0, Math.min(maxR - 1, Math.round(r))),
    Math.max(0, Math.min(maxC - 1, Math.round(c))),
  ]

  // Move column-first (horizontal), then row (vertical) — like A* on a grid
  const dr = gr - sr
  const dc = gc - sc
  const totalSteps = Math.abs(dr) + Math.abs(dc)
  if (totalSteps === 0) return [clamp(sr, sc)]

  // Horizontal leg
  const colStep = dc === 0 ? 0 : dc > 0 ? 1 : -1
  let r = sr, c = sc
  for (let i = 0; i < Math.abs(dc); i++) {
    path.push(clamp(r, c))
    c += colStep
  }
  // Vertical leg
  const rowStep = dr === 0 ? 0 : dr > 0 ? 1 : -1
  for (let i = 0; i < Math.abs(dr); i++) {
    path.push(clamp(r, c))
    r += rowStep
  }
  path.push(clamp(gr, gc))
  return path
}

function lerpAngle(a: number, b: number, t: number): number {
  let diff = b - a
  while (diff > 180) diff -= 360
  while (diff < -180) diff += 360
  return a + diff * t
}

function ProgressRing({ value, size = 64, strokeWidth = 5, color }: { value: number; size?: number; strokeWidth?: number; color: string }) {
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value / 100) * circumference
  return (
    <svg width={size} height={size} className="shrink-0">
      <circle
        cx={size / 2} cy={size / 2} r={radius}
        fill="none" stroke="oklch(0.20 0.01 250)" strokeWidth={strokeWidth}
      />
      <circle
        cx={size / 2} cy={size / 2} r={radius}
        fill="none" stroke={color} strokeWidth={strokeWidth}
        strokeDasharray={circumference} strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        className="transition-all duration-300"
      />
    </svg>
  )
}

export default function SimulatorPage() {
  const report = getPrimaryReport()
  const classes = getTerrainClasses()

  const [selectedTerrain, setSelectedTerrain] = useState("terrain_01")
  const [planner, setPlanner] = useState<PlannerType>("astar")
  const [placingMode, setPlacingMode] = useState<PlacingMode>(null)
  const [startPos, setStartPos] = useState<[number, number]>([520, 20])
  const [goalPos, setGoalPos] = useState<[number, number]>([520, 940])
  const [pathPoints, setPathPoints] = useState<[number, number][]>([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [roverProgress, setRoverProgress] = useState(0)
  const [speed, setSpeed] = useState(3)
  const [smoothHeading, setSmoothHeading] = useState(90)
  const animRef = useRef<number>(0)
  const lastTickRef = useRef(0)
  const pathLengthRef = useRef(0)

  const asset = getTerrainAsset(selectedTerrain)

  const [overlays, setOverlays] = useState<OverlayState>({
    segmentation: true,
    costmap: false,
    confidence: false,
    path: true,
    grid: true,
  })
  const [opacity, setOpacity] = useState<OverlayOpacity>({
    segmentation: 0.4,
    costmap: 0.5,
    confidence: 0.5,
  })

  const toggleOverlay = useCallback((layer: keyof OverlayState) => {
    setOverlays((prev) => ({ ...prev, [layer]: !prev[layer] }))
  }, [])

  const setLayerOpacity = useCallback((layer: keyof OverlayOpacity, val: number) => {
    setOpacity((prev) => ({ ...prev, [layer]: val }))
  }, [])

  // Auto-generate path when start/goal change
  const regeneratePath = useCallback((s: [number, number], g: [number, number], p: PlannerType) => {
    const path = generatePath(s, g, p)
    setPathPoints(path)
    setRoverProgress(0)
    setIsPlaying(false)
  }, [])

  const handleCanvasClick = useCallback(
    (row: number, col: number) => {
      if (placingMode === "start") {
        setStartPos([row, col])
        setPlacingMode("goal")
        if (goalPos) regeneratePath([row, col], goalPos, planner)
      } else if (placingMode === "goal") {
        setGoalPos([row, col])
        setPlacingMode(null)
        if (startPos) regeneratePath(startPos, [row, col], planner)
      }
    },
    [placingMode, startPos, goalPos, planner, regeneratePath]
  )

  const handleStartDrag = useCallback((pos: [number, number]) => {
    setStartPos(pos)
    if (goalPos) regeneratePath(pos, goalPos, planner)
  }, [goalPos, planner, regeneratePath])

  const handleGoalDrag = useCallback((pos: [number, number]) => {
    setGoalPos(pos)
    if (startPos) regeneratePath(startPos, pos, planner)
  }, [startPos, planner, regeneratePath])

  // Initial path generation
  useEffect(() => {
    if (startPos && goalPos) {
      regeneratePath(startPos, goalPos, planner)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Re-plan on planner change
  useEffect(() => {
    if (startPos && goalPos && pathPoints.length > 0) {
      regeneratePath(startPos, goalPos, planner)
    }
  }, [planner]) // eslint-disable-line react-hooks/exhaustive-deps

  pathLengthRef.current = pathPoints.length

  // Smooth animation loop - runs until rover reaches final path point
  useEffect(() => {
    if (!isPlaying || pathPoints.length === 0) return

    lastTickRef.current = performance.now()
    const step = (now: number) => {
      const delta = (now - lastTickRef.current) / 1000
      lastTickRef.current = now
      const targetLen = pathLengthRef.current
      if (targetLen <= 1) {
        setIsPlaying(false)
        return
      }
      const maxIdx = targetLen - 1
      setRoverProgress((prev) => {
        const next = prev + delta * speed * 30
        if (next >= maxIdx) {
          setIsPlaying(false)
          return maxIdx
        }
        return next
      })
      animRef.current = requestAnimationFrame(step)
    }
    animRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(animRef.current)
  }, [isPlaying, pathPoints.length, speed])

  // Compute interpolated position - ensure we reach the final goal
  const lastIdx = Math.max(0, pathPoints.length - 1)
  const clampedProgress = Math.min(roverProgress, lastIdx)
  const idx = Math.floor(clampedProgress)
  const frac = Math.min(1, clampedProgress - idx)
  const currentPt = pathPoints[idx] ?? null
  const nextPt = pathPoints[Math.min(idx + 1, pathPoints.length - 1)] ?? null
  const interpPos: [number, number] | null =
    currentPt && nextPt
      ? [
        currentPt[0] + (nextPt[0] - currentPt[0]) * frac,
        currentPt[1] + (nextPt[1] - currentPt[1]) * frac,
      ]
      : pathPoints.length > 0
        ? pathPoints[pathPoints.length - 1]
        : null

  // Compute target heading and smooth it
  const targetHeading =
    idx > 0 && pathPoints[idx - 1] && currentPt
      ? ((Math.atan2(currentPt[1] - pathPoints[idx - 1][1], -(currentPt[0] - pathPoints[idx - 1][0])) * 180) / Math.PI)
      : 90

  useEffect(() => {
    setSmoothHeading((prev) => lerpAngle(prev, targetHeading, 0.15))
  }, [targetHeading, roverProgress])

  const currentClass = useMemo(() => {
    return classes[Math.abs(idx) % classes.length]
  }, [classes, idx])

  const progressPct = pathPoints.length > 1 ? (roverProgress / (pathPoints.length - 1)) * 100 : 0
  const stepsRemaining = pathPoints.length > 0 ? Math.max(0, pathPoints.length - 1 - Math.floor(roverProgress)) : 0

  const roverState: RoverState = {
    position: interpPos ?? startPos ?? [0, 0],
    heading: smoothHeading,
    progress: Math.floor(roverProgress),
    totalSteps: pathPoints.length,
    terrainClass: currentClass?.name ?? "Unknown",
    terrainColor: currentClass?.color ?? "#888",
    confidence: 0.72 + Math.sin(roverProgress * 0.1) * 0.15,
    cost: 1.0 + Math.sin(roverProgress * 0.05) * 0.5,
  }

  const safetyScore = report.safety.safety_score
  const safetySeverity = safetyScore >= 0.8 ? "robust" : safetyScore >= 0.5 ? "moderate" : "critical"
  const avgConfidence = roverState.confidence

  return (
    <div className="space-y-6 w-full max-w-7xl mx-auto min-w-0">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground tracking-tight">
          Terrain Simulator
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Rover navigation simulation on real terrain with perception overlays
        </p>
      </div>

      {/* Toolbar */}
      <div className="hud-panel rounded-xl px-5 py-3 flex items-center gap-4 flex-wrap">
        {/* Terrain & Planner */}
        <div className="flex items-center gap-3">
          <select
            value={selectedTerrain}
            onChange={(e) => { setSelectedTerrain(e.target.value); setPathPoints([]); setRoverProgress(0) }}
            className="bg-input border border-border rounded-lg px-3 h-9 text-sm font-mono text-foreground"
          >
            {terrainAssets.map((a) => (
              <option key={a.id} value={a.id}>{a.label}</option>
            ))}
          </select>

          <select
            value={planner}
            onChange={(e) => setPlanner(e.target.value as PlannerType)}
            className="bg-input border border-border rounded-lg px-3 h-9 text-sm font-mono text-foreground"
          >
            <option value="astar">A* Planner</option>
            <option value="potential">Potential Field</option>
            <option value="rrt">RRT*</option>
          </select>
        </div>

        <div className="w-px h-6 bg-border" />

        {/* Placement */}
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={placingMode === "start" ? "default" : "outline"}
            onClick={() => setPlacingMode(placingMode === "start" ? null : "start")}
            className="text-sm h-9 px-4"
          >
            <MapPin className="w-4 h-4 mr-1.5 text-emerald-400" />
            {placingMode === "start" ? "Click Map..." : "Set Start"}
          </Button>
          {startPos && (
            <span className="text-xs font-mono text-emerald-400">({startPos[0]}, {startPos[1]})</span>
          )}
          <Button
            size="sm"
            variant={placingMode === "goal" ? "default" : "outline"}
            onClick={() => setPlacingMode(placingMode === "goal" ? null : "goal")}
            className="text-sm h-9 px-4"
          >
            <Flag className="w-4 h-4 mr-1.5 text-red-400" />
            {placingMode === "goal" ? "Click Map..." : "Set Goal"}
          </Button>
          {goalPos && (
            <span className="text-xs font-mono text-red-400">({goalPos[0]}, {goalPos[1]})</span>
          )}
        </div>

        <div className="w-px h-6 bg-border" />

        {/* Playback */}
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              if (!isPlaying) lastTickRef.current = performance.now()
              setIsPlaying(!isPlaying)
            }}
            className="text-sm h-9 px-4"
            disabled={pathPoints.length === 0}
          >
            {isPlaying ? <Pause className="w-4 h-4 mr-1.5" /> : <Play className="w-4 h-4 mr-1.5" />}
            {isPlaying ? "Pause" : "Play"}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => { setRoverProgress(0); setIsPlaying(false); setSmoothHeading(90) }}
            className="text-sm h-9 px-4"
          >
            <RotateCcw className="w-4 h-4 mr-1.5" />
            Reset
          </Button>
        </div>

        {/* Speed */}
        <div className="flex items-center gap-2.5 ml-auto">
          <span className="text-sm text-muted-foreground">Speed</span>
          <input
            type="range"
            min={1}
            max={10}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-24 h-1.5 accent-amber-500"
          />
          <span className="text-sm font-mono font-bold text-amber-400">{speed}x</span>
        </div>
      </div>

      {/* Main area: Canvas + Sidebar */}
      <div className="flex flex-col lg:flex-row gap-5">
        <div className="flex-1 min-w-0 w-full max-w-full">
          <div className="hud-panel rounded-xl overflow-hidden w-full">
            <TerrainCanvas
              asset={asset}
              canvasClassName="max-h-[55vh]"
              overlays={overlays}
              opacity={opacity}
              pathPoints={overlays.path ? pathPoints : undefined}
              roverPosition={interpPos}
              roverHeading={smoothHeading}
              roverProgress={roverProgress}
              startPos={startPos}
              goalPos={goalPos}
              onCanvasClick={placingMode ? handleCanvasClick : undefined}
              onStartDrag={handleStartDrag}
              onGoalDrag={handleGoalDrag}
            />
          </div>
        </div>

        <div className="w-full lg:min-w-[420px] lg:max-w-[480px] shrink-0 flex flex-col sm:flex-row gap-3">
          <OverlayControls
            overlays={overlays}
            opacity={opacity}
            onToggle={toggleOverlay}
            onOpacity={setLayerOpacity}
          />
          {pathPoints.length > 0 && <RoverTelemetry state={roverState} />}
        </div>
      </div>

      {/* Mission Dashboard */}
      {pathPoints.length > 0 && (
        <div className="space-y-4 w-full">
          <h2 className="text-lg font-bold text-foreground tracking-tight">Mission Dashboard</h2>

          {/* Metric cards - uniform height and alignment */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 auto-rows-fr">
            {/* Progress */}
            <div className="hud-panel rounded-xl p-5 flex items-center gap-4 min-h-[120px]">
              <ProgressRing
                value={progressPct}
                color={progressPct >= 100 ? "#10B981" : "#F59E0B"}
              />
              <div className="min-w-0">
                <div className="text-xs text-muted-foreground uppercase tracking-wide">Progress</div>
                <div className={cn(
                  "text-2xl font-mono font-bold mt-0.5",
                  progressPct >= 100 ? "text-emerald-400" : "text-amber-400"
                )}>
                  {progressPct.toFixed(0)}%
                </div>
              </div>
            </div>

            {/* Distance Remaining */}
            <div className="hud-panel rounded-xl p-5 min-h-[120px] flex flex-col justify-between">
              <div className="text-xs text-muted-foreground uppercase tracking-wide">Distance Remaining</div>
              <div className="text-2xl font-mono font-bold text-foreground mt-1">
                {stepsRemaining}
                <span className="text-sm font-normal text-muted-foreground ml-1.5">steps</span>
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                of {pathPoints.length} total
              </div>
            </div>

            {/* Avg Confidence */}
            <div className="hud-panel rounded-xl p-5 min-h-[120px] flex flex-col">
              <div className="text-xs text-muted-foreground uppercase tracking-wide">Avg. Confidence</div>
              <div className={cn(
                "text-2xl font-mono font-bold mt-1",
                avgConfidence >= 0.7 ? "text-emerald-400" : avgConfidence >= 0.4 ? "text-amber-400" : "text-red-400"
              )}>
                {(avgConfidence * 100).toFixed(1)}%
              </div>
              <div className="h-1.5 bg-secondary/40 rounded-full overflow-hidden mt-2">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${avgConfidence * 100}%`,
                    backgroundColor: avgConfidence >= 0.7 ? "#10B981" : avgConfidence >= 0.4 ? "#F59E0B" : "#EF4444",
                  }}
                />
              </div>
            </div>

            {/* Safety Score */}
            <div className="hud-panel rounded-xl p-5 min-h-[120px] flex flex-col">
              <div className="text-xs text-muted-foreground uppercase tracking-wide">Safety Score</div>
              <div className={cn(
                "text-2xl font-mono font-bold mt-1",
                safetySeverity === "robust" ? "text-emerald-400"
                  : safetySeverity === "moderate" ? "text-amber-400"
                    : "text-red-400"
              )}>
                {(safetyScore * 100).toFixed(1)}%
              </div>
              <span className={cn(
                "inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-md mt-2 self-start",
                safetySeverity === "robust" ? "text-emerald-400 bg-emerald-400/10"
                  : safetySeverity === "moderate" ? "text-amber-400 bg-amber-400/10"
                    : "text-red-400 bg-red-400/10"
              )}>
                {safetySeverity === "robust"
                  ? <><ShieldCheck className="w-3 h-3" />Safe</>
                  : safetySeverity === "moderate"
                    ? <><ShieldAlert className="w-3 h-3" />Moderate</>
                    : <><TrendingDown className="w-3 h-3" />Risky</>}
              </span>
            </div>
          </div>

          {/* Path details row - grid for consistent chip sizing */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
            <div className="hud-panel rounded-lg px-4 py-3 flex flex-col gap-0.5 min-w-0">
              <span className="text-xs text-muted-foreground uppercase tracking-wide truncate">Planner</span>
              <span className="text-sm font-mono font-bold text-amber-400 uppercase truncate">{planner}</span>
            </div>
            <div className="hud-panel rounded-lg px-4 py-3 flex flex-col gap-0.5 min-w-0">
              <span className="text-xs text-muted-foreground uppercase tracking-wide truncate">Steps</span>
              <span className="text-sm font-mono font-bold text-foreground">{pathPoints.length}</span>
            </div>
            <div className="hud-panel rounded-lg px-4 py-3 flex flex-col gap-0.5 min-w-0">
              <span className="text-xs text-muted-foreground uppercase tracking-wide truncate">Start</span>
              <span className="text-sm font-mono font-bold text-emerald-400 truncate">
                ({startPos[0]}, {startPos[1]})
              </span>
            </div>
            <div className="hud-panel rounded-lg px-4 py-3 flex flex-col gap-0.5 min-w-0">
              <span className="text-xs text-muted-foreground uppercase tracking-wide truncate">Goal</span>
              <span className="text-sm font-mono font-bold text-red-400 truncate">
                ({goalPos[0]}, {goalPos[1]})
              </span>
            </div>
            <div className="hud-panel rounded-lg px-4 py-3 flex flex-col gap-0.5 min-w-0">
              <span className="text-xs text-muted-foreground uppercase tracking-wide truncate">Path Cost</span>
              <span className="text-sm font-mono font-bold text-foreground">{report.navigation.path_cost.toFixed(0)}</span>
            </div>
            <div className="hud-panel rounded-lg px-4 py-3 flex flex-col gap-0.5 min-w-0">
              <span className="text-xs text-muted-foreground uppercase tracking-wide truncate">Obstacle Overlap</span>
              <span className="text-sm font-mono font-bold text-foreground">
                {(report.safety.obstacle_overlap_pct * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ChevronDown, ChevronRight, TrendingDown, TrendingUp } from "lucide-react"
import { getPrimaryReport, getExplanation, getTerrainClasses } from "@/lib/data"
import type { ExplanationSection } from "@/lib/types"
import { cn } from "@/lib/utils"

function SegmentationOverlay() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const report = getPrimaryReport()
  const classes = getTerrainClasses()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const W = 480
    const H = 320
    canvas.width = W
    canvas.height = H

    const bg = ctx.createLinearGradient(0, 0, 0, H)
    bg.addColorStop(0, "#1e3a5f")
    bg.addColorStop(0.35, "#5b8fb9")
    bg.addColorStop(0.4, "#c4a35a")
    bg.addColorStop(1, "#8b7355")
    ctx.fillStyle = bg
    ctx.fillRect(0, 0, W, H)

    const regions = [
      { cls: "sky", x: 0, y: 0, w: W, h: H * 0.38 },
      { cls: "landscape", x: 0, y: H * 0.38, w: W, h: H * 0.3 },
      { cls: "dry_grass", x: 0, y: H * 0.68, w: W * 0.45, h: H * 0.32 },
      { cls: "tree", x: W * 0.05, y: H * 0.2, w: W * 0.15, h: H * 0.35 },
      { cls: "tree", x: W * 0.75, y: H * 0.18, w: W * 0.18, h: H * 0.4 },
      { cls: "rock", x: W * 0.5, y: H * 0.7, w: W * 0.2, h: H * 0.15 },
      { cls: "log", x: W * 0.3, y: H * 0.78, w: W * 0.25, h: H * 0.08 },
      { cls: "dry_bush", x: W * 0.6, y: H * 0.55, w: W * 0.15, h: H * 0.15 },
      { cls: "flower", x: W * 0.42, y: H * 0.62, w: W * 0.08, h: H * 0.08 },
      { cls: "lush_bush", x: W * 0.2, y: H * 0.5, w: W * 0.12, h: H * 0.12 },
    ]

    regions.forEach((r) => {
      const tc = classes.find((t) => t.key === r.cls)
      if (!tc) return
      ctx.fillStyle = tc.color + "99"
      ctx.beginPath()
      const rx = 8
      ctx.moveTo(r.x + rx, r.y)
      ctx.lineTo(r.x + r.w - rx, r.y)
      ctx.quadraticCurveTo(r.x + r.w, r.y, r.x + r.w, r.y + rx)
      ctx.lineTo(r.x + r.w, r.y + r.h - rx)
      ctx.quadraticCurveTo(r.x + r.w, r.y + r.h, r.x + r.w - rx, r.y + r.h)
      ctx.lineTo(r.x + rx, r.y + r.h)
      ctx.quadraticCurveTo(r.x, r.y + r.h, r.x, r.y + r.h - rx)
      ctx.lineTo(r.x, r.y + rx)
      ctx.quadraticCurveTo(r.x, r.y, r.x + rx, r.y)
      ctx.closePath()
      ctx.fill()

      ctx.fillStyle = "#ffffffcc"
      ctx.font = "bold 10px Inter, sans-serif"
      ctx.textAlign = "center"
      ctx.fillText(tc.name, r.x + r.w / 2, r.y + r.h / 2 + 4)
    })

    ctx.strokeStyle = "rgba(255,255,255,0.05)"
    ctx.lineWidth = 0.5
    for (let x = 0; x < W; x += 20) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
    }
    for (let y = 0; y < H; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
    }
  }, [classes])

  return (
    <div className="space-y-4">
      <div className="glass-card rounded-xl overflow-hidden">
        <canvas ref={canvasRef} className="w-full" style={{ aspectRatio: "480/320" }} />
      </div>
      <div>
        <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
          Color Legend
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {classes.map((tc) => {
            const iouVal = report.metrics.per_class_iou[tc.key]
            return (
              <div key={tc.key} className="flex items-center gap-2 bg-secondary/50 rounded-lg px-3 py-2">
                <div className="w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: tc.color }} />
                <div className="flex flex-col min-w-0">
                  <span className="text-xs text-foreground truncate">{tc.name}</span>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {iouVal !== null ? (iouVal * 100).toFixed(1) + "%" : "N/A"}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function PathPlanning() {
  const costCanvasRef = useRef<HTMLCanvasElement>(null)
  const pathCanvasRef = useRef<HTMLCanvasElement>(null)
  const report = getPrimaryReport()

  useEffect(() => {
    const costCanvas = costCanvasRef.current
    if (costCanvas) {
      const ctx = costCanvas.getContext("2d")
      if (ctx) {
        const W = 300, H = 240
        costCanvas.width = W; costCanvas.height = H
        const cols = 20, rows = 16
        const cw = W / cols, ch = H / rows

        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const seed = Math.sin(r * 127.1 + c * 311.7) * 43758.5453
            const cost = seed - Math.floor(seed)
            let color: string
            if (cost > 0.85) color = `rgba(239, 68, 68, ${0.4 + cost * 0.4})`
            else if (cost > 0.5) color = `rgba(234, 179, 8, ${0.3 + cost * 0.3})`
            else color = `rgba(34, 197, 94, ${0.2 + cost * 0.3})`
            ctx.fillStyle = color
            ctx.fillRect(c * cw, r * ch, cw - 1, ch - 1)
          }
        }

        const grad = ctx.createLinearGradient(10, H - 20, W - 10, H - 20)
        grad.addColorStop(0, "#22c55e"); grad.addColorStop(0.5, "#eab308"); grad.addColorStop(1, "#ef4444")
        ctx.fillStyle = grad; ctx.fillRect(10, H - 18, W - 20, 8)
        ctx.fillStyle = "#ffffffaa"; ctx.font = "9px Inter, sans-serif"
        ctx.fillText("Low cost", 10, H - 22)
        ctx.textAlign = "right"; ctx.fillText("High cost", W - 10, H - 22)
      }
    }

    const pathCanvas = pathCanvasRef.current
    if (pathCanvas) {
      const ctx = pathCanvas.getContext("2d")
      if (ctx) {
        const W = 300, H = 240
        pathCanvas.width = W; pathCanvas.height = H
        const bg = ctx.createLinearGradient(0, 0, 0, H)
        bg.addColorStop(0, "#1e293b"); bg.addColorStop(0.4, "#334155"); bg.addColorStop(1, "#1e293b")
        ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H)

        for (let i = 0; i < 200; i++) {
          const s1 = Math.sin(i * 43.7) * 9999, s2 = Math.sin(i * 73.1) * 9999
          ctx.fillStyle = "rgba(100,116,139,0.3)"
          ctx.beginPath(); ctx.arc((s1 - Math.floor(s1)) * W, (s2 - Math.floor(s2)) * H, 1.5, 0, Math.PI * 2); ctx.fill()
        }

        const pts: [number, number][] = [
          [30, 200], [50, 180], [70, 160], [90, 145], [110, 130],
          [130, 115], [145, 100], [160, 85], [175, 75], [195, 60],
          [215, 50], [235, 45], [255, 42], [270, 40],
        ]

        ctx.strokeStyle = "rgba(34,197,94,0.3)"; ctx.lineWidth = 8
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1])
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1])
        ctx.stroke()

        ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2.5; ctx.setLineDash([6, 3])
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1])
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1])
        ctx.stroke(); ctx.setLineDash([])

        ctx.fillStyle = "#22c55e"; ctx.beginPath(); ctx.arc(30, 200, 8, 0, Math.PI * 2); ctx.fill()
        ctx.fillStyle = "#ffffff"; ctx.font = "bold 8px Inter, sans-serif"; ctx.textAlign = "center"; ctx.fillText("S", 30, 203)

        const gx = 270, gy = 40
        ctx.fillStyle = "#ef4444"; ctx.beginPath()
        for (let i = 0; i < 10; i++) {
          const r = i % 2 === 0 ? 10 : 5
          const a = (Math.PI / 5) * i - Math.PI / 2
          const x = gx + Math.cos(a) * r, y = gy + Math.sin(a) * r
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
        }
        ctx.closePath(); ctx.fill()
        ctx.fillStyle = "#ffffff"; ctx.font = "bold 8px Inter, sans-serif"; ctx.fillText("G", gx, gy + 3)

        ctx.fillStyle = "#94a3b8"; ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "left"
        ctx.fillText(`Path cost: ${report.navigation.path_cost} · ${report.navigation.path_length} steps`, 10, H - 8)
      }
    }
  }, [report])

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card rounded-xl overflow-hidden">
          <div className="p-3 border-b border-border">
            <h4 className="text-sm font-medium text-foreground">Traversability Cost Map</h4>
          </div>
          <canvas ref={costCanvasRef} className="w-full" style={{ aspectRatio: "300/240" }} />
        </div>
        <div className="glass-card rounded-xl overflow-hidden">
          <div className="p-3 border-b border-border">
            <h4 className="text-sm font-medium text-foreground">Planned Path (A*)</h4>
          </div>
          <canvas ref={pathCanvasRef} className="w-full" style={{ aspectRatio: "300/240" }} />
        </div>
      </div>
      <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
        <span>Path found: <span className={report.navigation.path_found ? "text-emerald-400" : "text-red-400"}>{report.navigation.path_found ? "Yes" : "No"}</span></span>
        <span>Cost: <span className="text-foreground font-mono">{report.navigation.path_cost}</span></span>
        <span>Length: <span className="text-foreground font-mono">{report.navigation.path_length} steps</span></span>
        <span>Safety: <span className="text-foreground font-mono">{(report.safety.safety_score * 100).toFixed(1)}%</span></span>
      </div>
    </div>
  )
}

function ConfidenceMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const report = getPrimaryReport()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const W = 480, H = 320
    canvas.width = W; canvas.height = H
    const cols = 32, rows = 22
    const cw = W / cols, ch = H / rows

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const seed = Math.sin(r * 127.1 + c * 311.7 + 42.0) * 43758.5453
        let conf = seed - Math.floor(seed)
        if (r < rows * 0.35) conf = Math.min(1, conf + 0.4)
        if (r > rows * 0.4 && r < rows * 0.65) conf = Math.min(1, conf + 0.2)
        if (c < 2 || c > cols - 3 || r > rows - 3) conf *= 0.6

        let red: number, green: number, blue: number
        if (conf < 0.5) {
          const t = conf / 0.5
          red = Math.floor(30 * (1 - t) + 6 * t)
          green = Math.floor(58 * (1 - t) + 182 * t)
          blue = Math.floor(138 * (1 - t) + 212 * t)
        } else {
          const t = (conf - 0.5) / 0.5
          red = Math.floor(6 * (1 - t) + 250 * t)
          green = Math.floor(182 * (1 - t) + 204 * t)
          blue = Math.floor(212 * (1 - t) + 21 * t)
        }
        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`
        ctx.fillRect(c * cw, r * ch, cw, ch)
      }
    }

    const grad = ctx.createLinearGradient(40, H - 24, W - 40, H - 24)
    grad.addColorStop(0, "#1e3a8a"); grad.addColorStop(0.5, "#06b6d4"); grad.addColorStop(1, "#facc15")
    ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(30, H - 34, W - 60, 24)
    ctx.fillStyle = grad; ctx.fillRect(40, H - 28, W - 80, 10)
    ctx.fillStyle = "#ffffffcc"; ctx.font = "10px Inter, sans-serif"
    ctx.textAlign = "left"; ctx.fillText("Low", 42, H - 30)
    ctx.textAlign = "right"; ctx.fillText("High", W - 42, H - 30)
    ctx.textAlign = "center"; ctx.fillText(`Mean: ${(report.segmentation.confidence_mean * 100).toFixed(1)}%`, W / 2, H - 30)
  }, [report])

  return (
    <div className="space-y-4">
      <div className="glass-card rounded-xl overflow-hidden">
        <canvas ref={canvasRef} className="w-full" style={{ aspectRatio: "480/320" }} />
      </div>
      <div className="flex gap-6 text-xs text-muted-foreground">
        <span>Mean: <span className="text-foreground font-mono">{(report.segmentation.confidence_mean * 100).toFixed(1)}%</span></span>
        <span>Std: <span className="text-foreground font-mono">\u00b1{(report.segmentation.confidence_std * 100).toFixed(1)}%</span></span>
      </div>
    </div>
  )
}

function ExplanationSectionCard({ section }: { section: ExplanationSection }) {
  const [classesExpanded, setClassesExpanded] = useState(false)
  const classes = getTerrainClasses()

  return (
    <div className="glass-card rounded-xl p-5 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <h4 className="text-sm font-semibold text-foreground">{section.title}</h4>
        {section.metric && (
          <span className="text-xs font-mono font-semibold text-primary bg-primary/10 px-2.5 py-1 rounded-full whitespace-nowrap shrink-0">
            {section.metric}
          </span>
        )}
      </div>

      <p className="text-sm text-muted-foreground leading-relaxed">
        {section.explanation.replace(/\*\*(.*?)\*\*/g, "$1")}
      </p>

      {section.best_class && section.worst_class && (
        <div className="flex gap-3">
          <div className="flex items-center gap-1.5 text-xs bg-emerald-500/10 text-emerald-400 px-2.5 py-1 rounded-full">
            <TrendingUp className="w-3 h-3" />
            Best: {section.best_class}
          </div>
          <div className="flex items-center gap-1.5 text-xs bg-red-500/10 text-red-400 px-2.5 py-1 rounded-full">
            <TrendingDown className="w-3 h-3" />
            Worst: {section.worst_class}
          </div>
        </div>
      )}

      {section.all_classes && (
        <div>
          <button
            onClick={() => setClassesExpanded(!classesExpanded)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {classesExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            All classes breakdown
          </button>
          {classesExpanded && (
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-2">
              {Object.entries(section.all_classes).map(([key, val]) => {
                const tc = classes.find((c) => c.key === key)
                return (
                  <div key={key} className="flex items-center gap-2 bg-secondary/50 rounded-lg px-2.5 py-1.5">
                    {tc && <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: tc.color }} />}
                    <span className="text-[11px] text-muted-foreground truncate">{tc?.name ?? key}</span>
                    <span className="text-[11px] font-mono text-foreground ml-auto">{val}</span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {section.perturbations && (
        <div className="grid grid-cols-2 gap-2 mt-1">
          {Object.entries(section.perturbations).map(([key, p]) => (
            <div key={key} className="flex items-center justify-between bg-secondary/50 rounded-lg px-3 py-2">
              <span className="text-xs text-muted-foreground capitalize">{key}</span>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-foreground">{p.drop}</span>
                <span className={cn(
                  "text-[10px] px-1.5 py-0.5 rounded-full",
                  p.verdict === "Minimal impact"
                    ? "bg-emerald-500/10 text-emerald-400"
                    : p.verdict === "Moderate impact"
                      ? "bg-yellow-500/10 text-yellow-400"
                      : "bg-red-500/10 text-red-400"
                )}>
                  {p.verdict}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Explanation() {
  const explanation = getExplanation()
  const modelExplanation = explanation.models[0]

  return (
    <div className="space-y-4">
      <div className="glass-card rounded-xl p-5">
        <h3 className="text-sm font-semibold text-primary mb-2">{explanation.title}</h3>
        <p className="text-sm text-muted-foreground leading-relaxed">{explanation.overview}</p>
      </div>
      {modelExplanation?.sections.map((section, i) => (
        <ExplanationSectionCard key={i} section={section} />
      ))}
    </div>
  )
}

function JsonViewer() {
  const [expanded, setExpanded] = useState(false)
  const report = getPrimaryReport()
  const toggleExpanded = useCallback(() => setExpanded((prev) => !prev), [])

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <button
        onClick={toggleExpanded}
        className="flex items-center gap-2 w-full px-4 py-3 text-left hover:bg-secondary/30 transition-colors"
      >
        {expanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
        <span className="text-sm font-medium text-foreground">Raw JSON Output</span>
        <span className="text-xs text-muted-foreground ml-1">benchmark_report.json</span>
      </button>
      {expanded && (
        <div className="border-t border-border p-4 max-h-96 overflow-auto">
          <pre className="text-xs font-mono text-muted-foreground whitespace-pre-wrap">
            <JsonSyntaxHighlight data={report} />
          </pre>
        </div>
      )}
    </div>
  )
}

function JsonSyntaxHighlight({ data }: { data: unknown }) {
  const json = JSON.stringify(data, null, 2)
  const highlighted = json
    .replace(/"([^"]+)":/g, '<span class="text-primary">"$1"</span>:')
    .replace(/: "([^"]+)"/g, ': <span class="text-emerald-400">"$1"</span>')
    .replace(/: (\d+\.?\d*)/g, ': <span class="text-yellow-400">$1</span>')
    .replace(/: (true|false)/g, ': <span class="text-cyan-400">$1</span>')
    .replace(/: (null)/g, ': <span class="text-red-400">$1</span>')
  return <code dangerouslySetInnerHTML={{ __html: highlighted }} />
}

export function ResultsPanel({ visible }: { visible: boolean }) {
  if (!visible) return null

  return (
    <div className={cn("space-y-4 animate-in fade-in slide-in-from-right-4 duration-500")}>
      <Tabs defaultValue="segmentation" className="w-full">
        <TabsList className="w-full justify-start bg-secondary/50 p-1">
          <TabsTrigger value="segmentation" className="text-xs">Segmentation Overlay</TabsTrigger>
          <TabsTrigger value="path" className="text-xs">Path Planning</TabsTrigger>
          <TabsTrigger value="confidence" className="text-xs">Confidence Map</TabsTrigger>
          <TabsTrigger value="explanation" className="text-xs">Explanation</TabsTrigger>
        </TabsList>
        <TabsContent value="segmentation" className="mt-4"><SegmentationOverlay /></TabsContent>
        <TabsContent value="path" className="mt-4"><PathPlanning /></TabsContent>
        <TabsContent value="confidence" className="mt-4"><ConfidenceMap /></TabsContent>
        <TabsContent value="explanation" className="mt-4"><Explanation /></TabsContent>
      </Tabs>
      <JsonViewer />
    </div>
  )
}

"use client"

import { useState, useMemo } from "react"
import { ChevronDown, ChevronRight, Trophy, ArrowUpDown, ShieldAlert, ShieldCheck, TrendingDown } from "lucide-react"
import { getAllReports, getTerrainAsset, getTerrainClasses } from "@/lib/data"
import { cn } from "@/lib/utils"

type SortKey = "model_name" | "mean_iou" | "pixel_accuracy" | "dice_coefficient" | "inference_time_ms" | "safety_score"
type SortDir = "asc" | "desc"

export default function ArenaPage() {
  const reports = getAllReports()
  const classes = getTerrainClasses()
  const [sortKey, setSortKey] = useState<SortKey>("mean_iou")
  const [sortDir, setSortDir] = useState<SortDir>("desc")
  const [expandedRow, setExpandedRow] = useState<string | null>(null)
  const [modelA, setModelA] = useState(reports[0]?.model_name ?? "")
  const [modelB, setModelB] = useState(reports[1]?.model_name ?? reports[0]?.model_name ?? "")

  const asset = getTerrainAsset("terrain_01")

  const sorted = useMemo(() => {
    const arr = [...reports]
    arr.sort((a, b) => {
      let av: number, bv: number
      switch (sortKey) {
        case "model_name": return sortDir === "asc" ? a.model_name.localeCompare(b.model_name) : b.model_name.localeCompare(a.model_name)
        case "mean_iou": av = a.metrics.mean_iou; bv = b.metrics.mean_iou; break
        case "pixel_accuracy": av = a.metrics.pixel_accuracy; bv = b.metrics.pixel_accuracy; break
        case "dice_coefficient": av = a.metrics.dice_coefficient; bv = b.metrics.dice_coefficient; break
        case "inference_time_ms": av = a.inference_time_ms; bv = b.inference_time_ms; break
        case "safety_score": av = a.safety.safety_score; bv = b.safety.safety_score; break
        default: return 0
      }
      return sortDir === "asc" ? av - bv : bv - av
    })
    return arr
  }, [reports, sortKey, sortDir])

  const bestValues = useMemo(() => {
    const best: Record<string, number> = {}
    const metrics = ["mean_iou", "pixel_accuracy", "dice_coefficient", "safety_score"]
    metrics.forEach((m) => {
      best[m] = Math.max(...reports.map((r) => {
        if (m === "safety_score") return r.safety.safety_score
        return (r.metrics as Record<string, number>)[m] ?? 0
      }))
    })
    best.inference_time_ms = Math.min(...reports.map((r) => r.inference_time_ms))
    return best
  }, [reports])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc")
    } else {
      setSortKey(key)
      setSortDir(key === "inference_time_ms" ? "asc" : "desc")
    }
  }

  const modelColors = ["#F59E0B", "#06B6D4", "#10B981", "#8B5CF6"]

  const robustnessAnalysis = useMemo(() => {
    const perturbations = Object.keys(reports[0]?.robustness ?? {})
    const perModel = reports.map((r, i) => {
      const drops = perturbations.map(p => r.robustness[p]?.miou_drop ?? 0)
      const avgDrop = drops.length > 0 ? drops.reduce((a, b) => a + b, 0) / drops.length : 0
      const worstPert = perturbations[drops.indexOf(Math.max(...drops))] ?? "none"
      const worstDrop = Math.max(...drops)
      return { model: r.model_name, color: modelColors[i % modelColors.length], avgDrop, worstPert, worstDrop, drops }
    })
    const perPerturbation = perturbations.map((p) => ({
      name: p,
      label: p.charAt(0).toUpperCase() + p.slice(1),
      models: reports.map((r, i) => ({
        name: r.model_name,
        color: modelColors[i % modelColors.length],
        drop: r.robustness[p]?.miou_drop ?? 0,
        mIoU: r.robustness[p]?.mean_iou ?? 0,
        pixelAcc: r.robustness[p]?.pixel_accuracy ?? 0,
      }))
    }))
    const maxDrop = Math.max(...perPerturbation.flatMap(p => p.models.map(m => Math.abs(m.drop))), 0.01)
    return { perModel, perPerturbation, maxDrop }
  }, [reports])

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground tracking-tight">
          Model Arena
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Side-by-side model comparison on real terrain
        </p>
      </div>

      {/* Split terrain view */}
      <div className="hud-panel rounded-xl p-5">
        <h2 className="text-base font-semibold text-foreground mb-4">Segmentation Comparison</h2>
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-3 flex-1">
            <span className="text-sm font-medium text-muted-foreground shrink-0">Model A</span>
            <select
              value={modelA}
              onChange={(e) => setModelA(e.target.value)}
              className="bg-input border border-border rounded-lg px-3 h-9 text-sm font-mono text-foreground flex-1 min-w-0"
            >
              {reports.map((r) => (
                <option key={r.model_name} value={r.model_name}>{r.model_name}</option>
              ))}
            </select>
          </div>
          <div className="text-xs font-bold text-muted-foreground bg-secondary/50 px-3 py-1.5 rounded-lg shrink-0">VS</div>
          <div className="flex items-center gap-3 flex-1">
            <span className="text-sm font-medium text-muted-foreground shrink-0">Model B</span>
            <select
              value={modelB}
              onChange={(e) => setModelB(e.target.value)}
              className="bg-input border border-border rounded-lg px-3 h-9 text-sm font-mono text-foreground flex-1 min-w-0"
            >
              {reports.map((r) => (
                <option key={r.model_name} value={r.model_name}>{r.model_name}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg overflow-hidden relative hud-panel">
            <img src={asset.maskOverlay} alt="Model A" className="w-full h-auto" />
            <div className="absolute bottom-2 left-2 px-2.5 py-1 rounded-md bg-black/70 text-xs font-mono text-amber-400">
              {modelA}
            </div>
          </div>
          <div className="rounded-lg overflow-hidden relative hud-panel">
            <img src={asset.maskOverlay} alt="Model B" className="w-full h-auto opacity-90" style={{ filter: "hue-rotate(30deg)" }} />
            <div className="absolute bottom-2 left-2 px-2.5 py-1 rounded-md bg-black/70 text-xs font-mono text-cyan-400">
              {modelB}
            </div>
          </div>
        </div>
      </div>

      {/* Comparison table */}
      <div className="hud-panel rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-border">
          <h2 className="text-base font-semibold text-foreground">Model Metrics</h2>
          <p className="text-sm text-muted-foreground mt-0.5">Click a column header to sort, click a row to expand details</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-secondary/20">
                {[
                  { key: "model_name" as SortKey, label: "Model" },
                  { key: "mean_iou" as SortKey, label: "mIoU" },
                  { key: "pixel_accuracy" as SortKey, label: "Pixel Acc" },
                  { key: "dice_coefficient" as SortKey, label: "Dice" },
                  { key: "inference_time_ms" as SortKey, label: "Latency" },
                  { key: "safety_score" as SortKey, label: "Safety" },
                ].map((col) => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wide cursor-pointer hover:text-foreground transition-colors"
                  >
                    <div className="flex items-center gap-1.5">
                      {col.label}
                      {sortKey === col.key ? (
                        <span className="text-primary text-sm">{sortDir === "asc" ? "↑" : "↓"}</span>
                      ) : (
                        <ArrowUpDown className="w-3 h-3 opacity-30" />
                      )}
                    </div>
                  </th>
                ))}
                <th className="px-4 py-3 w-10" />
              </tr>
            </thead>
            {sorted.map((r) => {
              const isExpanded = expandedRow === r.model_name
              return (
                <tbody key={r.model_name}>
                  <tr
                    className="border-b border-border/50 hover:bg-secondary/20 cursor-pointer transition-colors"
                    onClick={() => setExpandedRow(isExpanded ? null : r.model_name)}
                  >
                    <td className="px-4 py-3 text-sm font-mono font-medium text-foreground">{r.model_name}</td>
                    <td className={cn("px-4 py-3 text-sm font-mono", r.metrics.mean_iou === bestValues.mean_iou && "text-amber-400 font-bold")}>
                      {(r.metrics.mean_iou * 100).toFixed(1)}%
                      {r.metrics.mean_iou === bestValues.mean_iou && <Trophy className="w-3.5 h-3.5 inline ml-1.5" />}
                    </td>
                    <td className={cn("px-4 py-3 text-sm font-mono", r.metrics.pixel_accuracy === bestValues.pixel_accuracy && "text-amber-400 font-bold")}>
                      {(r.metrics.pixel_accuracy * 100).toFixed(1)}%
                    </td>
                    <td className={cn("px-4 py-3 text-sm font-mono", r.metrics.dice_coefficient === bestValues.dice_coefficient && "text-amber-400 font-bold")}>
                      {(r.metrics.dice_coefficient * 100).toFixed(1)}%
                    </td>
                    <td className={cn("px-4 py-3 text-sm font-mono", r.inference_time_ms === bestValues.inference_time_ms && "text-emerald-400 font-bold")}>
                      {r.inference_time_ms.toFixed(0)}ms
                    </td>
                    <td className={cn("px-4 py-3 text-sm font-mono", r.safety.safety_score === bestValues.safety_score && "text-amber-400 font-bold")}>
                      {(r.safety.safety_score * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3">
                      {isExpanded
                        ? <ChevronDown className="w-4 h-4 text-muted-foreground" />
                        : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr>
                      <td colSpan={7} className="px-5 py-5 bg-secondary/10 border-b border-border/30">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          {/* Per-Class IoU */}
                          <div className="hud-panel rounded-lg p-4">
                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Per-Class IoU</h3>
                            <div className="space-y-2.5">
                              {classes.map((c) => {
                                const val = r.metrics.per_class_iou[c.key]
                                return (
                                  <div key={c.key} className="flex items-center gap-3">
                                    <div className="w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: c.color }} />
                                    <span className="text-sm text-foreground flex-1">{c.name}</span>
                                    <span className="text-sm font-mono text-muted-foreground">
                                      {val === null ? "N/A" : (val * 100).toFixed(1) + "%"}
                                    </span>
                                  </div>
                                )
                              })}
                            </div>
                          </div>

                          {/* Navigation */}
                          <div className="hud-panel rounded-lg p-4">
                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Navigation</h3>
                            <div className="space-y-2.5">
                              {[
                                { l: "Path Found", v: r.navigation.path_found ? "YES" : "NO", color: r.navigation.path_found ? "text-emerald-400" : "text-red-400" },
                                { l: "Path Cost", v: r.navigation.path_cost.toFixed(0), color: "text-amber-400" },
                                { l: "Path Length", v: `${r.navigation.path_length} steps`, color: "text-amber-400" },
                                { l: "Safety Score", v: (r.safety.safety_score * 100).toFixed(1) + "%", color: "text-amber-400" },
                                { l: "Obstacle Overlap", v: (r.safety.obstacle_overlap_pct * 100).toFixed(1) + "%", color: "text-amber-400" },
                              ].map((item) => (
                                <div key={item.l} className="flex items-center justify-between">
                                  <span className="text-sm text-muted-foreground">{item.l}</span>
                                  <span className={cn("text-sm font-mono font-medium", item.color)}>{item.v}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Robustness */}
                          <div className="hud-panel rounded-lg p-4">
                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Robustness</h3>
                            <div className="space-y-2.5">
                              {Object.entries(r.robustness).map(([pert, res]) => (
                                <div key={pert} className="flex items-center justify-between">
                                  <span className="text-sm text-muted-foreground capitalize">{pert}</span>
                                  <span className={cn(
                                    "text-sm font-mono font-medium",
                                    res.miou_drop > 0 ? "text-red-400" : "text-emerald-400"
                                  )}>
                                    {res.miou_drop > 0 ? "-" : "+"}
                                    {(Math.abs(res.miou_drop) * 100).toFixed(2)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </tbody>
              )
            })}
          </table>
        </div>
      </div>

      {/* Robustness Comparison */}
      <div className="space-y-5">
        <div>
          <h2 className="text-lg font-bold text-foreground tracking-tight">Robustness Comparison</h2>
          <p className="text-sm text-muted-foreground mt-1">How each model degrades under environmental perturbations</p>
        </div>

        {/* Model summary cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {robustnessAnalysis.perModel.map((m) => {
            const severity = m.avgDrop > 0.05 ? "critical" : m.avgDrop > 0.02 ? "moderate" : "robust"
            return (
              <div key={m.model} className="hud-panel rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: m.color }} />
                  <span className="text-sm font-mono font-medium text-foreground truncate">{m.model}</span>
                </div>
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Avg. Drop</span>
                    <span className={cn(
                      "text-sm font-mono font-bold",
                      severity === "critical" ? "text-red-400" : severity === "moderate" ? "text-amber-400" : "text-emerald-400"
                    )}>
                      {(m.avgDrop * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Worst</span>
                    <span className="text-sm font-mono text-red-400 capitalize">{m.worstPert}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Verdict</span>
                    <span className={cn(
                      "inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-md",
                      severity === "critical" ? "text-red-400 bg-red-400/10"
                        : severity === "moderate" ? "text-amber-400 bg-amber-400/10"
                        : "text-emerald-400 bg-emerald-400/10"
                    )}>
                      {severity === "robust"
                        ? <><ShieldCheck className="w-3 h-3" />Robust</>
                        : severity === "moderate"
                        ? <><ShieldAlert className="w-3 h-3" />Moderate</>
                        : <><TrendingDown className="w-3 h-3" />Fragile</>}
                    </span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* Per-perturbation breakdown */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {robustnessAnalysis.perPerturbation.map((pert) => (
            <div key={pert.name} className="hud-panel rounded-xl p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-foreground capitalize">{pert.label}</h3>
                <span className="text-xs text-muted-foreground">mIoU drop</span>
              </div>
              <div className="space-y-3">
                {pert.models.map((m) => {
                  const absDrop = Math.abs(m.drop)
                  const barWidth = Math.min((absDrop / robustnessAnalysis.maxDrop) * 100, 100)
                  const isNegative = m.drop > 0
                  return (
                    <div key={m.name} className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: m.color }} />
                          <span className="text-sm text-foreground font-mono">{m.name}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-xs text-muted-foreground">
                            mIoU: {(m.mIoU * 100).toFixed(1)}%
                          </span>
                          <span className={cn(
                            "text-sm font-mono font-bold min-w-[60px] text-right",
                            isNegative ? "text-red-400" : "text-emerald-400"
                          )}>
                            {isNegative ? "-" : "+"}{(absDrop * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                      <div className="h-2 bg-secondary/40 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${barWidth}%`,
                            backgroundColor: isNegative
                              ? absDrop > 0.05 ? "#EF4444" : absDrop > 0.02 ? "#F59E0B" : "#10B981"
                              : "#10B981",
                          }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

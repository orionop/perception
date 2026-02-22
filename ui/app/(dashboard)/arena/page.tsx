"use client"

import { useState, useMemo } from "react"
import { ChevronDown, ChevronRight, Trophy, ArrowUpDown, TrendingUp } from "lucide-react"
import { getAllReports, getTerrainClasses } from "@/lib/data"
import { cn } from "@/lib/utils"
import Image from "next/image"

type SortKey = "model_name" | "mean_iou" | "pixel_accuracy" | "dice_coefficient" | "inference_time_ms" | "safety_score"
type SortDir = "asc" | "desc"

const V3_METRICS = { miou: 35, pixel: 55, classes: "sky, landscape" }
const ENSEMBLE_METRICS = { miou: 61, pixel: 81.2, classes: "sky, landscape, tree, rock, dry_grass, dry_bush" }

export default function ArenaPage() {
  const reports = getAllReports()
  const classes = getTerrainClasses()
  const [sortKey, setSortKey] = useState<SortKey>("mean_iou")
  const [sortDir, setSortDir] = useState<SortDir>("desc")
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

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

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground tracking-tight">
          Model Arena
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Side-by-side comparison: DINOv2 V3 (baseline) vs V6 Ensemble (best)
        </p>
      </div>

      {/* Visual Comparison — V3 vs Ensemble on image 0000120 */}
      <div className="hud-panel rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-foreground">Segmentation Comparison — Test Image 0000096</h2>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <span className="text-sm font-mono text-emerald-400 font-bold">+26% mIoU improvement</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          {/* V3 side */}
          <div className="rounded-lg overflow-hidden relative hud-panel">
            <div className="relative w-full aspect-video">
              <Image
                src="/terrain/v3_overlay.png"
                alt="V3 Segmentation"
                fill
                className="object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
            </div>
            <div className="absolute top-2 left-2 px-2.5 py-1 rounded-md bg-red-500/20 border border-red-500/30 text-xs font-mono text-red-400 font-bold">
              V3 — DINOv2 Baseline
            </div>
            <div className="absolute bottom-2 left-2 right-2 flex justify-between">
              <span className="px-2 py-1 rounded-md bg-black/70 text-xs font-mono text-red-400">35% mIoU</span>
              <span className="px-2 py-1 rounded-md bg-black/70 text-xs font-mono text-muted-foreground">55% Pixel Acc</span>
            </div>
          </div>

          {/* Ensemble side */}
          <div className="rounded-lg overflow-hidden relative hud-panel">
            <div className="relative w-full aspect-video">
              <Image
                src="/terrain/ensemble_overlay.png"
                alt="Ensemble Segmentation"
                fill
                className="object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
            </div>
            <div className="absolute top-2 left-2 px-2.5 py-1 rounded-md bg-emerald-500/20 border border-emerald-500/30 text-xs font-mono text-emerald-400 font-bold">
              Ensemble — V3+V5+V6+Bayesian
            </div>
            <div className="absolute bottom-2 left-2 right-2 flex justify-between">
              <span className="px-2 py-1 rounded-md bg-black/70 text-xs font-mono text-emerald-400 font-bold">61% mIoU</span>
              <span className="px-2 py-1 rounded-md bg-black/70 text-xs font-mono text-emerald-400">81.2% Pixel Acc</span>
            </div>
          </div>
        </div>

        {/* Improvement breakdown */}
        <div className="grid grid-cols-3 gap-3">
          <div className="hud-panel rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground uppercase tracking-wide mb-1">mIoU Gain</div>
            <div className="text-lg font-mono font-bold text-emerald-400">+26%</div>
            <div className="text-xs text-muted-foreground">35% → 61%</div>
          </div>
          <div className="hud-panel rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Pixel Accuracy Gain</div>
            <div className="text-lg font-mono font-bold text-emerald-400">+26.2%</div>
            <div className="text-xs text-muted-foreground">55% → 81.2%</div>
          </div>
          <div className="hud-panel rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Classes Detected</div>
            <div className="text-lg font-mono font-bold text-emerald-400">2 → 7</div>
            <div className="text-xs text-muted-foreground">V3 only saw sky+landscape</div>
          </div>
        </div>
      </div>

      {/* Model Metrics Table */}
      <div className="hud-panel rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-border">
          <h2 className="text-base font-semibold text-foreground">Model Metrics</h2>
          <p className="text-sm text-muted-foreground mt-0.5">Click a column to sort, click a row to expand</p>
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
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          {/* Per-Class IoU — top 5 only */}
                          <div className="hud-panel rounded-lg p-4">
                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Top Per-Class IoU</h3>
                            <div className="space-y-2.5">
                              {classes
                                .filter((c) => {
                                  const val = r.metrics.per_class_iou[c.key]
                                  return val !== null && val > 0
                                })
                                .sort((a, b) => (r.metrics.per_class_iou[b.key] ?? 0) - (r.metrics.per_class_iou[a.key] ?? 0))
                                .slice(0, 5)
                                .map((c) => {
                                  const val = r.metrics.per_class_iou[c.key]
                                  return (
                                    <div key={c.key} className="flex items-center gap-3">
                                      <div className="w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: c.color }} />
                                      <span className="text-sm text-foreground flex-1">{c.name}</span>
                                      <span className="text-sm font-mono text-amber-400 font-bold">
                                        {val === null ? "N/A" : (val * 100).toFixed(1) + "%"}
                                      </span>
                                    </div>
                                  )
                                })}
                            </div>
                          </div>

                          {/* Navigation + Robustness */}
                          <div className="hud-panel rounded-lg p-4">
                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Navigation & Robustness</h3>
                            <div className="space-y-2.5">
                              {[
                                { l: "Path Found", v: r.navigation.path_found ? "YES" : "NO", color: r.navigation.path_found ? "text-emerald-400" : "text-red-400" },
                                { l: "Path Cost", v: r.navigation.path_cost.toFixed(0), color: "text-amber-400" },
                                { l: "Safety Score", v: (r.safety.safety_score * 100).toFixed(1) + "%", color: "text-amber-400" },
                              ].map((item) => (
                                <div key={item.l} className="flex items-center justify-between">
                                  <span className="text-sm text-muted-foreground">{item.l}</span>
                                  <span className={cn("text-sm font-mono font-medium", item.color)}>{item.v}</span>
                                </div>
                              ))}
                              <div className="border-t border-border/30 pt-2 mt-2">
                                {Object.entries(r.robustness).map(([pert, res]) => (
                                  <div key={pert} className="flex items-center justify-between py-1">
                                    <span className="text-sm text-muted-foreground capitalize">{pert}</span>
                                    <span className={cn(
                                      "text-sm font-mono font-medium",
                                      res.miou_drop < 0 ? "text-emerald-400" : "text-red-400"
                                    )}>
                                      {res.miou_drop < 0 ? "" : "-"}{(Math.abs(res.miou_drop) * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                ))}
                              </div>
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
    </div>
  )
}

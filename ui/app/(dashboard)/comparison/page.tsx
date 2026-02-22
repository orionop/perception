"use client"

import { useState, useMemo, useCallback, Fragment } from "react"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { ChevronDown, ChevronRight, Download, ArrowUpDown, Trophy } from "lucide-react"
import { getAllReports, getTerrainClasses } from "@/lib/data"
import type { ModelReport } from "@/lib/types"
import { cn } from "@/lib/utils"

type SortKey =
  | "model_name"
  | "mean_iou"
  | "frequency_weighted_iou"
  | "dice_coefficient"
  | "pixel_accuracy"
  | "inference_time_ms"
  | "path_found"
  | "path_cost"
  | "safety_score"

function getValue(report: ModelReport, key: SortKey): number | boolean | string {
  switch (key) {
    case "model_name": return report.model_name
    case "mean_iou": return report.metrics.mean_iou
    case "frequency_weighted_iou": return report.metrics.frequency_weighted_iou
    case "dice_coefficient": return report.metrics.dice_coefficient
    case "pixel_accuracy": return report.metrics.pixel_accuracy
    case "inference_time_ms": return report.inference_time_ms
    case "path_found": return report.navigation.path_found
    case "path_cost": return report.navigation.path_cost
    case "safety_score": return report.safety.safety_score
  }
}

export default function ComparisonPage() {
  const reports = getAllReports()
  const classes = getTerrainClasses()
  const [sortKey, setSortKey] = useState<SortKey>("mean_iou")
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc")
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set())

  const handleSort = useCallback((key: SortKey) => {
    if (sortKey === key) setSortDir(sortDir === "asc" ? "desc" : "asc")
    else { setSortKey(key); setSortDir("desc") }
  }, [sortKey, sortDir])

  const toggleRow = useCallback((idx: number) => {
    setExpandedRows((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx); else next.add(idx)
      return next
    })
  }, [])

  const bestValues = useMemo(() => {
    const keys: SortKey[] = ["mean_iou", "frequency_weighted_iou", "dice_coefficient", "pixel_accuracy", "safety_score"]
    const bests: Record<string, number> = {}
    for (const k of keys) {
      bests[k] = Math.max(...reports.map(r => getValue(r, k) as number))
    }
    bests["inference_time_ms"] = Math.min(...reports.map(r => r.inference_time_ms))
    return bests
  }, [reports])

  const sorted = useMemo(() => {
    const arr = [...reports]
    arr.sort((a, b) => {
      const aVal = getValue(a, sortKey)
      const bVal = getValue(b, sortKey)
      if (typeof aVal === "boolean") return sortDir === "desc" ? (aVal ? -1 : 1) : aVal ? 1 : -1
      if (typeof aVal === "string") return sortDir === "desc" ? (bVal as string).localeCompare(aVal) : aVal.localeCompare(bVal as string)
      return sortDir === "desc" ? (bVal as number) - (aVal as number) : (aVal as number) - (bVal as number)
    })
    return arr
  }, [reports, sortKey, sortDir])

  const worstRobustness = useCallback((r: ModelReport) => {
    let worst = { key: "", drop: -Infinity }
    for (const [k, v] of Object.entries(r.robustness)) {
      if (v.miou_drop > worst.drop) worst = { key: k, drop: v.miou_drop }
    }
    return worst
  }, [])

  const exportCSV = useCallback(() => {
    const headers = ["Model", "mIoU", "fwIoU", "Dice", "Pixel Acc", "Latency (ms)", "Path Found", "Path Cost", "Safety Score"]
    const rows = reports.map((r) => [
      r.model_name, r.metrics.mean_iou, r.metrics.frequency_weighted_iou, r.metrics.dice_coefficient,
      r.metrics.pixel_accuracy, r.inference_time_ms, r.navigation.path_found, r.navigation.path_cost, r.safety.safety_score,
    ])
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n")
    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a"); a.href = url; a.download = "model_comparison.csv"; a.click()
    URL.revokeObjectURL(url)
  }, [reports])

  const exportJSON = useCallback(() => {
    const blob = new Blob([JSON.stringify(reports, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a"); a.href = url; a.download = "model_comparison.json"; a.click()
    URL.revokeObjectURL(url)
  }, [reports])

  function isBest(value: number, key: string) {
    return bestValues[key] !== undefined && value === bestValues[key]
  }

  function SortableHeader({ label, sortKeyName }: { label: string; sortKeyName: SortKey }) {
    return (
      <TableHead className="cursor-pointer hover:text-foreground transition-colors select-none" onClick={() => handleSort(sortKeyName)}>
        <div className="flex items-center gap-1">
          {label}
          <ArrowUpDown className={cn("w-3 h-3", sortKey === sortKeyName ? "text-primary" : "text-muted-foreground/50")} />
        </div>
      </TableHead>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground text-balance">Model Comparison</h1>
          <p className="text-sm text-muted-foreground mt-1">Compare segmentation models side by side</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={exportCSV} className="border-border text-foreground hover:bg-secondary">
            <Download className="w-4 h-4 mr-1.5" />CSV
          </Button>
          <Button variant="outline" size="sm" onClick={exportJSON} className="border-border text-foreground hover:bg-secondary">
            <Download className="w-4 h-4 mr-1.5" />JSON
          </Button>
        </div>
      </div>

      <div className="glass-card rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-transparent">
                <TableHead className="w-8" />
                <SortableHeader label="Model" sortKeyName="model_name" />
                <SortableHeader label="mIoU" sortKeyName="mean_iou" />
                <SortableHeader label="fwIoU" sortKeyName="frequency_weighted_iou" />
                <SortableHeader label="Dice" sortKeyName="dice_coefficient" />
                <SortableHeader label="Pixel Acc" sortKeyName="pixel_accuracy" />
                <SortableHeader label="Latency" sortKeyName="inference_time_ms" />
                <SortableHeader label="Path" sortKeyName="path_found" />
                <SortableHeader label="Safety" sortKeyName="safety_score" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {sorted.map((r, idx) => {
                const worst = worstRobustness(r)
                return (
                  <Fragment key={r.model_name}>
                    <TableRow className="border-border hover:bg-secondary/30 cursor-pointer transition-colors" onClick={() => toggleRow(idx)}>
                      <TableCell className="w-8">
                        {expandedRows.has(idx) ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
                      </TableCell>
                      <TableCell className="font-medium text-foreground">
                        <div className="flex items-center gap-2">
                          {r.model_name}
                          {isBest(r.metrics.mean_iou, "mean_iou") && <Trophy className="w-3.5 h-3.5 text-yellow-400" />}
                        </div>
                      </TableCell>
                      <TableCell className={cn("font-mono text-sm", isBest(r.metrics.mean_iou, "mean_iou") && "text-yellow-400 font-semibold")}>
                        {(r.metrics.mean_iou * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className={cn("font-mono text-sm", isBest(r.metrics.frequency_weighted_iou, "frequency_weighted_iou") && "text-yellow-400 font-semibold")}>
                        {(r.metrics.frequency_weighted_iou * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className={cn("font-mono text-sm", isBest(r.metrics.dice_coefficient, "dice_coefficient") && "text-yellow-400 font-semibold")}>
                        {(r.metrics.dice_coefficient * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className={cn("font-mono text-sm", isBest(r.metrics.pixel_accuracy, "pixel_accuracy") && "text-yellow-400 font-semibold")}>
                        {(r.metrics.pixel_accuracy * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell className={cn("font-mono text-sm", isBest(r.inference_time_ms, "inference_time_ms") && "text-yellow-400 font-semibold")}>
                        {r.inference_time_ms.toFixed(1)}ms
                      </TableCell>
                      <TableCell>
                        <span className={cn("text-xs font-medium px-2 py-0.5 rounded-full", r.navigation.path_found ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400")}>
                          {r.navigation.path_found ? "Yes" : "No"}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className={cn("font-mono text-sm", r.safety.safety_score > 0.7 ? "text-emerald-400" : r.safety.safety_score >= 0.5 ? "text-yellow-400" : "text-red-400",
                          isBest(r.safety.safety_score, "safety_score") && "font-semibold")}>
                          {r.safety.safety_score.toFixed(3)}
                        </span>
                      </TableCell>
                    </TableRow>

                    {expandedRows.has(idx) && (
                      <TableRow key={`${r.model_name}-expanded`} className="border-border bg-secondary/20">
                        <TableCell colSpan={9} className="p-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {/* Per-class IoU */}
                            <div>
                              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Per-Class IoU</h4>
                              <div className="space-y-2">
                                {classes.map((tc) => {
                                  const val = r.metrics.per_class_iou[tc.key]
                                  return (
                                    <div key={tc.key} className="flex items-center gap-3">
                                      <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: tc.color }} />
                                      <span className="text-xs text-muted-foreground w-24 shrink-0">{tc.name}</span>
                                      <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                                        <div className="h-full rounded-full transition-all" style={{
                                          width: val !== null ? `${val * 100}%` : "0%",
                                          backgroundColor: tc.color,
                                          opacity: val !== null ? 1 : 0.2,
                                        }} />
                                      </div>
                                      <span className="text-xs font-mono text-foreground w-12 text-right">
                                        {val !== null ? (val * 100).toFixed(1) + "%" : "N/A"}
                                      </span>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>

                            {/* Robustness */}
                            <div>
                              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Robustness</h4>
                              <div className="space-y-3">
                                {Object.entries(r.robustness).map(([key, result]) => (
                                  <div key={key} className="bg-secondary/50 rounded-lg px-3 py-2 space-y-1">
                                    <div className="flex items-center justify-between">
                                      <span className="text-xs text-muted-foreground capitalize">{key}</span>
                                      <span className={cn("text-xs font-mono font-semibold",
                                        Math.abs(result.miou_drop) < 0.03 ? "text-emerald-400" : Math.abs(result.miou_drop) < 0.05 ? "text-yellow-400" : "text-red-400"
                                      )}>
                                        {result.miou_drop > 0 ? "-" : "+"}{(Math.abs(result.miou_drop) * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                    <div className="flex gap-3 text-[10px] text-muted-foreground">
                                      <span>pAcc: {(result.pixel_accuracy * 100).toFixed(1)}%</span>
                                      <span>mIoU: {(result.mean_iou * 100).toFixed(1)}%</span>
                                      <span>{result.inference_time_ms.toFixed(0)}ms</span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Navigation & Safety */}
                            <div>
                              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Navigation & Safety</h4>
                              <div className="space-y-2 text-xs">
                                <div className="flex justify-between bg-secondary/50 rounded-lg px-3 py-2">
                                  <span className="text-muted-foreground">Path cost</span>
                                  <span className="font-mono text-foreground">{r.navigation.path_cost || "N/A"}</span>
                                </div>
                                <div className="flex justify-between bg-secondary/50 rounded-lg px-3 py-2">
                                  <span className="text-muted-foreground">Path length</span>
                                  <span className="font-mono text-foreground">{r.navigation.path_length || "N/A"} steps</span>
                                </div>
                                <div className="flex justify-between bg-secondary/50 rounded-lg px-3 py-2">
                                  <span className="text-muted-foreground">Obstacle overlap</span>
                                  <span className="font-mono text-foreground">{r.safety.obstacle_overlap_pct.toFixed(1)}%</span>
                                </div>
                                <div className="flex justify-between bg-secondary/50 rounded-lg px-3 py-2">
                                  <span className="text-muted-foreground">Avg confidence</span>
                                  <span className="font-mono text-foreground">{(r.safety.avg_confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex justify-between bg-secondary/50 rounded-lg px-3 py-2">
                                  <span className="text-muted-foreground">Worst perturbation</span>
                                  <span className="font-mono text-foreground capitalize">{worst.key} (-{(worst.drop * 100).toFixed(1)}%)</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}

"use client"

import { useMemo } from "react"
import { getPrimaryReport, getTerrainClasses } from "@/lib/data"

export function ConfusionMatrix() {
  const report = getPrimaryReport()
  const classes = getTerrainClasses()
  const matrix = report.metrics.confusion_matrix

  const maxVal = useMemo(() => {
    let m = 0
    for (const row of matrix) {
      for (const v of row) {
        if (v > m) m = v
      }
    }
    return m
  }, [matrix])

  function cellIntensity(value: number): string {
    if (value === 0) return "rgba(59, 130, 246, 0)"
    const t = Math.sqrt(value / maxVal)
    return `rgba(59, 130, 246, ${(t * 0.85).toFixed(3)})`
  }

  return (
    <div className="glass-card rounded-xl p-6">
      <h3 className="text-sm font-semibold text-foreground mb-1">
        Confusion Matrix
      </h3>
      <p className="text-xs text-muted-foreground mb-4">
        Predicted vs. ground truth class counts
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-[10px] font-mono">
          <thead>
            <tr>
              <th className="p-1 text-right text-muted-foreground font-normal w-20">
                <span className="text-[9px]">Pred &rarr;</span>
              </th>
              {classes.map((c) => (
                <th
                  key={c.key}
                  className="p-1 text-center text-muted-foreground font-normal"
                  style={{ minWidth: 36 }}
                >
                  <div
                    className="w-2 h-2 rounded-full mx-auto mb-0.5"
                    style={{ backgroundColor: c.color }}
                  />
                  <span className="block truncate max-w-[40px]">{c.name.slice(0, 4)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                <td className="p-1 text-right text-muted-foreground truncate max-w-[72px]">
                  {classes[ri]?.name ?? ri}
                </td>
                {row.map((val, ci) => (
                  <td
                    key={ci}
                    className="p-1 text-center text-foreground relative"
                    style={{ minWidth: 36 }}
                  >
                    <div
                      className="absolute inset-0.5 rounded-sm"
                      style={{ backgroundColor: cellIntensity(val) }}
                    />
                    <span className="relative z-10">
                      {val > 9999 ? `${(val / 1000).toFixed(0)}k` : val}
                    </span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-2 mt-4">
        <span className="text-[10px] text-muted-foreground">Low</span>
        <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-transparent via-blue-500/40 to-blue-500/85" />
        <span className="text-[10px] text-muted-foreground">High</span>
      </div>
    </div>
  )
}

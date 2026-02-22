"use client"

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts"
import { getPrimaryReport } from "@/lib/data"
import { cn } from "@/lib/utils"
import type { ModelReport } from "@/lib/types"

export function RobustnessChart({
  compact = false,
  report: reportProp,
}: { compact?: boolean; report?: ModelReport }) {
  const report = reportProp ?? getPrimaryReport()

  const chartData = Object.entries(report.robustness).map(([key, result]) => ({
    perturbation: key.charAt(0).toUpperCase() + key.slice(1),
    miou_drop: Math.max(0, result.miou_drop),
    mean_iou: result.mean_iou,
  }))

  return (
    <div className="hud-panel rounded-xl p-4 flex flex-col min-h-[180px]">
      <h3 className="text-sm font-semibold text-foreground mb-1">
        Robustness
      </h3>
      <p className="text-xs text-muted-foreground mb-3">
        mIoU drop under perturbations (brightness, blur, noise, contrast)
      </p>
      <div className={compact ? "h-[120px]" : "h-[200px]"}>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={chartData} cx="50%" cy="50%" outerRadius="70%">
            <PolarGrid stroke="oklch(0.20 0.01 250)" />
            <PolarAngleAxis
              dataKey="perturbation"
              tick={{ fill: "oklch(0.55 0.01 250)", fontSize: 9 }}
            />
            {!compact && (
              <PolarRadiusAxis
                angle={90}
                domain={[0, 0.05]}
                tick={{ fill: "oklch(0.40 0.01 250)", fontSize: 8 }}
                axisLine={false}
              />
            )}
            <Tooltip
              contentStyle={{
                backgroundColor: "oklch(0.11 0.01 250)",
                border: "1px solid oklch(0.22 0.01 250)",
                borderRadius: "4px",
                color: "oklch(0.95 0 0)",
                fontSize: "11px",
                fontFamily: "monospace",
              }}
              formatter={(value: number) => [
                (value * 100).toFixed(2) + "%",
                "mIoU Drop",
              ]}
            />
            <Radar
              name="mIoU Drop"
              dataKey="miou_drop"
              stroke="#F59E0B"
              fill="#F59E0B"
              fillOpacity={0.2}
              strokeWidth={1.5}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div className="grid grid-cols-2 gap-2 mt-3">
        {chartData.map((item) => (
          <div
            key={item.perturbation}
            className="flex items-center justify-between bg-secondary/40 rounded-lg px-3 py-2"
          >
            <span className="text-xs text-foreground font-medium">{item.perturbation}</span>
            <span
              className={cn(
                "text-xs font-mono tabular-nums",
                item.miou_drop > 0 ? "text-amber-400" : "text-emerald-400"
              )}
            >
              {item.miou_drop > 0 ? "-" : "+"}
              {(Math.abs(item.miou_drop) * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

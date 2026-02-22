"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import { getPrimaryReport, getTerrainClasses } from "@/lib/data"
import type { ModelReport } from "@/lib/types"

export function PerClassChart({
  compact = false,
  report: reportProp,
}: { compact?: boolean; report?: ModelReport }) {
  const report = reportProp ?? getPrimaryReport()
  const classes = getTerrainClasses()

  const chartData = classes.map((tc) => ({
    name: tc.name,
    iou: report.metrics.per_class_iou[tc.key] ?? 0,
    color: tc.color,
    isNull: report.metrics.per_class_iou[tc.key] === null,
  }))

  if (compact) {
    return (
      <div className="hud-panel rounded-xl p-4 flex flex-col min-h-[180px]">
        <h3 className="text-sm font-semibold text-foreground mb-1">
          Per-Class IoU
        </h3>
        <p className="text-xs text-muted-foreground mb-3">
          Intersection over Union by terrain class
        </p>
        <div className="flex gap-0.5 h-4 rounded overflow-hidden mb-3 flex-1 min-h-0">
          {chartData.map((d, i) => (
            <div
              key={i}
              className="h-full"
              style={{
                flex: d.isNull ? 0.3 : Math.max(0.3, d.iou * 10),
                backgroundColor: d.isNull ? "oklch(0.20 0 0)" : d.color,
                opacity: d.isNull ? 0.4 : 0.85,
              }}
              title={`${d.name}: ${d.isNull ? "N/A" : (d.iou * 100).toFixed(1) + "%"}`}
            />
          ))}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-3 gap-y-1.5">
          {chartData.map((d) => (
            <div key={d.name} className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: d.isNull ? "oklch(0.30 0 0)" : d.color }}
              />
              <span className="text-xs text-foreground truncate">{d.name}</span>
              <span className="text-xs font-mono text-muted-foreground tabular-nums ml-auto shrink-0">
                {d.isNull ? "N/A" : `${(d.iou * 100).toFixed(1)}%`}
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="hud-panel rounded p-4">
      <div className="text-[9px] uppercase tracking-[0.15em] text-muted-foreground mb-1">
        Per-Class IoU
      </div>
      <p className="text-[10px] text-muted-foreground mb-4">
        Intersection over Union by terrain class
      </p>
      <div className="h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 0, right: 20, bottom: 0, left: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="oklch(0.20 0.01 250 / 0.5)"
              horizontal={false}
            />
            <XAxis
              type="number"
              domain={[0, 1]}
              tick={{ fill: "oklch(0.50 0.01 250)", fontSize: 10 }}
              axisLine={{ stroke: "oklch(0.20 0.01 250)" }}
              tickLine={false}
            />
            <YAxis
              dataKey="name"
              type="category"
              width={80}
              tick={{ fill: "oklch(0.50 0.01 250)", fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "oklch(0.11 0.01 250)",
                border: "1px solid oklch(0.22 0.01 250)",
                borderRadius: "4px",
                color: "oklch(0.95 0 0)",
                fontSize: "11px",
                fontFamily: "monospace",
              }}
              formatter={(value: number, _name: string, props: { payload: { isNull: boolean } }) => {
                if (props.payload.isNull) return ["N/A", "IoU"]
                return [(value * 100).toFixed(1) + "%", "IoU"]
              }}
            />
            <Bar dataKey="iou" radius={[0, 2, 2, 0]} barSize={16}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.isNull ? "oklch(0.25 0 0)" : entry.color}
                  fillOpacity={entry.isNull ? 0.3 : 0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

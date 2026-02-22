"use client"

import { cn } from "@/lib/utils"

interface TelemetryReadoutProps {
  label: string
  value: string
  unit?: string
  status?: "good" | "warn" | "critical" | "neutral"
  compact?: boolean
}

export function TelemetryReadout({
  label,
  value,
  unit,
  status = "neutral",
  compact = false,
}: TelemetryReadoutProps) {
  return (
    <div className={cn("flex flex-col", compact ? "gap-0" : "gap-0.5")}>
      <div className="flex items-center gap-1.5">
        <div
          className={cn(
            "status-dot",
            status === "good" && "status-dot-green",
            status === "warn" && "status-dot-amber",
            status === "critical" && "status-dot-red",
            status === "neutral" && "status-dot-cyan"
          )}
        />
        <span className="telemetry-label">{label}</span>
      </div>
      <div className="flex items-baseline gap-1 pl-3.5">
        <span
          className={cn(
            "font-mono font-bold tabular-nums",
            compact ? "text-sm" : "text-lg",
            status === "good" && "text-emerald-400",
            status === "warn" && "text-amber-400",
            status === "critical" && "text-red-400",
            status === "neutral" && "telemetry-value"
          )}
        >
          {value}
        </span>
        {unit && (
          <span className="text-[9px] text-muted-foreground uppercase tracking-wider">
            {unit}
          </span>
        )}
      </div>
    </div>
  )
}

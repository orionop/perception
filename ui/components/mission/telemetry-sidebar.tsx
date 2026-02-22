"use client"

import { TelemetryReadout } from "./telemetry-readout"
import type { ModelReport } from "@/lib/types"

function getIoUStatus(v: number): "good" | "warn" | "critical" {
  if (v >= 0.5) return "good"
  if (v >= 0.15) return "warn"
  return "critical"
}

function getSafetyStatus(v: number): "good" | "warn" | "critical" {
  if (v > 0.7) return "good"
  if (v >= 0.5) return "warn"
  return "critical"
}

export function TelemetrySidebar({ report }: { report: ModelReport }) {
  return (
    <div className="hud-panel rounded p-3 space-y-3 w-44 shrink-0">
      <div className="text-[9px] uppercase tracking-[0.15em] text-muted-foreground border-b border-border pb-1.5">
        Telemetry
      </div>
      <TelemetryReadout
        label="Mean IoU"
        value={(report.metrics.mean_iou * 100).toFixed(1)}
        unit="%"
        status={getIoUStatus(report.metrics.mean_iou)}
        compact
      />
      <TelemetryReadout
        label="Pixel Acc"
        value={(report.metrics.pixel_accuracy * 100).toFixed(1)}
        unit="%"
        status={getIoUStatus(report.metrics.pixel_accuracy)}
        compact
      />
      <TelemetryReadout
        label="FW-IoU"
        value={(report.metrics.frequency_weighted_iou * 100).toFixed(1)}
        unit="%"
        status="neutral"
        compact
      />
      <TelemetryReadout
        label="Dice"
        value={(report.metrics.dice_coefficient * 100).toFixed(1)}
        unit="%"
        status="neutral"
        compact
      />
      <div className="border-t border-border pt-2" />
      <TelemetryReadout
        label="Safety"
        value={(report.safety.safety_score * 100).toFixed(1)}
        unit="%"
        status={getSafetyStatus(report.safety.safety_score)}
        compact
      />
      <TelemetryReadout
        label="Latency"
        value={report.inference_time_ms.toFixed(0)}
        unit="ms"
        status={report.inference_time_ms < 200 ? "good" : "warn"}
        compact
      />
      <TelemetryReadout
        label="Confidence"
        value={(report.segmentation.confidence_mean * 100).toFixed(1)}
        unit="%"
        status={report.segmentation.confidence_mean >= 0.7 ? "good" : "warn"}
        compact
      />
      <TelemetryReadout
        label="Path"
        value={report.navigation.path_found ? "FOUND" : "NONE"}
        status={report.navigation.path_found ? "good" : "critical"}
        compact
      />
      {report.navigation.path_found && (
        <TelemetryReadout
          label="Path Cost"
          value={report.navigation.path_cost.toFixed(0)}
          status="neutral"
          compact
        />
      )}
    </div>
  )
}

"use client"

import {
  Activity,
  Target,
  Gauge,
  Timer,
  Route,
  ShieldCheck,
  BarChart3,
  Brain,
} from "lucide-react"
import { getPrimaryReport } from "@/lib/data"

function getQualityBadge(miou: number) {
  if (miou >= 0.5) return { label: "Excellent", color: "bg-emerald-500/20 text-emerald-400" }
  if (miou >= 0.3) return { label: "Good", color: "bg-blue-500/20 text-blue-400" }
  if (miou >= 0.15) return { label: "Moderate", color: "bg-yellow-500/20 text-yellow-400" }
  return { label: "Poor", color: "bg-red-500/20 text-red-400" }
}

function getSafetyColor(score: number) {
  if (score > 0.7) return "text-emerald-400"
  if (score >= 0.5) return "text-yellow-400"
  return "text-red-400"
}

function getSafetyGaugeColor(score: number) {
  if (score > 0.7) return "#22c55e"
  if (score >= 0.5) return "#eab308"
  return "#ef4444"
}

function getConfidenceBadge(conf: number) {
  if (conf >= 0.85) return { label: "High Trust", color: "bg-emerald-500/20 text-emerald-400" }
  if (conf >= 0.7) return { label: "Moderate Trust", color: "bg-blue-500/20 text-blue-400" }
  return { label: "Low Trust", color: "bg-yellow-500/20 text-yellow-400" }
}

function SafetyGauge({ score }: { score: number }) {
  const angle = score * 180
  const color = getSafetyGaugeColor(score)

  return (
    <div className="relative w-20 h-10 overflow-hidden">
      <svg viewBox="0 0 100 50" className="w-full h-full">
        <path
          d="M 5 50 A 45 45 0 0 1 95 50"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          className="text-secondary"
        />
        <path
          d="M 5 50 A 45 45 0 0 1 95 50"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={`${(angle / 180) * 141.37} 141.37`}
          strokeLinecap="round"
        />
      </svg>
    </div>
  )
}

interface MetricCardProps {
  title: string
  value: string
  subtitle?: string
  icon: React.ReactNode
  badge?: { label: string; color: string }
  glowClass?: string
  children?: React.ReactNode
}

function MetricCard({
  title,
  value,
  subtitle,
  icon,
  badge,
  glowClass = "",
  children,
}: MetricCardProps) {
  return (
    <div
      className={`glass-card rounded-xl p-5 flex flex-col gap-3 transition-all duration-300 hover:scale-[1.02] ${glowClass}`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          {title}
        </span>
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-secondary">
          {icon}
        </div>
      </div>
      <div className="flex items-end gap-2">
        <span className="text-3xl font-bold text-foreground leading-none">{value}</span>
        {badge && (
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${badge.color}`}>
            {badge.label}
          </span>
        )}
      </div>
      {subtitle && (
        <span className="text-sm text-muted-foreground">{subtitle}</span>
      )}
      {children}
    </div>
  )
}

export function MetricCards() {
  const report = getPrimaryReport()
  const quality = getQualityBadge(report.metrics.mean_iou)
  const confidence = getConfidenceBadge(report.segmentation.confidence_mean)
  const fps = (1000 / report.inference_time_ms).toFixed(1)

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard
        title="Mean IoU"
        value={(report.metrics.mean_iou * 100).toFixed(1) + "%"}
        icon={<Target className="w-4 h-4 text-primary" />}
        badge={quality}
        glowClass="glow-blue"
      />
      <MetricCard
        title="Freq-Weighted IoU"
        value={(report.metrics.frequency_weighted_iou * 100).toFixed(1) + "%"}
        icon={<BarChart3 className="w-4 h-4 text-primary" />}
      />
      <MetricCard
        title="Pixel Accuracy"
        value={(report.metrics.pixel_accuracy * 100).toFixed(1) + "%"}
        icon={<Activity className="w-4 h-4 text-accent" />}
        glowClass="glow-emerald"
      />
      <MetricCard
        title="Dice Coefficient"
        value={(report.metrics.dice_coefficient * 100).toFixed(1) + "%"}
        icon={<Gauge className="w-4 h-4 text-primary" />}
      />
      <MetricCard
        title="Safety Score"
        value={report.safety.safety_score.toFixed(3)}
        icon={
          <ShieldCheck
            className={`w-4 h-4 ${getSafetyColor(report.safety.safety_score)}`}
          />
        }
        glowClass="glow-emerald"
      >
        <SafetyGauge score={report.safety.safety_score} />
      </MetricCard>
      <MetricCard
        title="Inference Time"
        value={report.inference_time_ms.toFixed(1) + "ms"}
        subtitle={`~${fps} FPS`}
        icon={<Timer className="w-4 h-4 text-primary" />}
      />
      <MetricCard
        title="Path Found"
        value={report.navigation.path_found ? "Yes" : "No"}
        subtitle={
          report.navigation.path_found
            ? `Cost: ${report.navigation.path_cost} · ${report.navigation.path_length} steps`
            : undefined
        }
        icon={<Route className="w-4 h-4 text-accent" />}
      />
      <MetricCard
        title="Confidence"
        value={(report.segmentation.confidence_mean * 100).toFixed(1) + "%"}
        subtitle={`\u00b1${(report.segmentation.confidence_std * 100).toFixed(1)}%`}
        icon={<Brain className="w-4 h-4 text-primary" />}
        badge={confidence}
      />
    </div>
  )
}

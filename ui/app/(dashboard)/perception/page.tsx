"use client"

import { useState, useCallback, useRef } from "react"
import { Upload, Loader2, ArrowRight, Eye, Map, Route, Shield, ChevronDown, ChevronRight, Activity, Navigation, ShieldCheck, Check, X, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { getPrimaryReport, getExplanation, getTerrainAsset, getTerrainClasses } from "@/lib/data"
import { cn } from "@/lib/utils"
import type { ExplanationSection } from "@/lib/types"

const API_URL = process.env.NEXT_PUBLIC_PERCEPTION_API_URL || "http://localhost:8000"

const PIPELINE_STEPS = [
  { key: "raw", label: "Raw Terrain", icon: Eye, desc: "Input terrain image" },
  { key: "segmented", label: "Segmentation", icon: Map, desc: "Semantic class overlay" },
  { key: "path", label: "Safe Path", icon: Shield, desc: "Planned navigation path" },
] as const

type StepKey = (typeof PIPELINE_STEPS)[number]["key"]

interface InferenceResult {
  raw: string
  segmentation: string
  costmap: string
  path: string
  path_found: boolean
  path_cost: number
  path_length: number
  inference_time_ms: number
  demo_mode?: boolean
  per_class_iou?: Record<string, number>
  mean_iou?: number
  pixel_accuracy?: number
  confidence_mean?: number
  confidence_std?: number
  class_distribution?: Record<string, number>
  image_iou?: number | null
}

export default function PerceptionLabPage() {
  const [inputImage, setInputImage] = useState<string | null>(null)
  const [inputFile, setInputFile] = useState<File | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [progress, setProgress] = useState(0)
  const [activeStep, setActiveStep] = useState<StepKey>("raw")
  const [pathReportOpen, setPathReportOpen] = useState(false)
  const [metricsSummaryOpen, setMetricsSummaryOpen] = useState(false)
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const report = getPrimaryReport()
  const explanation = getExplanation()
  const asset = getTerrainAsset("terrain_01")
  const classes = getTerrainClasses()

  const pathFound = inferenceResult ? inferenceResult.path_found : report.navigation.path_found
  const pathCost = inferenceResult ? inferenceResult.path_cost : report.navigation.path_cost
  const pathLength = inferenceResult ? inferenceResult.path_length : report.navigation.path_length

  const runPipeline = useCallback(async () => {
    if (!inputFile || !inputImage) return
    setIsRunning(true)
    setShowResults(false)
    setError(null)
    setProgress(5)

    // Simulate progress while waiting for actual inference
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) return 95
        return prev + Math.random() * 5 + 2 // Increase by 2-7%
      })
    }, 600)

    try {
      const formData = new FormData()
      formData.append("image", inputFile)

      const res = await fetch(`${API_URL}/api/infer`, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || res.statusText || "Inference failed")
      }

      const data = (await res.json()) as InferenceResult
      clearInterval(progressInterval)
      setProgress(100)

      // Slight delay to show 100% before revealing results
      setTimeout(() => {
        setInferenceResult(data)
        setShowResults(true)
        setActiveStep("path")
        setIsRunning(false)
      }, 500)

    } catch (e) {
      clearInterval(progressInterval)
      setError(e instanceof Error ? e.message : "Pipeline failed")
      setShowResults(false)
      setIsRunning(false)
    }
  }, [inputFile, inputImage])

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith("image/")) {
      setInputFile(file)
      setInputImage(URL.createObjectURL(file))
      setInferenceResult(null)
      setError(null)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setInputFile(file)
      setInputImage(URL.createObjectURL(file))
      setInferenceResult(null)
      setError(null)
    }
  }

  const stepImage: Record<StepKey, string> = {
    raw: inputImage ?? asset.terrain,
    segmented: inferenceResult?.segmentation
      ? `data:image/png;base64,${inferenceResult.segmentation}`
      : asset.maskOverlay,
    path: inferenceResult?.path
      ? `data:image/png;base64,${inferenceResult.path}`
      : asset.pathOverlay,
  }

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Page header */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">
            Perception Lab
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Full perception pipeline — from raw terrain to safe navigation path
          </p>
        </div>
        {showResults && inferenceResult && (
          <div className="flex items-center gap-4 text-sm font-mono text-muted-foreground">
            {inferenceResult.demo_mode && (
              <span className="text-amber-500 font-medium bg-amber-500/10 px-2.5 py-1 rounded-md">
                Demo mode (heuristic segmentation, no weights)
              </span>
            )}
            <span>Latency: <span className="text-amber-400 font-semibold">{inferenceResult.inference_time_ms.toFixed(0)}ms</span></span>
            {inferenceResult.inference_time_ms > 0 && (
              <span>FPS: <span className="text-amber-400 font-semibold">{(1000 / inferenceResult.inference_time_ms).toFixed(1)}</span></span>
            )}
          </div>
        )}
      </div>

      {/* Upload zone */}
      <div
        className={cn(
          "rounded-xl border-2 border-dashed p-8 flex flex-col items-center gap-4 cursor-pointer transition-all",
          inputImage
            ? "border-emerald-500/40 bg-emerald-500/5"
            : "border-border hover:border-amber-500/40 hover:bg-amber-500/5"
        )}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleFileDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
        {inputImage ? (
          <div className="flex items-center gap-5 w-full">
            <img src={inputImage} alt="Input" className="w-40 h-24 object-cover rounded-lg border border-border" />
            <div className="flex-1">
              <div className="text-base text-foreground font-medium">Terrain image loaded</div>
              <div className="text-sm text-muted-foreground mt-1">Click or drop a new file to replace</div>
            </div>
            <Button
              onClick={(e) => { e.stopPropagation(); runPipeline() }}
              disabled={isRunning}
              className="bg-primary hover:bg-primary/90 text-primary-foreground text-sm px-6 h-10"
            >
              {isRunning ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              {isRunning ? "Processing..." : "Run Pipeline"}
            </Button>
          </div>
        ) : (
          <>
            <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
              <Upload className="w-6 h-6 text-muted-foreground" />
            </div>
            <div className="text-center">
              <div className="text-base text-foreground font-medium">Drop terrain image or click to upload</div>
              <div className="text-sm text-muted-foreground mt-1">PNG / JPG — offroad terrain imagery</div>
            </div>
          </>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="hud-panel rounded-xl p-4 flex items-center gap-3 border border-red-500/30 bg-red-500/5">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-sm text-red-400">{error}</p>
          <p className="text-xs text-muted-foreground">Ensure the Perception API is running: <code className="font-mono">pip install &quot;.[api]&quot; &amp;&amp; uvicorn perception_engine.api.server:app --port 8000</code></p>
        </div>
      )}

      {/* Progress bar */}
      {isRunning && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm font-mono text-muted-foreground">
            <span>Running real inference...</span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div className="h-full bg-amber-500 rounded-full transition-all duration-500 ease-out" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {/* Pipeline stepper */}
      <div className="hud-panel rounded-xl p-2">
        <div className="flex items-center gap-1">
          {PIPELINE_STEPS.map((step, i) => {
            const Icon = step.icon
            const isActive = activeStep === step.key
            return (
              <div key={step.key} className="flex items-center flex-1">
                <button
                  onClick={() => showResults && setActiveStep(step.key)}
                  className={cn(
                    "flex-1 flex flex-col items-center gap-2 py-4 px-3 rounded-lg transition-all",
                    isActive
                      ? "bg-primary/10 border border-primary/30"
                      : "hover:bg-secondary/30 border border-transparent",
                    !showResults && !isRunning && "opacity-40"
                  )}
                >
                  <Icon className={cn("w-5 h-5", isActive ? "text-primary" : "text-muted-foreground")} />
                  <span className={cn(
                    "text-sm font-medium",
                    isActive ? "text-primary" : "text-muted-foreground"
                  )}>
                    {step.label}
                  </span>
                  <span className={cn(
                    "text-xs",
                    isActive ? "text-muted-foreground" : "text-muted-foreground/50"
                  )}>
                    {step.desc}
                  </span>
                </button>
                {i < PIPELINE_STEPS.length - 1 && (
                  <ArrowRight className="w-4 h-4 text-muted-foreground/30 shrink-0 mx-1" />
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Detail panel */}
      {(showResults || inputImage) && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* Main image viewer */}
          <div className="lg:col-span-2">
            <div className="hud-panel rounded-xl overflow-hidden">
              <img
                src={stepImage[activeStep]}
                alt={activeStep}
                className="w-full h-auto"
              />
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* Class Legend */}
            {activeStep === "segmented" && (
              <div className="hud-panel rounded-xl p-5">
                <h3 className="text-sm font-semibold text-foreground uppercase tracking-wide mb-4">Class Legend</h3>
                <div className="space-y-3">
                  {classes.map((c) => {
                    const iou = inferenceResult?.per_class_iou?.[c.key] ?? (inferenceResult ? null : report.metrics.per_class_iou[c.key])
                    const coverage = inferenceResult?.class_distribution?.[c.key]
                    return (
                      <div key={c.key} className="flex items-center gap-3">
                        <div className="w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: c.color }} />
                        <span className="text-sm text-foreground flex-1">{c.name}</span>
                        <span className="text-sm font-mono text-muted-foreground">
                          {coverage != null ? coverage.toFixed(1) + "%" : iou != null ? (iou * 100).toFixed(1) + "%" : "N/A"}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}



            {/* Key metrics */}
            {showResults && (
              <div className="grid grid-cols-1 gap-3">
                {/* Image IoU — prominent when available */}
                {inferenceResult?.image_iou != null && (
                  <div className="hud-panel rounded-xl p-4 flex items-center justify-between border border-amber-500/30 bg-amber-500/5">
                    <div>
                      <div className="text-xs text-muted-foreground uppercase tracking-wide">Image IoU</div>
                      <div className={cn(
                        "text-2xl font-mono font-bold mt-1",
                        inferenceResult.image_iou >= 0.6 ? "text-emerald-400"
                          : inferenceResult.image_iou >= 0.4 ? "text-amber-400"
                            : "text-red-400"
                      )}>
                        {(inferenceResult.image_iou * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="text-xs font-mono text-muted-foreground text-right">
                      <div>vs Ground Truth</div>
                      <div className="text-amber-400 font-semibold mt-1">{inferenceResult.image_iou.toFixed(4)}</div>
                    </div>
                  </div>
                )}
                <div className="hud-panel rounded-xl p-4 flex items-center justify-between">
                  <div>
                    <div className="text-xs text-muted-foreground uppercase tracking-wide">Path Found</div>
                    <div className={cn(
                      "text-xl font-mono font-bold mt-1",
                      pathFound ? "text-emerald-400" : "text-red-400"
                    )}>
                      {pathFound ? "YES" : "NO"}
                    </div>
                  </div>
                  <div className={cn(
                    "w-10 h-10 rounded-full flex items-center justify-center",
                    pathFound ? "bg-emerald-400/10" : "bg-red-400/10"
                  )}>
                    {pathFound
                      ? <Check className="w-5 h-5 text-emerald-400" />
                      : <X className="w-5 h-5 text-red-400" />}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="hud-panel rounded-xl p-4">
                    <div className="text-xs text-muted-foreground uppercase tracking-wide">Confidence</div>
                    <div className={cn(
                      "text-xl font-mono font-bold mt-1",
                      (inferenceResult?.confidence_mean ?? report.segmentation.confidence_mean) >= 0.8 ? "text-emerald-400"
                        : (inferenceResult?.confidence_mean ?? report.segmentation.confidence_mean) >= 0.5 ? "text-amber-400"
                          : "text-red-400"
                    )}>
                      {((inferenceResult?.confidence_mean ?? report.segmentation.confidence_mean) * 100).toFixed(1) + "%"}
                    </div>
                  </div>
                  <div className="hud-panel rounded-xl p-4">
                    <div className="text-xs text-muted-foreground uppercase tracking-wide">Mean IoU</div>
                    <div className={cn(
                      "text-xl font-mono font-bold mt-1",
                      (inferenceResult?.mean_iou ?? report.metrics.mean_iou) >= 0.8 ? "text-emerald-400"
                        : (inferenceResult?.mean_iou ?? report.metrics.mean_iou) >= 0.5 ? "text-amber-400"
                          : "text-red-400"
                    )}>
                      {((inferenceResult?.mean_iou ?? report.metrics.mean_iou) * 100).toFixed(1) + "%"}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Path Report — collapsible */}
            {activeStep === "path" && showResults && (
              <div className="hud-panel rounded-xl overflow-hidden">
                <button
                  onClick={() => setPathReportOpen(!pathReportOpen)}
                  className="flex items-center gap-3 w-full p-4 text-left hover:bg-secondary/20 transition-colors"
                >
                  {pathReportOpen
                    ? <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
                    : <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />}
                  <span className="text-sm font-medium text-foreground">Path Report</span>
                </button>
                {pathReportOpen && (
                  <div className="px-4 pb-4 space-y-3">
                    {[
                      { l: "Path Cost", v: pathCost.toFixed(0) },
                      { l: "Path Length", v: `${pathLength} steps` },
                      { l: "Obstacle Overlap", v: inferenceResult ? "N/A" : (report.safety.obstacle_overlap_pct * 100).toFixed(1) + "%" },
                    ].map((r) => (
                      <div key={r.l} className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">{r.l}</span>
                        <span className="text-sm font-mono text-amber-400 font-medium">{r.v}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Metrics Summary — collapsible */}
            {showResults && (
              <div className="hud-panel rounded-xl overflow-hidden">
                <button
                  onClick={() => setMetricsSummaryOpen(!metricsSummaryOpen)}
                  className="flex items-center gap-3 w-full p-4 text-left hover:bg-secondary/20 transition-colors"
                >
                  {metricsSummaryOpen
                    ? <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
                    : <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />}
                  <span className="text-sm font-medium text-foreground">Metrics Summary</span>
                </button>
                {metricsSummaryOpen && (
                  <div className="px-4 pb-4 space-y-3">
                    {[
                      { l: "Pixel Accuracy", v: ((inferenceResult?.pixel_accuracy ?? report.metrics.pixel_accuracy * 100)).toFixed(1) + "%" },
                      { l: "Confidence Mean", v: ((inferenceResult?.confidence_mean ?? report.segmentation.confidence_mean) * 100).toFixed(1) + "%" },
                      { l: "Confidence Std", v: "±" + ((inferenceResult?.confidence_std ?? report.segmentation.confidence_std) * 100).toFixed(1) + "%" },
                      { l: "Inference Time", v: inferenceResult ? inferenceResult.inference_time_ms.toFixed(0) + " ms" : "N/A" },
                    ].map((r) => (
                      <div key={r.l} className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">{r.l}</span>
                        <span className="text-sm font-mono text-amber-400 font-medium">{r.v}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Mission Briefing — only show with static demo data, hide during real inference */}
      {showResults && !inferenceResult && explanation.models[0] && (
        <MissionBriefing sections={explanation.models[0].sections} />
      )}
    </div>
  )
}

type BriefingTab = "performance" | "navigation" | "robustness"

const TAB_CONFIG: { key: BriefingTab; label: string; icon: typeof Activity; indices: number[]; accent: string }[] = [
  { key: "performance", label: "Performance", icon: Activity, indices: [0, 1, 2, 3], accent: "border-l-amber-500" },
  { key: "navigation", label: "Navigation", icon: Navigation, indices: [4, 5], accent: "border-l-cyan-500" },
  { key: "robustness", label: "Robustness", icon: ShieldCheck, indices: [6], accent: "border-l-emerald-500" },
]

function MissionBriefing({ sections }: { sections: ExplanationSection[] }) {
  const [activeTab, setActiveTab] = useState<BriefingTab>("performance")

  const findings = [
    { label: "mIoU", value: sections[0]?.metric, color: "text-amber-400" },
    { label: "Latency", value: sections[2]?.metric, color: "text-cyan-400" },
    { label: "Safety", value: sections[5]?.metric, color: "text-emerald-400" },
    { label: "Best Class", value: sections[1]?.best_class, color: "text-emerald-400" },
    { label: "Worst Class", value: sections[1]?.worst_class, color: "text-red-400" },
    { label: "Robustness", value: sections[6]?.metric, color: "text-amber-400" },
  ].filter(f => f.value)

  const activeConfig = TAB_CONFIG.find(t => t.key === activeTab)!

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-foreground tracking-tight">Mission Briefing</h2>
        <p className="text-sm text-muted-foreground mt-1">Detailed analysis of the perception pipeline results</p>
      </div>

      {/* Key Findings strip */}
      <div className="hud-panel rounded-xl p-5">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Key Findings</h3>
        <div className="flex flex-wrap gap-3">
          {findings.map((f) => (
            <div key={f.label} className="flex items-center gap-2 bg-secondary/40 rounded-lg px-3 py-2">
              <span className="text-xs text-muted-foreground uppercase tracking-wide">{f.label}</span>
              <span className={cn("text-sm font-mono font-bold", f.color)}>{f.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Category tabs */}
      <div className="flex gap-2">
        {TAB_CONFIG.map((tab) => {
          const Icon = tab.icon
          const isActive = activeTab === tab.key
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                "flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all flex-1 justify-center",
                isActive
                  ? "bg-primary/10 border border-primary/30 text-primary"
                  : "hud-panel text-muted-foreground hover:text-foreground hover:bg-secondary/30"
              )}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
              <span className="text-xs font-mono opacity-50">({tab.indices.length})</span>
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div className="space-y-3">
        {activeConfig.indices.map((idx) => {
          const section = sections[idx]
          if (!section) return null
          return <ExplanationCard key={idx} section={section} accent={activeConfig.accent} />
        })}
      </div>
    </div>
  )
}

function ExplanationCard({ section, accent }: { section: ExplanationSection; accent: string }) {
  const [expanded, setExpanded] = useState(false)
  const hasDetails = section.all_classes || section.perturbations

  return (
    <div className={cn("hud-panel rounded-xl p-5 border-l-[3px]", accent)}>
      <button
        onClick={() => hasDetails && setExpanded(!expanded)}
        className={cn("flex items-start gap-3 w-full text-left", hasDetails && "cursor-pointer")}
      >
        {hasDetails && (
          expanded
            ? <ChevronDown className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
            : <ChevronRight className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-sm font-semibold text-foreground">{section.title}</span>
            {section.metric && (
              <span className="text-sm font-mono font-bold text-amber-400 bg-amber-400/10 px-2.5 py-1 rounded-md">
                {section.metric}
              </span>
            )}
          </div>
          <p className="text-sm text-muted-foreground mt-2 leading-relaxed">{section.explanation}</p>
          {(section.best_class || section.worst_class) && (
            <div className="flex gap-3 mt-2">
              {section.best_class && (
                <span className="text-xs font-medium text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-md">
                  Best: {section.best_class}
                </span>
              )}
              {section.worst_class && (
                <span className="text-xs font-medium text-red-400 bg-red-400/10 px-2 py-1 rounded-md">
                  Worst: {section.worst_class}
                </span>
              )}
            </div>
          )}
        </div>
      </button>
      {expanded && section.all_classes && (
        <div className="mt-4 pl-7 grid grid-cols-2 gap-x-6 gap-y-2 border-t border-border/50 pt-4">
          {Object.entries(section.all_classes).map(([cls, desc]) => (
            <div key={cls} className="flex items-center justify-between text-sm">
              <span className="text-foreground capitalize">{cls.replace(/_/g, " ")}</span>
              <span className="font-mono text-muted-foreground">{desc}</span>
            </div>
          ))}
        </div>
      )}
      {expanded && section.perturbations && (
        <div className="mt-4 pl-7 space-y-2.5 border-t border-border/50 pt-4">
          {Object.entries(section.perturbations).map(([pert, info]) => (
            <div key={pert} className="flex items-center gap-4 text-sm">
              <span className="text-foreground capitalize w-20">{pert}</span>
              <span className="font-mono text-amber-400 w-14 text-right">{info.drop}</span>
              <span className={cn(
                "text-xs font-medium uppercase tracking-wide px-2 py-0.5 rounded-md",
                info.verdict.toLowerCase().includes("robust")
                  ? "text-emerald-400 bg-emerald-400/10"
                  : "text-red-400 bg-red-400/10"
              )}>
                {info.verdict}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

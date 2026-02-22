"use client"

import { useState } from "react"
import {
    Brain, Layers, Shield, BarChart3, Eye, Globe,
    ChevronDown, ChevronRight, Cpu, Zap, Target,
    TrendingUp, AlertTriangle, CheckCircle2, Sparkles
} from "lucide-react"
import { cn } from "@/lib/utils"

/* ─────────────────────── VERIFIED DATA (from hackathon_report.md) ─────────────────────── */

const VERSIONS = [
    { name: "V1", arch: "ResNet34 DeepLabV3+", miou: 16, pixel: 38.0, note: "Baseline CNN — overfit to dominant classes" },
    { name: "V2", arch: "ResNet50 DeepLabV3+", miou: 17, pixel: 40.0, note: "Deeper backbone — still texture-biased" },
    { name: "V3", arch: "DINOv2 ViT-S/14 + 2-block head", miou: 35, pixel: 55.0, note: "Foundation model backbone (+106% vs CNN)" },
    { name: "V4", arch: "V3 + Frequency Recalibration", miou: 45, pixel: 65.0, note: "Class rebalancing (+29%)" },
    { name: "V5", arch: "V4 + Multi-scale + 1D Spatial Prior", miou: 48, pixel: 70.0, note: "Spatial context (+7%)" },
    { name: "V6", arch: "DINOv2 + 3-block head + 4D Bayesian Fusion", miou: 52.1, pixel: 77.6, note: "Joint histogram fusion (+8.5%)" },
    { name: "Ens.", arch: "V3 + V5 + V6 Ensemble + Prior", miou: 61, pixel: 81.2, note: "Highest benchmark result" },
]

// Per-class IoU from the FULL 1,002-image test set (hackathon_report.md)
const PER_CLASS = [
    { cls: "Sky", iou: 99.0, color: "#C8E6FF" },
    { cls: "Landscape", iou: 72.7, color: "#C8E6B4" },
    { cls: "Tree", iou: 59.8, color: "#FF6400" },
    { cls: "Dry Bush", iou: 45.5, color: "#C80000" },
    { cls: "Dry Grass", iou: 44.3, color: "#00C800" },
]

const AUGMENTATIONS = [
    "Random horizontal flips (p=0.5)",
    "Color jitter (brightness=0.3, contrast=0.3, sat=0.2, hue=0.1)",
    "Random resized crops",
    "Inverse-frequency class weights (combat 42% landscape bias)",
    "Cosine Annealing LR with warm restarts",
]

// From Section 7 of hackathon_report.md — confusion matrix analysis
const CONFUSION_PAIRS = [
    { pair: "dry_grass → rock", pixels: "32.7M", pct: "6.29%", cause: "Identical HSV color profiles" },
    { pair: "landscape → rock", pixels: "32.0M", pct: "6.16%", cause: "Color overlap + adjacent positions" },
    { pair: "rock → landscape", pixels: "16.1M", pct: "3.10%", cause: "Reverse confusion at boundaries" },
    { pair: "dry_grass → landscape", pixels: "8.5M", pct: "1.65%", cause: "Both are flat, brown ground cover" },
    { pair: "landscape → dry_grass", pixels: "7.8M", pct: "1.49%", cause: "Reverse confusion" },
    { pair: "rock → dry_grass", pixels: "6.6M", pct: "1.27%", cause: "Texture similarity in arid conditions" },
]

const EDGE_CASES = [
    { scenario: "Rock vs Landscape (color-identical)", status: "Bayesian spatial prior mitigates; breaks down in flat boulder fields", ok: true },
    { scenario: "Sparse Dry Bush (0.4% of training)", status: "Frequency recalibration boosts confidence +1.5×, achieves 0.455 IoU", ok: true },
    { scenario: "Absent classes (ground_clutter, flower, log)", status: "Zero training samples → zero predictions (expected limitation)", ok: false },
    { scenario: "Lush Bush (15K pixels out of 519M total)", status: "Near-zero training → zero predictions", ok: false },
    { scenario: "Horizon line (sky ↔ landscape boundary)", status: "Near pixel-perfect — DINOv2 structural features excel here", ok: true },
    { scenario: "Dense boulder fields at same elevation", status: "Spatial prior assumption breaks down — hardest failure case", ok: false },
]


/* ─────────────────────── page ─────────────────────── */

export default function TechnicalPage() {
    return (
        <div className="space-y-8 max-w-6xl mx-auto pb-12">
            {/* Page header */}
            <div>
                <h1 className="text-2xl font-bold text-foreground tracking-tight">Command Center</h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Comprehensive analysis covering all evaluation criteria — Perception Engine v6
                </p>
            </div>

            {/* Quick stats strip — all verified */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                    { label: "Highest mIoU", value: "61%", icon: Target, color: "text-emerald-400" },
                    { label: "Pixel Accuracy", value: "81.2%", icon: CheckCircle2, color: "text-emerald-400" },
                    { label: "Test Images", value: "1,002", icon: Layers, color: "text-amber-400" },
                    { label: "Classes Evaluated", value: "10 / 10", icon: Shield, color: "text-emerald-400" },
                ].map((s) => (
                    <div key={s.label} className="hud-panel rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <s.icon className="w-4 h-4 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground uppercase tracking-wide">{s.label}</span>
                        </div>
                        <div className={cn("text-xl font-mono font-bold", s.color)}>{s.value}</div>
                    </div>
                ))}
            </div>

            {/* 1. Semantic Segmentation Model Development */}
            <Section
                icon={Brain}
                title="Semantic Segmentation Model Development"
                subtitle="Deep learning model assigning a class label to every pixel"
                accent="border-l-violet-500"
            >
                <div className="space-y-5">
                    <p className="text-sm text-muted-foreground leading-relaxed">
                        The system uses <span className="text-foreground font-medium">DINOv2 ViT-S/14</span> as a frozen backbone —
                        a Vision Transformer trained via self-supervised learning on 142M images. A custom segmentation head with
                        <span className="text-foreground font-medium"> 3 ConvNeXt-style blocks</span> (depthwise separable convolutions)
                        produces per-pixel class predictions across <span className="text-amber-400 font-mono">10 terrain classes</span>.
                    </p>

                    <div className="hud-panel rounded-lg p-4">
                        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Architecture Details</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                            {[
                                { l: "Backbone", v: "DINOv2 ViT-S/14 (frozen, 21M params)" },
                                { l: "Patch Size", v: "14×14 → 34×19 = 646 tokens" },
                                { l: "Embedding Dim", v: "384" },
                                { l: "Seg Head", v: "3× ConvNeXt blocks (~2.1M params)" },
                                { l: "Block 1", v: "7×7 depthwise sep. conv (broad context)" },
                                { l: "Block 2", v: "5×5 depthwise sep. conv (medium detail)" },
                                { l: "Block 3", v: "3×3 depthwise sep. conv (local boundaries)" },
                                { l: "Output", v: "10 classes, bilinear upsample to 960×540" },
                                { l: "Loss", v: "CE + inverse-frequency class weights" },
                                { l: "Optimizer", v: "AdamW (lr=3e-4, wd=0.01)" },
                                { l: "Scheduler", v: "Cosine Annealing w/ warm restarts" },
                                { l: "Training", v: "50 epochs, dual T4 GPUs (Kaggle)" },
                            ].map((r) => (
                                <div key={r.l} className="flex justify-between gap-2">
                                    <span className="text-muted-foreground shrink-0">{r.l}</span>
                                    <span className="font-mono text-amber-400 text-right">{r.v}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="hud-panel rounded-lg p-4">
                        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Top 5 Per-Class IoU (1,002-image test set)</h4>
                        <div className="space-y-2.5">
                            {PER_CLASS.map((c) => (
                                <div key={c.cls} className="flex items-center gap-3">
                                    <div className="w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: c.color }} />
                                    <span className="text-sm text-foreground w-28">{c.cls}</span>
                                    <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                                        <div
                                            className="h-full rounded-full transition-all"
                                            style={{
                                                width: `${c.iou}%`,
                                                backgroundColor: c.iou >= 60 ? "#10B981" : c.iou >= 30 ? "#F59E0B" : "#6B7280",
                                            }}
                                        />
                                    </div>
                                    <span className={cn(
                                        "text-sm font-mono w-14 text-right font-bold",
                                        c.iou >= 60 ? "text-emerald-400" : c.iou >= 30 ? "text-amber-400" : "text-muted-foreground"
                                    )}>
                                        {c.iou.toFixed(1)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </Section>

            {/* 2. Structured Training and Optimization Pipeline */}
            <Section
                icon={Cpu}
                title="Structured Training and Optimization Pipeline"
                subtitle="Systematic experimentation from V1 (16% mIoU) to V6 (52.1% mIoU)"
                accent="border-l-cyan-500"
            >
                <div className="space-y-5">
                    <div className="hud-panel rounded-lg p-4">
                        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Version Progression (V1 → Ensemble)</h4>
                        <div className="space-y-3">
                            {VERSIONS.map((v, i) => (
                                <div key={v.name} className="flex items-start gap-4">
                                    <div className="flex flex-col items-center">
                                        <div className={cn(
                                            "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold",
                                            i === VERSIONS.length - 1
                                                ? "bg-emerald-400/20 text-emerald-400 ring-2 ring-emerald-400/30"
                                                : "bg-secondary text-muted-foreground"
                                        )}>
                                            {v.name}
                                        </div>
                                        {i < VERSIONS.length - 1 && <div className="w-px h-6 bg-border mt-1" />}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-3 flex-wrap">
                                            <span className="text-sm font-medium text-foreground">{v.arch}</span>
                                            <span className={cn(
                                                "text-sm font-mono font-bold px-2 py-0.5 rounded-md",
                                                v.miou >= 50 ? "text-emerald-400 bg-emerald-400/10" : v.miou >= 30 ? "text-amber-400 bg-amber-400/10" : "text-red-400 bg-red-400/10"
                                            )}>
                                                {v.miou}% mIoU
                                            </span>
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1">{v.note}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="hud-panel rounded-lg p-4">
                        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                            Training Configuration & Augmentation
                        </h4>
                        <div className="space-y-2">
                            {AUGMENTATIONS.map((a, i) => (
                                <div key={i} className="flex items-center gap-2">
                                    <Sparkles className="w-3 h-3 text-cyan-400 shrink-0" />
                                    <span className="text-sm text-foreground">{a}</span>
                                </div>
                            ))}
                        </div>
                        <div className="mt-4 pt-3 border-t border-border/50">
                            <p className="text-xs text-muted-foreground">
                                DINOv2 backbone kept <span className="text-foreground font-medium">entirely frozen</span> —
                                only the 2.1M-param segmentation head is trained.
                                This prevents overfitting to synthetic textures and preserves domain-agnostic properties.
                            </p>
                        </div>
                    </div>
                </div>
            </Section>

            {/* 3. Generalization to Unseen Desert Environments */}
            <Section
                icon={Globe}
                title="Generalization to Unseen Desert Environments"
                subtitle="Performance on previously unseen terrain — test images from a different desert"
                accent="border-l-amber-500"
            >
                <div className="space-y-4">
                    <p className="text-sm text-muted-foreground leading-relaxed">
                        Test images come from a <span className="text-foreground font-medium">different desert environment</span> than the training data.
                        The model achieves <span className="text-amber-400 font-mono font-bold">0.5211 mIoU</span> on 1,002 unseen test images,
                        demonstrating <span className="text-foreground font-medium">stable generalization</span> rather than high variance on a lucky subset.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        {[
                            {
                                title: "Layer 1: Domain-Invariant Features",
                                desc: "DINOv2 (frozen, SSL-trained on 142M images) learns structural/semantic features that transfer across domains. Jump from 17% (CNN) to 35% (DINOv2) with zero tuning.",
                                metric: "+106% mIoU vs CNN",
                            },
                            {
                                title: "Layer 2: 4D Bayesian Spatial-Color Prior",
                                desc: "4D Joint Histogram P(class | H, S, V, Y) with 25,920 bins/class from training data. Encodes environment-invariant physical constraints (sky=top, rocks=elevated).",
                                metric: "+8.5% mIoU improvement",
                            },
                            {
                                title: "Layer 3: Class Frequency Recalibration",
                                desc: "Inverse-frequency weights boost underrepresented obstacle classes (rock: 3.5×, dry_grass: 0.6×), preventing suppression by dominant landscape class.",
                                metric: "+29% mIoU at V4",
                            },
                        ].map((s) => (
                            <div key={s.title} className="hud-panel rounded-lg p-4 flex flex-col">
                                <h4 className="text-sm font-semibold text-foreground mb-2">{s.title}</h4>
                                <p className="text-xs text-muted-foreground flex-1">{s.desc}</p>
                                <div className="mt-3 pt-2 border-t border-border/50">
                                    <span className="text-xs font-mono text-emerald-400 font-medium">{s.metric}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            {/* 4. Model Robustness and Edge-Case Handling */}
            <Section
                icon={Shield}
                title="Model Robustness and Edge-Case Handling"
                subtitle="Confusion matrix analysis and challenging scenario handling"
                accent="border-l-red-500"
            >
                <div className="space-y-5">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="hud-panel rounded-lg p-4">
                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Top Confusion Pairs (Quantified)</h4>
                            <div className="space-y-2.5">
                                {CONFUSION_PAIRS.map((r) => (
                                    <div key={r.pair} className="flex items-center gap-2 text-sm">
                                        <span className="text-foreground flex-1 truncate" title={r.pair}>{r.pair}</span>
                                        <span className="font-mono text-red-400 w-14 text-right shrink-0">{r.pct}</span>
                                    </div>
                                ))}
                            </div>
                            <div className="mt-3 pt-2 border-t border-border/50">
                                <p className="text-xs text-muted-foreground">
                                    Root cause: rock &amp; landscape share <span className="text-foreground">identical HSV color profiles</span> — physically the same geological material.
                                </p>
                            </div>
                        </div>

                        <div className="hud-panel rounded-lg p-4">
                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Edge Cases</h4>
                            <div className="space-y-2.5">
                                {EDGE_CASES.map((e) => (
                                    <div key={e.scenario} className="flex items-start gap-2">
                                        {e.ok
                                            ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                                            : <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                                        }
                                        <div>
                                            <div className="text-sm text-foreground font-medium">{e.scenario}</div>
                                            <div className="text-xs text-muted-foreground mt-0.5">{e.status}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </Section>

            {/* 5. Quantitative Evaluation and Performance Analysis */}
            <Section
                icon={BarChart3}
                title="Quantitative Evaluation and Performance Analysis"
                subtitle="mIoU, pixel accuracy, per-class IoU across all 6 model versions"
                accent="border-l-emerald-500"
            >
                <div className="space-y-5">
                    <div className="hud-panel rounded-lg p-4">
                        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">mIoU Progression (V1 → Ensemble)</h4>
                        <div className="flex items-end gap-2 h-40 px-2">
                            {VERSIONS.map((v) => (
                                <div key={v.name} className="flex-1 flex flex-col items-center gap-2">
                                    <span className={cn(
                                        "text-xs font-mono font-bold",
                                        v.miou >= 60 ? "text-emerald-400" : v.miou >= 45 ? "text-emerald-500" : v.miou >= 30 ? "text-amber-400" : "text-red-400"
                                    )}>
                                        {v.miou}%
                                    </span>
                                    <div
                                        className="w-full rounded-t-lg transition-all"
                                        style={{
                                            height: `${(v.miou / 70) * 100}%`,
                                            backgroundColor: v.miou >= 60 ? "#10B981" : v.miou >= 45 ? "#10B981" : v.miou >= 30 ? "#F59E0B" : "#EF4444",
                                            opacity: v.miou >= 60 ? 1 : 0.7,
                                        }}
                                    />
                                    <span className="text-[10px] font-medium text-muted-foreground">{v.name}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="hud-panel rounded-lg p-4">
                        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Benchmark Table (1,002 test images)</h4>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-border/50">
                                        <th className="text-left py-2 text-xs text-muted-foreground uppercase tracking-wide">Metric</th>
                                        <th className="text-right py-2 text-xs text-muted-foreground uppercase tracking-wide">V1</th>
                                        <th className="text-right py-2 text-xs text-muted-foreground uppercase tracking-wide">V2</th>
                                        <th className="text-right py-2 text-xs text-muted-foreground uppercase tracking-wide">V3</th>
                                        <th className="text-right py-2 text-xs text-muted-foreground uppercase tracking-wide">V6</th>
                                        <th className="text-right py-2 text-xs text-muted-foreground uppercase tracking-wide">Ens.</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {[
                                        { metric: "Mean IoU", values: ["16%", "17%", "35%", "52.1%", "61%"] },
                                        { metric: "Pixel Accuracy", values: ["~38%", "~40%", "~55%", "77.6%", "81.2%"] },
                                        { metric: "Backbone", values: ["ResNet34", "ResNet50", "DINOv2", "DINOv2", "Ensemble"] },
                                        { metric: "Key Innovation", values: ["Baseline", "Depth", "SSL", "Bayesian", "Fusion"] },
                                    ].map((row) => (
                                        <tr key={row.metric} className="border-b border-border/30">
                                            <td className="py-2.5 text-foreground">{row.metric}</td>
                                            {row.values.map((v, i) => (
                                                <td key={i} className="py-2.5 text-right font-mono text-amber-400">{v}</td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </Section>

            {/* 6. Visualization and Interpretability of Results */}
            <Section
                icon={Eye}
                title="Visualization and Interpretability of Results"
                subtitle="Visual outputs displaying segmentation predictions alongside input images"
                accent="border-l-pink-500"
            >
                <div className="space-y-5">
                    <p className="text-sm text-muted-foreground leading-relaxed">
                        The <span className="text-foreground font-medium">Perception Lab</span> page provides interactive visualization
                        of the full ensemble pipeline: upload any test image and view the segmentation overlay, cost map, and A* safe path.
                        For test images with available ground truth, per-image IoU is computed and displayed.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="hud-panel rounded-lg p-4">
                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Pipeline Outputs</h4>
                            <div className="space-y-3">
                                {[
                                    { stage: "1. Prediction Overlay", desc: "Color-coded segmentation mask overlaid on input image" },
                                    { stage: "2. Side-by-Side Comparison", desc: "Original vs predicted vs ground truth (when available)" },
                                    { stage: "3. Cost Map", desc: "Traversability grid: green (safe) → red (obstacle)" },
                                    { stage: "4. A* Safe Path", desc: "Optimal path avoiding obstacles" },
                                ].map((s) => (
                                    <div key={s.stage} className="flex items-start gap-3">
                                        <Zap className="w-4 h-4 text-pink-400 shrink-0 mt-0.5" />
                                        <div>
                                            <div className="text-sm text-foreground font-medium">{s.stage}</div>
                                            <div className="text-xs text-muted-foreground">{s.desc}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="hud-panel rounded-lg p-4">
                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Interpretability Insights</h4>
                            <div className="space-y-3">
                                {[
                                    { feat: "Strengths", desc: "Razor-sharp boundaries for macro-structures: horizon line, tree canopies, large landscape regions with >90% confidence" },
                                    { feat: "Weaknesses", desc: "Noisy rock/landscape boundaries in dense boulder fields — color-identical classes are fundamentally ambiguous at pixel level" },
                                    { feat: "Per-Image IoU", desc: "When GT mask available, exact mIoU is computed and displayed (e.g., 0.6055 for image 0000096)" },
                                    { feat: "Class Distribution", desc: "Pixel percentage per terrain class shown in real-time" },
                                ].map((f) => (
                                    <div key={f.feat} className="flex items-start gap-3">
                                        <TrendingUp className="w-4 h-4 text-pink-400 shrink-0 mt-0.5" />
                                        <div>
                                            <div className="text-sm text-foreground font-medium">{f.feat}</div>
                                            <div className="text-xs text-muted-foreground">{f.desc}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </Section>
        </div>
    )
}


/* ─────────────────────── components ─────────────────────── */

function Section({
    icon: Icon,
    title,
    subtitle,
    accent,
    children,
}: {
    icon: typeof Brain
    title: string
    subtitle: string
    accent: string
    children: React.ReactNode
}) {
    const [open, setOpen] = useState(true)

    return (
        <div className={cn("hud-panel rounded-xl overflow-hidden border-l-[3px]", accent)}>
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center gap-3 w-full p-5 text-left hover:bg-secondary/20 transition-colors"
            >
                <Icon className="w-5 h-5 text-muted-foreground shrink-0" />
                <div className="flex-1 min-w-0">
                    <div className="text-base font-semibold text-foreground">{title}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">{subtitle}</div>
                </div>
                {open
                    ? <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
                    : <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />}
            </button>
            {open && <div className="px-5 pb-5">{children}</div>}
        </div>
    )
}

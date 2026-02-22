"use client"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { Save, Check, ChevronDown, ChevronRight, FileCode, X } from "lucide-react"
import { getTerrainClasses } from "@/lib/data"

const CLASS_LIST = getTerrainClasses()

interface ConfigState {
  model: {
    name: string
    architecture: string
    backbone: string
    num_classes: number
    encoder_weights: string
    weights: string
  }
  cost_mapping: {
    traversable: number[]
    obstacle: number[]
    soft: number[]
    ignored: number[]
  }
  cost_values: {
    traversable: number
    soft: number
  }
  planner: {
    strategy: string
    allow_diagonal: boolean
    potential_field: { attractive_gain: number; repulsive_gain: number; repulsive_range: number; max_iterations: number }
    rrt_star: { max_iterations: number; step_size: number; goal_bias: number; rewire_radius: number }
  }
  safety: {
    weight_obstacle: number
    weight_confidence: number
    weight_cost: number
    max_acceptable_cost: number
  }
  preprocessing: {
    target_size: [number, number]
    mean: [number, number, number]
    std: [number, number, number]
  }
  robustness: {
    enabled: boolean
    brightness: { enabled: boolean; factor: number }
    blur: { enabled: boolean; radius: number }
    noise: { enabled: boolean; std: number }
    contrast: { enabled: boolean; factor: number }
  }
}

function configToYaml(c: ConfigState): string {
  const l: string[] = []
  l.push("# Perception Engine Configuration\n")
  l.push("device: auto\n")
  l.push("models:")
  l.push(`  - name: "${c.model.name}"`)
  l.push(`    architecture: "${c.model.architecture}"`)
  l.push(`    backbone: "${c.model.backbone}"`)
  l.push(`    num_classes: ${c.model.num_classes}`)
  l.push(`    encoder_weights: "${c.model.encoder_weights}"`)
  l.push(`    weights: "${c.model.weights}"\n`)

  const names = CLASS_LIST.map(cl => cl.key)
  l.push("class_names:")
  names.forEach(n => l.push(`  - ${n}`))
  l.push("")

  l.push("cost_mapping:")
  const catMap: Record<string, number[]> = { traversable: c.cost_mapping.traversable, obstacle: c.cost_mapping.obstacle, soft: c.cost_mapping.soft, ignored: c.cost_mapping.ignored }
  for (const [cat, ids] of Object.entries(catMap)) {
    l.push(`  ${cat}: [${ids.join(", ")}]`)
  }
  l.push("")

  l.push("cost_values:")
  l.push(`  traversable: ${c.cost_values.traversable}`)
  l.push("  obstacle: .inf")
  l.push(`  soft: ${c.cost_values.soft}`)
  l.push("  ignored: .inf\n")

  l.push("preprocessing:")
  l.push(`  target_size: [${c.preprocessing.target_size.join(", ")}]`)
  l.push("  normalize:")
  l.push(`    mean: [${c.preprocessing.mean.join(", ")}]`)
  l.push(`    std: [${c.preprocessing.std.join(", ")}]\n`)

  l.push("planner:")
  l.push(`  strategy: ${c.planner.strategy}`)
  l.push(`  allow_diagonal: ${c.planner.allow_diagonal}`)
  l.push("  start: null")
  l.push("  goal: null")
  l.push("  potential_field:")
  l.push(`    attractive_gain: ${c.planner.potential_field.attractive_gain}`)
  l.push(`    repulsive_gain: ${c.planner.potential_field.repulsive_gain}`)
  l.push(`    repulsive_range: ${c.planner.potential_field.repulsive_range}`)
  l.push(`    max_iterations: ${c.planner.potential_field.max_iterations}`)
  l.push("  rrt_star:")
  l.push(`    max_iterations: ${c.planner.rrt_star.max_iterations}`)
  l.push(`    step_size: ${c.planner.rrt_star.step_size}`)
  l.push(`    goal_bias: ${c.planner.rrt_star.goal_bias}`)
  l.push(`    rewire_radius: ${c.planner.rrt_star.rewire_radius}\n`)

  l.push("safety:")
  l.push(`  weight_obstacle: ${c.safety.weight_obstacle}`)
  l.push(`  weight_confidence: ${c.safety.weight_confidence}`)
  l.push(`  weight_cost: ${c.safety.weight_cost}`)
  l.push(`  max_acceptable_cost: ${c.safety.max_acceptable_cost}\n`)

  l.push("robustness:")
  l.push(`  enabled: ${c.robustness.enabled}`)
  const perturbs: string[] = []
  if (c.robustness.brightness.enabled) perturbs.push("brightness")
  if (c.robustness.blur.enabled) perturbs.push("blur")
  if (c.robustness.noise.enabled) perturbs.push("noise")
  if (c.robustness.contrast.enabled) perturbs.push("contrast")
  l.push("  perturbations:")
  perturbs.forEach(p => l.push(`    - ${p}`))
  l.push("  params:")
  l.push("    brightness:")
  l.push(`      factor: ${c.robustness.brightness.factor}`)
  l.push("    blur:")
  l.push(`      radius: ${c.robustness.blur.radius}`)
  l.push("    noise:")
  l.push(`      std: ${c.robustness.noise.std}`)
  l.push("    contrast:")
  l.push(`      factor: ${c.robustness.contrast.factor}`)

  return l.join("\n")
}

function ClassChips({
  category,
  selectedIds,
  allClasses,
  onChange,
}: {
  category: string
  selectedIds: number[]
  allClasses: typeof CLASS_LIST
  onChange: (ids: number[]) => void
}) {
  const availableClasses = allClasses.filter(c => selectedIds.includes(c.index) || true)

  return (
    <div className="space-y-2">
      <Label className="text-xs text-muted-foreground capitalize">{category} classes</Label>
      <div className="flex flex-wrap gap-1.5">
        {selectedIds.map((id) => {
          const cls = allClasses.find(c => c.index === id)
          if (!cls) return null
          return (
            <span key={id} className="inline-flex items-center gap-1 text-xs bg-secondary rounded-md px-2 py-1">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: cls.color }} />
              {cls.name}
              <button onClick={() => onChange(selectedIds.filter(i => i !== id))} className="hover:text-foreground text-muted-foreground">
                <X className="w-3 h-3" />
              </button>
            </span>
          )
        })}
        <Select onValueChange={(v) => { const id = parseInt(v); if (!selectedIds.includes(id)) onChange([...selectedIds, id]) }}>
          <SelectTrigger className="w-28 h-7 text-xs bg-secondary border-border text-muted-foreground">
            <SelectValue placeholder="+ Add" />
          </SelectTrigger>
          <SelectContent className="bg-card border-border">
            {availableClasses.filter(c => !selectedIds.includes(c.index)).map(c => (
              <SelectItem key={c.index} value={c.index.toString()}>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: c.color }} />
                  {c.name}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}

function YamlPreview({ config }: { config: ConfigState }) {
  const [expanded, setExpanded] = useState(false)
  const yaml = configToYaml(config)

  return (
    <section className="glass-card rounded-xl overflow-hidden">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-2 w-full px-6 py-4 text-left hover:bg-secondary/30 transition-colors">
        {expanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
        <FileCode className="w-4 h-4 text-primary" />
        <span className="text-sm font-medium text-foreground">YAML Preview</span>
        <span className="text-xs text-muted-foreground ml-1">experiment.yaml</span>
      </button>
      {expanded && (
        <div className="border-t border-border px-6 py-4 max-h-[500px] overflow-auto">
          <pre className="text-xs font-mono leading-relaxed text-muted-foreground whitespace-pre">
            {yaml.split("\n").map((line, i) => {
              if (line.startsWith("#")) return <div key={i} className="text-muted-foreground/50">{line}</div>
              if (line.match(/^\S+:/)) {
                const [k, ...rest] = line.split(":")
                return <div key={i}><span className="text-primary">{k}</span>:<span className="text-emerald-400">{rest.join(":")}</span></div>
              }
              if (line.match(/^\s+\S+:/)) {
                const match = line.match(/^(\s+)(\S+):(.*)$/)
                if (match) return <div key={i}><span>{match[1]}</span><span className="text-cyan-400">{match[2]}</span>:<span className="text-yellow-400">{match[3]}</span></div>
              }
              if (line.match(/^\s+-\s/)) return <div key={i} className="text-yellow-400">{line}</div>
              return <div key={i}>{line}</div>
            })}
          </pre>
        </div>
      )}
    </section>
  )
}

export default function SettingsPage() {
  const [saved, setSaved] = useState(false)
  const [config, setConfig] = useState<ConfigState>({
    model: {
      name: "deeplabv3plus_trained",
      architecture: "deeplabv3plus",
      backbone: "resnet34",
      num_classes: 10,
      encoder_weights: "imagenet",
      weights: "perception_engine/best_model.pth",
    },
    cost_mapping: {
      traversable: [8, 2, 4, 5],
      obstacle: [0, 7, 6],
      soft: [1, 3],
      ignored: [9],
    },
    cost_values: { traversable: 1.0, soft: 2.0 },
    planner: {
      strategy: "astar",
      allow_diagonal: false,
      potential_field: { attractive_gain: 1.0, repulsive_gain: 100.0, repulsive_range: 10, max_iterations: 5000 },
      rrt_star: { max_iterations: 5000, step_size: 5, goal_bias: 0.1, rewire_radius: 15 },
    },
    safety: { weight_obstacle: 0.4, weight_confidence: 0.3, weight_cost: 0.3, max_acceptable_cost: 1000.0 },
    preprocessing: { target_size: [288, 512], mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
    robustness: {
      enabled: true,
      brightness: { enabled: true, factor: 1.4 },
      blur: { enabled: true, radius: 3 },
      noise: { enabled: true, std: 25.0 },
      contrast: { enabled: true, factor: 0.5 },
    },
  })

  const handleSave = () => { setSaved(true); setTimeout(() => setSaved(false), 2000) }

  const updateConfig = useCallback(<K extends keyof ConfigState>(section: K, value: ConfigState[K]) => {
    setConfig(prev => ({ ...prev, [section]: value }))
  }, [])

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-foreground text-balance">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">Configure the perception engine experiment pipeline</p>
      </div>

      {/* Model Configuration */}
      <section className="glass-card rounded-xl p-6 space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Model Configuration</h2>
          <p className="text-xs text-muted-foreground mt-0.5">Select model architecture and weights</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Model Name</Label>
            <Input value={config.model.name} onChange={(e) => updateConfig("model", { ...config.model, name: e.target.value })} className="bg-input border-border text-foreground h-9 text-sm" />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Architecture</Label>
            <Select value={config.model.architecture} onValueChange={(v) => updateConfig("model", { ...config.model, architecture: v })}>
              <SelectTrigger className="bg-input border-border text-foreground h-9 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-card border-border">
                <SelectItem value="deeplabv3plus">DeepLabV3+</SelectItem>
                <SelectItem value="unet">U-Net</SelectItem>
                <SelectItem value="fpn">FPN</SelectItem>
                <SelectItem value="pspnet">PSPNet</SelectItem>
                <SelectItem value="manet">MAnet</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Backbone</Label>
            <Select value={config.model.backbone} onValueChange={(v) => updateConfig("model", { ...config.model, backbone: v })}>
              <SelectTrigger className="bg-input border-border text-foreground h-9 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-card border-border">
                <SelectItem value="resnet34">ResNet-34</SelectItem>
                <SelectItem value="resnet50">ResNet-50</SelectItem>
                <SelectItem value="resnet101">ResNet-101</SelectItem>
                <SelectItem value="efficientnet-b4">EfficientNet-B4</SelectItem>
                <SelectItem value="mobilenet_v2">MobileNetV2</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Encoder Weights</Label>
            <Select value={config.model.encoder_weights} onValueChange={(v) => updateConfig("model", { ...config.model, encoder_weights: v })}>
              <SelectTrigger className="bg-input border-border text-foreground h-9 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-card border-border">
                <SelectItem value="imagenet">ImageNet</SelectItem>
                <SelectItem value="noisy-student">Noisy Student</SelectItem>
                <SelectItem value="advprop">AdvProp</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground">Weights Path</Label>
          <Input value={config.model.weights} onChange={(e) => updateConfig("model", { ...config.model, weights: e.target.value })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
        </div>
      </section>

      {/* Cost Mapping */}
      <section className="glass-card rounded-xl p-6 space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Cost Mapping</h2>
          <p className="text-xs text-muted-foreground mt-0.5">Assign terrain classes to traversal cost categories</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {(["traversable", "obstacle", "soft", "ignored"] as const).map(cat => (
            <ClassChips
              key={cat}
              category={cat}
              selectedIds={config.cost_mapping[cat]}
              allClasses={CLASS_LIST}
              onChange={(ids) => updateConfig("cost_mapping", { ...config.cost_mapping, [cat]: ids })}
            />
          ))}
        </div>
        <div className="grid grid-cols-2 gap-4 mt-2">
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Traversable cost</Label>
            <Input type="number" step="0.1" value={config.cost_values.traversable} onChange={(e) => updateConfig("cost_values", { ...config.cost_values, traversable: parseFloat(e.target.value) || 1 })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Soft terrain cost</Label>
            <Input type="number" step="0.1" value={config.cost_values.soft} onChange={(e) => updateConfig("cost_values", { ...config.cost_values, soft: parseFloat(e.target.value) || 2 })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
        </div>
      </section>

      {/* Planner Settings */}
      <section className="glass-card rounded-xl p-6 space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Planner Settings</h2>
          <p className="text-xs text-muted-foreground mt-0.5">Path planning algorithm configuration</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Strategy</Label>
            <Select value={config.planner.strategy} onValueChange={(v) => updateConfig("planner", { ...config.planner, strategy: v })}>
              <SelectTrigger className="bg-input border-border text-foreground h-9 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-card border-border">
                <SelectItem value="astar">A*</SelectItem>
                <SelectItem value="potential_field">Potential Field</SelectItem>
                <SelectItem value="rrt_star">RRT*</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center justify-between sm:justify-start gap-4">
            <Label className="text-xs text-muted-foreground">Diagonal Movement</Label>
            <Switch checked={config.planner.allow_diagonal} onCheckedChange={(v) => updateConfig("planner", { ...config.planner, allow_diagonal: v })} />
          </div>
        </div>

        {config.planner.strategy === "potential_field" && (
          <div className="grid grid-cols-2 gap-4 mt-2 p-3 bg-secondary/30 rounded-lg">
            <h4 className="col-span-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Potential Field Parameters</h4>
            {([
              ["attractive_gain", "Attractive Gain", 0.1, 10, 0.1],
              ["repulsive_gain", "Repulsive Gain", 1, 500, 1],
              ["repulsive_range", "Repulsive Range", 1, 50, 1],
              ["max_iterations", "Max Iterations", 100, 20000, 100],
            ] as const).map(([key, label, min, max, step]) => (
              <div key={key} className="space-y-2">
                <Label className="text-xs text-muted-foreground">{label}</Label>
                <Input type="number" min={min} max={max} step={step}
                  value={config.planner.potential_field[key]}
                  onChange={(e) => updateConfig("planner", { ...config.planner, potential_field: { ...config.planner.potential_field, [key]: parseFloat(e.target.value) || 0 } })}
                  className="bg-input border-border text-foreground h-9 text-sm font-mono"
                />
              </div>
            ))}
          </div>
        )}

        {config.planner.strategy === "rrt_star" && (
          <div className="grid grid-cols-2 gap-4 mt-2 p-3 bg-secondary/30 rounded-lg">
            <h4 className="col-span-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">RRT* Parameters</h4>
            {([
              ["max_iterations", "Max Iterations", 100, 20000, 100],
              ["step_size", "Step Size", 1, 20, 1],
              ["goal_bias", "Goal Bias", 0.01, 0.5, 0.01],
              ["rewire_radius", "Rewire Radius", 1, 50, 1],
            ] as const).map(([key, label, min, max, step]) => (
              <div key={key} className="space-y-2">
                <Label className="text-xs text-muted-foreground">{label}</Label>
                <Input type="number" min={min} max={max} step={step}
                  value={config.planner.rrt_star[key]}
                  onChange={(e) => updateConfig("planner", { ...config.planner, rrt_star: { ...config.planner.rrt_star, [key]: parseFloat(e.target.value) || 0 } })}
                  className="bg-input border-border text-foreground h-9 text-sm font-mono"
                />
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Safety Weights */}
      <section className="glass-card rounded-xl p-6 space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Safety Scoring</h2>
          <p className="text-xs text-muted-foreground mt-0.5">Weighted safety score computation</p>
        </div>
        <div className="space-y-4">
          {([
            ["weight_obstacle", "Obstacle Weight"],
            ["weight_confidence", "Confidence Weight"],
            ["weight_cost", "Cost Weight"],
          ] as const).map(([key, label]) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs text-muted-foreground">{label}</Label>
                <span className="text-xs font-mono text-foreground">{config.safety[key].toFixed(2)}</span>
              </div>
              <Slider
                value={[config.safety[key]]}
                onValueChange={([v]) => updateConfig("safety", { ...config.safety, [key]: v })}
                min={0} max={1} step={0.05}
              />
            </div>
          ))}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Max Acceptable Cost</Label>
            <Input type="number" step="50"
              value={config.safety.max_acceptable_cost}
              onChange={(e) => updateConfig("safety", { ...config.safety, max_acceptable_cost: parseFloat(e.target.value) || 1000 })}
              className="bg-input border-border text-foreground h-9 text-sm font-mono"
            />
          </div>
        </div>
      </section>

      {/* Preprocessing */}
      <section className="glass-card rounded-xl p-6 space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Preprocessing</h2>
          <p className="text-xs text-muted-foreground mt-0.5">Image preprocessing parameters</p>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Target Height</Label>
            <Input type="number" value={config.preprocessing.target_size[0]} onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, target_size: [parseInt(e.target.value) || 288, config.preprocessing.target_size[1]] })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Target Width</Label>
            <Input type="number" value={config.preprocessing.target_size[1]} onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, target_size: [config.preprocessing.target_size[0], parseInt(e.target.value) || 512] })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Mean R</Label>
            <Input type="number" step="0.001" value={config.preprocessing.mean[0]} onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, mean: [parseFloat(e.target.value) || 0, config.preprocessing.mean[1], config.preprocessing.mean[2]] })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Mean G</Label>
            <Input type="number" step="0.001" value={config.preprocessing.mean[1]} onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, mean: [config.preprocessing.mean[0], parseFloat(e.target.value) || 0, config.preprocessing.mean[2]] })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Mean B</Label>
            <Input type="number" step="0.001" value={config.preprocessing.mean[2]} onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, mean: [config.preprocessing.mean[0], config.preprocessing.mean[1], parseFloat(e.target.value) || 0] })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Std R</Label>
            <Input type="number" step="0.001" value={config.preprocessing.std[0]} onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, std: [parseFloat(e.target.value) || 0, config.preprocessing.std[1], config.preprocessing.std[2]] })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Std G</Label>
            <Input type="number" step="0.001" value={config.preprocessing.std[1]} onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, std: [config.preprocessing.std[0], parseFloat(e.target.value) || 0, config.preprocessing.std[2]] })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Std B</Label>
            <Input type="number" step="0.001" value={config.preprocessing.std[2]} onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, std: [config.preprocessing.std[0], config.preprocessing.std[1], parseFloat(e.target.value) || 0] })} className="bg-input border-border text-foreground h-9 text-sm font-mono" />
          </div>
        </div>
      </section>

      {/* Robustness Testing */}
      <section className="glass-card rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-foreground">Robustness Testing</h2>
            <p className="text-xs text-muted-foreground mt-0.5">Perturbation tests and severity</p>
          </div>
          <Switch checked={config.robustness.enabled} onCheckedChange={(v) => updateConfig("robustness", { ...config.robustness, enabled: v })} />
        </div>
        {config.robustness.enabled && (
          <div className="space-y-5">
            {/* Brightness */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm text-foreground">Brightness</Label>
                <Switch checked={config.robustness.brightness.enabled} onCheckedChange={(v) => updateConfig("robustness", { ...config.robustness, brightness: { ...config.robustness.brightness, enabled: v } })} />
              </div>
              {config.robustness.brightness.enabled && (
                <div className="flex items-center gap-4">
                  <span className="text-xs text-muted-foreground w-12">Factor</span>
                  <Slider value={[config.robustness.brightness.factor]} onValueChange={([v]) => updateConfig("robustness", { ...config.robustness, brightness: { ...config.robustness.brightness, factor: v } })} min={0.5} max={3.0} step={0.1} className="flex-1" />
                  <span className="text-xs font-mono text-muted-foreground w-10 text-right">{config.robustness.brightness.factor.toFixed(1)}</span>
                </div>
              )}
            </div>
            {/* Blur */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm text-foreground">Blur</Label>
                <Switch checked={config.robustness.blur.enabled} onCheckedChange={(v) => updateConfig("robustness", { ...config.robustness, blur: { ...config.robustness.blur, enabled: v } })} />
              </div>
              {config.robustness.blur.enabled && (
                <div className="flex items-center gap-4">
                  <span className="text-xs text-muted-foreground w-12">Radius</span>
                  <Slider value={[config.robustness.blur.radius]} onValueChange={([v]) => updateConfig("robustness", { ...config.robustness, blur: { ...config.robustness.blur, radius: v } })} min={1} max={15} step={1} className="flex-1" />
                  <span className="text-xs font-mono text-muted-foreground w-10 text-right">{config.robustness.blur.radius}</span>
                </div>
              )}
            </div>
            {/* Noise */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm text-foreground">Noise</Label>
                <Switch checked={config.robustness.noise.enabled} onCheckedChange={(v) => updateConfig("robustness", { ...config.robustness, noise: { ...config.robustness.noise, enabled: v } })} />
              </div>
              {config.robustness.noise.enabled && (
                <div className="flex items-center gap-4">
                  <span className="text-xs text-muted-foreground w-12">Std</span>
                  <Slider value={[config.robustness.noise.std]} onValueChange={([v]) => updateConfig("robustness", { ...config.robustness, noise: { ...config.robustness.noise, std: v } })} min={1} max={100} step={1} className="flex-1" />
                  <span className="text-xs font-mono text-muted-foreground w-10 text-right">{config.robustness.noise.std.toFixed(0)}</span>
                </div>
              )}
            </div>
            {/* Contrast */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm text-foreground">Contrast</Label>
                <Switch checked={config.robustness.contrast.enabled} onCheckedChange={(v) => updateConfig("robustness", { ...config.robustness, contrast: { ...config.robustness.contrast, enabled: v } })} />
              </div>
              {config.robustness.contrast.enabled && (
                <div className="flex items-center gap-4">
                  <span className="text-xs text-muted-foreground w-12">Factor</span>
                  <Slider value={[config.robustness.contrast.factor]} onValueChange={([v]) => updateConfig("robustness", { ...config.robustness, contrast: { ...config.robustness.contrast, factor: v } })} min={0.1} max={2.0} step={0.05} className="flex-1" />
                  <span className="text-xs font-mono text-muted-foreground w-10 text-right">{config.robustness.contrast.factor.toFixed(2)}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </section>

      <YamlPreview config={config} />

      <Button onClick={handleSave} className="bg-primary hover:bg-primary/90 text-primary-foreground font-medium px-8" disabled={saved}>
        {saved ? (<><Check className="w-4 h-4 mr-2" />Saved</>) : (<><Save className="w-4 h-4 mr-2" />Save Config</>)}
      </Button>
    </div>
  )
}

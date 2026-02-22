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
    name: string; architecture: string; backbone: string
    num_classes: number; encoder_weights: string; weights: string
  }
  cost_mapping: { traversable: number[]; obstacle: number[]; soft: number[]; ignored: number[] }
  cost_values: { traversable: number; soft: number }
  planner: {
    strategy: string; allow_diagonal: boolean
    potential_field: { attractive_gain: number; repulsive_gain: number; repulsive_range: number; max_iterations: number }
    rrt_star: { max_iterations: number; step_size: number; goal_bias: number; rewire_radius: number }
  }
  safety: { weight_obstacle: number; weight_confidence: number; weight_cost: number; max_acceptable_cost: number }
  preprocessing: { target_size: [number, number]; mean: [number, number, number]; std: [number, number, number] }
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
  l.push("# Perception Engine — Mission Configuration\n")
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
  for (const [cat, ids] of Object.entries(catMap)) l.push(`  ${cat}: [${ids.join(", ")}]`)
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

function ClassChips({ category, selectedIds, onChange }: { category: string; selectedIds: number[]; onChange: (ids: number[]) => void }) {
  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium text-muted-foreground capitalize">{category} classes</Label>
      <div className="flex flex-wrap gap-2">
        {selectedIds.map((id) => {
          const cls = CLASS_LIST.find(c => c.index === id)
          if (!cls) return null
          return (
            <span key={id} className="inline-flex items-center gap-1.5 text-sm bg-secondary rounded-lg px-2.5 py-1 border border-border">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: cls.color }} />
              {cls.name}
              <button onClick={() => onChange(selectedIds.filter(i => i !== id))} className="hover:text-foreground text-muted-foreground ml-0.5">
                <X className="w-3.5 h-3.5" />
              </button>
            </span>
          )
        })}
        <Select onValueChange={(v) => { const id = parseInt(v); if (!selectedIds.includes(id)) onChange([...selectedIds, id]) }}>
          <SelectTrigger className="w-24 h-8 text-sm bg-secondary border-border text-muted-foreground rounded-lg">
            <SelectValue placeholder="+ Add" />
          </SelectTrigger>
          <SelectContent className="bg-card border-border">
            {CLASS_LIST.filter(c => !selectedIds.includes(c.index)).map(c => (
              <SelectItem key={c.index} value={c.index.toString()}>
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
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

function Section({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section className="hud-panel rounded-xl p-6 space-y-5">
      <div>
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>
      </div>
      {children}
    </section>
  )
}

export default function ConfigPage() {
  const [saved, setSaved] = useState(false)
  const [yamlOpen, setYamlOpen] = useState(false)
  const [config, setConfig] = useState<ConfigState>({
    model: { name: "deeplabv3plus_trained", architecture: "deeplabv3plus", backbone: "resnet34", num_classes: 10, encoder_weights: "imagenet", weights: "perception_engine/best_model.pth" },
    cost_mapping: { traversable: [8, 2, 4, 5], obstacle: [0, 7, 6], soft: [1, 3], ignored: [9] },
    cost_values: { traversable: 1.0, soft: 2.0 },
    planner: {
      strategy: "astar", allow_diagonal: false,
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

  const yaml = configToYaml(config)

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground tracking-tight">Configuration</h1>
        <p className="text-sm text-muted-foreground mt-1">Configure experiment pipeline parameters and model settings</p>
      </div>

      {/* Model */}
      <Section title="Model" subtitle="Architecture and weights selection">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Model Name</Label>
            <Input
              value={config.model.name}
              onChange={(e) => updateConfig("model", { ...config.model, name: e.target.value })}
              className="bg-input border-border text-foreground h-9 text-sm font-mono"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Architecture</Label>
            <Select value={config.model.architecture} onValueChange={(v) => updateConfig("model", { ...config.model, architecture: v })}>
              <SelectTrigger className="bg-input border-border text-foreground h-9 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-card border-border">
                {["deeplabv3plus", "unet", "fpn", "pspnet", "manet"].map(a => <SelectItem key={a} value={a}>{a}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Backbone</Label>
            <Select value={config.model.backbone} onValueChange={(v) => updateConfig("model", { ...config.model, backbone: v })}>
              <SelectTrigger className="bg-input border-border text-foreground h-9 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-card border-border">
                {["resnet34", "resnet50", "resnet101", "efficientnet-b4", "mobilenet_v2"].map(b => <SelectItem key={b} value={b}>{b}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Encoder Weights</Label>
            <Select value={config.model.encoder_weights} onValueChange={(v) => updateConfig("model", { ...config.model, encoder_weights: v })}>
              <SelectTrigger className="bg-input border-border text-foreground h-9 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-card border-border">
                {["imagenet", "noisy-student", "advprop"].map(w => <SelectItem key={w} value={w}>{w}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="space-y-1.5">
          <Label className="text-sm font-medium text-muted-foreground">Weights Path</Label>
          <Input
            value={config.model.weights}
            onChange={(e) => updateConfig("model", { ...config.model, weights: e.target.value })}
            className="bg-input border-border text-foreground h-9 text-sm font-mono"
          />
        </div>
      </Section>

      {/* Cost Mapping */}
      <Section title="Cost Mapping" subtitle="Assign terrain classes to traversal categories">
        <div className="grid grid-cols-2 gap-5">
          {(["traversable", "obstacle", "soft", "ignored"] as const).map(cat => (
            <ClassChips key={cat} category={cat} selectedIds={config.cost_mapping[cat]} onChange={(ids) => updateConfig("cost_mapping", { ...config.cost_mapping, [cat]: ids })} />
          ))}
        </div>
        <div className="grid grid-cols-2 gap-4 pt-2">
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Traversable cost</Label>
            <Input
              type="number" step="0.1"
              value={config.cost_values.traversable}
              onChange={(e) => updateConfig("cost_values", { ...config.cost_values, traversable: parseFloat(e.target.value) || 1 })}
              className="bg-input border-border text-foreground h-9 text-sm font-mono"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Soft terrain cost</Label>
            <Input
              type="number" step="0.1"
              value={config.cost_values.soft}
              onChange={(e) => updateConfig("cost_values", { ...config.cost_values, soft: parseFloat(e.target.value) || 2 })}
              className="bg-input border-border text-foreground h-9 text-sm font-mono"
            />
          </div>
        </div>
      </Section>

      {/* Planner */}
      <Section title="Planner" subtitle="Path planning algorithm and parameters">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Strategy</Label>
            <Select value={config.planner.strategy} onValueChange={(v) => updateConfig("planner", { ...config.planner, strategy: v })}>
              <SelectTrigger className="bg-input border-border text-foreground h-9 text-sm"><SelectValue /></SelectTrigger>
              <SelectContent className="bg-card border-border">
                <SelectItem value="astar">A*</SelectItem>
                <SelectItem value="potential_field">Potential Field</SelectItem>
                <SelectItem value="rrt_star">RRT*</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-end gap-3 pb-0.5">
            <div className="flex items-center gap-3">
              <Switch
                checked={config.planner.allow_diagonal}
                onCheckedChange={(v) => updateConfig("planner", { ...config.planner, allow_diagonal: v })}
              />
              <Label className="text-sm text-foreground">Allow diagonal movement</Label>
            </div>
          </div>
        </div>

        {config.planner.strategy === "potential_field" && (
          <div className="grid grid-cols-2 gap-4 p-4 bg-secondary/20 rounded-lg border border-border/50">
            <div className="col-span-2 text-sm font-medium text-muted-foreground">Potential Field Parameters</div>
            {([
              ["attractive_gain", "Attractive Gain", 0.1, 10, 0.1],
              ["repulsive_gain", "Repulsive Gain", 1, 500, 1],
              ["repulsive_range", "Repulsive Range", 1, 50, 1],
              ["max_iterations", "Max Iterations", 100, 20000, 100],
            ] as const).map(([key, label, min, max, step]) => (
              <div key={key} className="space-y-1.5">
                <Label className="text-sm font-medium text-muted-foreground">{label}</Label>
                <Input
                  type="number" min={min} max={max} step={step}
                  value={config.planner.potential_field[key]}
                  onChange={(e) => updateConfig("planner", { ...config.planner, potential_field: { ...config.planner.potential_field, [key]: parseFloat(e.target.value) || 0 } })}
                  className="bg-input border-border text-foreground h-9 text-sm font-mono"
                />
              </div>
            ))}
          </div>
        )}

        {config.planner.strategy === "rrt_star" && (
          <div className="grid grid-cols-2 gap-4 p-4 bg-secondary/20 rounded-lg border border-border/50">
            <div className="col-span-2 text-sm font-medium text-muted-foreground">RRT* Parameters</div>
            {([
              ["max_iterations", "Max Iterations", 100, 20000, 100],
              ["step_size", "Step Size", 1, 20, 1],
              ["goal_bias", "Goal Bias", 0.01, 0.5, 0.01],
              ["rewire_radius", "Rewire Radius", 1, 50, 1],
            ] as const).map(([key, label, min, max, step]) => (
              <div key={key} className="space-y-1.5">
                <Label className="text-sm font-medium text-muted-foreground">{label}</Label>
                <Input
                  type="number" min={min} max={max} step={step}
                  value={config.planner.rrt_star[key]}
                  onChange={(e) => updateConfig("planner", { ...config.planner, rrt_star: { ...config.planner.rrt_star, [key]: parseFloat(e.target.value) || 0 } })}
                  className="bg-input border-border text-foreground h-9 text-sm font-mono"
                />
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* Safety Scoring */}
      <Section title="Safety Scoring" subtitle="Weighted components for the composite safety score">
        <div className="space-y-5">
          {([["weight_obstacle", "Obstacle Weight"], ["weight_confidence", "Confidence Weight"], ["weight_cost", "Cost Weight"]] as const).map(([key, label]) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium text-muted-foreground">{label}</Label>
                <span className="text-sm font-mono font-semibold text-amber-400">{config.safety[key].toFixed(2)}</span>
              </div>
              <Slider
                value={[config.safety[key]]}
                onValueChange={([v]) => updateConfig("safety", { ...config.safety, [key]: v })}
                min={0} max={1} step={0.05}
              />
            </div>
          ))}
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Max Acceptable Cost</Label>
            <Input
              type="number" step="50"
              value={config.safety.max_acceptable_cost}
              onChange={(e) => updateConfig("safety", { ...config.safety, max_acceptable_cost: parseFloat(e.target.value) || 1000 })}
              className="bg-input border-border text-foreground h-9 text-sm font-mono"
            />
          </div>
        </div>
      </Section>

      {/* Preprocessing */}
      <Section title="Preprocessing" subtitle="Image normalization and target resolution">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Target Height</Label>
            <Input
              type="number"
              value={config.preprocessing.target_size[0]}
              onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, target_size: [parseInt(e.target.value) || 288, config.preprocessing.target_size[1]] })}
              className="bg-input border-border text-foreground h-9 text-sm font-mono"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-sm font-medium text-muted-foreground">Target Width</Label>
            <Input
              type="number"
              value={config.preprocessing.target_size[1]}
              onChange={(e) => updateConfig("preprocessing", { ...config.preprocessing, target_size: [config.preprocessing.target_size[0], parseInt(e.target.value) || 512] })}
              className="bg-input border-border text-foreground h-9 text-sm font-mono"
            />
          </div>
        </div>
        <div>
          <Label className="text-sm font-medium text-muted-foreground mb-2 block">Normalize Mean (R, G, B)</Label>
          <div className="grid grid-cols-3 gap-4">
            {[0, 1, 2].map(i => (
              <Input
                key={`mean-${i}`}
                type="number" step="0.001"
                value={config.preprocessing.mean[i]}
                onChange={(e) => { const m = [...config.preprocessing.mean] as [number, number, number]; m[i] = parseFloat(e.target.value) || 0; updateConfig("preprocessing", { ...config.preprocessing, mean: m }) }}
                className="bg-input border-border text-foreground h-9 text-sm font-mono"
              />
            ))}
          </div>
        </div>
        <div>
          <Label className="text-sm font-medium text-muted-foreground mb-2 block">Normalize Std (R, G, B)</Label>
          <div className="grid grid-cols-3 gap-4">
            {[0, 1, 2].map(i => (
              <Input
                key={`std-${i}`}
                type="number" step="0.001"
                value={config.preprocessing.std[i]}
                onChange={(e) => { const s = [...config.preprocessing.std] as [number, number, number]; s[i] = parseFloat(e.target.value) || 0; updateConfig("preprocessing", { ...config.preprocessing, std: s }) }}
                className="bg-input border-border text-foreground h-9 text-sm font-mono"
              />
            ))}
          </div>
        </div>
      </Section>

      {/* Robustness Testing */}
      <Section title="Robustness Testing" subtitle="Perturbation tests and severity levels">
        <div className="flex items-center justify-between">
          <span className="text-sm text-foreground">Enable robustness testing</span>
          <Switch
            checked={config.robustness.enabled}
            onCheckedChange={(v) => updateConfig("robustness", { ...config.robustness, enabled: v })}
          />
        </div>
        {config.robustness.enabled && (
          <div className="space-y-4 pt-1">
            {([
              { key: "brightness" as const, label: "Brightness", param: "factor", min: 0.5, max: 3.0, step: 0.1 },
              { key: "blur" as const, label: "Blur", param: "radius", min: 1, max: 15, step: 1 },
              { key: "noise" as const, label: "Noise", param: "std", min: 1, max: 100, step: 1 },
              { key: "contrast" as const, label: "Contrast", param: "factor", min: 0.1, max: 2.0, step: 0.05 },
            ]).map(({ key, label, param, min, max, step }) => (
              <div key={key} className="space-y-2.5 p-4 bg-secondary/10 rounded-lg border border-border/40">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">{label}</span>
                  <Switch
                    checked={config.robustness[key].enabled}
                    onCheckedChange={(v) => updateConfig("robustness", { ...config.robustness, [key]: { ...config.robustness[key], enabled: v } })}
                  />
                </div>
                {config.robustness[key].enabled && (
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-muted-foreground w-14 shrink-0 capitalize">{param}</span>
                    <Slider
                      value={[(config.robustness[key] as Record<string, number>)[param]]}
                      onValueChange={([v]) => updateConfig("robustness", { ...config.robustness, [key]: { ...config.robustness[key], [param]: v } })}
                      min={min} max={max} step={step} className="flex-1"
                    />
                    <span className="text-sm font-mono font-semibold text-amber-400 w-12 text-right">
                      {((config.robustness[key] as Record<string, number>)[param]).toFixed(step < 1 ? 2 : 0)}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* YAML Preview */}
      <div className="hud-panel rounded-xl overflow-hidden">
        <button
          onClick={() => setYamlOpen(!yamlOpen)}
          className="flex items-center gap-3 w-full px-5 py-4 text-left hover:bg-secondary/20 transition-colors"
        >
          {yamlOpen
            ? <ChevronDown className="w-4 h-4 text-muted-foreground" />
            : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
          <FileCode className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-foreground">YAML Preview</span>
          <span className="text-sm text-muted-foreground ml-1">experiment.yaml</span>
        </button>
        {yamlOpen && (
          <div className="border-t border-border p-5 max-h-[500px] overflow-auto">
            <pre className="text-sm font-mono leading-relaxed text-muted-foreground whitespace-pre">
              {yaml.split("\n").map((line, i) => {
                if (line.startsWith("#")) return <div key={i} className="text-muted-foreground/40">{line}</div>
                if (line.match(/^\S+:/)) {
                  const [k, ...rest] = line.split(":")
                  return <div key={i}><span className="text-primary">{k}</span>:<span className="text-emerald-400">{rest.join(":")}</span></div>
                }
                if (line.match(/^\s+\S+:/)) {
                  const match = line.match(/^(\s+)(\S+):(.*)$/)
                  if (match) return <div key={i}><span>{match[1]}</span><span className="text-cyan-400">{match[2]}</span>:<span className="text-amber-400">{match[3]}</span></div>
                }
                if (line.match(/^\s+-\s/)) return <div key={i} className="text-amber-400">{line}</div>
                return <div key={i}>{line}</div>
              })}
            </pre>
          </div>
        )}
      </div>

      {/* Save button */}
      <Button
        onClick={handleSave}
        className="bg-primary hover:bg-primary/90 text-primary-foreground text-sm font-medium px-8 h-10"
        disabled={saved}
      >
        {saved
          ? (<><Check className="w-4 h-4 mr-2" />Saved</>)
          : (<><Save className="w-4 h-4 mr-2" />Save Configuration</>)}
      </Button>
    </div>
  )
}

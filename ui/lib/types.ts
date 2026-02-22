export interface SegmentationInfo {
  mask_shape: [number, number];
  num_classes: number;
  confidence_mean: number;
  confidence_std: number;
  class_distribution: Record<string, number>;
}

export interface NavigationInfo {
  path_found: boolean;
  path_cost: number;
  path_length: number;
}

export interface SafetyInfo {
  safety_score: number;
  obstacle_overlap_pct: number;
  avg_confidence: number;
  path_cost: number;
}

export interface MetricsInfo {
  pixel_accuracy: number;
  mean_iou: number;
  frequency_weighted_iou: number;
  dice_coefficient: number;
  per_class_iou: Record<string, number | null>;
  confusion_matrix: number[][];
}

export interface RobustnessResult {
  pixel_accuracy: number;
  mean_iou: number;
  miou_drop: number;
  per_class_iou: Record<string, number | null>;
  inference_time_ms: number;
}

export interface ModelReport {
  model_name: string;
  inference_time_ms: number;
  segmentation: SegmentationInfo;
  navigation: NavigationInfo;
  safety: SafetyInfo;
  metrics: MetricsInfo;
  robustness: Record<string, RobustnessResult>;
}

export interface ModelConfig {
  name: string;
  architecture: string;
  backbone: string;
  num_classes: number;
  encoder_weights: string;
  weights: string;
}

export interface CostMapping {
  traversable: number[];
  obstacle: number[];
  soft: number[];
  ignored: number[];
}

export interface CostValues {
  traversable: number;
  obstacle: number;
  soft: number;
  ignored: number;
}

export interface PlannerConfig {
  strategy: string;
  allow_diagonal: boolean;
  start: [number, number] | null;
  goal: [number, number] | null;
  potential_field: {
    attractive_gain: number;
    repulsive_gain: number;
    repulsive_range: number;
    max_iterations: number;
  };
  rrt_star: {
    max_iterations: number;
    step_size: number;
    goal_bias: number;
    rewire_radius: number;
  };
}

export interface SafetyConfig {
  weight_obstacle: number;
  weight_confidence: number;
  weight_cost: number;
  max_acceptable_cost: number;
}

export interface RobustnessConfig {
  enabled: boolean;
  perturbations: string[];
  params: Record<string, Record<string, number>>;
}

export interface ExperimentConfig {
  device: string;
  models: ModelConfig[];
  class_names: string[];
  mask_value_mapping: Record<string, number>;
  cost_mapping: CostMapping;
  cost_values: CostValues;
  preprocessing: {
    target_size: [number, number];
    normalize: { mean: number[]; std: number[] };
  };
  planner: PlannerConfig;
  safety: SafetyConfig;
  robustness: RobustnessConfig;
  dataset: {
    image_dir: string;
    gt_dir: string;
    image_ext: string;
    gt_ext: string;
    max_samples: number | null;
  };
  output: {
    save_visualizations: boolean;
    output_dir: string;
  };
}

export interface BenchmarkReport {
  num_models: number;
  reports: ModelReport[];
  config: ExperimentConfig;
}

export interface PerturbationVerdict {
  drop: string;
  verdict: string;
}

export interface ExplanationSection {
  title: string;
  metric?: string;
  explanation: string;
  best_class?: string;
  worst_class?: string;
  all_classes?: Record<string, string>;
  perturbations?: Record<string, PerturbationVerdict>;
}

export interface ModelExplanation {
  model_name: string;
  sections: ExplanationSection[];
}

export interface ExplanationReport {
  title: string;
  overview: string;
  models: ModelExplanation[];
}

export interface TerrainClass {
  name: string;
  key: string;
  color: string;
  index: number;
}

export interface TerrainAsset {
  id: string;
  label: string;
  terrain: string;
  maskOverlay: string;
  pathOverlay: string;
  confidence: string;
}

export type OverlayLayer = "segmentation" | "costmap" | "confidence" | "path" | "grid"

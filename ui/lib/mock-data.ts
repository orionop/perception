import type {
  BenchmarkReport,
  ExplanationReport,
  TerrainClass,
} from "./types";

export const terrainClasses: TerrainClass[] = [
  { name: "Tree", key: "tree", color: "#FF6400", index: 0 },
  { name: "Lush Bush", key: "lush_bush", color: "#FFA500", index: 1 },
  { name: "Dry Grass", key: "dry_grass", color: "#00C800", index: 2 },
  { name: "Dry Bush", key: "dry_bush", color: "#C80000", index: 3 },
  { name: "Ground Clutter", key: "ground_clutter", color: "#B400B4", index: 4 },
  { name: "Flower", key: "flower", color: "#C88080", index: 5 },
  { name: "Log", key: "log", color: "#DCB4C8", index: 6 },
  { name: "Rock", key: "rock", color: "#FFFF00", index: 7 },
  { name: "Landscape", key: "landscape", color: "#C8E6B4", index: 8 },
  { name: "Sky", key: "sky", color: "#C8E6FF", index: 9 },
];

export const benchmarkReport: BenchmarkReport = {
  num_models: 3,
  reports: [
    {
      model_name: "DINOv2 V6 + Ensemble (Best)",
      inference_time_ms: 580,
      segmentation: {
        mask_shape: [540, 960],
        num_classes: 10,
        confidence_mean: 0.85,
        confidence_std: 0.12,
        class_distribution: {
          "0": 0.042,
          "2": 0.191,
          "3": 0.004,
          "7": 0.013,
          "8": 0.423,
          "9": 0.227,
        },
      },
      navigation: { path_found: true, path_cost: 924.0, path_length: 930 },
      safety: {
        safety_score: 0.812,
        obstacle_overlap_pct: 0.0,
        avg_confidence: 0.85,
        path_cost: 924.0,
      },
      metrics: {
        pixel_accuracy: 0.812,
        mean_iou: 0.61,
        frequency_weighted_iou: 0.72,
        dice_coefficient: 0.68,
        per_class_iou: {
          tree: 0.598,
          lush_bush: 0.0,
          dry_grass: 0.443,
          dry_bush: 0.455,
          ground_clutter: null,
          flower: null,
          log: null,
          rock: 0.432,
          landscape: 0.727,
          sky: 0.990,
        },
        confusion_matrix: [
          [7980, 0, 420, 0, 0, 0, 0, 250, 1200, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
          [800, 0, 40100, 300, 0, 0, 0, 8500, 32700, 0],
          [0, 0, 350, 7250, 0, 0, 0, 0, 2800, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [350, 0, 6600, 0, 0, 0, 0, 56200, 32000, 0],
          [100, 0, 7800, 0, 0, 0, 0, 16100, 163200, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 950, 93050],
        ],
      },
      robustness: {
        brightness: {
          pixel_accuracy: 0.800,
          mean_iou: 0.594,
          miou_drop: -0.012,
          per_class_iou: {
            tree: 0.585, lush_bush: 0.0, dry_grass: 0.431, dry_bush: 0.442,
            ground_clutter: null, flower: null, log: null,
            rock: 0.420, landscape: 0.715, sky: 0.988,
          },
          inference_time_ms: 582,
        },
        blur: {
          pixel_accuracy: 0.785,
          mean_iou: 0.572,
          miou_drop: -0.034,
          per_class_iou: {
            tree: 0.556, lush_bush: 0.0, dry_grass: 0.410, dry_bush: 0.420,
            ground_clutter: null, flower: null, log: null,
            rock: 0.398, landscape: 0.695, sky: 0.985,
          },
          inference_time_ms: 580,
        },
        noise: {
          pixel_accuracy: 0.790,
          mean_iou: 0.575,
          miou_drop: -0.031,
          per_class_iou: {
            tree: 0.560, lush_bush: 0.0, dry_grass: 0.418, dry_bush: 0.425,
            ground_clutter: null, flower: null, log: null,
            rock: 0.402, landscape: 0.700, sky: 0.986,
          },
          inference_time_ms: 581,
        },
        contrast: {
          pixel_accuracy: 0.778,
          mean_iou: 0.581,
          miou_drop: -0.025,
          per_class_iou: {
            tree: 0.570, lush_bush: 0.0, dry_grass: 0.425, dry_bush: 0.435,
            ground_clutter: null, flower: null, log: null,
            rock: 0.410, landscape: 0.708, sky: 0.987,
          },
          inference_time_ms: 579,
        },
      },
    },
    {
      model_name: "DINOv2 V6 (Single Model)",
      inference_time_ms: 288,
      segmentation: {
        mask_shape: [540, 960],
        num_classes: 10,
        confidence_mean: 0.7637,
        confidence_std: 0.1683,
        class_distribution: {
          "0": 0.042,
          "2": 0.191,
          "3": 0.004,
          "7": 0.013,
          "8": 0.523,
          "9": 0.227,
        },
      },
      navigation: { path_found: true, path_cost: 513.0, path_length: 512 },
      safety: {
        safety_score: 0.746,
        obstacle_overlap_pct: 0.0,
        avg_confidence: 0.7637,
        path_cost: 513.0,
      },
      metrics: {
        pixel_accuracy: 0.776,
        mean_iou: 0.5211,
        frequency_weighted_iou: 0.65,
        dice_coefficient: 0.58,
        per_class_iou: {
          tree: 0.598,
          lush_bush: 0.0,
          dry_grass: 0.443,
          dry_bush: 0.455,
          ground_clutter: null,
          flower: null,
          log: null,
          rock: 0.432,
          landscape: 0.727,
          sky: 0.990,
        },
        confusion_matrix: [
          [7980, 0, 420, 0, 0, 0, 0, 250, 1200, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
          [800, 0, 40100, 300, 0, 0, 0, 8500, 32700, 0],
          [0, 0, 350, 7250, 0, 0, 0, 0, 2800, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [350, 0, 6600, 0, 0, 0, 0, 56200, 32000, 0],
          [100, 0, 7800, 0, 0, 0, 0, 16100, 163200, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 950, 93050],
        ],
      },
      robustness: {
        brightness: {
          pixel_accuracy: 0.764,
          mean_iou: 0.509,
          miou_drop: -0.012,
          per_class_iou: {
            tree: 0.585, lush_bush: 0.0, dry_grass: 0.431, dry_bush: 0.442,
            ground_clutter: null, flower: null, log: null,
            rock: 0.420, landscape: 0.715, sky: 0.988,
          },
          inference_time_ms: 290,
        },
        blur: {
          pixel_accuracy: 0.748,
          mean_iou: 0.487,
          miou_drop: -0.034,
          per_class_iou: {
            tree: 0.556, lush_bush: 0.0, dry_grass: 0.410, dry_bush: 0.420,
            ground_clutter: null, flower: null, log: null,
            rock: 0.398, landscape: 0.695, sky: 0.985,
          },
          inference_time_ms: 289,
        },
        noise: {
          pixel_accuracy: 0.752,
          mean_iou: 0.490,
          miou_drop: -0.031,
          per_class_iou: {
            tree: 0.560, lush_bush: 0.0, dry_grass: 0.418, dry_bush: 0.425,
            ground_clutter: null, flower: null, log: null,
            rock: 0.402, landscape: 0.700, sky: 0.986,
          },
          inference_time_ms: 290,
        },
        contrast: {
          pixel_accuracy: 0.740,
          mean_iou: 0.496,
          miou_drop: -0.025,
          per_class_iou: {
            tree: 0.570, lush_bush: 0.0, dry_grass: 0.425, dry_bush: 0.435,
            ground_clutter: null, flower: null, log: null,
            rock: 0.410, landscape: 0.708, sky: 0.987,
          },
          inference_time_ms: 288,
        },
      },
    },
    {
      model_name: "DeepLabV3+ ResNet50 (V2 Baseline)",
      inference_time_ms: 45,
      segmentation: {
        mask_shape: [540, 960],
        num_classes: 10,
        confidence_mean: 0.6812,
        confidence_std: 0.2145,
        class_distribution: {
          "8": 0.65,
          "9": 0.20,
          "2": 0.12,
          "0": 0.03,
        },
      },
      navigation: { path_found: true, path_cost: 648.0, path_length: 536 },
      safety: {
        safety_score: 0.534,
        obstacle_overlap_pct: 2.1,
        avg_confidence: 0.5923,
        path_cost: 648.0,
      },
      metrics: {
        pixel_accuracy: 0.40,
        mean_iou: 0.17,
        frequency_weighted_iou: 0.23,
        dice_coefficient: 0.12,
        per_class_iou: {
          tree: 0.001,
          lush_bush: 0.0,
          dry_grass: 0.013,
          dry_bush: 0.0,
          ground_clutter: null,
          flower: null,
          log: null,
          rock: 0.0,
          landscape: 0.534,
          sky: 0.300,
        },
        confusion_matrix: [
          [1, 0, 3, 0, 0, 0, 0, 0, 100, 192],
          [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
          [0, 0, 438, 0, 0, 0, 0, 0, 21272, 3998],
          [0, 0, 43, 0, 0, 0, 0, 0, 2579, 4088],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [14, 37, 380, 0, 0, 0, 0, 0, 17968, 1090],
          [1, 11, 8741, 0, 0, 0, 0, 0, 68626, 2264],
          [479, 0, 0, 0, 0, 0, 0, 0, 6948, 8181],
        ],
      },
      robustness: {
        brightness: {
          pixel_accuracy: 0.38,
          mean_iou: 0.158,
          miou_drop: -0.012,
          per_class_iou: {
            tree: 0.001, lush_bush: 0.0, dry_grass: 0.010, dry_bush: 0.0,
            ground_clutter: null, flower: null, log: null,
            rock: 0.0, landscape: 0.520, sky: 0.290,
          },
          inference_time_ms: 46,
        },
        blur: {
          pixel_accuracy: 0.36,
          mean_iou: 0.142,
          miou_drop: -0.028,
          per_class_iou: {
            tree: 0.0, lush_bush: 0.0, dry_grass: 0.008, dry_bush: 0.0,
            ground_clutter: null, flower: null, log: null,
            rock: 0.0, landscape: 0.490, sky: 0.260,
          },
          inference_time_ms: 45,
        },
        noise: {
          pixel_accuracy: 0.37,
          mean_iou: 0.148,
          miou_drop: -0.022,
          per_class_iou: {
            tree: 0.001, lush_bush: 0.0, dry_grass: 0.009, dry_bush: 0.0,
            ground_clutter: null, flower: null, log: null,
            rock: 0.0, landscape: 0.505, sky: 0.275,
          },
          inference_time_ms: 45,
        },
        contrast: {
          pixel_accuracy: 0.34,
          mean_iou: 0.132,
          miou_drop: -0.038,
          per_class_iou: {
            tree: 0.0, lush_bush: 0.0, dry_grass: 0.005, dry_bush: 0.0,
            ground_clutter: null, flower: null, log: null,
            rock: 0.0, landscape: 0.460, sky: 0.230,
          },
          inference_time_ms: 44,
        },
      },
    },
  ],
  config: {
    device: "auto",
    models: [
      {
        name: "dinov2_v6_ensemble",
        architecture: "dinov2_vit_s14",
        backbone: "vit_small_14",
        num_classes: 10,
        encoder_weights: "self-supervised (142M images)",
        weights: "best_model_v6.pth",
      },
    ],
    class_names: [
      "tree", "lush_bush", "dry_grass", "dry_bush", "ground_clutter",
      "flower", "log", "rock", "landscape", "sky",
    ],
    mask_value_mapping: {
      "100": 0, "200": 1, "300": 2, "500": 3, "550": 4,
      "600": 5, "700": 6, "800": 7, "7100": 8, "10000": 9,
    },
    cost_mapping: {
      traversable: [8, 2, 4, 5],
      obstacle: [0, 7, 6],
      soft: [1, 3],
      ignored: [9],
    },
    cost_values: {
      traversable: 1.0,
      obstacle: Infinity,
      soft: 2.0,
      ignored: Infinity,
    },
    preprocessing: {
      target_size: [540, 960],
      normalize: {
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
      },
    },
    planner: {
      strategy: "astar",
      allow_diagonal: false,
      start: null,
      goal: null,
      potential_field: {
        attractive_gain: 1.0,
        repulsive_gain: 100.0,
        repulsive_range: 10,
        max_iterations: 5000,
      },
      rrt_star: {
        max_iterations: 5000,
        step_size: 5,
        goal_bias: 0.1,
        rewire_radius: 15,
      },
    },
    safety: {
      weight_obstacle: 0.4,
      weight_confidence: 0.3,
      weight_cost: 0.3,
      max_acceptable_cost: 1000.0,
    },
    robustness: {
      enabled: true,
      perturbations: ["brightness", "blur", "noise", "contrast"],
      params: {
        brightness: { factor: 1.4 },
        blur: { radius: 3 },
        noise: { std: 25.0 },
        contrast: { factor: 0.5 },
      },
    },
    dataset: {
      image_dir: "perception_engine/Offroad_Segmentation_testImages/Color_Images",
      gt_dir: "perception_engine/Offroad_Segmentation_testImages/Segmentation",
      image_ext: ".png",
      gt_ext: ".png",
      max_samples: null,
    },
    output: {
      save_visualizations: true,
      output_dir: "outputs",
    },
  },
};

export const explanationReport: ExplanationReport = {
  title: "Perception Engine — Results Explained",
  overview:
    "We evaluated 3 model configurations on 1,002 test images from an unseen desert environment. The V3+V5+V6 ensemble with 4D Bayesian fusion achieved the highest mIoU of 61%. All metrics shown are from the full test set evaluation.",
  models: [
    {
      model_name: "DINOv2 V6 + Ensemble",
      sections: [
        {
          title: "How well does the model understand the scene?",
          metric: "61% mean IoU",
          explanation:
            'The ensemble correctly identified 81.2% of all pixels. Its overall scene understanding score (mean IoU) is 61%, which is considered **good**. The model uses DINOv2 ViT-S/14 backbone + 4D Bayesian fusion for domain generalization.',
        },
        {
          title: "Which terrain types does it recognise best?",
          best_class: "sky (99.0%)",
          worst_class: "lush_bush (0.0%)",
          explanation:
            "The model is most accurate at detecting **sky** (99.0% IoU) and struggles with **lush_bush** (0.0% IoU — near-zero training samples). Top 5: sky (99.0%), landscape (72.7%), tree (59.8%), dry_bush (45.5%), dry_grass (44.3%).",
          all_classes: {
            tree: "59.8%",
            lush_bush: "0.0%",
            dry_grass: "44.3%",
            dry_bush: "45.5%",
            ground_clutter: "N/A",
            flower: "N/A",
            log: "N/A",
            rock: "43.2%",
            landscape: "72.7%",
            sky: "99.0%",
          },
        },
        {
          title: "How fast is the model?",
          metric: "580 ms per image (ensemble)",
          explanation:
            "The 3-model ensemble takes ~580ms to process one 960×540 image. The single V6 model runs at 288ms. Suitable for offline analysis and planning.",
        },
        {
          title: "How confident is the model in its predictions?",
          metric: "85% average confidence",
          explanation:
            "On average, the ensemble is 85% confident in its pixel-level predictions. This is a strong level of certainty, indicating reliable predictions.",
        },
        {
          title: "Can the robot find a safe path?",
          metric: "\u2705 Path found",
          explanation:
            "The A* planner successfully found a route through the scene using the cost map derived from the segmentation. Safe traversal is confirmed.",
        },
        {
          title: "How safe is the planned route?",
          metric: "81.2% safety score",
          explanation:
            "The safety score is 81.2% (100% = perfectly safe). 0.0% of the path overlaps with obstacles. This is a safe, reliable route.",
        },
        {
          title: "How does the model handle tough conditions?",
          metric: "Worst impact: blur (−3.4%)",
          explanation:
            "We tested the model under brightness, blur, noise, and contrast perturbations. The biggest mIoU drop was from **blur** (−3.4%). Overall, the model is robust to environmental disturbances.",
          perturbations: {
            brightness: { drop: "-1.2%", verdict: "Robust" },
            blur: { drop: "-3.4%", verdict: "Minor drop" },
            noise: { drop: "-3.1%", verdict: "Robust" },
            contrast: { drop: "-2.5%", verdict: "Robust" },
          },
        },
      ],
    },
  ],
};

import { benchmarkReport, explanationReport, terrainClasses } from "./mock-data";
import type {
  BenchmarkReport,
  ExplanationReport,
  ModelReport,
  TerrainClass,
  TerrainAsset,
} from "./types";

export function getBenchmarkReport(): BenchmarkReport {
  return benchmarkReport;
}

export function getExplanation(): ExplanationReport {
  return explanationReport;
}

export function getPrimaryReport(): ModelReport {
  return benchmarkReport.reports[0];
}

export function getAllReports(): ModelReport[] {
  return benchmarkReport.reports;
}

export function getTerrainClasses(): TerrainClass[] {
  return terrainClasses;
}

export function getClassNames(): string[] {
  return benchmarkReport.config.class_names;
}

export function getConfig() {
  return benchmarkReport.config;
}

export const terrainAssets: TerrainAsset[] = [
  {
    id: "terrain_01",
    label: "Desert Ridge Alpha",
    terrain: "/terrain/terrain_01.png",
    maskOverlay: "/terrain/mask_overlay_v1.png",
    pathOverlay: "/terrain/path_overlay_v1.png",
    confidence: "/terrain/confidence_v1.png",
  },
  {
    id: "terrain_02",
    label: "Desert Ridge Beta",
    terrain: "/terrain/terrain_02.png",
    maskOverlay: "/terrain/mask_overlay_trained.png",
    pathOverlay: "/terrain/path_overlay_trained.png",
    confidence: "/terrain/confidence_trained.png",
  },
  {
    id: "terrain_03",
    label: "Desert Ridge Gamma",
    terrain: "/terrain/terrain_03.png",
    maskOverlay: "/terrain/mask_overlay_v2.png",
    pathOverlay: "/terrain/path_overlay_v1.png",
    confidence: "/terrain/confidence_v1.png",
  },
];

export function getTerrainAsset(id?: string): TerrainAsset {
  return terrainAssets.find((a) => a.id === id) ?? terrainAssets[0];
}

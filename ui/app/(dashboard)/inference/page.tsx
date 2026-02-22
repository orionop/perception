"use client"

import { useState, useCallback } from "react"
import { ImageUpload } from "@/components/inference/image-upload"
import { ResultsPanel } from "@/components/inference/results-panel"
import { Loader2, Play, Clock, Cpu } from "lucide-react"
import { Button } from "@/components/ui/button"
import { getPrimaryReport } from "@/lib/data"

export default function InferencePage() {
  const [inputImage, setInputImage] = useState<File | null>(null)
  const [groundTruth, setGroundTruth] = useState<File | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [progress, setProgress] = useState(0)

  const report = getPrimaryReport()

  const runInference = useCallback(() => {
    if (!inputImage) return
    setIsRunning(true)
    setShowResults(false)
    setProgress(0)

    const stages = [
      { p: 15, delay: 300 },
      { p: 35, delay: 600 },
      { p: 55, delay: 900 },
      { p: 75, delay: 1200 },
      { p: 90, delay: 1600 },
      { p: 100, delay: 2000 },
    ]

    stages.forEach(({ p, delay }) => {
      setTimeout(() => setProgress(p), delay)
    })

    setTimeout(() => {
      setIsRunning(false)
      setShowResults(true)
    }, 2200)
  }, [inputImage])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground text-balance">Inference</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Upload terrain images and run segmentation inference
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ImageUpload
          label="Input Image"
          description="PNG or JPG, terrain / off-road imagery"
          file={inputImage}
          onFileSelect={setInputImage}
        />
        <ImageUpload
          label="Ground Truth Mask"
          description="Optional segmentation ground truth mask"
          file={groundTruth}
          onFileSelect={setGroundTruth}
          optional
        />
      </div>

      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <Button
          onClick={runInference}
          disabled={!inputImage || isRunning}
          className="bg-primary hover:bg-primary/90 text-primary-foreground font-medium px-8 py-5"
        >
          {isRunning ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Running Inference...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Run Inference
            </>
          )}
        </Button>

        {isRunning && (
          <div className="flex-1 max-w-md space-y-1">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Processing...</span>
              <span>{progress}%</span>
            </div>
            <div className="h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {showResults && (
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <Clock className="w-3.5 h-3.5" />
              <span className="font-mono">{report.inference_time_ms.toFixed(1)}ms</span>
            </div>
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <Cpu className="w-3.5 h-3.5" />
              <span className="font-mono">{(1000 / report.inference_time_ms).toFixed(1)} FPS</span>
            </div>
          </div>
        )}
      </div>

      <ResultsPanel visible={showResults} />
    </div>
  )
}

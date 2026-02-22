"use client"

import { useCallback, useState } from "react"
import { Upload, ImageIcon, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface ImageUploadProps {
  label: string
  description: string
  onFileSelect: (file: File | null) => void
  file: File | null
  optional?: boolean
}

export function ImageUpload({
  label,
  description,
  onFileSelect,
  file,
  optional,
}: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)

  const handleFile = useCallback(
    (f: File) => {
      onFileSelect(f)
      const reader = new FileReader()
      reader.onloadend = () => setPreview(reader.result as string)
      reader.readAsDataURL(f)
    },
    [onFileSelect]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const f = e.dataTransfer.files[0]
      if (f && (f.type === "image/png" || f.type === "image/jpeg")) {
        handleFile(f)
      }
    },
    [handleFile]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const clear = useCallback(() => {
    onFileSelect(null)
    setPreview(null)
  }, [onFileSelect])

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-medium text-foreground">{label}</span>
        {optional && (
          <span className="text-xs text-muted-foreground">(optional)</span>
        )}
      </div>
      {preview ? (
        <div className="relative glass-card rounded-xl overflow-hidden">
          <img
            src={preview}
            alt={`Uploaded ${label}`}
            className="w-full h-48 object-cover"
          />
          <button
            onClick={clear}
            className="absolute top-2 right-2 w-7 h-7 rounded-full bg-background/80 backdrop-blur flex items-center justify-center hover:bg-background transition-colors"
          >
            <X className="w-4 h-4 text-foreground" />
          </button>
        </div>
      ) : (
        <label
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={cn(
            "flex flex-col items-center justify-center gap-3 h-48 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200",
            isDragging
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-secondary/30"
          )}
        >
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-secondary">
            {isDragging ? (
              <ImageIcon className="w-6 h-6 text-primary" />
            ) : (
              <Upload className="w-6 h-6 text-muted-foreground" />
            )}
          </div>
          <div className="text-center">
            <p className="text-sm text-foreground font-medium">
              Drop image here or click to upload
            </p>
            <p className="text-xs text-muted-foreground mt-1">{description}</p>
          </div>
          <input
            type="file"
            accept="image/png,image/jpeg"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) handleFile(f)
            }}
          />
        </label>
      )}
    </div>
  )
}

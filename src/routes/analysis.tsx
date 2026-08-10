import { createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type ChangeEvent,
} from "react";
import {
  Upload,
  ImageIcon,
  Video,
  X,
  Check,
  Loader2,
  Circle,
  Sparkles,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { analysisStore, type AnalysisResult } from "@/lib/analysis-store";
import { useAuthState } from "@/lib/auth-store";
import { toast } from "sonner";

export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "SatyaDrishti | New Analysis" },
      {
        name: "description",
        content:
          "Upload an image or video and run the SatyaDrishti multi-agent deepfake detection pipeline.",
      },
    ],
  }),
  component: AnalysisPage,
});

const STAGES = [
  { id: "orchestrator", label: "Orchestrator", detail: "Routing task to agents" },
  { id: "image", label: "Image Analysis", detail: "Frequency & artifact scan" },
  { id: "video", label: "Video Analysis", detail: "Temporal consistency checks" },
  { id: "confidence", label: "Confidence Evaluation", detail: "Per-agent scoring" },
  { id: "fusion", label: "Fusion Decision", detail: "Weighted verdict" },
  { id: "explain", label: "Explanation Generation", detail: "Human-readable rationale" },
  { id: "result", label: "Final Result", detail: "Report ready" },
] as const;

function AnalysisPage() {
  const navigate = useNavigate();
  const { user } = useAuthState();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [running, setRunning] = useState(false);
  const [stageIdx, setStageIdx] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const fileType: "image" | "video" | null = file
    ? file.type.startsWith("video")
      ? "video"
      : "image"
    : null;

  const setSelected = useCallback(
    (f: File | null) => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setFile(f);
      setPreviewUrl(f ? URL.createObjectURL(f) : null);
      setStageIdx(-1);
    },
    [previewUrl],
  );

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setSelected(f);
  }

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) setSelected(f);
  }

  async function analyze() {
    if (!file || !fileType) return;
    setRunning(true);
    setStageIdx(0); // Orchestrator stage

    try {
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

      // 1. Upload to Flask backend
      const formData = new FormData();
      formData.append("file", file);

      const uploadResponse = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        const data = await uploadResponse.json().catch(() => ({}));
        throw new Error(data.error || "Failed to upload file to backend server.");
      }

      const uploadData = await uploadResponse.json();
      const backendId = uploadData.id;
      const backendMediaType = uploadData.media_type as "image" | "video";

      // 2. Call analyze endpoint on Flask backend (Orchestrator Agent)
      const analyzeResponse = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_id: backendId,
          media_type: backendMediaType,
          user: user ? { id: user.id, name: user.name, email: user.email } : null,
        }),
      });

      if (!analyzeResponse.ok) {
        const data = await analyzeResponse.json().catch(() => ({}));
        throw new Error(data.error || "Failed to analyze media file on backend.");
      }

      const analyzeData = await analyzeResponse.json();
      const fusionResult = analyzeData.fusion_result;

      // 3. Run UI simulation of stages for user experience
      for (let i = 1; i < STAGES.length; i++) {
        await new Promise((r) => setTimeout(r, 700));
        setStageIdx(i);
      }
      await new Promise((r) => setTimeout(r, 300));

      const result: AnalysisResult = {
        id: backendId,
        fileName: file.name,
        fileType: backendMediaType,
        mediaUrl: uploadData.filename ? `${API_URL}/api/analysis/uploads/${uploadData.filename}` : previewUrl!,
        verdict: fusionResult.verdict as "REAL" | "DEEPFAKE",
        confidence: fusionResult.confidence,
        explanation: fusionResult.explanation,
        agents: fusionResult.agents,
        createdAt: Date.now(),
        confidenceLevel: fusionResult.confidence_level,
        evidence: fusionResult.evidence || [],
        recommendation: fusionResult.recommendation,
        timeline: fusionResult.timeline || [],
        reportUrl: fusionResult.report_url ? `${API_URL}${fusionResult.report_url}` : undefined,
      };
      
      analysisStore.save(result);
      setRunning(false);
      navigate({ to: "/result" });
      toast.success("Analysis pipeline completed!");

    } catch (err: any) {
      console.error("Upload/Analysis error:", err);
      toast.error(err.message || "An error occurred during file analysis.");
      setRunning(false);
      setStageIdx(-1);
    }
  }

  return (
    <AppShell
      title="New analysis"
      description="Upload media and run the multi-agent detection pipeline."
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        {/* Upload + preview */}
        <div className="space-y-6">
          <div className="rounded-2xl border border-border/70 bg-card p-6">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold">Upload media</h3>
                <p className="text-xs text-muted-foreground">
                  Supports JPG, PNG, MP4, MOV. Max 100 MB.
                </p>
              </div>
              {file && (
                <Button variant="ghost" size="sm" onClick={() => setSelected(null)}>
                  <X className="mr-1 h-4 w-4" /> Clear
                </Button>
              )}
            </div>

            {!file ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                className={cn(
                  "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors",
                  dragOver
                    ? "border-primary/60 bg-primary/5"
                    : "border-border/80 hover:border-primary/40 hover:bg-secondary/40",
                )}
              >
                <div className="grid h-12 w-12 place-items-center rounded-full border border-border/70 bg-secondary/50 text-primary">
                  <Upload className="h-5 w-5" />
                </div>
                <p className="mt-4 text-sm font-medium">
                  Drag & drop your file here
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  or click to browse from your device
                </p>
                <div className="mt-5 flex gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (inputRef.current) inputRef.current.accept = "image/*";
                      inputRef.current?.click();
                    }}
                  >
                    <ImageIcon className="mr-1.5 h-4 w-4" /> Upload image
                  </Button>
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (inputRef.current) inputRef.current.accept = "video/*";
                      inputRef.current?.click();
                    }}
                  >
                    <Video className="mr-1.5 h-4 w-4" /> Upload video
                  </Button>
                </div>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*,video/*"
                  className="hidden"
                  onChange={onPick}
                />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="overflow-hidden rounded-xl border border-border/70 bg-black/40">
                  {fileType === "image" ? (
                    <img
                      src={previewUrl!}
                      alt={file.name}
                      className="mx-auto max-h-[420px] w-full object-contain"
                    />
                  ) : (
                    <video
                      src={previewUrl!}
                      controls
                      className="mx-auto max-h-[420px] w-full"
                    />
                  )}
                </div>
                <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">{file.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB · {fileType}
                    </div>
                  </div>
                  <Button onClick={analyze} disabled={running}>
                    {running ? (
                      <>
                        <Loader2 className="mr-1.5 h-4 w-4 animate-spin" /> Analyzing
                      </>
                    ) : (
                      <>
                        <Sparkles className="mr-1.5 h-4 w-4" /> Analyze
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Workflow panel */}
        <div className="rounded-2xl border border-border/70 bg-card p-6">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold">Detection pipeline</h3>
              <p className="text-xs text-muted-foreground">
                Live status of the agent workflow.
              </p>
            </div>
            {running && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-2.5 py-1 text-[11px] text-primary">
                <Loader2 className="h-3 w-3 animate-spin" /> Running
              </span>
            )}
          </div>

          <ol className="relative space-y-4 pl-2">
            {STAGES.map((s, i) => {
              const done = i < stageIdx || (!running && stageIdx === STAGES.length - 1 && i <= stageIdx);
              const active = running && i === stageIdx;
              return (
                <li key={s.id} className="relative pl-8">
                  {i < STAGES.length - 1 && (
                    <span
                      className={cn(
                        "absolute left-[11px] top-6 h-full w-px",
                        done ? "bg-primary/50" : "bg-border/70",
                      )}
                    />
                  )}
                  <span
                    className={cn(
                      "absolute left-0 top-0 grid h-6 w-6 place-items-center rounded-full border",
                      done
                        ? "border-primary bg-primary text-primary-foreground"
                        : active
                          ? "border-primary/60 bg-primary/10 text-primary"
                          : "border-border/70 bg-secondary/40 text-muted-foreground",
                    )}
                  >
                    {done ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : active ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Circle className="h-2 w-2 fill-current" />
                    )}
                  </span>
                  <div>
                    <div
                      className={cn(
                        "text-sm font-medium",
                        done || active ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      {s.label}
                    </div>
                    <div className="text-xs text-muted-foreground">{s.detail}</div>
                  </div>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </AppShell>
  );
}
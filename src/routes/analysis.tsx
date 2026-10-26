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
import { analysisStore, type AnalysisResult, type AnalysisVerdict, getCanonicalVerdict } from "@/lib/analysis-store";
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

function detectClientMediaType(file: File): "image" | "video" | null {
  const mime = file.type.toLowerCase();
  const name = file.name.toLowerCase();
  const ext = name.split(".").pop() || "";

  const imageExts = ["jpg", "jpeg", "png", "webp", "gif", "bmp"];
  const videoExts = ["mp4", "mov", "avi", "mkv", "webm"];

  if (mime.startsWith("image/") || imageExts.includes(ext)) {
    return "image";
  }
  if (mime.startsWith("video/") || videoExts.includes(ext)) {
    return "video";
  }
  return null;
}

function getPipelineStages(mediaType: "image" | "video" | null) {
  if (mediaType === "video") {
    return [
      { id: "orchestrator", label: "Orchestrator", detail: "Automatically detects media type and routes the analysis" },
      { id: "video", label: "Video Analysis", detail: "Temporal consistency checks" },
      { id: "confidence", label: "Confidence Evaluation", detail: "Per-agent scoring" },
      { id: "fusion", label: "Fusion Decision", detail: "Weighted verdict" },
      { id: "explain", label: "Explanation Generation", detail: "Human-readable rationale" },
      { id: "result", label: "Final Result", detail: "Report ready" },
    ];
  }
  if (mediaType === "image") {
    return [
      { id: "orchestrator", label: "Orchestrator", detail: "Automatically detects media type and routes the analysis" },
      { id: "image", label: "Image Analysis", detail: "Frequency & artifact scan" },
      { id: "confidence", label: "Confidence Evaluation", detail: "Per-agent scoring" },
      { id: "fusion", label: "Fusion Decision", detail: "Weighted verdict" },
      { id: "explain", label: "Explanation Generation", detail: "Human-readable rationale" },
      { id: "result", label: "Final Result", detail: "Report ready" },
    ];
  }
  return [
    { id: "orchestrator", label: "Orchestrator", detail: "Automatically detects media type and routes the analysis" },
    { id: "specialist", label: "Specialist Analysis", detail: "Image or Video Agent pipeline (auto-routed)" },
    { id: "confidence", label: "Confidence Evaluation", detail: "Per-agent scoring" },
    { id: "fusion", label: "Fusion Decision", detail: "Weighted verdict" },
    { id: "explain", label: "Explanation Generation", detail: "Human-readable rationale" },
    { id: "result", label: "Final Result", detail: "Report ready" },
  ];
}

function AnalysisPage() {
  const navigate = useNavigate();
  const { user, isInitialized } = useAuthState();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [running, setRunning] = useState(false);
  const [stageIdx, setStageIdx] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isInitialized && !user) {
      navigate({ to: "/login" });
    }
  }, [user, isInitialized, navigate]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const fileType = file ? detectClientMediaType(file) : null;
  const currentStages = getPipelineStages(fileType);

  const setSelected = useCallback(
    (f: File | null) => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (!f) {
        setFile(null);
        setPreviewUrl(null);
        setStageIdx(-1);
        return;
      }

      if (f.size > 50 * 1024 * 1024) {
        toast.error("File exceeds maximum allowed size of 50MB.");
        return;
      }

      const detected = detectClientMediaType(f);
      if (!detected) {
        toast.error("Unsupported file format. Please upload an image (JPEG, PNG, WEBP) or video (MP4, MOV, AVI).");
        return;
      }

      setFile(f);
      setPreviewUrl(URL.createObjectURL(f));
      setStageIdx(-1);
    },
    [previewUrl],
  );

  const reset = useCallback(() => {
    setSelected(null);
  }, [setSelected]);

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setSelected(f);
  }

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) setSelected(f);
    if (e.target) e.target.value = "";
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
      if (user?.id) {
        formData.append("user_id", String(user.id));
        formData.append("user", JSON.stringify({ id: user.id, name: user.name, email: user.email }));
      }

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

      // 2. Call analyze endpoint on Flask backend (Orchestrator Agent auto-routes)
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
      for (let i = 1; i < currentStages.length; i++) {
        await new Promise((r) => setTimeout(r, 700));
        setStageIdx(i);
      }
      await new Promise((r) => setTimeout(r, 300));

      const { canonicalVerdict, provisionalSuggestion, isBinaryFallback, reliabilityWarning } = getCanonicalVerdict(fusionResult);
      const isNotSuitable = canonicalVerdict === "NOT_SUITABLE";
      const isUncertain = canonicalVerdict === "UNCERTAIN";
      const isUncertified = isBinaryFallback || isUncertain || isNotSuitable;
      const rawConf = typeof fusionResult.confidence === "number" ? fusionResult.confidence : null;
      const confidence = isUncertified ? null : (rawConf !== null ? Math.round(rawConf) : null);

      let confidenceLevel = fusionResult.confidence_level;
      if (isNotSuitable) {
        confidenceLevel = "Not Performed";
      } else if (isBinaryFallback) {
        confidenceLevel = "Provisional";
      } else if (isUncertain) {
        confidenceLevel = "Uncertified";
      } else if (confidence !== null) {
        confidenceLevel = confidence >= 80 ? "High" : confidence >= 60 ? "Medium" : "Low";
      }

      const effectiveMediaUrl = uploadData.media_url
        ? (uploadData.media_url.startsWith("http") ? uploadData.media_url : `${API_URL}${uploadData.media_url.startsWith("/") ? "" : "/"}${uploadData.media_url}`)
        : uploadData.saved_filename
        ? `${API_URL}/api/analysis/uploads/${uploadData.saved_filename}?token=${uploadData.token || ""}&file_id=${backendId}`
        : `${API_URL}/api/analysis/uploads/${backendId}_${file.name}?file_id=${backendId}`;

      const result: AnalysisResult = {
        id: backendId,
        fileName: file.name,
        saved_filename: uploadData.saved_filename || `${backendId}_${file.name}`,
        token: uploadData.token,
        fileType: backendMediaType,
        mediaUrl: effectiveMediaUrl,
        verdict: canonicalVerdict,
        prediction: canonicalVerdict,
        canonicalVerdict: canonicalVerdict,
        primaryVerdict: canonicalVerdict,
        provisionalSuggestion: provisionalSuggestion,
        originalVerdict: fusionResult.original_verdict || fusionResult.originalVerdict || (isUncertain ? "UNCERTAIN" : canonicalVerdict),
        isBinaryFallback: isBinaryFallback,
        isEstimate: isBinaryFallback || isUncertain,
        reliabilityWarning: reliabilityWarning || fusionResult.reliability_warning,
        confidence: confidence,
        confidenceLevel: confidenceLevel || "Uncertified",
        fakeProbability: isNotSuitable ? undefined : fusionResult.fake_probability,
        realProbability: isNotSuitable ? undefined : fusionResult.real_probability,
        rawModelScore: isNotSuitable ? undefined : fusionResult.raw_model_score,
        uncertainty: isNotSuitable ? undefined : fusionResult.uncertainty,
        reason: fusionResult.reason,
        userFacingStatement: fusionResult.user_facing_statement,
        explanation: fusionResult.explanation,
        analysisStatus: fusionResult.analysis_status || (isNotSuitable ? "Not performed" : "Completed"),
        faceDetected: fusionResult.face_detected !== undefined ? fusionResult.face_detected : !isNotSuitable,
        faceBbox: fusionResult.face_bbox || null,
        evidenceCards: fusionResult.evidence_cards || [],
        qualityGate: fusionResult.quality_gate,
        features: fusionResult.features,
        mediaCategory: fusionResult.media_category || (isNotSuitable ? "Document / Non-face image" : "Facial Media"),
        agents: fusionResult.agents,
        createdAt: Date.now(),
        evidence: fusionResult.evidence || [],
        recommendation: fusionResult.recommendation,
        timeline: fusionResult.timeline || [],
        reportUrl: fusionResult.report_url ? `${API_URL}${fusionResult.report_url}` : undefined,
        visualAnalysis: fusionResult.visual_analysis,
        evidenceBreakdown: fusionResult.evidence_breakdown,
        approxCostUsd: fusionResult.approx_cost_usd ?? 0.0,
        providersRan: fusionResult.providers_ran || [],
        naturalLanguageAnswer: analyzeData.natural_language_answer,
      };
      
      analysisStore.save(result, user?.id);
      setRunning(false);
      navigate({ to: "/result", search: { id: backendId } });
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
      title="New Analysis Workspace"
      description="Upload digital media to execute multi-agent forensic verification."
      actions={
        file ? (
          <Button
            size="sm"
            variant="outline"
            className="h-8 text-xs border-slate-700 text-slate-300 hover:bg-slate-800"
            onClick={reset}
            disabled={running}
          >
            Clear Selected Media
          </Button>
        ) : null
      }
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        {/* Inspection Input & Preview */}
        <div className="space-y-4">
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
            <div className="mb-4">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Media Inspection Input
              </h2>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Supported formats: JPEG, PNG, WEBP, MP4, MOV, AVI (Max 50MB)
              </p>
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
                  "flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed px-6 py-12 text-center transition-colors",
                  dragOver
                    ? "border-sky-500 bg-sky-500/5"
                    : "border-slate-700 bg-slate-950/40 hover:border-sky-500/50 hover:bg-slate-900/40",
                )}
              >
                <div className="grid h-10 w-10 place-items-center rounded border border-slate-700 bg-slate-800/80 text-sky-400">
                  <Upload className="h-4 w-4" />
                </div>
                <p className="mt-3 text-xs font-medium text-slate-200">
                  Drag and drop your media file here
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  or select a file from your local file system
                </p>
                <p className="mt-1 text-[10px] text-sky-400/90 font-medium">
                  Images and videos are automatically detected
                </p>
                <div className="mt-4">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs border-slate-700 text-slate-300 hover:bg-slate-800"
                    onClick={(e) => {
                      e.stopPropagation();
                      inputRef.current?.click();
                    }}
                  >
                    <Upload className="mr-1.5 h-3.5 w-3.5 text-slate-400" /> Select Media
                  </Button>
                </div>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp,image/gif,image/bmp,video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska,.jpg,.jpeg,.png,.webp,.gif,.bmp,.mp4,.mov,.avi,.mkv,.webm"
                  className="hidden"
                  onChange={onPick}
                />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="overflow-hidden rounded border border-border/80 bg-slate-950/80">
                  {fileType === "image" ? (
                    <img
                      src={previewUrl!}
                      alt={file.name}
                      className="mx-auto max-h-[380px] w-full object-contain"
                    />
                  ) : (
                    <video
                      src={previewUrl!}
                      controls
                      className="mx-auto max-h-[380px] w-full"
                    />
                  )}
                </div>
                <div className="flex items-center justify-between gap-3 border-t border-border/60 pt-3">
                  <div className="min-w-0 space-y-1">
                    <div className="truncate text-xs font-semibold text-slate-200 font-mono">{file.name}</div>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] text-slate-400">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </span>
                      <span className="text-slate-600">·</span>
                      <span className="inline-flex items-center rounded border border-sky-500/30 bg-sky-500/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-sky-400">
                        Media detected: {fileType === "image" ? "Image" : "Video"}
                      </span>
                    </div>
                  </div>
                  <Button
                    onClick={analyze}
                    disabled={running}
                    size="sm"
                    className="h-9 px-4 bg-sky-600 text-white hover:bg-sky-500 font-medium text-xs"
                  >
                    {running ? (
                      <>
                        <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> Executing Pipeline...
                      </>
                    ) : (
                      <>
                        <Sparkles className="mr-1.5 h-3.5 w-3.5" /> Start Analysis
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Workflow panel */}
        <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
          <div className="mb-5 flex items-center justify-between border-b border-border/60 pb-3">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Agent Execution Pipeline
              </h3>
              <p className="text-[11px] text-slate-400">
                Multi-agent orchestration sequence.
              </p>
            </div>
            {running && (
              <span className="inline-flex items-center gap-1.5 rounded border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 text-[10px] font-medium text-sky-400">
                <Loader2 className="h-3 w-3 animate-spin" /> RUNNING
              </span>
            )}
          </div>

          <ol className="relative space-y-4 pl-1">
            {currentStages.map((s, i) => {
              const done = i < stageIdx || (!running && stageIdx === currentStages.length - 1 && i <= stageIdx);
              const active = running && i === stageIdx;
              return (
                <li key={s.id} className="relative pl-7">
                  {i < currentStages.length - 1 && (
                    <span
                      className={cn(
                        "absolute left-[9px] top-5 h-full w-px",
                        done ? "bg-sky-500/50" : "bg-slate-800",
                      )}
                    />
                  )}
                  <span
                    className={cn(
                      "absolute left-0 top-0.5 grid h-5 w-5 place-items-center rounded text-[10px] font-mono border",
                      done
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                        : active
                          ? "border-sky-500/50 bg-sky-500/10 text-sky-400"
                          : "border-slate-800 bg-slate-950/60 text-slate-400",
                    )}
                  >
                    {done ? (
                      <Check className="h-3 w-3" />
                    ) : active ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <span>{String(i + 1).padStart(2, "0")}</span>
                    )}
                  </span>
                  <div>
                    <div
                      className={cn(
                        "text-xs font-medium",
                        done || active ? "text-slate-200" : "text-slate-400",
                      )}
                    >
                      {s.label}
                    </div>
                    <div className="text-[11px] text-slate-400">{s.detail}</div>
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
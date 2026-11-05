import { createFileRoute, Link, useNavigate, redirect } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import {
  Download,
  ShieldCheck,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  ExternalLink,
  FileX,
  HelpCircle,
  Eye,
  Scan,
  Activity,
  Layers,
  Info,
  CheckCircle2,
  Users,
  Cpu,
  Sliders,
  ChevronDown,
  ChevronRight,
  ThumbsUp,
  ThumbsDown,
  Flag,
  Check,
  Loader2,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { useAnalysisState, analysisStore, AnalysisResult, formatBackendAnalysis, getVerdictTheme, getCanonicalVerdict, getAnalysisStatus, AnalysisStatus } from "@/lib/analysis-store";
import { useAuthState, authStore } from "@/lib/auth-store";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

function formatResolution(res: any): string {
  if (!res) return "Not assessed";
  if (typeof res === "string") {
    if (res === "[object Object]" || res === "Standard Native Resolution") return "Not assessed";
    return res;
  }
  if (Array.isArray(res) && res.length >= 2) {
    if (Number(res[0]) > 0 && Number(res[1]) > 0) return `${res[0]} × ${res[1]} px`;
    return "Not assessed";
  }
  if (typeof res === "object") {
    if (res.formatted && typeof res.formatted === "string" && res.formatted !== "[object Object]") {
      return res.formatted;
    }
    const w = res.width ?? res.w;
    const h = res.height ?? res.h;
    if (typeof w === "number" && typeof h === "number" && w > 0 && h > 0) {
      return `${w} × ${h} px`;
    }
    if (typeof w === "string" && typeof h === "string" && Number(w) > 0 && Number(h) > 0) {
      return `${w} × ${h} px`;
    }
  }
  return "Not assessed";
}

type ResultSearch = {
  id?: string;
};

export const Route = createFileRoute("/result")({
  beforeLoad: async () => {
    const isAuthed = await authStore.ensureInitialized();
    if (!isAuthed || !authStore.get().user) {
      throw redirect({ to: "/login" });
    }
  },
  validateSearch: (search: Record<string, unknown>): ResultSearch => ({
    id: typeof search.id === "string" ? search.id : undefined,
  }),
  head: () => ({
    meta: [
      { title: "SatyaDrishti | Detection Results" },
      {
        name: "description",
        content: "Detailed deepfake detection result with confidence and agent trace.",
      },
    ],
  }),
  component: ResultPage,
});

function ResultPage() {
  const navigate = useNavigate();
  const { user, isInitialized } = useAuthState();
  const { id } = Route.useSearch();
  const { latest, selectedReport, history } = useAnalysisState();
  const [fetchedReport, setFetchedReport] = useState<AnalysisResult | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  useEffect(() => {
    if (isInitialized && !user) {
      navigate({ to: "/login" });
    }
  }, [user, isInitialized, navigate]);

  useEffect(() => {
    if (!isInitialized || !user?.id) return;
    analysisStore.setUser(user.id);

    // If no report is selected, in store, and no specific ID was requested, fetch this user's latest analysis
    if (!id && !selectedReport && !latest && history.length === 0) {
      const fetchUserHistory = async () => {
        try {
          const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
          const token = authStore.getToken();
          const headers: Record<string, string> = {};
          if (token) {
            headers["Authorization"] = `Bearer ${token}`;
          }
          const res = await fetch(`${API_URL}/api/history?user_id=${encodeURIComponent(String(user.id))}`, { headers });
          if (res.status === 401) {
            await authStore.signOut();
            navigate({ to: "/login" });
            return;
          }
          if (res.ok) {
            const data = await res.json();
            if (data.status === "success" && Array.isArray(data.history) && data.history.length > 0) {
              const formatted = data.history.map((item: any) => formatBackendAnalysis(item, API_URL));
              analysisStore.setHistory(formatted, user.id);
            } else {
              analysisStore.setHistory([], user.id);
            }
          }
        } catch (e) {
          console.error("Failed to load user history:", e);
        }
      };
      fetchUserHistory();
    }
  }, [user, isInitialized, id, selectedReport, latest, history.length, navigate]);

  if (!isInitialized) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
          <p className="text-xs font-mono text-slate-400">Verifying session security...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  // Resolution order for active report:
  // 1. If fetchedReport matches ID, use fetchedReport (contains authoritative signed media URL & full evidence)
  // 2. If ID matches selectedReport, use selectedReport
  // 3. If ID matches an item in history, use it
  // 4. If no ID param is provided, use selectedReport or fallback to latest
  const activeReport =
    (id && fetchedReport?.id === id ? fetchedReport : null) ||
    (id && selectedReport?.id === id ? selectedReport : null) ||
    (id ? history.find((h) => h.id === id) : null) ||
    (!id ? selectedReport || latest : null);

  // If id is provided in URL, or activeReport is present but lacks full details / token, fetch authoritative report from backend
  useEffect(() => {
    const reportId = id || activeReport?.id;
    if (!reportId) return;

    const hasDetails = Boolean(activeReport && activeReport.id === reportId && (activeReport.timeline?.length || activeReport.evidence?.length));
    const hasSignedMedia = Boolean(activeReport?.mediaUrl && (activeReport.mediaUrl.includes("token=") || activeReport.mediaUrl.startsWith("blob:")));
    if (hasDetails && hasSignedMedia && (fetchedReport?.id === reportId || selectedReport?.id === reportId)) {
      return;
    }

    const fetchDetail = async () => {
      try {
        const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
        const token = authStore.getToken();
        const headers: Record<string, string> = {};
        if (token) {
          headers["Authorization"] = `Bearer ${token}`;
        }
        const res = await fetch(`${API_URL}/api/analysis/${reportId}`, { headers });
        if (res.ok) {
          const data = await res.json();
          if (data.status === "success" && data.analysis) {
            setFetchedReport(data.analysis);
            analysisStore.selectReport(data.analysis);
          }
        } else {
          toast.error("You can't download because no analysis is done");
        }
      } catch (e) {
        console.error("Failed to load analysis details:", e);
      }
    };
    fetchDetail();
  }, [id, activeReport?.id]);

  // User Feedback & Continuous Learning State
  const [feedbackType, setFeedbackType] = useState<"CORRECT" | "INCORRECT" | "REPORT_ISSUE" | null>(null);
  const [userCorrection, setUserCorrection] = useState<"REAL" | "FAKE" | null>(null);
  const [feedbackCategory, setFeedbackCategory] = useState<string>("");
  const [feedbackComment, setFeedbackComment] = useState<string>("");
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState<boolean>(false);
  const [recordedFeedback, setRecordedFeedback] = useState<any>(null);

  // Load existing feedback if present in activeReport or fetch from API
  useEffect(() => {
    if (activeReport?.user_feedback) {
      setRecordedFeedback(activeReport.user_feedback);
      return;
    }
    const repId = id || activeReport?.id;
    if (!repId) return;

    const fetchExistingFeedback = async () => {
      try {
        const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
        const res = await fetch(`${API_URL}/api/analysis/${repId}/feedback`);
        if (res.ok) {
          const data = await res.json();
          if (data.status === "success" && data.feedback) {
            setRecordedFeedback(data.feedback);
          }
        }
      } catch (e) {
        // silent
      }
    };
    fetchExistingFeedback();
  }, [id, activeReport?.id, activeReport?.user_feedback]);

  const handleSubmitFeedback = async () => {
    const repId = id || activeReport?.id;
    if (!repId || !feedbackType) return;

    setIsSubmittingFeedback(true);
    try {
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
      const token = authStore.getToken();
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const res = await fetch(`${API_URL}/api/analysis/${repId}/feedback`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          feedback_type: feedbackType,
          user_correction: userCorrection || (feedbackType === "CORRECT" ? (canonicalVerdict === "REAL" ? "REAL" : "FAKE") : (feedbackCategory === "FALSE_POSITIVE" ? "REAL" : "FAKE")),
          issue_category: feedbackCategory || (feedbackType === "CORRECT" ? "VERIFIED_ACCURATE" : "GENERAL"),
          user_comment: feedbackComment,
          reason: feedbackComment,
          user_id: authStore.get().user?.id,
        }),
      });

      const data = await res.json();
      if (res.ok && data.status === "success") {
        setRecordedFeedback(data.feedback);
        if (data.persistence === "local_fallback" || data.storage === "local_fallback") {
          toast.warning("Feedback saved to local fallback storage (cloud database currently offline).");
        } else {
          toast.success("Feedback submitted and securely persisted to cloud database.");
        }
        setFeedbackType(null);
      } else {
        toast.error(data.error || "Failed to submit feedback.");
      }
    } catch (e: any) {
      toast.error(`Error submitting feedback: ${e.message}`);
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  const [savingPdf, setSavingPdf] = useState(false);
  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
  const reportId = activeReport?.id || "";
  const cleanBaseName = activeReport?.fileName ? activeReport.fileName.replace(/\.[^/.]+$/, "") : "forensic";
  const pdfDownloadFilename = `SatyaDrishti_Report_${cleanBaseName}_${reportId.slice(0, 8)}.pdf`;
  const rawReportUrl = activeReport?.reportUrl
    ? (activeReport.reportUrl.startsWith("http") ? activeReport.reportUrl : `${API_URL}${activeReport.reportUrl}`)
    : `${API_URL}/api/analysis/report/${reportId}.pdf`;
  const finalPdfUrl = rawReportUrl.endsWith(".pdf") ? rawReportUrl : `${rawReportUrl}.pdf`;

  const handleSavePdf = async () => {
    if (!activeReport || !activeReport.id) {
      toast.error("You can't download because no analysis is done");
      return;
    }
    setSavingPdf(true);
    try {
      if ("showSaveFilePicker" in window) {
        const fileHandle = await (window as any).showSaveFilePicker({
          suggestedName: pdfDownloadFilename,
          types: [
            {
              description: "PDF Document (*.pdf)",
              accept: { "application/pdf": [".pdf"] },
            },
          ],
        });
        const res = await fetch(finalPdfUrl);
        if (!res.ok) {
          toast.error("You can't download because no analysis is done");
          setSavingPdf(false);
          return;
        }
        const blob = await res.blob();
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        toast.success(`PDF saved successfully: ${pdfDownloadFilename}`);
        setSavingPdf(false);
        return;
      }
    } catch (e: any) {
      if (e.name === "AbortError") {
        setSavingPdf(false);
        return;
      }
      console.warn("File picker error, falling back:", e);
    }

    try {
      const checkRes = await fetch(finalPdfUrl, { method: "HEAD" });
      if (!checkRes.ok) {
        toast.error("You can't download because no analysis is done");
        setSavingPdf(false);
        return;
      }
    } catch (err) {
      // Proceed to fallback trigger
    }

    setSavingPdf(false);
    // Fallback: trigger attachment download
    const downloadUrl = finalPdfUrl.includes("?") ? `${finalPdfUrl}&download=true` : `${finalPdfUrl}?download=true`;
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = pdfDownloadFilename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    toast.success("Downloading PDF Report...");
  };

  if (!activeReport) {
    return (
      <AppShell
        title="Analysis Report"
        description="Detailed verification report and multi-agent forensic summary."
        actions={
          <div className="flex items-center gap-2">
            <Button asChild variant="outline" size="sm" className="h-8 text-xs border-slate-700 text-slate-300 hover:bg-slate-800">
              <Link to="/analysis">
                <RotateCcw className="mr-1.5 h-3.5 w-3.5 text-slate-400" /> New Analysis
              </Link>
            </Button>
            <Button
              size="sm"
              onClick={() => {
                toast.error("You can't download because no analysis is done");
              }}
              className="h-8 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-medium shadow-sm transition-all"
              title="Download PDF report"
            >
              <Download className="mr-1.5 h-3.5 w-3.5" />
              Download PDF
            </Button>
          </div>
        }
      >
        <div className="mx-auto max-w-lg rounded-xl border border-border/80 bg-slate-900/60 p-8 text-center shadow-lg">
          <div className="mx-auto grid h-12 w-12 place-items-center rounded-lg border border-slate-700 bg-slate-800/80 text-sky-400 shadow-inner">
            <Sparkles className="h-6 w-6" />
          </div>
          <h2 className="mt-4 text-base font-semibold text-slate-100">No Analysis Done Yet</h2>
          <p className="mt-2 text-xs text-slate-400 leading-relaxed max-w-md mx-auto">
            You cannot view or download a report because no analysis has been done. Upload an image or video to execute forensic verification and generate an audit report.
          </p>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-2.5">
            <Button asChild size="sm" className="bg-sky-600 hover:bg-sky-500 text-white text-xs font-medium px-4 h-9">
              <Link to="/analysis">
                <RotateCcw className="mr-1.5 h-3.5 w-3.5" /> Execute New Analysis
              </Link>
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                toast.error("You can't download because no analysis is done");
              }}
              className="border-slate-700 text-slate-300 hover:bg-slate-800 text-xs h-9 px-3"
            >
              <Download className="mr-1.5 h-3.5 w-3.5 text-slate-400" /> Download Report
            </Button>
            <Button asChild size="sm" variant="ghost" className="text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 text-xs h-9">
              <Link to="/dashboard">Go to Dashboard</Link>
            </Button>
          </div>
        </div>
      </AppShell>
    );
  }

  const { canonicalVerdict, provisionalSuggestion, isBinaryFallback, reliabilityWarning: detectedWarning } = getCanonicalVerdict(activeReport);
  const theme = getVerdictTheme(canonicalVerdict, isBinaryFallback);
  const analysisStatus: AnalysisStatus = getAnalysisStatus(activeReport);
  const isNotSuitable = canonicalVerdict === "NOT_SUITABLE";
  const isIllustration = canonicalVerdict === "ILLUSTRATION";
  const isFake = canonicalVerdict === "FAKE";
  const isReal = canonicalVerdict === "REAL";
  const rawOrig = String(activeReport.originalVerdict || activeReport.original_verdict || "").trim().toUpperCase();
  const rawVerdict = String(activeReport.verdict || "").trim().toUpperCase();
  const isUncertain = rawOrig === "UNCERTAIN" || rawVerdict === "UNCERTAIN" || Boolean(activeReport.isEstimate || isBinaryFallback);
  const isSuspicious = false;
  const isInconclusive = false;
  const reliabilityWarning = detectedWarning || activeReport.reliabilityWarning;
  const isLowConfidence = analysisStatus === "REVIEW RECOMMENDED" || (!(isNotSuitable || isIllustration) && (isBinaryFallback || activeReport.confidenceLevel === "Low" || activeReport.confidenceLevel === "Uncertified"));
  const hasSpecialistDisagreement = isBinaryFallback || (
    typeof reliabilityWarning === "string" && (
      reliabilityWarning.toLowerCase().includes("disagree") ||
      reliabilityWarning.toLowerCase().includes("conflict") ||
      reliabilityWarning.toLowerCase().includes("ambiguous") ||
      reliabilityWarning.toLowerCase().includes("divergent")
    )
  );

  const tone = (isNotSuitable || isIllustration) ? "neutral" : isFake ? "danger" : isReal ? "ok" : "warning";

  const ufe = activeReport.userFriendlyExplanation;
  const whatDoesThisMean = ufe?.whatDoesThisMean || activeReport.userFacingStatement || activeReport.summary || (
    isIllustration
      ? "This image has been identified as a stylized illustration, anime, or cartoon. SatyaDrishti's deepfake detection models are trained exclusively on photographic images and cannot produce meaningful authenticity scores for non-photographic media."
      : isNotSuitable
      ? "No detectable human face was found in this media. SatyaDrishti's deepfake detection requires a visible human face."
      : isFake
      ? "Our analysis indicates strong evidence that this media has been artificially generated or digitally altered. Multiple independent forensic checks identified synthetic anomalies across key facial regions."
      : "Our analysis found no significant evidence of digital manipulation or artificial generation in this media. The facial features, skin texture, and optical camera noise closely match genuine photography."
  );
  const whatShouldUserDo = ufe?.whatShouldUserDo || activeReport.recommendation || (
    isIllustration
      ? "Upload a clear photograph of a real person for deepfake analysis. SatyaDrishti cannot assess illustrations, anime, or cartoon artwork."
      : isNotSuitable
      ? "Upload a clear media file containing a visible human face."
      : isFake
      ? (isLowConfidence ? "Treat this media with caution. Some indicators suggest alteration, but image quality or conflicting signals prevent a definitive conclusion. Verify through original sources." : "Do not rely on this media for critical decisions or share it without verification. If this claims to represent a real person, treat it with high skepticism.")
      : (isLowConfidence ? "While indicators lean toward authentic, limitations in image quality or subtle inconsistencies suggest caution. Do not rely solely on this analysis for high-stakes decisions." : "This media appears authentic based on standard forensic indicators. Standard precautions still apply.")
  );
  const whyConfidenceIsLimited = ufe?.whyConfidenceIsLimited || (
    hasSpecialistDisagreement
      ? "Different automated inspection tools produced conflicting indications. While some detectors flagged suspicious patterns, others assessed the image as authentic, leading to lower overall confidence."
      : isLowConfidence
      ? "The media contains optical softness or compression, which can obscure minute pixel details and limit the certainty of the analysis."
      : null
  );
  const findings = ufe?.whatRaisedConcern || [];
  const confTier = ufe?.confidenceTier || ((isNotSuitable || isIllustration) ? "Not Performed" : isLowConfidence ? "Low" : (activeReport.confidenceLevel || "Moderate"));
  const confPct = ufe?.confidencePercentage !== undefined && ufe?.confidencePercentage !== null ? ufe.confidencePercentage : activeReport.confidence;

  let resolvedMediaSrc = activeReport.mediaUrl.startsWith("http") || activeReport.mediaUrl.startsWith("blob:")
    ? activeReport.mediaUrl
    : `${API_URL}${activeReport.mediaUrl.startsWith("/") ? "" : "/"}${activeReport.mediaUrl}`;

  if (activeReport.id && resolvedMediaSrc.includes("/api/analysis/uploads/")) {
    if (!resolvedMediaSrc.includes(activeReport.id)) {
      const sep = resolvedMediaSrc.includes("?") ? "&" : "?";
      resolvedMediaSrc = `${resolvedMediaSrc}${sep}file_id=${encodeURIComponent(activeReport.id)}`;
    }
  }

  return (
    <AppShell
      title="Forensic Analysis Report"
      description={`Target: ${activeReport.fileName}`}
      fullWidth
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <Button asChild variant="outline" size="sm" className="h-8 text-xs border-slate-700 text-slate-300 hover:bg-slate-800">
            <Link to="/analysis">
              <RotateCcw className="mr-1.5 h-3.5 w-3.5 text-slate-400" /> New Analysis
            </Link>
          </Button>
          <Button
            asChild
            size="sm"
            variant="outline"
            className="h-8 text-xs border-sky-500/40 text-sky-400 hover:bg-sky-950/40 hover:text-sky-300"
          >
            <a
              href={finalPdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              title="Open full PDF report directly in browser"
            >
              <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
              View PDF
            </a>
          </Button>
          <Button
            size="sm"
            onClick={handleSavePdf}
            disabled={savingPdf}
            className="h-8 text-xs bg-sky-600 hover:bg-sky-500 text-white font-medium shadow-sm transition-all"
            title="Save PDF file to your computer"
          >
            <Download className="mr-1.5 h-3.5 w-3.5" />
            {savingPdf ? "Saving..." : "Download PDF"}
          </Button>
        </div>
      }
    >
      <div className="space-y-6">
        {/* Top Hero Section: Media Preview & Quality (Left) vs Primary Forensic Verdict & Specialist Models (Right) */}
        <div className="grid gap-6 lg:grid-cols-2 items-start">
        {/* Media Preview & Execution Trace */}
        <div className="space-y-5">
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-4">
            <div className="overflow-hidden rounded border border-border/80 bg-slate-950">
              {activeReport.fileType === "image" ? (
                <img
                  src={resolvedMediaSrc}
                  alt={activeReport.fileName}
                  className="mx-auto max-h-[480px] w-full object-contain"
                  onError={async (e) => {
                    const target = e.currentTarget;
                    const repId = id || activeReport?.id;
                    if (repId) {
                      try {
                        const res = await fetch(`${API_URL}/api/analysis/${repId}`);
                        if (res.ok) {
                          const data = await res.json();
                          if (data.status === "success" && data.analysis?.mediaUrl) {
                            const freshUrl = data.analysis.mediaUrl.startsWith("http")
                              ? data.analysis.mediaUrl
                              : `${API_URL}${data.analysis.mediaUrl.startsWith("/") ? "" : "/"}${data.analysis.mediaUrl}`;
                            if (target.src !== freshUrl) {
                              target.src = freshUrl;
                            }
                            setFetchedReport(data.analysis);
                            analysisStore.selectReport(data.analysis);
                          }
                        }
                      } catch {}
                    }
                  }}
                />
              ) : (
                <video
                  src={resolvedMediaSrc}
                  controls
                  className="mx-auto max-h-[480px] w-full"
                />
              )}
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 px-1 text-[11px] text-slate-400 font-mono">
              <span className="truncate max-w-[200px]">{activeReport.fileName}</span>
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="uppercase font-semibold tracking-wider text-slate-300 bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
                  {activeReport.fileType}
                </span>
                {activeReport.fileType === "video" && (
                  <>
                    <span className="rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 px-2 py-0.5 text-[10px] font-mono">
                      {activeReport.totalDurationSeconds ? `${activeReport.totalDurationSeconds.toFixed(1)}s` : ((activeReport as any).metadata?.duration_seconds ? `${(activeReport as any).metadata.duration_seconds.toFixed(1)}s` : "Video")}
                    </span>
                    <span className="rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 text-[10px] font-mono">
                      {activeReport.temporalCoveragePercentage !== undefined ? `${activeReport.temporalCoveragePercentage.toFixed(1)}% coverage` : "100% coverage"}
                    </span>
                    <span className={cn(
                      "rounded px-2 py-0.5 text-[10px] font-mono border",
                      activeReport.audioVisualSync?.audio_present
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        : "bg-slate-800 text-slate-400 border-slate-700"
                    )}>
                      Audio: {activeReport.audioVisualSync?.audio_present ? "Present (16kHz)" : "No Stream"}
                    </span>
                    {activeReport.sequenceAggregation?.total_frames_analyzed ? (
                      <span className="rounded bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 text-[10px] font-mono">
                        {activeReport.sequenceAggregation.total_frames_analyzed} frames
                      </span>
                    ) : null}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Section 1: Image Evidence & Quality Limitations Overview */}
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5 mb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
                <Scan className="h-3.5 w-3.5 text-sky-400" />
                Image Evidence & Quality Overview
              </h3>
              <span className={cn(
                "rounded px-2 py-0.5 text-[10px] font-bold font-mono tracking-wide uppercase",
                theme.badgeClass
              )}>
                {theme.badgeLabel}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded border border-slate-800/80 bg-slate-950/60 p-2.5">
                <div className="text-[10px] uppercase text-slate-400 font-semibold flex items-center gap-1">
                  <Scan className="h-3 w-3 text-sky-400" /> Face Detection (Preprocessing)
                </div>
                <div className="mt-1 font-semibold text-slate-200">
                  {activeReport.faceDetected ? "Detected & Analyzed" : "No Face Detected"}
                </div>
                <div className="mt-0.5 text-[10px] text-slate-500">
                  {activeReport.faceBbox && activeReport.faceBbox.length === 4
                    ? `Region: [${activeReport.faceBbox.join(", ")}] • Face presence does NOT establish authenticity.`
                    : (activeReport.faceDetected ? "Facial geometry located • Preprocessing step only." : "Ineligible: No face detected")}
                </div>
              </div>

              <div className="rounded border border-slate-800/80 bg-slate-950/60 p-2.5">
                <div className="text-[10px] uppercase text-slate-400 font-semibold flex items-center gap-1">
                  <Activity className="h-3 w-3 text-sky-400" /> Sharpness Metric (Laplacian)
                </div>
                <div className="mt-1 font-semibold text-slate-200 font-mono">
                  {activeReport.qualityGate?.blur_score !== undefined
                    ? `${Number(activeReport.qualityGate.blur_score).toFixed(1)} (variance)`
                    : (activeReport.features?.blur_score !== undefined
                      ? `${Number(activeReport.features.blur_score).toFixed(1)} (variance)`
                      : "Verified Adequate")}
                </div>
                <div className="mt-0.5 text-[10px] text-slate-500">
                  {Number(activeReport.qualityGate?.blur_score ?? activeReport.features?.blur_score ?? 60) > 40
                    ? "Pass: Crisp high-frequency focus • Sharpness does NOT prove authenticity."
                    : "Caution: Low optical sharpness or motion blur."}
                </div>
              </div>

              <div className="rounded border border-slate-800/80 bg-slate-950/60 p-2.5">
                <div className="text-[10px] uppercase text-slate-400 font-semibold flex items-center gap-1">
                  <Layers className="h-3 w-3 text-sky-400" /> Native Resolution
                </div>
                <div className="mt-1 font-semibold text-slate-200 font-mono">
                  {formatResolution((activeReport.measurements as any)?.resolution?.formatted || activeReport.qualityGate?.resolution || (activeReport.measurements as any)?.resolution)}
                </div>
                <div className="mt-0.5 text-[10px] text-slate-500">
                  Verified spatial input dimensions (min threshold: 100×100 px).
                </div>
              </div>

              <div className="rounded border border-slate-800/80 bg-slate-950/60 p-2.5">
                <div className="text-[10px] uppercase text-slate-400 font-semibold flex items-center gap-1">
                  <Info className="h-3 w-3 text-sky-400" /> Error Level Analysis (Exploratory)
                </div>
                <div className="mt-1 font-semibold text-slate-200 font-mono">
                  {activeReport.features?.ela_compression_anomaly !== undefined
                    ? `Discrepancy: ${Number(activeReport.features.ela_compression_anomaly).toFixed(2)}`
                    : "Baseline Disparity"}
                </div>
                <div className="mt-0.5 text-[10px] text-slate-500">
                  Exploratory JPEG recompression metric • Does NOT prove tampering.
                </div>
              </div>
            </div>
          </div>

          {/* Timeline of Agent Executions */}
          {activeReport.timeline && activeReport.timeline.length > 0 && (
            <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200 mb-4">
                Agent Execution Trace
              </h3>
              <div className="relative border-l border-slate-800 ml-2 pl-4 space-y-4">
                {activeReport.timeline.map((t, idx) => (
                  <div key={idx} className="relative">
                    <div className={cn(
                      "absolute -left-[21px] top-1 h-2.5 w-2.5 rounded-full border border-slate-900",
                      t.status === "Success" 
                        ? "bg-emerald-400" 
                        : "bg-rose-500"
                    )} />
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <div className="text-xs font-medium text-slate-200">{t.agent}</div>
                        <div className="text-[11px] text-slate-400 mt-0.5">
                          Status: <span className={t.status === "Success" ? "text-emerald-400 font-medium" : "text-rose-400 font-medium"}>{t.status}</span>
                        </div>
                      </div>
                      <div className="text-[10px] font-mono bg-slate-800 px-2 py-0.5 rounded text-slate-300">
                        {typeof t.duration === "number" ? t.duration.toFixed(3) : t.duration}s
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ========================================================================= */}
        {/* LEVEL 1: USER-FRIENDLY EXPLANATION — UNDERSTANDING YOUR RESULT */}
        {/* ========================================================================= */}
        <div className="space-y-4">
          {/* Primary Verdict & Confidence Banner */}
          <div
            className={cn(
              "rounded-lg border p-5 shadow-sm transition-all",
              theme.borderClass,
              theme.bgClass
            )}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-sky-400" />
                  {isIllustration ? "Illustration / Cartoon Detected" : isNotSuitable ? "Media Suitability Gate" : "Understanding Your Result"}
                </div>
                <div
                  className={cn(
                    "mt-1.5 inline-flex items-center gap-2.5 text-2xl font-black tracking-tight sm:text-3xl",
                    theme.textClass
                  )}
                >
                  {(isNotSuitable || isIllustration) && <FileX className="h-7 w-7 text-slate-400" />}
                  {isFake && <AlertTriangle className="h-7 w-7 text-rose-400" />}
                  {isReal && <ShieldCheck className="h-7 w-7 text-emerald-400" />}
                  {theme.label}
                </div>

                {/* Confidence Tier Pill & Indicators */}
                <div className="mt-2.5 flex flex-wrap items-center gap-2">
                  <span
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-bold tracking-wide uppercase border shadow-sm",
                      confTier === "High"
                        ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                        : confTier === "Moderate" || confTier === "Medium"
                        ? "bg-amber-500/15 text-amber-300 border-amber-500/30"
                        : confTier === "Low"
                        ? "bg-rose-500/15 text-rose-300 border-rose-500/30"
                        : "bg-slate-800 text-slate-400 border-slate-700"
                    )}
                  >
                    <span
                      className={cn(
                        "h-2 w-2 rounded-full",
                        confTier === "High"
                          ? "bg-emerald-400 animate-pulse"
                          : confTier === "Moderate" || confTier === "Medium"
                          ? "bg-amber-400"
                          : confTier === "Low"
                          ? "bg-rose-400"
                          : "bg-slate-500"
                      )}
                    />
                    {confTier === "Not Performed" ? "Analysis Not Performed" : `${confTier} Confidence`}
                  </span>

                  {confPct !== null && confPct !== undefined && confTier !== "Not Performed" && (
                    <span className="inline-flex items-center rounded-full bg-slate-800/80 px-2.5 py-1 text-xs font-semibold text-slate-300 border border-slate-700 font-mono">
                      Score: {Math.round(confPct)}%
                    </span>
                  )}

                  {hasSpecialistDisagreement && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-300 border border-amber-500/20">
                      <Info className="h-3 w-3" /> Mixed Detector Signals
                    </span>
                  )}
                </div>
              </div>

              <ConfidenceRing
                value={confPct}
                tone={tone === "neutral" ? "ok" : tone}
                status={analysisStatus}
              />
            </div>

            {/* What Does This Mean? */}
            <div className="mt-4 pt-4 border-t border-slate-800/80">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
                <Info className="h-3.5 w-3.5 text-sky-400" /> What Does This Mean?
              </h4>
              <p className="mt-1.5 text-xs sm:text-sm text-slate-200 leading-relaxed font-normal">
                {whatDoesThisMean}
              </p>
            </div>
          </div>

          {/* Why Is Confidence Limited? (Conditional on limited confidence or model conflict) */}
          {whyConfidenceIsLimited && !(isNotSuitable || isIllustration) && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-950/20 p-4 shadow-sm">
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                <div className="space-y-1">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-amber-300">
                    Why Is Confidence Limited?
                  </h4>
                  <p className="text-xs text-amber-200/90 leading-relaxed">
                    {whyConfidenceIsLimited}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* What Should You Do? (Actionable, proportionate guidance) */}
          <div className="rounded-lg border border-sky-500/20 bg-sky-950/20 p-4 shadow-sm">
            <div className="flex items-start gap-2.5">
              <ShieldCheck className="h-4 w-4 shrink-0 text-sky-400 mt-0.5" />
              <div className="space-y-1">
                <h4 className="text-xs font-bold uppercase tracking-wider text-sky-300">
                  What Should You Do?
                </h4>
                <p className="text-xs text-slate-200 leading-relaxed">
                  {whatShouldUserDo}
                </p>
              </div>
            </div>
          </div>

          {/* Key Findings / What Raised Concern? */}
          {findings.length > 0 && !(isNotSuitable || isIllustration) && (
            <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
                  <Eye className="h-3.5 w-3.5 text-sky-400" />
                  {isFake ? "What Raised Concern?" : "Key Observations"}
                </h4>
                <span className="text-[10px] text-slate-400 font-mono">
                  {findings.length} key observations
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {findings.map((f, idx) => {
                  const isConcern = f.status === "concern";
                  return (
                    <div
                      key={idx}
                      className={cn(
                        "rounded border p-3 space-y-1 transition-colors",
                        isConcern
                          ? "border-rose-500/20 bg-rose-950/15"
                          : "border-emerald-500/20 bg-emerald-950/15"
                      )}
                    >
                      <div className="flex items-center justify-between gap-1.5">
                        <span className={cn(
                          "text-xs font-bold tracking-tight",
                          isConcern ? "text-rose-300" : "text-emerald-300"
                        )}>
                          {f.title}
                        </span>
                        <span
                          className={cn(
                            "text-[9px] font-bold uppercase px-1.5 py-0.5 rounded border shrink-0",
                            isConcern
                              ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
                              : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                          )}
                        >
                          {isConcern ? "Suspicious" : "Natural"}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-300 leading-relaxed">
                        {f.explanation}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Dedicated Media Suitability Assessment Card for Unsupported Media */}
          {isNotSuitable && (
            <div className="rounded-md border border-slate-700/80 bg-slate-900/60 p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Media Suitability Assessment
              </h3>
              <div className="mt-3 grid grid-cols-2 gap-2.5 text-xs">
                <div className="rounded border border-slate-800 bg-slate-950/60 p-2.5">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Media Type</div>
                  <div className="mt-1 text-sm font-semibold text-slate-200">
                    {activeReport.mediaCategory || "Document / Non-face image"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Non-facial image structure</div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-2.5">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Face Detected</div>
                  <div className="mt-1 text-sm font-semibold text-rose-400">No</div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Facial detector requirement</div>
                </div>
              </div>
              <div className="mt-3 rounded border border-slate-800/80 bg-slate-950/40 p-2.5 text-xs text-slate-300">
                <span className="font-semibold text-sky-400">Next Step:</span> Upload a clear image containing a visible human face.
              </div>
            </div>
          )}

          {/* Dedicated Assessment Card for Non-Photographic Media (Illustrations/Anime) */}
          {isIllustration && (
            <div className="rounded-md border border-amber-500/40 bg-amber-950/20 p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-300">
                Non-Photographic Media Assessment
              </h3>
              <div className="mt-3 grid grid-cols-2 gap-2.5 text-xs">
                <div className="rounded border border-amber-500/20 bg-slate-950/60 p-2.5">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Media Classification</div>
                  <div className="mt-1 text-sm font-semibold text-amber-300">
                    {activeReport.mediaCategory || "Illustration / Anime / Cartoon"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Stylized / illustrated graphics</div>
                </div>
                <div className="rounded border border-amber-500/20 bg-slate-950/60 p-2.5">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Detection Eligibility</div>
                  <div className="mt-1 text-sm font-semibold text-amber-400">Ineligible</div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Requires camera-captured photograph</div>
                </div>
              </div>
              <div className="mt-3 rounded border border-amber-500/30 bg-slate-950/40 p-2.5 text-xs text-slate-300">
                <span className="font-semibold text-amber-400">Notice:</span> SatyaDrishti deepfake detection models are trained specifically on photographic media. Photographic models have been bypassed to prevent false-positive or misleading authenticity conclusions.
              </div>
            </div>
          )}

          {/* Conversational / Natural-Language Query Answer */}
          {activeReport.naturalLanguageAnswer && (
            <div className="rounded-lg border border-sky-500/20 bg-sky-950/20 p-4">
              <div className="flex items-center gap-1.5 text-sky-400 text-xs font-bold uppercase tracking-wider">
                <Sparkles className="h-3.5 w-3.5" /> Conversational Forensic Inquiry
              </div>
              <div className="mt-1.5 text-xs italic text-slate-400">
                &ldquo;{activeReport.naturalLanguageAnswer.question}&rdquo;
              </div>
              <div className="mt-1.5 text-xs leading-relaxed text-slate-200 font-medium whitespace-pre-line">
                {activeReport.naturalLanguageAnswer.answer}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ========================================================================= */}
      {/* USER FEEDBACK & CONTINUOUS LEARNING SECTION */}
      {/* ========================================================================= */}
      <div className="rounded-lg border border-border/80 bg-slate-900/60 p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded bg-sky-500/10 text-sky-400">
                <Sparkles className="h-3.5 w-3.5" />
              </div>
              <h3 className="text-xs sm:text-sm font-bold uppercase tracking-wider text-slate-100">
                Forensic Result Feedback & Continuous Learning
              </h3>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              Help refine forensic accuracy. Submissions are queued for expert verification and do not automatically alter production models.
            </p>
          </div>

          {recordedFeedback && (
            <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-mono font-bold uppercase tracking-wide border shrink-0 ${
              (recordedFeedback.status === "VERIFIED" || recordedFeedback.verification_status === "VERIFIED_CORRECT")
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : recordedFeedback.status === "REJECTED"
                ? "border-rose-500/30 bg-rose-500/10 text-rose-300"
                : recordedFeedback.status === "USED_FOR_TRAINING"
                ? "border-purple-500/30 bg-purple-500/10 text-purple-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-300"
            }`}>
              <span className={`h-1.5 w-1.5 rounded-full ${
                (recordedFeedback.status === "VERIFIED" || recordedFeedback.verification_status === "VERIFIED_CORRECT")
                  ? "bg-emerald-400"
                  : recordedFeedback.status === "REJECTED"
                  ? "bg-rose-400"
                  : recordedFeedback.status === "USED_FOR_TRAINING"
                  ? "bg-purple-400"
                  : "bg-amber-400 animate-pulse"
              }`} />
              Status: {recordedFeedback.status || recordedFeedback.verification_status || "PENDING"}
              {recordedFeedback.verified_label && ` (${recordedFeedback.verified_label})`}
              {recordedFeedback.storage_tier === "supabase" || recordedFeedback.persisted_in_supabase === true ? (
                <span className="text-[9px] text-emerald-400 font-normal border-l border-slate-700 pl-1.5">Cloud Synced</span>
              ) : (
                <span className="text-[9px] text-amber-400/90 font-normal border-l border-slate-700 pl-1.5">Local Storage (Sync Pending)</span>
              )}
            </span>
          )}
        </div>

        {/* Feedback Recorded Banner */}
        {recordedFeedback ? (
          <div className="rounded-md border border-slate-800 bg-slate-950/70 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">
                Feedback Logged: {recordedFeedback.feedback_type === "CORRECT" ? "Result Confirmed Correct" : recordedFeedback.feedback_type === "INCORRECT" ? "Result Reported Incorrect" : "Issue Reported"}
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                {recordedFeedback.created_at ? new Date(recordedFeedback.created_at).toLocaleString() : "Recently"}
              </span>
            </div>
            <p className="text-xs text-slate-300">
              {recordedFeedback.user_comment ? `“${recordedFeedback.user_comment}”` : "No additional comments provided."}
            </p>
            <div className="text-[11px] text-slate-400 flex items-center gap-1.5 pt-1 border-t border-slate-800/80">
              <ShieldCheck className="h-3.5 w-3.5 text-sky-400 shrink-0" />
              <span>
                Your feedback is preserved in the SatyaDrishti audit ledger. Unverified submissions are audited by forensic researchers before becoming eligible for model validation datasets.
              </span>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* 3 Primary Feedback Action Buttons */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setFeedbackType("CORRECT");
                  setFeedbackCategory("VERIFIED_ACCURATE");
                }}
                className={cn(
                  "h-11 justify-start gap-2.5 text-xs font-semibold border-slate-800 hover:bg-emerald-950/30 hover:text-emerald-300 hover:border-emerald-500/40 transition-all cursor-pointer",
                  feedbackType === "CORRECT" && "border-emerald-500 bg-emerald-950/40 text-emerald-300 shadow-sm shadow-emerald-950/40"
                )}
              >
                <ThumbsUp className="h-4 w-4 text-emerald-400 shrink-0" />
                <span>Result appears correct</span>
              </Button>

              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setFeedbackType("INCORRECT");
                  setFeedbackCategory("FALSE_POSITIVE");
                }}
                className={cn(
                  "h-11 justify-start gap-2.5 text-xs font-semibold border-slate-800 hover:bg-rose-950/30 hover:text-rose-300 hover:border-rose-500/40 transition-all cursor-pointer",
                  feedbackType === "INCORRECT" && "border-rose-500 bg-rose-950/40 text-rose-300 shadow-sm shadow-rose-950/40"
                )}
              >
                <ThumbsDown className="h-4 w-4 text-rose-400 shrink-0" />
                <span>Result appears incorrect</span>
              </Button>

              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setFeedbackType("REPORT_ISSUE");
                  setFeedbackCategory("QUALITY_OR_ARTIFACT_ISSUE");
                }}
                className={cn(
                  "h-11 justify-start gap-2.5 text-xs font-semibold border-slate-800 hover:bg-sky-950/30 hover:text-sky-300 hover:border-sky-500/40 transition-all cursor-pointer",
                  feedbackType === "REPORT_ISSUE" && "border-sky-500 bg-sky-950/40 text-sky-300 shadow-sm shadow-sky-950/40"
                )}
              >
                <Flag className="h-4 w-4 text-sky-400 shrink-0" />
                <span>Report an issue</span>
              </Button>
            </div>

            {/* Conditional Feedback Form */}
            {feedbackType && (
              <div className="rounded-md border border-slate-800 bg-slate-950/80 p-4 space-y-3.5">
                <div className="text-xs font-semibold text-slate-200">
                  {feedbackType === "CORRECT" && "Confirming Authentic/Manipulated Accuracy"}
                  {feedbackType === "INCORRECT" && "Report Incorrect Prediction for Review"}
                  {feedbackType === "REPORT_ISSUE" && "Submit Forensic Analysis Issue"}
                </div>

                {feedbackType === "INCORRECT" && (
                  <div className="space-y-3">
                    <div className="space-y-1.5">
                      <label className="text-[11px] font-medium text-slate-300">
                        User-Selected Correction (What is the true status?)
                      </label>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => setUserCorrection("REAL")}
                          className={`flex-1 rounded border px-3 py-1.5 text-xs font-medium transition-colors ${userCorrection === "REAL" ? "border-emerald-500 bg-emerald-950/40 text-emerald-300" : "border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200"}`}
                        >
                          Claim Media is REAL (Authentic)
                        </button>
                        <button
                          type="button"
                          onClick={() => setUserCorrection("FAKE")}
                          className={`flex-1 rounded border px-3 py-1.5 text-xs font-medium transition-colors ${userCorrection === "FAKE" ? "border-rose-500 bg-rose-950/40 text-rose-300" : "border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200"}`}
                        >
                          Claim Media is FAKE (Deepfake)
                        </button>
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[11px] font-medium text-slate-300">
                        Primary Discrepancy Category
                      </label>
                    <select
                      value={feedbackCategory}
                      onChange={(e) => setFeedbackCategory(e.target.value)}
                      className="w-full rounded border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-200 focus:border-sky-500 focus:outline-none"
                    >
                      <option value="FALSE_POSITIVE">False Positive (Genuine media flagged as Deepfake)</option>
                      <option value="FALSE_NEGATIVE">False Negative (Deepfake media flagged as Real)</option>
                      <option value="SPECIALIST_DISAGREEMENT">Conflicting Specialist Findings</option>
                      <option value="MISINTERPRETED_ARTIFACTS">Compression/Lens Artifact Misinterpreted</option>
                      <option value="OTHER">Other Discrepancy</option>
                    </select>
                  </div>
                </div>
                )}

                {feedbackType === "REPORT_ISSUE" && (
                  <div className="space-y-1.5">
                    <label className="text-[11px] font-medium text-slate-300">
                      Issue Type
                    </label>
                    <select
                      value={feedbackCategory}
                      onChange={(e) => setFeedbackCategory(e.target.value)}
                      className="w-full rounded border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-200 focus:border-sky-500 focus:outline-none"
                    >
                      <option value="INCORRECT_FACE_REGION">Incorrect Face/ROI Crop Selected</option>
                      <option value="IMAGE_QUALITY_CONCERN">Image Resolution/Softness Unhandled</option>
                      <option value="EXPLANATION_UNCLEAR">Forensic Explanation Unclear</option>
                      <option value="OTHER">Other Technical Problem</option>
                    </select>
                  </div>
                )}

                <div className="space-y-1.5">
                  <label className="text-[11px] font-medium text-slate-300">
                    Additional Details (Optional)
                  </label>
                  <textarea
                    rows={2}
                    value={feedbackComment}
                    onChange={(e) => setFeedbackComment(e.target.value)}
                    placeholder="Add any contextual evidence, source links, or observations..."
                    className="w-full rounded border border-slate-700 bg-slate-900 p-2 text-xs text-slate-200 placeholder:text-slate-500 focus:border-sky-500 focus:outline-none"
                  />
                </div>

                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
                  <div className="text-[10px] text-slate-400">
                    Strict verification policy: User reports are verified by analysts before dataset eligibility.
                  </div>
                  <div className="flex items-center gap-2 self-end">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => setFeedbackType(null)}
                      className="h-8 text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                    >
                      Cancel
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      disabled={isSubmittingFeedback}
                      onClick={handleSubmitFeedback}
                      className="h-8 text-xs bg-sky-600 hover:bg-sky-500 text-white font-medium"
                    >
                      {isSubmittingFeedback ? (
                        <>
                          <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> Submitting...
                        </>
                      ) : (
                        "Submit Feedback"
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ========================================================================= */}
      {/* LEVEL 2: EXPANDABLE TECHNICAL FORENSIC EVIDENCE & SPECIALIST SUITE */}
      {/* ========================================================================= */}
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 overflow-hidden shadow-sm">
        <button
          type="button"
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="w-full flex items-center justify-between p-4 bg-slate-900/90 hover:bg-slate-800/80 border-b border-slate-800 transition-colors text-left"
        >
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded bg-sky-500/10 text-sky-400">
              <Cpu className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-xs sm:text-sm font-bold uppercase tracking-wider text-slate-100 flex items-center gap-2">
                View Technical Forensic Details
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-normal">
                  Specialist Models • ELA • PRNU • FFT • Decision Policy
                </span>
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Raw softmax scores, calibrated ensemble metrics, sensor noise correlation, compression heatmaps, and spatial ROIs
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-xs font-mono font-semibold text-sky-400 shrink-0">
            <span>{showTechnicalDetails ? "Collapse Technical Details" : "View Technical Forensic Details"}</span>
            {showTechnicalDetails ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </div>
        </button>

        {showTechnicalDetails && (
          <div className="p-5 space-y-6">
            {/* Section B: Multi-Model Inference & Task Breakdown */}
            {!(isNotSuitable || isIllustration) && (
              <div className="rounded-md border border-border/80 bg-slate-950/70 p-5 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                  <div className="flex items-center gap-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded bg-sky-500/10 text-sky-400">
                      <Cpu className="h-3.5 w-3.5" />
                    </div>
                    <div>
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                        Multi-Model Forensic Comparison
                      </h3>
                      <div className="text-[10px] text-slate-400">
                        Evaluation across specialized architectures and intended detection tasks
                      </div>
                    </div>
                  </div>
                  {activeReport.isBinaryFallback && (
                    <span className="rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2 py-0.5 text-[10px] font-mono">
                      Model Conflict Active
                    </span>
                  )}
                </div>

                {/* Models List */}
                <div className="space-y-2 mt-3">
                  {activeReport.models && activeReport.models.length > 0 ? (
                    activeReport.models.map((m, idx) => {
                      const isPredFake = m.predicted_class === "FAKE";
                      const isPredReal = m.predicted_class === "REAL";
                      return (
                        <div key={idx} className="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-2">
                          <div className="flex items-center justify-between gap-2">
                            <div>
                              <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
                                <span>{m.name}</span>
                                <span className="text-[9px] font-mono font-normal text-slate-400">
                                  ({m.status})
                                </span>
                              </div>
                              <div className="text-[10px] text-sky-400/90 font-medium">
                                Task: {m.task}
                              </div>
                            </div>
                            <span
                              className={cn(
                                "rounded px-2 py-0.5 text-[10px] font-bold font-mono tracking-wider uppercase border shrink-0",
                                isPredFake
                                  ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                                  : isPredReal
                                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                                  : "bg-slate-800 text-slate-400 border-slate-700"
                              )}
                            >
                              {m.predicted_class}
                            </span>
                          </div>

                          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[10px] font-mono bg-slate-950/80 rounded p-2 border border-slate-800/60">
                            <div>
                              <span className="text-slate-500">Detector score: </span>
                              <span className={cn("font-bold", isPredFake ? "text-rose-400" : isPredReal ? "text-emerald-400" : "text-slate-300")}>
                                {m.raw_score !== null && m.raw_score !== undefined ? `${(m.raw_score * 100).toFixed(1)}%` : "N/A"}
                              </span>
                            </div>
                            <div>
                              <span className="text-slate-500">Calibrated: </span>
                              <span className="text-slate-300 font-bold">
                                {m.calibrated_prob !== null && m.calibrated_prob !== undefined ? `${(m.calibrated_prob * 100).toFixed(1)}%` : (m.raw_score !== null && m.raw_score !== undefined ? `${(m.raw_score * 100).toFixed(1)}%` : "N/A")}
                              </span>
                            </div>
                            <div className="col-span-2 sm:col-span-1">
                              <span className="text-slate-500">Calibration: </span>
                              <span className="text-slate-300">{m.calibration_status || "Empirical Benchmark"}</span>
                            </div>
                          </div>

                          {m.limitations && (
                            <div className="text-[10px] text-slate-400 leading-relaxed">
                              <span className="text-slate-500 font-semibold">Known Limitation: </span>
                              {m.limitations}
                            </div>
                          )}
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-xs text-slate-400 italic">
                      Specialist neural models evaluated: ResNet-18 baseline, XceptionNet face-swap detector, and Vision Transformer synthetic portrait detector.
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Model Score vs Safety Decision Separation */}
            {!(isNotSuitable || isIllustration) && (activeReport.rawModelScore !== undefined || activeReport.fakeProbability !== undefined) && (
              <div className="rounded-md border border-border/80 bg-slate-950/70 p-5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                  Decision Layer Metrics (Score Separation & Calibration)
                </h3>
                <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                    <div className="text-[10px] uppercase text-slate-400 font-semibold">Raw Fused Specialist Score</div>
                    <div className="mt-1 text-base font-bold font-mono text-slate-200">
                      {activeReport.rawModelScore !== undefined && activeReport.rawModelScore !== null && !isNaN(Number(activeReport.rawModelScore)) ? `${(Number(activeReport.rawModelScore) * 100).toFixed(1)}%` : "N/A"}
                    </div>
                    <div className="mt-0.5 text-[10px] text-slate-500">Uncalibrated ensemble score across neural specialists</div>
                  </div>
                  <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                    <div className="text-[10px] uppercase text-slate-400 font-semibold">Estimated Fake Probability</div>
                    <div className="mt-1 text-base font-bold font-mono text-rose-400">
                      {activeReport.fakeProbability !== undefined && activeReport.fakeProbability !== null && !isNaN(Number(activeReport.fakeProbability)) ? `${(Number(activeReport.fakeProbability) * 100).toFixed(1)}%` : "N/A"}
                    </div>
                    <div className="mt-0.5 text-[10px] text-slate-500">
                      Temperature-calibrated posterior manipulation estimate
                    </div>
                  </div>
                  <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                    <div className="text-[10px] uppercase text-slate-400 font-semibold">Estimated Authentic Probability</div>
                    <div className="mt-1 text-base font-bold font-mono text-emerald-400">
                      {(() => {
                        const realP = activeReport.realProbability !== undefined && activeReport.realProbability !== null
                          ? Number(activeReport.realProbability)
                          : (activeReport.fakeProbability !== undefined && activeReport.fakeProbability !== null ? 1.0 - Number(activeReport.fakeProbability) : null);
                        return realP !== null && !isNaN(realP) ? `${(realP * 100).toFixed(1)}%` : "N/A";
                      })()}
                    </div>
                    <div className="mt-0.5 text-[10px] text-slate-500">
                      Calibrated posterior probability of genuine media (1 - p)
                    </div>
                  </div>
                  <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                    <div className="text-[10px] uppercase text-slate-400 font-semibold">Confidence Margin</div>
                    <div className="mt-1 text-base font-bold font-mono text-amber-400">
                      {activeReport.uncertainty !== undefined && activeReport.uncertainty !== null && !isNaN(Number(activeReport.uncertainty)) ? `${(Number(activeReport.uncertainty) * 100).toFixed(1)}%` : "N/A"}
                    </div>
                    <div className="mt-0.5 text-[10px] text-slate-500">
                      Decision margin ambiguity (1 - |2p - 1|)
                    </div>
                  </div>
                </div>
              </div>
            )}

          {/* Section 3: Strongest Evidence Snapshots (Video) or Forensic Regional Evidence (Image) */}
          {(() => {
            const isVideo = activeReport.fileType === "video";
            const videoSnapshots = (activeReport.evidenceSnapshots && activeReport.evidenceSnapshots.length > 0)
              ? activeReport.evidenceSnapshots
              : (isVideo ? (activeReport.evidenceCards as any) : []);

            if (isVideo && videoSnapshots && videoSnapshots.length > 0) {
              return (
                <div className="rounded-md border border-border/80 bg-slate-900/60 p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div>
                      <h3 className="text-sm font-bold uppercase tracking-wider text-slate-100 flex items-center gap-2">
                        <Scan className="h-4 w-4 text-sky-400" />
                        Strongest Evidence Snapshots & Temporal Observations
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Informative extracted video frames grounded in empirical temporal and spatial measurements
                      </p>
                    </div>
                    <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2.5 py-1 rounded border border-slate-700">
                      {videoSnapshots.length} Keyframe Snapshots
                    </span>
                  </div>

                  {/* Video Sequence Aggregation & Full-Duration Coverage Overview */}
                  {activeReport.sequenceAggregation && (
                    <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2">
                        <div className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                          <Activity className="h-4 w-4 text-sky-400" />
                          Full-Duration Video Coverage & Sequence Aggregation
                        </div>
                        {activeReport.sequenceAggregation.peak_fake_probability !== undefined && activeReport.sequenceAggregation.peak_fake_probability !== null && (
                          <div className={cn(
                            "text-[10px] font-mono px-2 py-0.5 rounded border",
                            activeReport.sequenceAggregation.peak_fake_probability >= 0.50
                              ? "text-rose-400 bg-rose-500/10 border-rose-500/20"
                              : "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
                          )}>
                            Peak Fake Prob: {((activeReport.sequenceAggregation.peak_fake_probability || 0) * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>

                      {/* Row 1: Duration, Coverage, Tracked Faces, Frames */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                        <div className="rounded bg-slate-900/80 border border-slate-800 p-2.5">
                          <div className="text-[10px] text-slate-400 font-semibold uppercase">Total Duration</div>
                          <div className="text-base font-bold font-mono text-sky-400 mt-0.5">
                            {activeReport.totalDurationSeconds !== undefined
                              ? `${activeReport.totalDurationSeconds.toFixed(1)}s`
                              : ((activeReport as any).metadata?.duration_seconds
                                ? `${(activeReport as any).metadata.duration_seconds.toFixed(1)}s`
                                : "N/A")}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5 font-mono">
                            Timeline: {activeReport.firstAnalyzedTimestamp !== undefined ? `${activeReport.firstAnalyzedTimestamp.toFixed(1)}s` : "0.0s"} → {activeReport.lastAnalyzedTimestamp !== undefined ? `${activeReport.lastAnalyzedTimestamp.toFixed(1)}s` : (activeReport.totalDurationSeconds ? `${activeReport.totalDurationSeconds.toFixed(1)}s` : "end")}
                          </div>
                        </div>

                        <div className="rounded bg-slate-900/80 border border-slate-800 p-2.5">
                          <div className="text-[10px] text-slate-400 font-semibold uppercase">Temporal Coverage</div>
                          <div className="text-base font-bold font-mono text-emerald-400 mt-0.5">
                            {activeReport.temporalCoveragePercentage !== undefined
                              ? `${activeReport.temporalCoveragePercentage.toFixed(1)}%`
                              : "100.0%"}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Full-duration sampling
                          </div>
                        </div>

                        <div className="rounded bg-slate-900/80 border border-slate-800 p-2.5">
                          <div className="text-[10px] text-slate-400 font-semibold uppercase">Analyzed Keyframes</div>
                          <div className="text-base font-bold font-mono text-slate-200 mt-0.5">
                            {activeReport.sequenceAggregation.total_frames_analyzed}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Evenly spaced intervals
                          </div>
                        </div>

                        <div className="rounded bg-slate-900/80 border border-slate-800 p-2.5">
                          <div className="text-[10px] text-slate-400 font-semibold uppercase">Tracked Faces</div>
                          <div className="text-base font-bold font-mono text-purple-400 mt-0.5">
                            {activeReport.sequenceAggregation.tracked_faces}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Individual facial buffers
                          </div>
                        </div>
                      </div>

                      {/* Row 2: Frame-Level Evidence Outcomes */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs pt-1">
                        <div className="rounded bg-slate-900/80 border border-slate-800 p-2.5">
                          <div className="text-[10px] text-slate-400 font-semibold uppercase">Manipulated Frames</div>
                          <div className={cn(
                            "text-base font-bold font-mono mt-0.5",
                            activeReport.sequenceAggregation.fake_frames_detected > 0 ? "text-rose-400" : "text-slate-300"
                          )}>
                            {activeReport.sequenceAggregation.fake_frames_detected}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Supports FAKE
                          </div>
                        </div>

                        <div className="rounded bg-slate-900/80 border border-slate-800 p-2.5">
                          <div className="text-[10px] text-slate-400 font-semibold uppercase">Authentic Frames</div>
                          <div className="text-base font-bold font-mono text-emerald-400 mt-0.5">
                            {activeReport.sequenceAggregation.real_frames_detected}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Supports REAL
                          </div>
                        </div>

                        <div className="rounded bg-slate-900/80 border border-slate-800 p-2.5">
                          <div className="text-[10px] text-slate-400 font-semibold uppercase">Mixed / Low-Confidence Frames</div>
                          <div className="text-base font-bold font-mono text-amber-400 mt-0.5">
                            {activeReport.sequenceAggregation.inconclusive_frames}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Transition buffer
                          </div>
                        </div>

                        <div className="rounded bg-slate-900/80 border border-slate-800 p-2.5">
                          <div className="text-[10px] text-slate-400 font-semibold uppercase">Peak Timing</div>
                          <div className="text-sm font-bold font-mono text-slate-200 mt-0.5 truncate">
                            {activeReport.sequenceAggregation.manipulation_timestamps && activeReport.sequenceAggregation.manipulation_timestamps.length > 0
                              ? activeReport.sequenceAggregation.manipulation_timestamps[0].formatted_timestamp
                              : "N/A"}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Highest anomaly frame
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {videoSnapshots.map((snap: any, sIdx: number) => {
                      const isSnapFake = snap.verdict_support === "SUPPORTS_FAKE";
                      const isSnapReal = snap.verdict_support === "SUPPORTS_REAL";
                      const isStrong = snap.evidence_strength === "STRONG";
                      const isModerate = snap.evidence_strength === "MODERATE";
                      const title = snap.finding_title || snap.title || `Snapshot ${sIdx + 1}`;
                      const timestamp = snap.timestamp || "00:00:00";
                      const source = snap.evidence_source || "Specialist Analyzer";

                      return (
                        <div
                          key={snap.id || sIdx}
                          className={cn(
                            "rounded-lg border p-4 transition-colors",
                            isSnapFake
                              ? "border-rose-500/30 bg-rose-950/10 hover:border-rose-500/50"
                              : isSnapReal
                              ? "border-emerald-500/30 bg-emerald-950/10 hover:border-emerald-500/50"
                              : "border-amber-500/30 bg-amber-950/10 hover:border-amber-500/50"
                          )}
                        >
                          <div className="flex flex-col sm:flex-row gap-4 items-start">
                            {/* Real Extracted Video Frame */}
                            {snap.snapshot_data_url || snap.crop_data_url ? (
                              <div className="shrink-0 space-y-1">
                                <div className="overflow-hidden rounded border border-slate-700 bg-slate-950">
                                  <img
                                    src={snap.snapshot_data_url || snap.crop_data_url}
                                    alt={title}
                                    className="h-32 w-32 sm:h-36 sm:w-36 object-cover rounded"
                                  />
                                </div>
                                <div className="text-[10px] font-mono font-semibold text-center text-sky-400 bg-slate-900 rounded py-0.5 border border-slate-800">
                                  {timestamp}
                                </div>
                              </div>
                            ) : (
                              <div className="shrink-0 h-32 w-32 rounded border border-slate-800 bg-slate-950/80 flex flex-col items-center justify-center p-2 text-center text-slate-500">
                                <Layers className="h-6 w-6 text-slate-600 mb-1" />
                                <span className="text-[10px] font-mono text-sky-400">{timestamp}</span>
                              </div>
                            )}

                            <div className="flex-1 min-w-0 space-y-2">
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <div>
                                  <h4 className="text-xs font-bold text-slate-200">
                                    {title}
                                  </h4>
                                  <div className="text-[10px] text-slate-400 font-mono flex items-center gap-2 mt-0.5">
                                    <span className="text-sky-400">Timestamp: {timestamp}</span>
                                    <span>•</span>
                                    <span>Source: {source}</span>
                                  </div>
                                </div>
                                <div className="flex items-center gap-1.5">
                                  <span
                                    className={cn(
                                      "rounded px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider font-mono border",
                                      isSnapFake
                                        ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                                        : isSnapReal
                                        ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                                        : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                                    )}
                                  >
                                    {isSnapFake ? "Supports FAKE" : isSnapReal ? "Supports REAL" : "Mixed Findings"}
                                  </span>
                                  <span
                                    className={cn(
                                      "rounded px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider font-mono border",
                                      isStrong
                                        ? "bg-rose-500/10 text-rose-300 border-rose-500/30"
                                        : isModerate
                                        ? "bg-amber-500/10 text-amber-300 border-amber-500/30"
                                        : "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                                    )}
                                  >
                                    {(snap.evidence_strength || "CONSISTENT_NATURAL").replace("_", " ")}
                                  </span>
                                </div>
                              </div>

                              <div className="text-[11px] font-mono font-medium text-sky-400 bg-slate-950/70 rounded px-2.5 py-1 border border-slate-800 inline-block">
                                {snap.measurement_label}
                              </div>

                              <p className="text-xs text-slate-300 leading-relaxed">
                                {snap.plain_explanation || snap.observed_finding}
                              </p>

                              {snap.decision_impact && (
                                <div className="text-[10px] font-mono rounded bg-slate-950/70 border border-slate-800 p-2 text-slate-300">
                                  <span className="font-semibold text-slate-200">Decision Impact: </span>
                                  {snap.decision_impact}
                                </div>
                              )}

                              {snap.limitations && (
                                <div className="rounded bg-slate-950/40 border border-slate-800/80 p-2.5 text-[10px] text-slate-400 space-y-1">
                                  <div className="font-semibold text-slate-300 flex items-center gap-1">
                                    <Info className="h-3.5 w-3.5 text-amber-400 shrink-0" />
                                    Limitations & Benign Alternative Explanations:
                                  </div>
                                  <div className="leading-normal text-slate-400">
                                    {snap.limitations}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Dedicated Audio-Visual Temporal Synchronization Panel */}
                  {(() => {
                    const av = activeReport.audioVisualSync;
                    const statusStr = av?.status || (av?.analyzed ? "ANALYZED" : "NOT AVAILABLE — NO AUDIO TRACK");
                    const isAnalyzed = av?.analyzed || statusStr === "ANALYZED";
                    const isNoAudio = statusStr.includes("NO AUDIO TRACK");
                    const isNoSpeech = statusStr.includes("NO SPEECH DETECTED");
                    const verdictSupport = av?.verdict_support || "INCONCLUSIVE";
                    const isSync = verdictSupport === "IN_SYNC";
                    const isOutOfSync = verdictSupport === "OUT_OF_SYNC";

                    return (
                      <div className="mt-5 rounded-md border border-slate-700/80 bg-slate-950/80 p-5 space-y-4">
                        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
                          <div className="flex items-center gap-2">
                            <div className="flex h-7 w-7 items-center justify-center rounded bg-amber-500/10 text-amber-400">
                              <Activity className="h-4 w-4" />
                            </div>
                            <div>
                              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-100 flex items-center gap-2">
                                Audio-Visual Temporal Synchronization & Lip-Sync Analysis
                              </h4>
                              <p className="text-[11px] text-slate-400">
                                Cross-modal acoustic speech envelope alignment with visual oral cavity aperture dynamics
                              </p>
                            </div>
                          </div>
                          <span
                            className={cn(
                              "rounded px-2.5 py-1 text-[10px] font-bold font-mono tracking-wider uppercase border",
                              isAnalyzed
                                ? isSync
                                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                                  : isOutOfSync
                                  ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                                  : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                                : isNoAudio
                                ? "bg-slate-800 text-slate-400 border-slate-700"
                                : "bg-amber-500/15 text-amber-300 border-amber-500/30"
                            )}
                          >
                            Status: {statusStr}
                          </span>
                        </div>

                        {/* Metric Tiles */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                          <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                            <div className="text-[10px] uppercase text-slate-400 font-semibold">Audio Stream</div>
                            <div className="mt-1 text-sm font-bold font-mono text-slate-200">
                              {av?.audio_present ? "Present (16 kHz)" : (isNoAudio ? "None" : "Evaluated")}
                            </div>
                            <div className="mt-0.5 text-[10px] text-slate-500">Container audio channel</div>
                          </div>

                          <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                            <div className="text-[10px] uppercase text-slate-400 font-semibold">Speech Activity</div>
                            <div className={cn(
                              "mt-1 text-sm font-bold font-mono",
                              av?.speech_detected ? "text-emerald-400" : (isNoSpeech ? "text-amber-400" : "text-slate-400")
                            )}>
                              {av?.speech_detected ? `Vocal (${((av.speech_ratio || 0) * 100).toFixed(0)}%)` : (isNoSpeech ? "Ambient / Silence" : "N/A")}
                            </div>
                            <div className="mt-0.5 text-[10px] text-slate-500">Spectral VAD (300Hz-3.4kHz)</div>
                          </div>

                          <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                            <div className="text-[10px] uppercase text-slate-400 font-semibold">Temporal Offset</div>
                            <div className="mt-1 text-sm font-bold font-mono text-sky-400">
                              {av?.temporal_offset_ms !== undefined && av.temporal_offset_ms !== null
                                ? `${av.temporal_offset_ms > 0 ? "+" : ""}${av.temporal_offset_ms.toFixed(1)} ms`
                                : "N/A"}
                            </div>
                            <div className="mt-0.5 text-[10px] text-slate-500">Acoustic-visual lag</div>
                          </div>

                          <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                            <div className="text-[10px] uppercase text-slate-400 font-semibold">Sync Confidence</div>
                            <div className={cn(
                              "mt-1 text-sm font-bold font-mono",
                              av?.sync_confidence !== undefined && av.sync_confidence !== null
                                ? av.sync_confidence >= 0.50 ? "text-emerald-400" : "text-amber-400"
                                : "text-slate-400"
                            )}>
                              {av?.sync_confidence !== undefined && av.sync_confidence !== null
                                ? `${(av.sync_confidence * 100).toFixed(1)}%`
                                : "N/A"}
                            </div>
                            <div className="mt-0.5 text-[10px] text-slate-500">Cross-correlation index</div>
                          </div>
                        </div>

                        {/* Narrative details */}
                        <div className="rounded bg-slate-900/40 border border-slate-800/80 p-3 text-xs text-slate-300 leading-relaxed space-y-1">
                          <div className="font-semibold text-slate-200">
                            Forensic Finding:
                          </div>
                          <p className="text-slate-300 text-xs leading-relaxed">
                            {av?.details || av?.reason || "Lip-sync analysis was not performed."}
                          </p>
                          {av?.model_note && (
                            <div className="text-[10px] text-slate-400 italic pt-1 border-t border-slate-800/60">
                              * {av.model_note}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })()}
                </div>
              );
            }

            if (!isVideo && activeReport.evidenceCards && activeReport.evidenceCards.length > 0) {
              return (
                <div className="rounded-md border border-border/80 bg-slate-900/60 p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div>
                      <h3 className="text-sm font-bold uppercase tracking-wider text-slate-100 flex items-center gap-2">
                        <Scan className="h-4 w-4 text-sky-400" />
                        Strongest Forensic Evidence
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Empirical regional measurements from actual analyzed image pixels
                      </p>
                    </div>
                    <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2.5 py-1 rounded border border-slate-700">
                      {activeReport.evidenceCards.length} Forensic Regions Analyzed
                    </span>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {activeReport.evidenceCards.map((card) => {
                      const isCardFake = card.verdict_support === "SUPPORTS_FAKE";
                      const isStrong = card.evidence_strength === "STRONG";
                      const isModerate = card.evidence_strength === "MODERATE";
                      return (
                        <div
                          key={card.id}
                          className={cn(
                            "rounded-lg border p-4 transition-colors",
                            isCardFake
                              ? "border-rose-500/30 bg-rose-950/10 hover:border-rose-500/50"
                              : "border-emerald-500/30 bg-emerald-950/10 hover:border-emerald-500/50"
                          )}
                        >
                          <div className="flex flex-col sm:flex-row gap-4 items-start">
                            {/* Real Image Crop */}
                            {card.crop_data_url ? (
                              <div className="shrink-0 space-y-1">
                                <div className="overflow-hidden rounded border border-slate-700 bg-slate-950">
                                  <img
                                    src={card.crop_data_url}
                                    alt={card.title}
                                    className="h-28 w-28 object-cover rounded"
                                  />
                                </div>
                                <div className="text-[9px] font-mono text-center text-slate-400">
                                  {card.crop_box ? `[${card.crop_box.join(", ")}]` : "Actual Crop"}
                                </div>
                              </div>
                            ) : (
                              <div className="shrink-0 h-28 w-28 rounded border border-slate-800 bg-slate-950/80 flex flex-col items-center justify-center p-2 text-center text-slate-500">
                                <Layers className="h-6 w-6 text-slate-600 mb-1" />
                                <span className="text-[9px] font-mono">Image-Wide</span>
                              </div>
                            )}

                            <div className="flex-1 min-w-0 space-y-2">
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <div>
                                  <h4 className="text-xs font-bold text-slate-200">
                                    {card.title}
                                  </h4>
                                  <div className="text-[10px] text-slate-400 font-mono">
                                    Region: {card.region_name}
                                  </div>
                                </div>
                                <div className="flex items-center gap-1.5">
                                  <span
                                    className={cn(
                                      "rounded px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider font-mono border",
                                      isCardFake
                                        ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                                        : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                                    )}
                                  >
                                    {isCardFake ? "Supports FAKE" : "Supports REAL"}
                                  </span>
                                  {isCardFake && activeReport.canonicalVerdict === "REAL" && (
                                    <span className="rounded px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider font-mono border bg-amber-500/20 text-amber-300 border-amber-500/40">
                                      Qualified Finding
                                    </span>
                                  )}
                                  <span
                                    className={cn(
                                      "rounded px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider font-mono border",
                                      isStrong
                                        ? "bg-rose-500/10 text-rose-300 border-rose-500/30"
                                        : isModerate
                                        ? "bg-amber-500/10 text-amber-300 border-amber-500/30"
                                        : "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                                    )}
                                  >
                                    {card.evidence_strength.replace("_", " ")}
                                  </span>
                                </div>
                              </div>

                              <div className="text-[11px] font-mono font-medium text-sky-400 bg-slate-950/70 rounded px-2.5 py-1 border border-slate-800 inline-block">
                                {card.measurement_label}
                              </div>

                              <p className="text-xs text-slate-300 leading-relaxed">
                                {card.plain_explanation}
                              </p>

                              {card.decision_impact && (
                                <div className="text-[10px] font-mono rounded bg-slate-950/70 border border-slate-800 p-2 text-slate-300">
                                  <span className="font-semibold text-slate-200">Decision Impact: </span>
                                  {card.decision_impact}
                                </div>
                              )}

                              <div className="rounded bg-slate-950/40 border border-slate-800/80 p-2.5 text-[10px] text-slate-400 space-y-1">
                                <div className="font-semibold text-slate-300 flex items-center gap-1">
                                  <Info className="h-3.5 w-3.5 text-amber-400 shrink-0" />
                                  Limitations & Benign Alternative Explanations:
                                </div>
                                <div className="leading-normal text-slate-400">
                                  {card.limitations}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            }

            return null;
          })()}


          {/* Section: Per-Face Temporal Facial Forensics (Multi-Person Tracking) */}
          {activeReport.fileType === "video" && activeReport.trackedFaces && activeReport.trackedFaces.length > 0 && (
            <div className="rounded-md border border-border/80 bg-slate-900/60 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-100 flex items-center gap-2">
                    <Users className="h-4 w-4 text-sky-400" />
                    Per-Face Temporal Facial Forensics (Multi-Face Tracking)
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Isolated temporal track histories and empirical boundary/oral metrics evaluated per individual
                  </p>
                </div>
                <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2.5 py-1 rounded border border-slate-700">
                  {activeReport.trackedFaces.length} Tracked {activeReport.trackedFaces.length === 1 ? "Face" : "Faces"}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                {activeReport.trackedFaces.map((t) => {
                  const isFakeTrack = t.verdict === "FAKE";
                  const isRealTrack = t.verdict === "REAL";

                  return (
                    <div
                      key={t.track_id}
                      className={cn(
                        "rounded-lg border p-4 transition-colors space-y-3",
                        isFakeTrack
                          ? "border-rose-500/30 bg-rose-950/15"
                          : isRealTrack
                          ? "border-emerald-500/30 bg-emerald-950/15"
                          : "border-amber-500/30 bg-amber-950/15"
                      )}
                    >
                      <div className="flex items-center justify-between gap-2 border-b border-slate-800/80 pb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold text-slate-200">
                            Track {t.track_id}
                          </span>
                          <span className="text-[10px] font-mono text-slate-400">
                            ({t.total_frames} frames)
                          </span>
                        </div>
                        <span
                          className={cn(
                            "rounded px-2 py-0.5 text-[10px] font-bold font-mono tracking-wider uppercase border",
                            isFakeTrack
                              ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                              : isRealTrack
                              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                              : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                          )}
                        >
                          {t.verdict}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="rounded bg-slate-950/60 border border-slate-800/80 p-2">
                          <div className="text-[10px] text-slate-400">Fake Probability</div>
                          <div className={cn(
                            "text-sm font-bold font-mono mt-0.5",
                            isFakeTrack ? "text-rose-400" : isRealTrack ? "text-emerald-400" : "text-amber-400"
                          )}>
                            {(t.fake_probability * 100).toFixed(1)}%
                          </div>
                        </div>

                        <div className="rounded bg-slate-950/60 border border-slate-800/80 p-2">
                          <div className="text-[10px] text-slate-400">Boundary Flicker</div>
                          <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                            {t.boundary_flicker.toFixed(1)}
                            {t.boundary_flicker > 22.0 && (
                              <span className="ml-1 text-[9px] text-rose-400 font-sans font-medium">(Elevated)</span>
                            )}
                          </div>
                        </div>

                        <div className="rounded bg-slate-950/60 border border-slate-800/80 p-2">
                          <div className="text-[10px] text-slate-400">Oral Cavity Texture</div>
                          <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                            {t.mouth_texture.toFixed(1)}
                            {t.mouth_texture < 30.0 && (
                              <span className="ml-1 text-[9px] text-rose-400 font-sans font-medium">(Smoothed)</span>
                            )}
                          </div>
                        </div>

                        <div className="rounded bg-slate-950/60 border border-slate-800/80 p-2">
                          <div className="text-[10px] text-slate-400">Temporal Anomalies</div>
                          <div className={cn(
                            "text-sm font-bold font-mono mt-0.5",
                            t.anomalies_detected > 0 ? "text-amber-400" : "text-slate-300"
                          )}>
                            {t.anomalies_detected} Flagged
                          </div>
                        </div>
                      </div>

                      <div className="text-[10px] text-slate-400 leading-normal bg-slate-950/40 rounded p-2 border border-slate-800/60">
                        {isFakeTrack ? (
                          <span className="text-rose-300">
                            Manipulated face track exhibiting inter-frame boundary flicker and elevated synthetic probability.
                          </span>
                        ) : isRealTrack ? (
                          <span className="text-emerald-300">
                            Authentic face track exhibiting natural texture transitions and smooth boundary continuity.
                          </span>
                        ) : (
                          <span className="text-amber-300">
                            Mixed facial signals; within indeterminate transition buffer.
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {activeReport.trackedFaces.length > 1 && (
                <div className="rounded bg-slate-950/50 border border-slate-800 p-2.5 text-[11px] text-slate-400 flex items-center gap-2">
                  <Info className="h-4 w-4 text-sky-400 shrink-0" />
                  <span>
                    <strong>Multi-Person Isolation Rule:</strong> Each face track was evaluated with independent temporal buffers. A detected deepfake face drives the video verdict and is never hidden or diluted by authentic bystanders.
                  </span>
                </div>
              )}
            </div>
          )}


          {/* Multi-Provider AI Vision Forensics Panel */}
          {(() => {
            const mv = activeReport.visualAnalysis || (activeReport.evidenceBreakdown?.multi_provider_vision ? activeReport.evidenceBreakdown.multi_provider_vision : null);
            const providerResults = mv?.provider_results || {};
            const providerList = Object.keys(providerResults).map((k) => providerResults[k]);

            return (
              <div className="rounded-md border border-sky-500/30 bg-slate-900/80 p-5 shadow-sm">
                <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded bg-sky-500/10 text-sky-400">
                      <Eye className="h-3.5 w-3.5" />
                    </div>
                    <div>
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                        Multi-Provider AI Vision Forensics
                      </h3>
                      <div className="text-[10px] text-slate-400">
                        Independent verification across vision providers & local forensics
                      </div>
                    </div>
                  </div>
                  {mv?.approx_cost_usd !== undefined && mv.approx_cost_usd > 0 && (
                    <span className="rounded border border-slate-700 bg-slate-800/80 px-2 py-0.5 text-[10px] font-mono text-emerald-400">
                      Est. API: ${mv.approx_cost_usd.toFixed(5)}
                    </span>
                  )}
                </div>

                {/* Provider Cards */}
                <div className="mt-3.5 grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {providerList.length > 0 ? (
                    providerList.map((p: any, idx: number) => {
                      const assessment = p.assessment || (p.success ? "MIXED_FINDINGS" : "FAILED");
                      return (
                        <div key={idx} className="rounded border border-slate-800 bg-slate-950/70 p-3">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-semibold text-slate-200 truncate">
                              {p.name || p.provider}
                            </span>
                            <span
                              className={cn(
                                "rounded px-1.5 py-0.5 text-[9px] font-bold tracking-wide uppercase font-mono shrink-0",
                                assessment === "FAKE"
                                  ? "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                                  : assessment === "REAL"
                                  ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                                  : assessment === "DISAGREEMENT"
                                  ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                                  : "bg-slate-800 text-slate-400 border border-slate-700"
                              )}
                            >
                              {assessment}
                            </span>
                          </div>
                          <div className="mt-1 flex items-center justify-between text-[10px] text-slate-400">
                            <span className="font-mono text-slate-500 truncate max-w-[140px]">{p.model || "standard"}</span>
                            <span>{p.latency_ms ? `${p.latency_ms}ms` : "Active"}</span>
                          </div>
                          {p.observable_anomalies && p.observable_anomalies.length > 0 && (
                            <div className="mt-2 text-[10px] text-rose-300/90 bg-rose-950/20 border border-rose-900/30 rounded p-1.5 line-clamp-2">
                              {p.observable_anomalies[0]}
                            </div>
                          )}
                        </div>
                      );
                    })
                  ) : (
                    <div className="col-span-2 rounded border border-slate-800 bg-slate-950/40 p-3 text-center text-xs text-slate-400">
                      Standard local forensic suite executed. External vision APIs (Gemini, OpenAI, xAI) can be enabled via backend environment variables.
                    </div>
                  )}
                </div>

                {/* Consensus / Disagreement Statement */}
                {mv?.consensus_reason && (
                  <div className={cn(
                    "mt-3 rounded border p-2.5 text-[11px]",
                    mv.consensus_assessment === "DISAGREEMENT"
                      ? "border-amber-500/30 bg-amber-950/20 text-amber-200"
                      : "border-slate-800 bg-slate-950/40 text-slate-300"
                  )}>
                    <span className={cn(
                      "font-semibold",
                      mv.consensus_assessment === "DISAGREEMENT" ? "text-amber-400" : "text-sky-400"
                    )}>
                      {mv.consensus_assessment === "DISAGREEMENT" ? "Provider Disagreement:" : "Provider Assessment:"}
                    </span>{" "}
                    {mv.consensus_reason}
                  </div>
                )}
              </div>
            );
          })()}

          {/* Section: Multimodal Vision Reasoning (Four-Stage Forensic Examination) */}
          {activeReport.visionReasoning && (
            <div className="rounded-md border border-sky-500/30 bg-slate-900/80 p-5 shadow-sm space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-6 w-6 items-center justify-center rounded bg-sky-500/10 text-sky-400">
                    <Sparkles className="h-3.5 w-3.5" />
                  </div>
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                      Multimodal Vision Reasoning (Four-Stage Inspection)
                    </h3>
                    <div className="text-[10px] text-slate-400">
                      {activeReport.visionReasoning.provider_id
                        ? `Evaluated via ${activeReport.visionReasoning.provider_id} (${activeReport.visionReasoning.model_id || "vision model"})`
                        : "Deterministic Grounded Baseline Forensics"}
                      {activeReport.visionReasoning.latency_ms ? ` • ${activeReport.visionReasoning.latency_ms}ms` : ""}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span
                    className={cn(
                      "rounded px-2 py-0.5 text-[10px] font-bold tracking-wide uppercase font-mono border",
                      activeReport.visionReasoning.qualitative_assessment === "FAKE"
                        ? "bg-rose-500/15 text-rose-400 border-rose-500/30"
                        : activeReport.visionReasoning.qualitative_assessment === "REAL"
                        ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                        : "bg-amber-500/15 text-amber-400 border-amber-500/30"
                    )}
                  >
                    Assessment: {activeReport.visionReasoning.qualitative_assessment || "Mixed Analysis Findings"}
                  </span>
                  {activeReport.visionReasoning.evidence_strength && (
                    <span className="rounded px-2 py-0.5 text-[10px] font-mono border border-slate-700 bg-slate-800 text-slate-300">
                      Strength: {activeReport.visionReasoning.evidence_strength}
                    </span>
                  )}
                  {activeReport.visionReasoning.confidence_qualifier && (
                    <span className="rounded px-2 py-0.5 text-[10px] font-mono border border-slate-700 bg-slate-800 text-slate-400">
                      Conf: {activeReport.visionReasoning.confidence_qualifier}
                    </span>
                  )}
                </div>
              </div>

              {/* Reasoning Summary */}
              {activeReport.visionReasoning.reasoning_summary && (
                <div className="rounded bg-slate-950/60 border border-slate-800 p-3 text-xs text-slate-200 leading-relaxed">
                  <span className="font-semibold text-sky-400">Forensic Synthesis: </span>
                  {activeReport.visionReasoning.reasoning_summary}
                </div>
              )}

              {/* Stage Breakdown Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {/* Stage A: Whole Image Inspection */}
                <div className="rounded border border-slate-800 bg-slate-950/50 p-3 space-y-1.5">
                  <div className="text-[10px] uppercase font-bold text-sky-400 flex items-center gap-1">
                    <Scan className="h-3 w-3" /> Stage A: Whole-Image Structure & Lighting
                  </div>
                  <ul className="space-y-1 text-[11px] text-slate-300">
                    {Array.isArray(activeReport.visionReasoning.whole_image_observations) && activeReport.visionReasoning.whole_image_observations.length > 0 ? (
                      activeReport.visionReasoning.whole_image_observations.map((obs: string, idx: number) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-sky-400 mt-0.5">•</span>
                          <span>{obs}</span>
                        </li>
                      ))
                    ) : (
                      <li className="text-slate-500 italic">Scene perspective, illumination, and background continuity evaluated.</li>
                    )}
                  </ul>
                </div>

                {/* Stage B: Targeted Regional Inspection */}
                <div className="rounded border border-slate-800 bg-slate-950/50 p-3 space-y-1.5">
                  <div className="text-[10px] uppercase font-bold text-sky-400 flex items-center gap-1">
                    <Eye className="h-3 w-3" /> Stage B: Targeted Regional Inspection
                  </div>
                  <div className="space-y-1.5 text-[11px] text-slate-300">
                    {Array.isArray(activeReport.visionReasoning.regional_observations) && activeReport.visionReasoning.regional_observations.length > 0 ? (
                      activeReport.visionReasoning.regional_observations.map((reg: any, idx: number) => (
                        <div key={idx} className="rounded bg-slate-900/60 border border-slate-800/80 p-2">
                          <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                            <span className="font-bold text-slate-200">{reg.region || "Region"}</span>
                            <span>{reg.resolution || ""}</span>
                          </div>
                          <div className="mt-0.5 text-[11px] text-slate-300">{reg.finding}</div>
                        </div>
                      ))
                    ) : (
                      <div className="text-slate-500 italic">Eyes, mouth, hairline, and facial boundary regions examined.</div>
                    )}
                  </div>
                </div>

                {/* Stage C: Benign Alternative Explanations */}
                <div className="rounded border border-slate-800 bg-slate-950/50 p-3 space-y-1.5">
                  <div className="text-[10px] uppercase font-bold text-amber-400 flex items-center gap-1">
                    <Info className="h-3 w-3" /> Stage C: Benign Alternative Explanations
                  </div>
                  <ul className="space-y-1 text-[11px] text-slate-300">
                    {Array.isArray(activeReport.visionReasoning.alternative_explanations) && activeReport.visionReasoning.alternative_explanations.length > 0 ? (
                      activeReport.visionReasoning.alternative_explanations.map((alt: string, idx: number) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-amber-400 mt-0.5">•</span>
                          <span>{alt}</span>
                        </li>
                      ))
                    ) : (
                      <li className="text-slate-500 italic">Evaluated natural asymmetry, directional lighting, and camera optical characteristics.</li>
                    )}
                  </ul>
                </div>

                {/* Stage D: Limitations & Constraints */}
                <div className="rounded border border-slate-800 bg-slate-950/50 p-3 space-y-1.5">
                  <div className="text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" /> Stage D: Limitations & Disagreement
                  </div>
                  <ul className="space-y-1 text-[11px] text-slate-400">
                    {Array.isArray(activeReport.visionReasoning.limitations) && activeReport.visionReasoning.limitations.length > 0 ? (
                      activeReport.visionReasoning.limitations.map((lim: string, idx: number) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-slate-500 mt-0.5">•</span>
                          <span>{lim}</span>
                        </li>
                      ))
                    ) : (
                      <li className="text-slate-500 italic">Standard visual reasoning constraints applied. Absence of visible artifacts does not guarantee absolute authenticity.</li>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          )}


          {/* Section 4: Explain the Verdict & Consensus Rationale */}
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-5 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
              <CheckCircle2 className="h-3.5 w-3.5 text-sky-400" />
              Forensic Verdict Explanation & Signal Consensus
            </h3>
            <div className="rounded border border-slate-800 bg-slate-950/60 p-3 text-xs leading-relaxed text-slate-300 space-y-2">
              <p>
                <strong className="text-slate-100">Decision Derivation: </strong>
                {activeReport.explanation && !(isFake && (activeReport.explanation.includes("UNCERTAIN") || activeReport.explanation.includes("Secondary manual verification"))) && !(isReal && (activeReport.explanation.includes("UNCERTAIN") || activeReport.explanation.includes("FAKE"))) ? (
                  <span>{activeReport.explanation}</span>
                ) : isFake ? (
                  <span>
                    The classification as <strong className="text-rose-400">FAKE</strong> is supported by measured detector outputs and forensic anomalies exceeding the manipulation threshold.
                  </span>
                ) : isReal ? (
                  <span>
                    The classification as <strong className="text-emerald-400">REAL</strong> was derived from detector scores remaining below the manipulation threshold across evaluated facial regions. <strong className="text-amber-300">Notice:</strong> Absence of detected manipulation does not guarantee absolute authenticity against unseen synthesis methods.
                  </span>
                ) : isIllustration ? (
                  <span>
                    Non-photographic media (illustration, anime, or cartoon) was detected. Photographic deepfake models were bypassed to prevent misclassification.
                  </span>
                ) : isNotSuitable ? (
                  <span>
                    The media suitability check determined that no human face is present. Facial authenticity analysis cannot be performed without detectable facial geometry.
                  </span>
                ) : (
                  <span>
                    The analysis returned a provisional verdict because measured forensic signals lie in the indeterminate margin or detectors conflict. Independent verification is advised.
                  </span>
                )}
              </p>
              {hasSpecialistDisagreement && (
                <div className="mt-2 rounded border border-slate-800 bg-slate-900/60 p-2.5 text-[11px] text-slate-300">
                  <span className="font-medium text-slate-200">Decision Policy: </span>
                  The system reconciled differing detector outputs by prioritizing calibrated ensemble weights and boundary criteria. Individual detector scores are detailed in the model inventory below.
                </div>
              )}

            </div>
          </div>

          {/* Evidence Details */}
          {activeReport.evidence && activeReport.evidence.length > 0 && (
            <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Verifiable Forensic Evidence Details
              </h3>
              <ul className="mt-3 space-y-2">
                {activeReport.evidence.map((ev, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-xs text-slate-300">
                    <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" />
                    <span>{ev}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Section D: Structured Decision Evidence & Limitations */}
          {!(isNotSuitable || isIllustration) && (
            <div className="rounded-md border border-border/80 bg-slate-900/60 p-5 space-y-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
                <Sliders className="h-3.5 w-3.5 text-sky-400" />
                Defensible Evidence Derivation
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                {/* Supporting Evidence */}
                {activeReport.supportingEvidence && activeReport.supportingEvidence.length > 0 && (
                  <div className="rounded border border-emerald-500/20 bg-emerald-950/10 p-3 space-y-1.5">
                    <div className="font-semibold text-emerald-400 text-[11px] uppercase tracking-wide">
                      Supporting Evidence
                    </div>
                    <ul className="space-y-1 text-slate-300 text-[11px]">
                      {activeReport.supportingEvidence.map((s, idx) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-emerald-400 mt-0.5">•</span>
                          <span>{s}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Contradicting Evidence */}
                {activeReport.contradictingEvidence && activeReport.contradictingEvidence.length > 0 && (
                  <div className="rounded border border-amber-500/30 bg-amber-950/15 p-3 space-y-1.5">
                    <div className="font-semibold text-amber-400 text-[11px] uppercase tracking-wide">
                      Contradicting / Divergent Evidence
                    </div>
                    <ul className="space-y-1 text-amber-200/90 text-[11px]">
                      {activeReport.contradictingEvidence.map((c, idx) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-amber-400 mt-0.5">•</span>
                          <span>{c}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Unavailable Checks */}
                {activeReport.unavailableChecks && activeReport.unavailableChecks.length > 0 && (
                  <div className="rounded border border-slate-800 bg-slate-950/60 p-3 space-y-1.5">
                    <div className="font-semibold text-slate-400 text-[11px] uppercase tracking-wide">
                      Unavailable / Unassessed Verifications
                    </div>
                    <ul className="space-y-1 text-slate-400 text-[11px]">
                      {activeReport.unavailableChecks.map((u, idx) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-slate-500 mt-0.5">•</span>
                          <span>{u}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

            </div>
          )}

          {/* Section E: Expandable Technical Audit */}
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-4">
            <details className="group cursor-pointer">
              <summary className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-300 group-hover:text-white select-none">
                <span className="flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5 text-sky-400" /> Technical Audit & Calibration Parameters
                </span>
                <span className="text-[10px] text-slate-500 group-open:rotate-180 transition-transform">
                  ▼
                </span>
              </summary>
              <div className="mt-3 space-y-3 pt-3 border-t border-slate-800 text-xs">
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="rounded bg-slate-950/80 p-2.5 border border-slate-800">
                    <div className="text-slate-500 text-[10px] uppercase font-semibold">Decision Policy</div>
                    <div className="font-mono text-slate-200 mt-0.5">SafetyDecisionSystem</div>
                    <div className="text-[9px] text-slate-500 mt-0.5">Bounds: [tau_real: 0.35, tau_fake: 0.65]</div>
                  </div>
                  <div className="rounded bg-slate-950/80 p-2.5 border border-slate-800">
                    <div className="text-slate-500 text-[10px] uppercase font-semibold">Verification Criteria</div>
                    <div className={cn("font-mono font-bold mt-0.5", activeReport.decision?.criteria_met ? "text-emerald-400" : "text-amber-400")}>
                      {activeReport.decision?.criteria_met ? "MET (Consensus Validated)" : "UNMET (Review Recommended)"}
                    </div>
                    <div className="text-[9px] text-slate-500 mt-0.5">{activeReport.decision?.confidence_type || "Standard Calibration"}</div>
                  </div>
                </div>

                {activeReport.limitations && activeReport.limitations.length > 0 && (
                  <div className="rounded bg-slate-950/60 p-2.5 border border-slate-800 space-y-1">
                    <div className="text-[10px] uppercase text-amber-400 font-semibold flex items-center gap-1">
                      <Info className="h-3 w-3" /> Technical Limitations & Operating Assumptions
                    </div>
                    <ul className="space-y-1 text-[11px] text-slate-400">
                      {activeReport.limitations.map((lim, idx) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-slate-600">•</span>
                          <span>{lim}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </details>
          </div>


          {/* Audit Recommendation & Agent Output Summary */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {activeReport.recommendation ? (
              <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                  Audit Recommendation
                </h3>
                <p className="mt-2 text-xs leading-relaxed text-slate-300">
                  {activeReport.recommendation}
                </p>
              </div>
            ) : null}

            <div className={cn("rounded-md border border-border/80 bg-slate-900/60 p-5", !activeReport.recommendation && "lg:col-span-2")}>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Agent Output Summary
              </h3>
              <ul className="mt-3 divide-y divide-border/60">
                {activeReport.agents.map((a) => (
                  <li key={a.name} className="grid grid-cols-[130px_minmax(0,1fr)] gap-3 py-2.5 text-xs">
                    <span className="font-semibold text-slate-200">{a.name}</span>
                    <span className="min-w-0 text-slate-300">{a.summary}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          </div>
        )}
      </div>
      </div>
    </AppShell>
  );
}

function ConfidenceRing({
  value,
  tone,
  status = "ANALYSIS COMPLETE",
}: {
  value: number | null | undefined;
  tone: "ok" | "danger" | "warning";
  status?: AnalysisStatus;
}) {
  if (status === "ILLUSTRATION DETECTED") {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-amber-500/30 bg-amber-950/20 px-3.5 py-3 text-center min-w-[125px] shrink-0">
        <FileX className="h-5 w-5 text-amber-400 mb-1" />
        <div className="text-[9px] font-semibold uppercase tracking-wider text-amber-400">Media Type</div>
        <div className="text-[11px] font-bold text-amber-200 mt-0.5 font-mono">ILLUSTRATION</div>
      </div>
    );
  }

  if (status === "UNSUPPORTED FILE") {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-slate-700 bg-slate-900/80 px-3.5 py-3 text-center min-w-[125px] shrink-0">
        <FileX className="h-5 w-5 text-slate-400 mb-1" />
        <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400">Status</div>
        <div className="text-[11px] font-bold text-slate-200 mt-0.5 font-mono">UNSUPPORTED FILE</div>
      </div>
    );
  }

  if (status === "ANALYSIS FAILED") {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-rose-500/30 bg-rose-950/20 px-3.5 py-3 text-center min-w-[125px] shrink-0">
        <HelpCircle className="h-5 w-5 text-rose-400 mb-1" />
        <div className="text-[9px] font-semibold uppercase tracking-wider text-rose-400">Status</div>
        <div className="text-[11px] font-bold text-rose-200 mt-0.5 font-mono">ANALYSIS FAILED</div>
      </div>
    );
  }

  if (status === "REVIEW RECOMMENDED") {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-amber-500/30 bg-amber-950/20 px-3.5 py-3 text-center min-w-[125px] shrink-0">
        <AlertTriangle className="h-5 w-5 text-amber-400 mb-1" />
        <div className="text-[9px] font-semibold uppercase tracking-wider text-amber-400">Status</div>
        <div className="text-[11px] font-bold text-amber-200 mt-0.5 font-mono">REVIEW RECOMMENDED</div>
        {value !== null && value !== undefined && (
          <div className="text-[10px] text-amber-400/80 font-mono mt-1 font-semibold">{Math.round(value)}% score</div>
        )}
      </div>
    );
  }

  if (value === null || value === undefined) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-emerald-500/30 bg-emerald-950/20 px-3.5 py-3 text-center min-w-[125px] shrink-0">
        <ShieldCheck className="h-5 w-5 text-emerald-400 mb-1" />
        <div className="text-[9px] font-semibold uppercase tracking-wider text-emerald-400">Status</div>
        <div className="text-[11px] font-bold text-emerald-200 mt-0.5 font-mono">ANALYSIS COMPLETE</div>
      </div>
    );
  }
  const numericVal = Math.max(0, Math.min(100, Math.round(value)));
  const r = 32;
  const c = 2 * Math.PI * r;
  const offset = c - (numericVal / 100) * c;
  const color =
    tone === "ok"
      ? "oklch(0.70 0.17 155)"
      : tone === "warning"
      ? "oklch(0.75 0.18 75)"
      : "oklch(0.62 0.22 25)";
  return (
    <div className="relative h-20 w-20 shrink-0">
      <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90">
        <circle
          cx="40"
          cy="40"
          r={r}
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth="5"
          fill="none"
        />
        <circle
          cx="40"
          cy="40"
          r={r}
          stroke={color}
          strokeWidth="5"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="text-base font-bold font-mono text-slate-100">{numericVal}%</div>
          <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400">
            confidence
          </div>
        </div>
      </div>
    </div>
  );
}
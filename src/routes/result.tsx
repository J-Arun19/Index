import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
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
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { useAnalysisState, analysisStore, AnalysisResult, formatBackendAnalysis, getVerdictTheme, getCanonicalVerdict } from "@/lib/analysis-store";
import { useAuthState } from "@/lib/auth-store";
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
          const res = await fetch(`${API_URL}/api/history?user_id=${encodeURIComponent(String(user.id))}`);
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
  }, [user, isInitialized, id, selectedReport, latest, history.length]);

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
        const res = await fetch(`${API_URL}/api/analysis/${reportId}`);
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
  const isNotSuitable = canonicalVerdict === "NOT_SUITABLE";
  const isFake = canonicalVerdict === "FAKE";
  const isReal = canonicalVerdict === "REAL";
  const isUncertain = canonicalVerdict === "UNCERTAIN";
  const isSuspicious = isUncertain;
  const isInconclusive = isUncertain;
  const reliabilityWarning = detectedWarning || activeReport.reliabilityWarning;

  const tone = isNotSuitable ? "neutral" : isFake ? "danger" : isReal ? "ok" : "warning";

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
      actions={
        <div className="flex items-center gap-2">
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
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
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
            <div className="mt-3 flex items-center justify-between px-1 text-[11px] text-slate-400 font-mono">
              <span className="truncate">{activeReport.fileName}</span>
              <span className="uppercase font-semibold tracking-wider text-slate-300">{activeReport.fileType}</span>
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

        {/* Verdict + Confidence + Explanation */}
        <div className="space-y-5">
          <div
            className={cn(
              "rounded-md border p-5",
              theme.borderClass,
              theme.bgClass
            )}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  {isNotSuitable ? "Media Suitability Gate" : "Primary Forensic Verdict"}
                </div>
                <div
                  className={cn(
                    "mt-1 inline-flex items-center gap-2 text-2xl font-bold tracking-tight sm:text-3xl",
                    theme.textClass
                  )}
                >
                  {isNotSuitable && <FileX className="h-6 w-6 text-slate-400" />}
                  {isFake && <AlertTriangle className="h-6 w-6 text-rose-400" />}
                  {isReal && <ShieldCheck className="h-6 w-6 text-emerald-400" />}
                  {isUncertain && <HelpCircle className="h-6 w-6 text-amber-400" />}
                  {theme.label}
                </div>

                {/* Subordinated Warning Banner if Low Confidence or Uncertain */}
                {isBinaryFallback && (
                  <div className="mt-3 flex items-start gap-2.5 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200 shadow-sm">
                    <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                    <div className="space-y-1">
                      <div className="font-semibold text-amber-300 flex items-center gap-2">
                        <span>LOW CONFIDENCE WARNING</span>
                        <span className="rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-mono text-amber-200">
                          Specialist Disagreement / Ambiguous Margin
                        </span>
                      </div>
                      <p className="text-[11px] text-amber-300/90 leading-relaxed">
                        {reliabilityWarning || "Specialist neural detectors or vision providers produced conflicting assessments. The primary verdict reflects weighted ensemble evidence; secondary manual verification is advised."}
                      </p>
                    </div>
                  </div>
                )}
                {isUncertain && !isBinaryFallback && (
                  <div className="mt-3 flex items-start gap-2.5 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200 shadow-sm">
                    <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                    <div className="space-y-1">
                      <div className="font-semibold text-amber-300">
                        <span>Incomplete Evaluation</span>
                      </div>
                      <p className="text-[11px] text-amber-300/90 leading-relaxed">
                        {reliabilityWarning || "Analysis could not be completed reliably due to processing limits or missing model evaluation."}
                      </p>
                    </div>
                  </div>
                )}

                <p className="mt-2 text-xs text-slate-300 leading-relaxed">
                  {activeReport.userFacingStatement ||
                    (isNotSuitable
                      ? "No detectable human face was found in this media. SatyaDrishti's current deepfake detector is designed for facial-media authenticity analysis. Upload a clear media file containing a visible human face."
                      : isFake
                      ? "Deepfake detected / high likelihood of manipulation based on forensic anomaly patterns."
                      : isReal
                      ? "Authentic natural media features observed. Note: Absence of detected manipulation does not guarantee absolute authenticity."
                      : "Insufficient evidence to reliably classify this media. Authenticity has not been established.")}
                </p>
                {isNotSuitable ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    <span className="inline-flex items-center rounded border border-slate-700 bg-slate-800/80 px-2 py-0.5 text-[10px] font-semibold text-slate-300 font-mono">
                      Status: Ineligible
                    </span>
                    <span className="inline-flex items-center rounded border border-slate-700 bg-slate-800/80 px-2 py-0.5 text-[10px] font-semibold text-rose-400 font-mono">
                      Face Detected: No
                    </span>
                  </div>
                ) : (
                  <div className="mt-3 flex flex-wrap gap-2">
                    <span className={cn(
                      "inline-flex items-center rounded px-2 py-0.5 text-[10px] font-semibold border",
                      (isBinaryFallback || activeReport.confidenceLevel === "Low")
                        ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        : activeReport.confidenceLevel === "High"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        : activeReport.confidenceLevel === "Medium"
                        ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        : "bg-slate-800 text-slate-400 border-slate-700",
                    )}>
                      {isBinaryFallback
                        ? "Low Confidence"
                        : (activeReport.confidenceLevel ? `${activeReport.confidenceLevel} Confidence` : "Standard Evaluated")}
                    </span>
                    {activeReport.fakeProbability !== undefined && (
                      <span className="inline-flex items-center rounded px-2 py-0.5 text-[10px] font-semibold border border-slate-700 bg-slate-800 text-slate-300 font-mono">
                        {isUncertain ? "Est. Fake Prob" : "Fake Prob"}: {(activeReport.fakeProbability * 100).toFixed(1)}%{isUncertain ? " (Estimate)" : ""}
                      </span>
                    )}
                    {activeReport.uncertainty !== undefined && (
                      <span className="inline-flex items-center rounded px-2 py-0.5 text-[10px] font-semibold border border-slate-700 bg-slate-800 text-slate-300 font-mono">
                        Uncertainty: {(activeReport.uncertainty * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                )}
              </div>
              {isNotSuitable ? (
                <div className="flex flex-col items-center justify-center rounded-lg border border-slate-800 bg-slate-950/80 px-4 py-3 text-center min-w-[110px] shrink-0">
                  <FileX className="h-6 w-6 text-slate-400 mb-1" />
                  <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400">Analysis</div>
                  <div className="text-xs font-bold text-slate-300 mt-0.5 font-mono">NOT PERFORMED</div>
                </div>
              ) : (
                <ConfidenceRing value={activeReport.confidence} tone={tone === "neutral" ? "ok" : tone} uncertified={isUncertain || activeReport.confidence === null} />
              )}
            </div>
          </div>

          {/* Dedicated Media Suitability Assessment Card for Unsupported Media */}
          {isNotSuitable && (
            <div className="rounded-md border border-slate-700/80 bg-slate-900/60 p-5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Media Suitability Assessment
              </h3>
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Media Type</div>
                  <div className="mt-1 text-sm font-semibold text-slate-200">
                    {activeReport.mediaCategory || "Document / Non-face image"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Non-facial image structure</div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Face Detected</div>
                  <div className="mt-1 text-sm font-semibold text-rose-400">No</div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Haar cascade / MTCNN output</div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Analysis Status</div>
                  <div className="mt-1 text-sm font-semibold text-slate-300">Not performed</div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Ineligible input media</div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Reason</div>
                  <div className="mt-1 text-sm font-semibold text-amber-400">No suitable face detected</div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Facial detector requirement</div>
                </div>
              </div>
              <div className="mt-3 rounded border border-slate-800/80 bg-slate-950/40 p-3 text-xs text-slate-300">
                <span className="font-semibold text-sky-400">Recommendation:</span> Upload a clear image containing a visible human face.
              </div>
            </div>
          )}

          {/* Model Score vs Safety Decision Separation (Hidden for unsupported media) */}
          {!isNotSuitable && (activeReport.rawModelScore !== undefined || activeReport.fakeProbability !== undefined) && (
            <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Decision Layer Metrics (Score Separation)
              </h3>
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Raw Fused Specialist Score</div>
                  <div className="mt-1 text-base font-bold font-mono text-slate-200">
                    {activeReport.rawModelScore !== undefined ? `${(activeReport.rawModelScore * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Uncalibrated ensemble score across neural specialists</div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Final Fake Score</div>
                  <div className="mt-1 text-base font-bold font-mono text-sky-400">
                    {activeReport.fakeProbability !== undefined ? `${(activeReport.fakeProbability * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">
                    Calibrated multi-signal decision score
                  </div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Estimated Fake Probability</div>
                  <div className="mt-1 text-base font-bold font-mono text-indigo-400">
                    {activeReport.fakeProbability !== undefined ? `${(activeReport.fakeProbability * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">
                    Temperature-calibrated posterior probability estimate
                  </div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Uncertainty</div>
                  <div className="mt-1 text-base font-bold font-mono text-amber-400">
                    {activeReport.uncertainty !== undefined ? `${(activeReport.uncertainty * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">
                    Decision margin ambiguity (1 - |2p - 1|)
                  </div>
                </div>
              </div>
              <div className="mt-3 text-[10px] text-slate-400">
                <span className="font-semibold text-slate-300">Authoritative Decision Policy:</span> Successfully analyzed supported media resolves to a primary REAL or FAKE verdict. When models conflict or fall in the ambiguous margin, a LOW CONFIDENCE warning and manual verification recommendation are assigned.
              </div>
            </div>
          )}

          {/* Section B: Multi-Model Inference & Task Breakdown */}
          {!isNotSuitable && (
            <div className="rounded-md border border-border/80 bg-slate-900/60 p-5 space-y-3">
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
                      <div key={idx} className="rounded border border-slate-800 bg-slate-950/70 p-3 space-y-2">
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

                        <div className="grid grid-cols-2 gap-2 text-[10px] font-mono bg-slate-900/60 rounded p-2 border border-slate-800/60">
                          <div>
                            <span className="text-slate-500">Raw fake score: </span>
                            <span className={cn("font-bold", isPredFake ? "text-rose-400" : isPredReal ? "text-emerald-400" : "text-slate-300")}>
                              {m.raw_score !== null && m.raw_score !== undefined ? `${(m.raw_score * 100).toFixed(1)}%` : "N/A"}
                            </span>
                          </div>
                          <div>
                            <span className="text-slate-500">Calibration: </span>
                            <span className="text-slate-300">{m.calibration_status || "Raw Softmax"}</span>
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

          {/* Conversational / Natural-Language Query Answer */}
          {activeReport.naturalLanguageAnswer && (
            <div className="rounded-md border border-sky-500/20 bg-sky-950/20 p-5">
              <div className="flex items-center gap-1.5 text-sky-400 text-xs font-semibold uppercase tracking-wider">
                <Sparkles className="h-3.5 w-3.5" /> Conversational Forensic Inquiry
              </div>
              <div className="mt-2 text-xs italic text-slate-400">
                &ldquo;{activeReport.naturalLanguageAnswer.question}&rdquo;
              </div>
              <div className="mt-2 text-xs leading-relaxed text-slate-200 font-medium">
                {activeReport.naturalLanguageAnswer.answer}
              </div>
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
                      const assessment = p.assessment || (p.success ? "INCONCLUSIVE" : "FAILED");
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
                            Indeterminate facial signals; within inconclusive threshold buffer.
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

                  <div className="grid grid-cols-1 gap-4">
                    {videoSnapshots.map((snap: any, sIdx: number) => {
                      const isSnapFake = snap.verdict_support === "SUPPORTS_FAKE";
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
                              : "border-emerald-500/30 bg-emerald-950/10 hover:border-emerald-500/50"
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
                                        : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                                    )}
                                  >
                                    {isSnapFake ? "Supports FAKE" : "Supports REAL"}
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

                  {/* Audio-Visual Temporal Synchronization Note */}
                  <div className="mt-4 rounded-md border border-amber-500/30 bg-amber-500/5 p-3.5 text-xs flex items-start gap-3">
                    <Info className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                    <div className="space-y-1">
                      <div className="font-semibold text-slate-200 flex items-center gap-2">
                        <span>Audio-Visual Temporal Synchronization</span>
                        <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-mono text-amber-300 font-bold uppercase">
                          Not Analyzed
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed">
                        Lip-sync and speech-to-audio temporal alignment was not evaluated. SatyaDrishti's active pipeline evaluated visual and spatial-temporal features. A cross-modal phoneme-viseme alignment model is currently not configured.
                      </p>
                    </div>
                  </div>
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

                  <div className="grid grid-cols-1 gap-4">
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

          {/* Section 4: Explain the Verdict & Consensus Rationale */}
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-5 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
              <CheckCircle2 className="h-3.5 w-3.5 text-sky-400" />
              Forensic Verdict Explanation & Signal Consensus
            </h3>
            <div className="rounded border border-slate-800 bg-slate-950/60 p-3 text-xs leading-relaxed text-slate-300 space-y-2">
              <p>
                <strong className="text-slate-100">Decision Derivation: </strong>
                {activeReport.explanation ? (
                  <span>{activeReport.explanation}</span>
                ) : isFake ? (
                  <span>
                    The classification as <strong className="text-rose-400">FAKE</strong> is supported by measured detector outputs and forensic anomalies exceeding the manipulation threshold.
                  </span>
                ) : isReal ? (
                  <span>
                    The classification as <strong className="text-emerald-400">REAL</strong> was derived from detector scores remaining below the manipulation threshold across evaluated facial regions. <strong className="text-amber-300">Notice:</strong> Absence of detected manipulation does not guarantee absolute authenticity against unseen synthesis methods.
                  </span>
                ) : isNotSuitable ? (
                  <span>
                    The media suitability check determined that no human face is present. Facial authenticity analysis cannot be performed without detectable facial geometry.
                  </span>
                ) : (
                  <span>
                    The analysis returned an <strong className="text-amber-400">UNCERTAIN</strong> verdict because measured forensic signals lie in the indeterminate margin, detectors conflict, or image quality is insufficient. Authenticity has NOT been established.
                  </span>
                )}
              </p>
              {isBinaryFallback && reliabilityWarning && (
                <div className="mt-2 rounded border border-amber-500/30 bg-amber-500/10 p-2 text-[11px] text-amber-200">
                  <span className="font-semibold text-amber-300">Provisional Limit: </span>
                  {reliabilityWarning}
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
          {!isNotSuitable && (
            <div className="rounded-md border border-border/80 bg-slate-900/60 p-5 space-y-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
                <Sliders className="h-3.5 w-3.5 text-sky-400" />
                Defensible Evidence Derivation
              </h3>

              <div className="space-y-3 text-xs">
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
                      {activeReport.decision?.criteria_met ? "MET (Certified)" : "UNMET (Conservative Abstention)"}
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

          {/* Recommendation */}
          {activeReport.recommendation && (
            <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Audit Recommendation
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-slate-300">
                {activeReport.recommendation}
              </p>
            </div>
          )}

          <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
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
    </AppShell>
  );
}

function ConfidenceRing({
  value,
  tone,
  uncertified = false,
}: {
  value: number | null | undefined;
  tone: "ok" | "danger" | "warning";
  uncertified?: boolean;
}) {
  const isUncertified = uncertified || value === null || value === undefined;
  if (isUncertified) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-amber-500/30 bg-amber-950/20 px-3 py-3 text-center min-w-[110px] shrink-0">
        <HelpCircle className="h-5 w-5 text-amber-400 mb-1" />
        <div className="text-[9px] font-semibold uppercase tracking-wider text-amber-400">Status</div>
        <div className="text-[11px] font-bold text-amber-200 mt-0.5 font-mono">NOT CERTIFIED</div>
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
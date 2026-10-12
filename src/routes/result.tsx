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
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { useAnalysisState, analysisStore, AnalysisResult, formatBackendAnalysis } from "@/lib/analysis-store";
import { useAuthState } from "@/lib/auth-store";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

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
  // 1. If ID matches an item in history, use it
  // 2. If ID matches selectedReport, use selectedReport
  // 3. If fetchedReport matches ID, use fetchedReport
  // 4. If no ID param is provided, use selectedReport or fallback to latest
  const activeReport =
    (id ? history.find((h) => h.id === id) : null) ||
    (id && selectedReport?.id === id ? selectedReport : null) ||
    (id && fetchedReport?.id === id ? fetchedReport : null) ||
    (!id ? selectedReport || latest : null);

  // If id is provided in URL, but we don't have full details in memory, fetch from backend API
  useEffect(() => {
    if (!id) return;
    if (activeReport && activeReport.id === id && (activeReport.timeline?.length || activeReport.evidence?.length)) {
      return;
    }

    const fetchDetail = async () => {
      try {
        const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
        const res = await fetch(`${API_URL}/api/analysis/${id}`);
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
  }, [id, activeReport]);

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

  const isNotSuitable = activeReport.verdict === "NOT_SUITABLE";
  const isFake = !isNotSuitable && (activeReport.verdict === "DEEPFAKE" || activeReport.verdict === "FAKE");
  const isUncertain = !isNotSuitable && activeReport.verdict === "UNCERTAIN";
  const isLikelyReal = !isNotSuitable && !isFake && !isUncertain;

  const tone = isNotSuitable ? "neutral" : isFake ? "danger" : isUncertain ? "warning" : "ok";

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
                  src={activeReport.mediaUrl}
                  alt={activeReport.fileName}
                  className="mx-auto max-h-[480px] w-full object-contain"
                />
              ) : (
                <video
                  src={activeReport.mediaUrl}
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
              isNotSuitable && "border-slate-700 bg-slate-900/90 shadow-sm",
              isFake && "border-rose-500/30 bg-rose-500/5",
              isUncertain && "border-amber-500/30 bg-amber-500/5",
              isLikelyReal && "border-emerald-500/30 bg-emerald-500/5",
            )}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  {isNotSuitable ? "Media Suitability Gate" : "Forensic Safety Verdict"}
                </div>
                <div
                  className={cn(
                    "mt-1 inline-flex items-center gap-2 text-2xl font-bold tracking-tight sm:text-3xl",
                    isNotSuitable && "text-slate-200",
                    isFake && "text-rose-400",
                    isUncertain && "text-amber-400",
                    isLikelyReal && "text-emerald-400",
                  )}
                >
                  {isNotSuitable && <FileX className="h-6 w-6 text-slate-400" />}
                  {isFake && <AlertTriangle className="h-6 w-6 text-rose-400" />}
                  {isUncertain && <AlertTriangle className="h-6 w-6 text-amber-400" />}
                  {isLikelyReal && <ShieldCheck className="h-6 w-6 text-emerald-400" />}
                  {isNotSuitable
                    ? "NOT SUITABLE FOR FACE ANALYSIS"
                    : isFake
                    ? "FAKE"
                    : isUncertain
                    ? "UNCERTAIN"
                    : "LIKELY REAL"}
                </div>
                <p className="mt-1.5 text-xs text-slate-300 leading-relaxed">
                  {isNotSuitable &&
                    (activeReport.userFacingStatement ||
                      "No detectable human face was found in this image. SatyaDrishti's current image deepfake detector is designed for facial-media authenticity analysis. Upload a clear image containing a visible human face.")}
                  {isFake && (activeReport.userFacingStatement || "Deepfake detected / high likelihood of manipulation.")}
                  {isUncertain && (activeReport.userFacingStatement || "Insufficient evidence to determine authenticity reliably. Additional verification is recommended.")}
                  {isLikelyReal && (activeReport.userFacingStatement || "No strong evidence of manipulation was detected. This does not prove authenticity.")}
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
                  activeReport.confidenceLevel && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      <span className={cn(
                        "inline-flex items-center rounded px-2 py-0.5 text-[10px] font-semibold border",
                        activeReport.confidenceLevel === "High" && "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
                        activeReport.confidenceLevel === "Medium" && "bg-amber-500/10 text-amber-400 border-amber-500/20",
                        activeReport.confidenceLevel === "Low" && "bg-rose-500/10 text-rose-400 border-rose-500/20",
                      )}>
                        {activeReport.confidenceLevel} Confidence
                      </span>
                      {activeReport.uncertainty !== undefined && (
                        <span className="inline-flex items-center rounded px-2 py-0.5 text-[10px] font-semibold border border-slate-700 bg-slate-800 text-slate-300 font-mono">
                          Uncertainty: {(activeReport.uncertainty * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                  )
                )}
              </div>
              {isNotSuitable ? (
                <div className="flex flex-col items-center justify-center rounded-lg border border-slate-800 bg-slate-950/80 px-4 py-3 text-center min-w-[110px] shrink-0">
                  <FileX className="h-6 w-6 text-slate-400 mb-1" />
                  <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400">Analysis</div>
                  <div className="text-xs font-bold text-slate-300 mt-0.5 font-mono">NOT PERFORMED</div>
                </div>
              ) : (
                <ConfidenceRing value={activeReport.confidence} tone={tone === "neutral" ? "ok" : tone} />
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
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Raw Classifier Score</div>
                  <div className="mt-1 text-base font-bold font-mono text-slate-200">
                    {activeReport.rawModelScore !== undefined ? `${(activeReport.rawModelScore * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Uncalibrated neural probability</div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[10px] uppercase text-slate-400 font-semibold">Calibrated Probability</div>
                  <div className="mt-1 text-base font-bold font-mono text-sky-400">
                    {activeReport.fakeProbability !== undefined ? `${(activeReport.fakeProbability * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-500">Temperature scaled on validation set</div>
                </div>
              </div>
              <div className="mt-2 text-[10px] text-slate-400">
                <span className="font-semibold text-slate-300">Safety Policy:</span> Samples between 35% and 65% calibrated probability or with contradictory evidence automatically abstain to UNCERTAIN.
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

          <div className="rounded-md border border-border/80 bg-slate-900/60 p-5">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              Technical Rationale & Reason
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-300">
              {activeReport.reason || activeReport.explanation}
            </p>
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
}: {
  value: number;
  tone: "ok" | "danger" | "warning";
}) {
  const r = 32;
  const c = 2 * Math.PI * r;
  const offset = c - (value / 100) * c;
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
          <div className="text-base font-bold font-mono text-slate-100">{value}%</div>
          <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-400">
            confidence
          </div>
        </div>
      </div>
    </div>
  );
}
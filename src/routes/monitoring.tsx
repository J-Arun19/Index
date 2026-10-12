import { useState, useRef, useEffect } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { 
  Play, 
  Square, 
  Video, 
  AlertTriangle, 
  ShieldCheck, 
  Activity, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  Cpu, 
  Layers, 
  Camera, 
  Save 
} from "lucide-react";

import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useAuthState } from "@/lib/auth-store";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/monitoring")({
  component: RealTimeMonitoringPage,
});

interface FrameAnalysisResult {
  status?: string;
  id: string;
  timestamp: string;
  verdict: "REAL" | "LIKELY_REAL" | "UNCERTAIN" | "SUSPICIOUS" | "DEEPFAKE" | "INVALID_FRAME" | "ERROR";
  detection_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  latency_ms: number;
  agents: {
    validation: string;
    image: string;
    confidence: string;
    artifact: string;
    fusion: string;
  };
}

interface SessionStats {
  frames_received: number;
  frames_analyzed: number;
  frames_skipped: number;
  deepfake_detections: number;
  average_confidence: number;
  average_latency_ms: number;
  final_verdict: string;
  final_risk: string;
}

interface MonitoringSessionSummary {
  session_id: string;
  user_name: string;
  status: string;
  started_at: string;
  ended_at?: string;
  frames_received: number;
  frames_analyzed: number;
  frames_skipped: number;
  deepfake_detections: number;
  average_confidence: number;
  average_latency_ms: number;
  final_verdict: string;
  final_risk: string;
}

export function RealTimeMonitoringPage() {
  const navigate = useNavigate();
  const { user, isInitialized } = useAuthState();

  useEffect(() => {
    if (isInitialized && !user) {
      navigate({ to: "/login" });
    }
  }, [user, isInitialized, navigate]);

  // Stream & Hardware state
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Statistics counters
  const [framesReceived, setFramesReceived] = useState(0);
  const [framesAnalyzed, setFramesAnalyzed] = useState(0);
  const [framesSkipped, setFramesSkipped] = useState(0);
  const [deepfakeDetections, setDeepfakeDetections] = useState(0);
  const [avgConfidence, setAvgConfidence] = useState(0);
  const [avgLatency, setAvgLatency] = useState(0);

  // Real-time state
  const [currentVerdict, setCurrentVerdict] = useState<string>("IDLE");
  const [currentScore, setCurrentScore] = useState<number>(0);
  const [currentRisk, setCurrentRisk] = useState<string>("LOW");
  const [currentAgents, setCurrentAgents] = useState<{ [key: string]: string }>({
    validation: "IDLE",
    image: "IDLE",
    confidence: "IDLE",
    artifact: "IDLE",
    fusion: "IDLE",
  });

  // Temporal window (Size N = 5)
  const [windowPredictions, setWindowPredictions] = useState<FrameAnalysisResult[]>([]);
  const [timeline, setTimeline] = useState<FrameAnalysisResult[]>([]);

  // Session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionSummary, setSessionSummary] = useState<SessionStats | null>(null);
  const [isSavingSession, setIsSavingSession] = useState(false);
  const [sessionSavedSuccess, setSessionSavedSuccess] = useState(false);

  // Refs for non-blocking sampling loop
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<any>(null);
  const isProcessingRef = useRef<boolean>(false);

  // Refs for tracking aggregate stats inside async loop
  const statsRef = useRef({
    received: 0,
    analyzed: 0,
    skipped: 0,
    deepfakes: 0,
    totalScoreSum: 0,
    totalLatencySum: 0,
  });

  // Clean up stream on unmount
  useEffect(() => {
    return () => {
      stopMonitoringCleanup();
    };
  }, []);

  const stopMonitoringCleanup = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    isProcessingRef.current = false;
  };

  const handleStartMonitoring = async () => {
    setCameraError(null);
    setSessionSummary(null);
    setSessionSavedSuccess(false);

    // Reset stats
    statsRef.current = {
      received: 0,
      analyzed: 0,
      skipped: 0,
      deepfakes: 0,
      totalScoreSum: 0,
      totalLatencySum: 0,
    };
    setFramesReceived(0);
    setFramesAnalyzed(0);
    setFramesSkipped(0);
    setDeepfakeDetections(0);
    setAvgConfidence(0);
    setAvgLatency(0);
    setWindowPredictions([]);
    setTimeline([]);
    setCurrentVerdict("ANALYZING");

    // 1. Access Camera via MediaDevices API
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Browser does not support camera access via MediaDevices API.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });

      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // 2. Initialize Monitoring Session in Backend
      try {
        const res = await fetch("/api/monitor/start-session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: user?.id || null,
            user_name: user?.name || "Anonymous Auditor",
          }),
        });
        const data = await res.json();
        if (data.session_id) {
          setSessionId(data.session_id);
        }
      } catch (err) {
        console.warn("Could not create remote monitoring session, fallback to local:", err);
      }

      setIsMonitoring(true);

      // 3. Start Controlled Frame Sampling Loop (ANALYSIS_FPS = 2 -> 500ms interval)
      timerRef.current = setInterval(() => {
        captureAndAnalyzeFrame();
      }, 500);
    } catch (err: any) {
      console.error("Camera access error:", err);
      let msg = "Failed to access camera.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        msg = "Camera permission denied. Please allow camera access in browser settings.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        msg = "No video camera detected on your device.";
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        msg = "Camera is currently in use by another application.";
      } else if (err.message) {
        msg = err.message;
      }
      setCameraError(msg);
      stopMonitoringCleanup();
      setIsMonitoring(false);
    }
  };

  const handleStopMonitoring = async () => {
    stopMonitoringCleanup();
    setIsMonitoring(false);

    const s = statsRef.current;
    const finalAvgScore = s.analyzed > 0 ? s.totalScoreSum / s.analyzed : 0;
    const finalAvgLatency = s.analyzed > 0 ? s.totalLatencySum / s.analyzed : 0;

    // Determine final aggregated verdict & risk
    let finalVerdict = "REAL";
    let finalRisk = "LOW";
    if (s.deepfakes > 0) {
      finalVerdict = "DEEPFAKE";
      finalRisk = finalAvgScore >= 80 ? "CRITICAL" : "HIGH";
    } else if (finalAvgScore >= 30) {
      finalVerdict = "SUSPICIOUS";
      finalRisk = "MEDIUM";
    }

    const summary: SessionStats = {
      frames_received: s.received,
      frames_analyzed: s.analyzed,
      frames_skipped: s.skipped,
      deepfake_detections: s.deepfakes,
      average_confidence: Number(finalAvgScore.toFixed(2)),
      average_latency_ms: Number(finalAvgLatency.toFixed(1)),
      final_verdict: finalVerdict,
      final_risk: finalRisk,
    };

    setSessionSummary(summary);
    setCurrentVerdict("STOPPED");

    // Call backend to finalize session record
    if (sessionId) {
      try {
        await fetch("/api/monitor/stop-session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            stats: summary,
          }),
        });
      } catch (err) {
        console.error("Failed to stop remote session:", err);
      }
    }
  };

  const captureAndAnalyzeFrame = async () => {
    statsRef.current.received += 1;
    setFramesReceived(statsRef.current.received);

    // Active Inference Guard: If backend inference is busy, SKIP frame
    if (isProcessingRef.current) {
      statsRef.current.skipped += 1;
      setFramesSkipped(statsRef.current.skipped);
      return;
    }

    if (!videoRef.current || !canvasRef.current) return;

    isProcessingRef.current = true;

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video.videoWidth === 0 || video.videoHeight === 0) {
        isProcessingRef.current = false;
        return;
      }

      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        isProcessingRef.current = false;
        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameBase64 = canvas.toDataURL("image/jpeg", 0.75);

      // Call Backend Real-Time Endpoint
      const res = await fetch("/api/monitor/analyze-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frame: frameBase64,
          session_id: sessionId,
          timestamp: new Date().toLocaleTimeString(),
        }),
      });

      const data: FrameAnalysisResult = await res.json();

      if (data.status === "success" && data.verdict !== "INVALID_FRAME") {
        statsRef.current.analyzed += 1;
        statsRef.current.totalScoreSum += data.detection_score;
        statsRef.current.totalLatencySum += data.latency_ms;

        if (data.verdict === "DEEPFAKE" || data.risk_level === "HIGH" || data.risk_level === "CRITICAL") {
          statsRef.current.deepfakes += 1;
        }

        setFramesAnalyzed(statsRef.current.analyzed);
        setDeepfakeDetections(statsRef.current.deepfakes);
        setAvgConfidence(Number((statsRef.current.totalScoreSum / statsRef.current.analyzed).toFixed(1)));
        setAvgLatency(Number((statsRef.current.totalLatencySum / statsRef.current.analyzed).toFixed(0)));

        setCurrentVerdict(data.verdict);
        setCurrentScore(data.detection_score);
        setCurrentRisk(data.risk_level);
        if (data.agents) {
          setCurrentAgents(data.agents);
        }

        // Update Temporal Evidence Sliding Window (N = 5)
        setWindowPredictions((prev) => {
          const nextWin = [...prev, data].slice(-5);
          return nextWin;
        });

        // Add to Timeline (last 10 items)
        setTimeline((prev) => [data, ...prev].slice(0, 10));
      } else {
        statsRef.current.skipped += 1;
        setFramesSkipped(statsRef.current.skipped);
      }
    } catch (err) {
      console.error("Frame analysis request failed:", err);
      statsRef.current.skipped += 1;
      setFramesSkipped(statsRef.current.skipped);
    } finally {
      isProcessingRef.current = false;
    }
  };

  // Calculate Stable Temporal Risk over last N=5 frames
  const calculateTemporalRisk = () => {
    if (windowPredictions.length === 0) return { verdict: currentVerdict, risk: currentRisk, score: currentScore };
    
    const deepfakes = windowPredictions.filter((f) => f.verdict === "DEEPFAKE" || f.detection_score >= 60).length;
    const suspicious = windowPredictions.filter((f) => f.verdict === "SUSPICIOUS" || f.detection_score >= 30).length;
    const avgWinScore = windowPredictions.reduce((acc, f) => acc + f.detection_score, 0) / windowPredictions.length;

    let stableVerdict = "REAL";
    let stableRisk = "LOW";

    if (deepfakes >= 3 || avgWinScore >= 60) {
      stableVerdict = "DEEPFAKE";
      stableRisk = avgWinScore >= 80 ? "CRITICAL" : "HIGH";
    } else if (suspicious >= 2 || avgWinScore >= 30) {
      stableVerdict = "SUSPICIOUS";
      stableRisk = "MEDIUM";
    }

    return { verdict: stableVerdict, risk: stableRisk, score: Number(avgWinScore.toFixed(1)) };
  };

  const temporalState = calculateTemporalRisk();

  return (
    <AppShell
      title="Real-Time Monitoring Console"
      description="Continuous stream sampling with temporal evidence aggregation."
    >
      <div className="space-y-5">
        {/* Hidden Canvas for Frame Capture */}
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* Camera Permission / Error Banner */}
        {cameraError && (
          <div className="flex items-center gap-3 rounded-md border border-rose-500/40 bg-rose-500/10 p-3.5 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" />
            <div className="flex-1">
              <span className="font-semibold text-rose-200">Camera Access Error:</span> {cameraError}
            </div>
          </div>
        )}

        {/* Temporal Risk Alert Banner */}
        {isMonitoring && (temporalState.verdict === "DEEPFAKE" || temporalState.verdict === "SUSPICIOUS") && (
          <div className="flex items-center gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 p-3.5 text-amber-300">
            <AlertTriangle className="h-5 w-5 shrink-0 text-amber-400" />
            <div className="flex-1">
              <div className="font-semibold text-xs text-amber-200 uppercase tracking-wider">⚠ Media Manipulation Flagged</div>
              <div className="text-[11px] opacity-90 mt-0.5">
                Detection Score: {temporalState.score}% | Risk Level: {temporalState.risk} | 
                Sustained evidence accumulated across {windowPredictions.length}-frame temporal window.
              </div>
            </div>
          </div>
        )}

        {/* Main Grid: Video Stream + Controls */}
        <div className="grid gap-5 lg:grid-cols-3">
          {/* Video Stream Container (2 cols) */}
          <div className="lg:col-span-2 overflow-hidden rounded-md border border-border/80 bg-slate-900/60 p-4 space-y-3">
            <div className="flex flex-row items-center justify-between">
              <div>
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <Video className="h-4 w-4 text-sky-400" /> Live Media Stream Input
                </h2>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Native MediaDevices Feed | Controlled Sampling Rate (2 FPS)
                </p>
              </div>
              <div>
                {isMonitoring ? (
                  <span className="inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[10px] font-semibold border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    LIVE (2 FPS)
                  </span>
                ) : (
                  <span className="inline-flex items-center rounded px-2 py-0.5 text-[10px] font-medium border border-slate-700 bg-slate-800 text-slate-400">
                    STREAM OFFLINE
                  </span>
                )}
              </div>
            </div>

            <div className="relative aspect-video w-full overflow-hidden rounded border border-border/80 bg-slate-950 flex items-center justify-center">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={cn("h-full w-full object-cover", !isMonitoring && "hidden")}
              />

              {!isMonitoring && (
                <div className="flex flex-col items-center gap-3 text-slate-400 p-6 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded border border-slate-700 bg-slate-800 text-slate-300">
                    <Camera className="h-6 w-6 text-sky-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-xs text-slate-200">Camera Feed Inactive</div>
                    <div className="text-[11px] text-slate-400 mt-0.5 max-w-sm">
                      Click "Start Stream Monitoring" below to grant camera access and initiate continuous deepfake risk sampling.
                    </div>
                  </div>
                </div>
              )}

              {/* Live Overlays */}
              {isMonitoring && (
                <div className="absolute top-3 left-3 flex flex-wrap gap-2">
                  <span className={cn(
                    "px-2 py-0.5 text-[10px] font-bold rounded uppercase tracking-wider text-white border",
                    (temporalState.verdict === "REAL" || temporalState.verdict === "LIKELY_REAL") && "bg-emerald-600 border-emerald-500",
                    (temporalState.verdict === "SUSPICIOUS" || temporalState.verdict === "UNCERTAIN") && "bg-amber-600 border-amber-500",
                    temporalState.verdict === "DEEPFAKE" && "bg-rose-600 border-rose-500"
                  )}>
                    Verdict: {temporalState.verdict}
                  </span>
                  <span className="px-2 py-0.5 text-[10px] font-mono font-medium rounded bg-slate-900/90 text-slate-200 border border-slate-700">
                    Score: {currentScore.toFixed(1)}%
                  </span>
                  <span className={cn(
                    "px-2 py-0.5 text-[10px] font-semibold rounded bg-slate-900/90 border",
                    temporalState.risk === "LOW" && "text-emerald-400 border-emerald-500/30",
                    temporalState.risk === "MEDIUM" && "text-amber-400 border-amber-500/30",
                    (temporalState.risk === "HIGH" || temporalState.risk === "CRITICAL") && "text-rose-400 border-rose-500/30"
                  )}>
                    Risk: {temporalState.risk}
                  </span>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="pt-1">
              {!isMonitoring ? (
                <Button onClick={handleStartMonitoring} className="w-full h-9 bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs gap-2">
                  <Play className="h-3.5 w-3.5" /> Start Stream Monitoring
                </Button>
              ) : (
                <Button onClick={handleStopMonitoring} variant="destructive" className="w-full h-9 font-medium text-xs gap-2 bg-rose-600 hover:bg-rose-500">
                  <Square className="h-3.5 w-3.5" /> Stop Stream Monitoring
                </Button>
              )}
            </div>
          </div>

          {/* Current Real-Time Metrics (1 col) */}
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-4 flex flex-col justify-between space-y-4">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-sky-400" /> Live Telemetry Status
              </h2>
              <p className="text-[11px] text-slate-400 mt-0.5">
                5-Frame Sliding Temporal Window
              </p>
            </div>

            <div className="rounded border border-border/80 bg-slate-950/60 p-3.5 text-center space-y-1">
              <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Current Temporal Verdict</div>
              <div className={cn(
                "text-2xl font-bold tracking-tight",
                (temporalState.verdict === "REAL" || temporalState.verdict === "LIKELY_REAL") && "text-emerald-400",
                (temporalState.verdict === "SUSPICIOUS" || temporalState.verdict === "UNCERTAIN") && "text-amber-400",
                temporalState.verdict === "DEEPFAKE" && "text-rose-400",
                temporalState.verdict === "IDLE" && "text-slate-400"
              )}>
                {temporalState.verdict}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2.5">
              <div className="rounded border border-border/80 bg-slate-950/40 p-2.5">
                <div className="text-[10px] uppercase font-semibold text-slate-400">Score</div>
                <div className="mt-0.5 text-lg font-bold text-slate-100 font-mono">{currentScore.toFixed(1)}%</div>
              </div>
              <div className="rounded border border-border/80 bg-slate-950/40 p-2.5">
                <div className="text-[10px] uppercase font-semibold text-slate-400">Risk</div>
                <div className={cn(
                  "mt-0.5 text-lg font-bold",
                  temporalState.risk === "LOW" && "text-emerald-400",
                  temporalState.risk === "MEDIUM" && "text-amber-400",
                  (temporalState.risk === "HIGH" || temporalState.risk === "CRITICAL") && "text-rose-400"
                )}>
                  {temporalState.risk}
                </div>
              </div>
            </div>

            {/* Agent Execution Pipeline Checklist */}
            <div className="space-y-2">
              <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5 text-sky-400" /> Pipeline Modules
              </div>
              <div className="space-y-1 text-xs font-mono">
                <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[11px]">
                  <span className="text-slate-400">Media Validation</span>
                  <span className="text-emerald-400 font-semibold">{currentAgents.validation || "ACTIVE"}</span>
                </div>
                <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[11px]">
                  <span className="text-slate-400">Image Agent</span>
                  <span className="text-emerald-400 font-semibold">{currentAgents.image || "IDLE"}</span>
                </div>
                <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[11px]">
                  <span className="text-slate-400">Confidence Eval</span>
                  <span className="text-emerald-400 font-semibold">{currentAgents.confidence || "IDLE"}</span>
                </div>
                <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[11px]">
                  <span className="text-slate-400">Artifact Agent</span>
                  <span className="text-slate-400 font-semibold">{currentAgents.artifact || "SKIPPED"}</span>
                </div>
                <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[11px]">
                  <span className="text-slate-400">Fusion Agent</span>
                  <span className="text-slate-400 font-semibold">{currentAgents.fusion || "SKIPPED"}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Operational Telemetry Grid */}
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-3.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Frames Received</div>
            <div className="mt-1 text-xl font-bold text-slate-100 font-mono">{framesReceived}</div>
          </div>
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-3.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Frames Analyzed</div>
            <div className="mt-1 text-xl font-bold text-sky-400 font-mono">{framesAnalyzed}</div>
          </div>
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-3.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Frames Skipped</div>
            <div className="mt-1 text-xl font-bold text-amber-400 font-mono">{framesSkipped}</div>
          </div>
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-3.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Deepfake Detections</div>
            <div className="mt-1 text-xl font-bold text-rose-400 font-mono">{deepfakeDetections}</div>
          </div>
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-3.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Average Score</div>
            <div className="mt-1 text-xl font-bold text-slate-100 font-mono">{avgConfidence}%</div>
          </div>
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-3.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Average Latency</div>
            <div className="mt-1 text-xl font-bold text-slate-100 font-mono">{avgLatency} ms</div>
          </div>
        </div>

        {/* Detection Event Log Timeline */}
        <div className="rounded-md border border-border/80 bg-slate-900/60">
          <div className="flex items-center justify-between border-b border-border/80 px-4 py-3">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Clock className="h-3.5 w-3.5 text-sky-400" /> Detection Telemetry Stream (Recent Samples)
            </h3>
          </div>
          <div className="p-4">
            {timeline.length === 0 ? (
              <div className="py-6 text-center text-xs text-slate-400">
                No telemetry samples recorded yet. Initiate stream monitoring to log live events.
              </div>
            ) : (
              <div className="space-y-1.5">
                {timeline.map((item, i) => (
                  <div
                    key={item.id || i}
                    className="flex items-center justify-between rounded border border-border/60 bg-slate-950/40 px-3.5 py-2 text-xs font-mono"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-slate-400 text-[11px]">{item.timestamp}</span>
                      <span className={cn(
                        "rounded px-2 py-0.5 text-[10px] font-semibold border",
                        (item.verdict === "REAL" || item.verdict === "LIKELY_REAL") && "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
                        (item.verdict === "SUSPICIOUS" || item.verdict === "UNCERTAIN") && "bg-amber-500/10 text-amber-400 border-amber-500/20",
                        item.verdict === "DEEPFAKE" && "bg-rose-500/10 text-rose-400 border-rose-500/20"
                      )}>
                        {item.verdict}
                      </span>
                    </div>

                    <div className="flex items-center gap-5 text-[11px]">
                      <div>
                        <span className="text-slate-400 mr-1">Score:</span>
                        <span className="font-semibold text-slate-200">{item.detection_score}%</span>
                      </div>
                      <div>
                        <span className="text-slate-400 mr-1">Risk:</span>
                        <span className={cn(
                          "font-semibold",
                          item.risk_level === "LOW" && "text-emerald-400",
                          item.risk_level === "MEDIUM" && "text-amber-400",
                          (item.risk_level === "HIGH" || item.risk_level === "CRITICAL") && "text-rose-400"
                        )}>
                          {item.risk_level}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400 mr-1">Latency:</span>
                        <span className="text-slate-300">{item.latency_ms}ms</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Session Summary Panel */}
        {sessionSummary && (
          <div className="rounded-md border border-sky-500/30 bg-slate-900/80 p-5 space-y-4">
            <div className="border-b border-border/80 pb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">Monitoring Session Summary</h3>
              <p className="text-[11px] text-slate-400">Aggregated telemetry for completed real-time session</p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="rounded bg-slate-950/60 p-3 border border-border/60">
                <div className="text-slate-400 text-[10px] uppercase font-semibold">Frames Analyzed</div>
                <div className="text-lg font-bold text-slate-100 mt-0.5 font-mono">{sessionSummary.frames_analyzed}</div>
              </div>
              <div className="rounded bg-slate-950/60 p-3 border border-border/60">
                <div className="text-slate-400 text-[10px] uppercase font-semibold">Frames Skipped</div>
                <div className="text-lg font-bold text-amber-400 mt-0.5 font-mono">{sessionSummary.frames_skipped}</div>
              </div>
              <div className="rounded bg-slate-950/60 p-3 border border-border/60">
                <div className="text-slate-400 text-[10px] uppercase font-semibold">Deepfake Detections</div>
                <div className="text-lg font-bold text-rose-400 mt-0.5 font-mono">{sessionSummary.deepfake_detections}</div>
              </div>
              <div className="rounded bg-slate-950/60 p-3 border border-border/60">
                <div className="text-slate-400 text-[10px] uppercase font-semibold">Avg Latency</div>
                <div className="text-lg font-bold text-slate-100 mt-0.5 font-mono">{sessionSummary.average_latency_ms} ms</div>
              </div>
            </div>

            <div className="flex items-center justify-between rounded border border-border/80 bg-slate-950/40 p-3.5">
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Session Final Verdict & Risk</div>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-lg font-bold text-slate-100">{sessionSummary.final_verdict}</span>
                  <span className="rounded px-2 py-0.5 text-[10px] font-semibold border border-slate-700 text-slate-300 font-mono">
                    {sessionSummary.final_risk} RISK
                  </span>
                </div>
              </div>
              {sessionSavedSuccess ? (
                <span className="inline-flex items-center gap-1.5 rounded px-3 py-1 text-xs font-semibold text-emerald-400 border border-emerald-500/30 bg-emerald-500/10">
                  <CheckCircle2 className="h-3.5 w-3.5" /> Session Saved
                </span>
              ) : (
                <Button size="sm" onClick={() => setSessionSavedSuccess(true)} className="h-8 text-xs bg-sky-600 hover:bg-sky-500 text-white gap-1.5 font-medium">
                  <Save className="h-3.5 w-3.5" /> Save Session Record
                </Button>
              )}
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}

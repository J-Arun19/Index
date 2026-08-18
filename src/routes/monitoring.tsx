import { useState, useRef, useEffect } from "react";
import { createFileRoute } from "@tanstack/react-router";
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
  id: string;
  timestamp: string;
  verdict: "REAL" | "SUSPICIOUS" | "DEEPFAKE" | "INVALID_FRAME" | "ERROR";
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

export function RealTimeMonitoringPage() {
  const { user } = useAuthState();

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
      title="Real-Time Deepfake Monitoring"
      description="Live camera stream with continuous deepfake-risk analysis and temporal evidence aggregation."
    >
      <div className="space-y-6">
        {/* Hidden Canvas for Frame Capture */}
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* Camera Permission / Error Banner */}
        {cameraError && (
          <div className="flex items-center gap-3 rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
            <AlertTriangle className="h-5 w-5 shrink-0" />
            <div className="flex-1">
              <span className="font-semibold">Camera Access Error:</span> {cameraError}
            </div>
          </div>
        )}

        {/* Temporal Risk Alert Banner */}
        {isMonitoring && (temporalState.verdict === "DEEPFAKE" || temporalState.verdict === "SUSPICIOUS") && (
          <div className="flex items-center gap-4 rounded-lg border border-amber-500/50 bg-amber-500/10 p-4 text-amber-500 animate-pulse">
            <AlertTriangle className="h-6 w-6 shrink-0" />
            <div className="flex-1">
              <div className="font-bold text-base">⚠ POSSIBLE MANIPULATION DETECTED</div>
              <div className="text-xs opacity-90 mt-0.5">
                Detection Score: {temporalState.score}% | Risk Level: {temporalState.risk} | 
                Sustained evidence accumulated across {windowPredictions.length}-frame temporal window. Further verification recommended.
              </div>
            </div>
          </div>
        )}

        {/* Main Grid: Video Stream + Controls */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Video Stream Container (2 cols) */}
          <Card className="lg:col-span-2 overflow-hidden border-border/70 bg-card/80">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <div>
                <CardTitle className="text-lg font-bold flex items-center gap-2">
                  <Video className="h-5 w-5 text-primary" /> Camera Stream
                </CardTitle>
                <CardDescription className="text-xs">
                  Native MediaDevices input | Controlled Sampling (2 FPS)
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                {isMonitoring ? (
                  <Badge variant="outline" className="gap-1.5 border-emerald-500/40 bg-emerald-500/10 text-emerald-400">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 animate-ping" />
                    LIVE (2 FPS)
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-muted-foreground">
                    OFFLINE
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-border/60 bg-black/90 flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={cn("h-full w-full object-cover", !isMonitoring && "hidden")}
                />

                {!isMonitoring && (
                  <div className="flex flex-col items-center gap-3 text-muted-foreground p-6 text-center">
                    <div className="flex h-16 w-16 items-center justify-center rounded-full bg-secondary/80 text-foreground">
                      <Camera className="h-8 w-8 text-primary" />
                    </div>
                    <div>
                      <div className="font-semibold text-foreground">Camera Monitoring Inactive</div>
                      <div className="text-xs text-muted-foreground mt-1 max-w-sm">
                        Click "START MONITORING" below to request camera permission and initiate real-time deepfake analysis.
                      </div>
                    </div>
                  </div>
                )}

                {/* Live Overlays */}
                {isMonitoring && (
                  <div className="absolute top-3 left-3 flex flex-wrap gap-2">
                    <Badge className={cn(
                      "px-2.5 py-1 text-xs font-bold",
                      temporalState.verdict === "REAL" && "bg-emerald-600 text-white",
                      temporalState.verdict === "SUSPICIOUS" && "bg-amber-600 text-white",
                      temporalState.verdict === "DEEPFAKE" && "bg-rose-600 text-white"
                    )}>
                      Verdict: {temporalState.verdict}
                    </Badge>
                    <Badge variant="secondary" className="bg-black/70 backdrop-blur text-white border border-white/10">
                      Detection Score: {currentScore.toFixed(1)}%
                    </Badge>
                    <Badge variant="outline" className={cn(
                      "bg-black/70 backdrop-blur",
                      temporalState.risk === "LOW" && "text-emerald-400 border-emerald-500/40",
                      temporalState.risk === "MEDIUM" && "text-amber-400 border-amber-500/40",
                      (temporalState.risk === "HIGH" || temporalState.risk === "CRITICAL") && "text-rose-400 border-rose-500/40"
                    )}>
                      Risk: {temporalState.risk}
                    </Badge>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-4">
                {!isMonitoring ? (
                  <Button onClick={handleStartMonitoring} className="flex-1 bg-primary text-primary-foreground gap-2 font-semibold">
                    <Play className="h-4 w-4" /> START MONITORING
                  </Button>
                ) : (
                  <Button onClick={handleStopMonitoring} variant="destructive" className="flex-1 gap-2 font-semibold">
                    <Square className="h-4 w-4" /> STOP MONITORING
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Current Real-Time Metrics (1 col) */}
          <Card className="border-border/70 bg-card/80 flex flex-col justify-between">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg font-bold flex items-center gap-2">
                <ShieldCheck className="h-5 w-5 text-primary" /> Live Risk Status
              </CardTitle>
              <CardDescription className="text-xs">
                Temporal Evidence Aggregation (5-frame window)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="rounded-xl border border-border/70 bg-secondary/30 p-4 text-center space-y-2">
                <div className="text-xs text-muted-foreground font-medium uppercase tracking-wider">Current Temporal Verdict</div>
                <div className={cn(
                  "text-3xl font-extrabold tracking-tight",
                  temporalState.verdict === "REAL" && "text-emerald-400",
                  temporalState.verdict === "SUSPICIOUS" && "text-amber-400",
                  temporalState.verdict === "DEEPFAKE" && "text-rose-400",
                  temporalState.verdict === "IDLE" && "text-muted-foreground"
                )}>
                  {temporalState.verdict}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg border border-border/60 bg-card p-3">
                  <div className="text-xs text-muted-foreground">Detection Score</div>
                  <div className="mt-1 text-xl font-bold text-foreground">{currentScore.toFixed(1)}%</div>
                </div>
                <div className="rounded-lg border border-border/60 bg-card p-3">
                  <div className="text-xs text-muted-foreground">Risk Level</div>
                  <div className={cn(
                    "mt-1 text-xl font-bold",
                    temporalState.risk === "LOW" && "text-emerald-400",
                    temporalState.risk === "MEDIUM" && "text-amber-400",
                    (temporalState.risk === "HIGH" || temporalState.risk === "CRITICAL") && "text-rose-400"
                  )}>
                    {temporalState.risk}
                  </div>
                </div>
              </div>

              {/* Agent Execution Timeline Status */}
              <div className="space-y-2">
                <div className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                  <Activity className="h-3.5 w-3.5 text-primary" /> Agent Execution Pipeline
                </div>
                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center justify-between rounded bg-secondary/50 px-2.5 py-1.5">
                    <span className="text-muted-foreground">Media Validation</span>
                    <Badge variant="outline" className="text-[10px] py-0 px-1.5 border-emerald-500/40 text-emerald-400">
                      {currentAgents.validation || "ACTIVE"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between rounded bg-secondary/50 px-2.5 py-1.5">
                    <span className="text-muted-foreground">Image Agent (ResNet-18)</span>
                    <Badge variant="outline" className="text-[10px] py-0 px-1.5 border-emerald-500/40 text-emerald-400">
                      {currentAgents.image || "IDLE"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between rounded bg-secondary/50 px-2.5 py-1.5">
                    <span className="text-muted-foreground">Confidence Evaluation</span>
                    <Badge variant="outline" className="text-[10px] py-0 px-1.5 border-emerald-500/40 text-emerald-400">
                      {currentAgents.confidence || "IDLE"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between rounded bg-secondary/50 px-2.5 py-1.5">
                    <span className="text-muted-foreground">Artifact Agent</span>
                    <Badge variant="outline" className={cn(
                      "text-[10px] py-0 px-1.5",
                      currentAgents.artifact === "Success" ? "border-amber-500/40 text-amber-400" : "text-muted-foreground border-border/50"
                    )}>
                      {currentAgents.artifact || "SKIPPED"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between rounded bg-secondary/50 px-2.5 py-1.5">
                    <span className="text-muted-foreground">Fusion Agent</span>
                    <Badge variant="outline" className={cn(
                      "text-[10px] py-0 px-1.5",
                      currentAgents.fusion === "Success" ? "border-amber-500/40 text-amber-400" : "text-muted-foreground border-border/50"
                    )}>
                      {currentAgents.fusion || "SKIPPED"}
                    </Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Monitoring Statistics Grid */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
          <Card className="border-border/70 bg-card/60">
            <CardContent className="p-4">
              <div className="text-xs font-medium text-muted-foreground">Frames Received</div>
              <div className="mt-1 text-2xl font-bold text-foreground">{framesReceived}</div>
            </CardContent>
          </Card>
          <Card className="border-border/70 bg-card/60">
            <CardContent className="p-4">
              <div className="text-xs font-medium text-muted-foreground">Frames Analyzed</div>
              <div className="mt-1 text-2xl font-bold text-primary">{framesAnalyzed}</div>
            </CardContent>
          </Card>
          <Card className="border-border/70 bg-card/60">
            <CardContent className="p-4">
              <div className="text-xs font-medium text-muted-foreground">Frames Skipped</div>
              <div className="mt-1 text-2xl font-bold text-amber-400">{framesSkipped}</div>
            </CardContent>
          </Card>
          <Card className="border-border/70 bg-card/60">
            <CardContent className="p-4">
              <div className="text-xs font-medium text-muted-foreground">Deepfake Detections</div>
              <div className="mt-1 text-2xl font-bold text-rose-400">{deepfakeDetections}</div>
            </CardContent>
          </Card>
          <Card className="border-border/70 bg-card/60">
            <CardContent className="p-4">
              <div className="text-xs font-medium text-muted-foreground">Average Score</div>
              <div className="mt-1 text-2xl font-bold text-foreground">{avgConfidence}%</div>
            </CardContent>
          </Card>
          <Card className="border-border/70 bg-card/60">
            <CardContent className="p-4">
              <div className="text-xs font-medium text-muted-foreground">Average Latency</div>
              <div className="mt-1 text-2xl font-bold text-foreground">{avgLatency} ms</div>
            </CardContent>
          </Card>
        </div>

        {/* Detection Timeline */}
        <Card className="border-border/70 bg-card/80">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-bold flex items-center gap-2">
              <Clock className="h-4 w-4 text-primary" /> Detection Timeline (Recent Frames)
            </CardTitle>
          </CardHeader>
          <CardContent>
            {timeline.length === 0 ? (
              <div className="py-8 text-center text-xs text-muted-foreground">
                No frames analyzed yet. Start monitoring to populate live timeline.
              </div>
            ) : (
              <div className="space-y-2">
                {timeline.map((item, i) => (
                  <div
                    key={item.id || i}
                    className="flex items-center justify-between rounded-lg border border-border/50 bg-secondary/20 px-4 py-2.5 text-xs transition-colors hover:bg-secondary/40"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-muted-foreground">{item.timestamp}</span>
                      <Badge className={cn(
                        "font-semibold text-[11px]",
                        item.verdict === "REAL" && "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
                        item.verdict === "SUSPICIOUS" && "bg-amber-500/20 text-amber-400 border-amber-500/30",
                        item.verdict === "DEEPFAKE" && "bg-rose-500/20 text-rose-400 border-rose-500/30"
                      )}>
                        {item.verdict}
                      </Badge>
                    </div>

                    <div className="flex items-center gap-6">
                      <div>
                        <span className="text-muted-foreground mr-1">Score:</span>
                        <span className="font-semibold text-foreground">{item.detection_score}%</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground mr-1">Risk:</span>
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
                        <span className="text-muted-foreground mr-1">Latency:</span>
                        <span className="font-mono text-foreground">{item.latency_ms} ms</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Session Summary Modal / Card */}
        {sessionSummary && (
          <Card className="border-primary/50 bg-card p-6 shadow-xl">
            <CardHeader className="px-0 pt-0 pb-4 border-b border-border/60">
              <CardTitle className="text-lg font-bold text-foreground">Monitoring Session Summary</CardTitle>
              <CardDescription className="text-xs">Aggregated metrics for completed real-time session</CardDescription>
            </CardHeader>
            <CardContent className="px-0 pt-4 space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                <div className="rounded-lg bg-secondary/40 p-3">
                  <div className="text-muted-foreground">Frames Analyzed</div>
                  <div className="text-lg font-bold text-foreground mt-1">{sessionSummary.frames_analyzed}</div>
                </div>
                <div className="rounded-lg bg-secondary/40 p-3">
                  <div className="text-muted-foreground">Frames Skipped</div>
                  <div className="text-lg font-bold text-foreground mt-1">{sessionSummary.frames_skipped}</div>
                </div>
                <div className="rounded-lg bg-secondary/40 p-3">
                  <div className="text-muted-foreground">Deepfake Detections</div>
                  <div className="text-lg font-bold text-rose-400 mt-1">{sessionSummary.deepfake_detections}</div>
                </div>
                <div className="rounded-lg bg-secondary/40 p-3">
                  <div className="text-muted-foreground">Avg Latency</div>
                  <div className="text-lg font-bold text-foreground mt-1">{sessionSummary.average_latency_ms} ms</div>
                </div>
              </div>

              <div className="flex items-center justify-between rounded-lg border border-border/60 p-4">
                <div>
                  <div className="text-xs text-muted-foreground">Overall Session Verdict & Risk</div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xl font-extrabold text-foreground">{sessionSummary.final_verdict}</span>
                    <Badge variant="outline">{sessionSummary.final_risk} RISK</Badge>
                  </div>
                </div>
                {sessionSavedSuccess ? (
                  <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 gap-1 py-1.5 px-3">
                    <CheckCircle2 className="h-4 w-4" /> Session Saved
                  </Badge>
                ) : (
                  <Button size="sm" onClick={() => setSessionSavedSuccess(true)} className="gap-2">
                    <Save className="h-4 w-4" /> Save Session Summary
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </AppShell>
  );
}

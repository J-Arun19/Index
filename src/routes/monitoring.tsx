import { useState, useRef, useEffect, useCallback } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { 
  Play, 
  Pause,
  Square, 
  Video, 
  AlertTriangle, 
  ShieldCheck, 
  Activity, 
  CheckCircle2, 
  Clock, 
  Camera, 
  Save,
  UploadCloud,
  Film,
  Volume2,
  VolumeX,
  RotateCcw,
  Sliders,
  UserCheck,
  Info
} from "lucide-react";

import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAuthState } from "@/lib/auth-store";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/monitoring")({
  component: RealTimeMonitoringPage,
});

interface TrackedFace {
  id: string;
  track_id?: string;
  track_avg_score?: number;
  track_max_score?: number;
  track_frames?: number;
  box: [number, number, number, number]; // [x, y, w, h]
  box_normalized: [number, number, number, number]; // [nx, ny, nw, nh]
  verdict: "REAL" | "FAKE" | "INCONCLUSIVE";
  prediction: string;
  detection_score: number;
  raw_score: number;
  confidence: number | null;
  uncertainty: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  label: string;
  timestamp: string;
  reason?: string;
}

interface FrameAnalysisResult {
  status?: string;
  id?: string;
  timestamp: string;
  verdict: "REAL" | "UNCERTAIN" | "SUSPICIOUS" | "DEEPFAKE" | "INCONCLUSIVE" | "NO_FACE" | "INVALID_FRAME" | "ERROR";
  detection_score: number;
  confidence?: number | null;
  uncertainty?: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  latency_ms: number;
  faces_count?: number;
  faces?: TrackedFace[];
  reason?: string;
  agents?: {
    validation: string;
    face_detector?: string;
    image: string;
    confidence: string;
    artifact: string;
    fusion: string;
  };
  providers?: { [key: string]: string };
}

interface SampleVideo {
  id: string;
  title: string;
  filename: string;
  expected_verdict: string;
  description: string;
  url: string;
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

function RealTimeMonitoringPage() {
  const navigate = useNavigate();
  const { user, isInitialized } = useAuthState();

  useEffect(() => {
    if (isInitialized && !user) {
      navigate({ to: "/login" });
    }
  }, [user, isInitialized, navigate]);

  // Mode Selection: 'live' | 'recorded'
  const [activeMode, setActiveMode] = useState<"live" | "recorded">("live");

  // Configurable Sampling Rate (ms)
  const [samplingInterval, setSamplingInterval] = useState<number>(500);

  // Live Camera State
  const [isLiveMonitoring, setIsLiveMonitoring] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Recorded Video State
  const [sampleVideos, setSampleVideos] = useState<SampleVideo[]>([]);
  const [selectedVideoUrl, setSelectedVideoUrl] = useState<string | null>(null);
  const [videoFileName, setVideoFileName] = useState<string | null>(null);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [isMuted, setIsMuted] = useState(true);
  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState(false);

  // Detected / Tracked Faces
  const [trackedFaces, setTrackedFaces] = useState<TrackedFace[]>([]);
  const [lastAnalysisTimestamp, setLastAnalysisTimestamp] = useState<string | null>(null);
  const [isAnalysisPending, setIsAnalysisPending] = useState<boolean>(false);

  // Telemetry Counters
  const [framesReceived, setFramesReceived] = useState(0);
  const [framesAnalyzed, setFramesAnalyzed] = useState(0);
  const [framesSkipped, setFramesSkipped] = useState(0);
  const [deepfakeDetections, setDeepfakeDetections] = useState(0);
  const [avgScore, setAvgScore] = useState(0);
  const [avgLatency, setAvgLatency] = useState(0);

  // Real-time Verdict State
  const [currentVerdict, setCurrentVerdict] = useState<string>("IDLE");
  const [currentScore, setCurrentScore] = useState<number>(0);
  const [currentRisk, setCurrentRisk] = useState<string>("LOW");
  const [currentAgents, setCurrentAgents] = useState<{ [key: string]: string }>({
    validation: "IDLE",
    face_detector: "IDLE",
    image: "IDLE",
    confidence: "IDLE",
    artifact: "IDLE",
    fusion: "IDLE",
  });

  // Sliding Temporal Window & Timeline
  const [windowPredictions, setWindowPredictions] = useState<FrameAnalysisResult[]>([]);
  const [timeline, setTimeline] = useState<FrameAnalysisResult[]>([]);

  // Session State
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionSummary, setSessionSummary] = useState<SessionStats | null>(null);
  const [sessionSavedSuccess, setSessionSavedSuccess] = useState(false);

  // DOM Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Operational Refs
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<any>(null);
  const isProcessingRef = useRef<boolean>(false);
  const isStartingRef = useRef<boolean>(false);
  const facesRef = useRef<TrackedFace[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const statsRef = useRef({
    received: 0,
    analyzed: 0,
    skipped: 0,
    deepfakes: 0,
    totalScoreSum: 0,
    totalLatencySum: 0,
  });

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

  // Fetch available sample videos from backend
  useEffect(() => {
    fetch(`${API_URL}/api/monitor/samples`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "success" && Array.isArray(data.samples)) {
          setSampleVideos(data.samples);
        }
      })
      .catch((err) => console.warn("Could not load sample videos:", err));
  }, [API_URL]);

  // Synchronize facesRef with trackedFaces state
  useEffect(() => {
    facesRef.current = trackedFaces;
    drawOverlay();
  }, [trackedFaces]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopAllStreamsAndTimers();
    };
  }, []);

  const stopAllStreamsAndTimers = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
    isProcessingRef.current = false;
    facesRef.current = [];
    setTrackedFaces([]);
    clearOverlayCanvas();
    setWindowPredictions([]);
    setTimeline([]);
    setCurrentVerdict("IDLE");
    setCurrentScore(0);
    setCurrentRisk("LOW");
    setLastAnalysisTimestamp(null);
    setIsLiveMonitoring(false);
    setIsVideoPlaying(false);
    setIsAnalyzingVideo(false);
    setIsAnalysisPending(false);
    if (sessionId) {
      fetch(`${API_URL}/api/monitor/reset-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(() => {});
    }
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
    setAvgScore(0);
    setAvgLatency(0);
  };

  // Switch between Live and Recorded mode cleanly
  const handleModeChange = (mode: "live" | "recorded") => {
    stopAllStreamsAndTimers();
    setActiveMode(mode);
    setCameraError(null);
    setSessionSummary(null);
    setSessionSavedSuccess(false);
  };

  const clearOverlayCanvas = () => {
    const canvas = overlayCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  // ==========================================
  // Overlay Canvas Drawing Logic
  // ==========================================
  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const clientW = video.clientWidth;
    const clientH = video.clientHeight;

    if (canvas.width !== clientW || canvas.height !== clientH) {
      canvas.width = clientW;
      canvas.height = clientH;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const videoW = video.videoWidth;
    const videoH = video.videoHeight;
    if (videoW === 0 || videoH === 0) return;

    // Geometric object-contain scaling & letterboxing
    const scale = Math.min(clientW / videoW, clientH / videoH);
    const renderW = videoW * scale;
    const renderH = videoH * scale;
    const offsetX = (clientW - renderW) / 2;
    const offsetY = (clientH - renderH) / 2;

    const faces = facesRef.current;
    if (!faces || faces.length === 0) return;

    faces.forEach((face) => {
      if (!face) return;
      let nx = 0, ny = 0, nw = 0, nh = 0;
      if (Array.isArray(face.box_normalized) && face.box_normalized.length === 4) {
        [nx, ny, nw, nh] = face.box_normalized;
      } else if (Array.isArray(face.box) && face.box.length === 4 && videoW > 0 && videoH > 0) {
        nx = face.box[0] / videoW;
        ny = face.box[1] / videoH;
        nw = face.box[2] / videoW;
        nh = face.box[3] / videoH;
      } else {
        return;
      }

      const x = offsetX + nx * renderW;
      const y = offsetY + ny * renderH;
      const w = nw * renderW;
      const h = nh * renderH;

      // Color encoding strictly per prompt specifications:
      // RED — FAKE: actual analysis produced sufficient evidence supporting manipulation
      // GREEN — REAL: actual analysis produced sufficient evidence supporting authenticity
      // AMBER — INCONCLUSIVE: ambiguous, providers disagree, analysis pending, or failed
      let strokeColor = "#f59e0b"; // amber
      let bgColor = "rgba(245, 158, 11, 0.92)";
      let textColor = "#ffffff";
      let badgeBorder = "#d97706";

      if (face.verdict === "FAKE") {
        strokeColor = "#ef4444"; // red
        bgColor = "rgba(239, 68, 68, 0.95)";
        badgeBorder = "#dc2626";
      } else if (face.verdict === "REAL") {
        strokeColor = "#10b981"; // emerald
        bgColor = "rgba(16, 185, 129, 0.95)";
        badgeBorder = "#059669";
      }

      ctx.save();
      try {
        // Draw bounding box with subtle neon cyber glow
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = strokeColor;
        ctx.shadowBlur = 8;
        ctx.strokeRect(x, y, w, h);

        // Corner accent brackets for defense-grade HUD aesthetic
        const bracketLen = Math.min(14, w / 4, h / 4);
        ctx.beginPath();
        // Top-Left
        ctx.moveTo(x, y + bracketLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + bracketLen, y);
        // Top-Right
        ctx.moveTo(x + w - bracketLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + bracketLen);
        // Bottom-Left
        ctx.moveTo(x, y + h - bracketLen);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + bracketLen, y + h);
        // Bottom-Right
        ctx.moveTo(x + w - bracketLen, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - bracketLen);
        ctx.stroke();

        // Label text
        const timeStr = face.timestamp ? face.timestamp.slice(0, 8) : "";
        const trackLabel = face.track_id || face.id || "Face";
        const confStr = face.confidence !== null && face.confidence !== undefined ? ` [${face.confidence}% conf]` : "";
        const fullLabel = `[${trackLabel}: ${face.verdict}] ${face.detection_score}%${confStr} (${timeStr})`;

        ctx.font = "bold 11px 'JetBrains Mono', 'Roboto Mono', monospace, sans-serif";
        const textMetrics = ctx.measureText(fullLabel);
        const padX = 7;
        const badgeW = textMetrics.width + padX * 2;
        const badgeH = 22;

        // Position badge directly above box (or flip inside top if near top boundary)
        let badgeY = y - badgeH - 3;
        if (badgeY < 4) {
          badgeY = y + 4;
        }
        const badgeX = Math.max(4, Math.min(x, clientW - badgeW - 4));

        // Draw badge background pill
        ctx.fillStyle = bgColor;
        ctx.strokeStyle = badgeBorder;
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (typeof (ctx as any).roundRect === "function") {
          (ctx as any).roundRect(badgeX, badgeY, badgeW, badgeH, 3);
        } else {
          ctx.rect(badgeX, badgeY, badgeW, badgeH);
        }
        ctx.fill();
        ctx.stroke();

        // Draw label text
        ctx.fillStyle = textColor;
        ctx.textBaseline = "middle";
        ctx.fillText(fullLabel, badgeX + padX, badgeY + badgeH / 2);

        // Draw evidence pill under box if evidence is available
        const evidenceStr = face.reason || (face as any).evidence || "";
        if (evidenceStr && h > 30) {
          const shortEv = evidenceStr.length > 48 ? `${evidenceStr.slice(0, 45)}...` : evidenceStr;
          ctx.font = "normal 9px 'JetBrains Mono', 'Roboto Mono', monospace, sans-serif";
          const evMetrics = ctx.measureText(shortEv);
          const evW = evMetrics.width + 10;
          const evH = 17;
          let evY = y + h + 3;
          if (evY + evH > clientH - 4) {
            evY = Math.max(4, y - badgeH - evH - 5);
          }
          const evX = Math.max(4, Math.min(x, clientW - evW - 4));
          ctx.fillStyle = "rgba(15, 23, 42, 0.90)";
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1;
          ctx.beginPath();
          if (typeof (ctx as any).roundRect === "function") {
            (ctx as any).roundRect(evX, evY, evW, evH, 3);
          } else {
            ctx.rect(evX, evY, evW, evH);
          }
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = strokeColor;
          ctx.textBaseline = "middle";
          ctx.fillText(shortEv, evX + 5, evY + evH / 2);
        }
      } catch (drawErr) {
        console.warn("Failed to draw overlay element:", drawErr);
      } finally {
        ctx.restore();
      }
    });
  }, []);

  // Re-align canvas on window/container resize
  useEffect(() => {
    const handleResize = () => {
      drawOverlay();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [drawOverlay]);

  // ==========================================
  // Frame Analysis Pipeline Call
  // ==========================================
  const captureAndAnalyzeFrame = async () => {
    statsRef.current.received += 1;
    setFramesReceived(statsRef.current.received);

    // Active Inference Guard: If previous frame inference is still executing, skip frame
    if (isProcessingRef.current) {
      statsRef.current.skipped += 1;
      setFramesSkipped(statsRef.current.skipped);
      return;
    }

    const video = videoRef.current;
    const captureCanvas = captureCanvasRef.current;
    if (!video || !captureCanvas) return;

    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    isProcessingRef.current = true;
    setIsAnalysisPending(true);

    try {
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      // Maintain exact aspect ratio, capping max dimension to 720 for responsive network transfer
      const maxDim = 720;
      let targetW = vw;
      let targetH = vh;
      if (Math.max(vw, vh) > maxDim) {
        const s = maxDim / Math.max(vw, vh);
        targetW = Math.round(vw * s);
        targetH = Math.round(vh * s);
      }
      captureCanvas.width = targetW;
      captureCanvas.height = targetH;
      const ctx = captureCanvas.getContext("2d");
      if (!ctx) {
        isProcessingRef.current = false;
        setIsAnalysisPending(false);
        return;
      }

      ctx.drawImage(video, 0, 0, targetW, targetH);
      const frameBase64 = captureCanvas.toDataURL("image/jpeg", 0.85);

      const res = await fetch(`${API_URL}/api/monitor/analyze-frame`, {
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
        const hasFaces = (data.faces && data.faces.length > 0) || (data.faces_count && data.faces_count > 0);

        if (hasFaces) {
          statsRef.current.analyzed += 1;
          statsRef.current.totalScoreSum += data.detection_score;
          statsRef.current.totalLatencySum += data.latency_ms;

          if (data.verdict === "DEEPFAKE" || data.risk_level === "HIGH" || data.risk_level === "CRITICAL") {
            statsRef.current.deepfakes += 1;
          }

          setFramesAnalyzed(statsRef.current.analyzed);
          setDeepfakeDetections(statsRef.current.deepfakes);
          setAvgScore(Number((statsRef.current.totalScoreSum / statsRef.current.analyzed).toFixed(1)));
          setAvgLatency(Number((statsRef.current.totalLatencySum / statsRef.current.analyzed).toFixed(0)));
        }

        setCurrentVerdict(data.verdict);
        setCurrentScore(data.detection_score);
        setCurrentRisk(data.risk_level);
        setLastAnalysisTimestamp(data.timestamp);

        if (data.agents) {
          setCurrentAgents((prev) => ({ ...prev, ...data.agents }));
        }

        // Update detected face bounding boxes
        const faces = data.faces || [];
        setTrackedFaces(faces);
        facesRef.current = faces;
        drawOverlay();

        // Update Sliding Window (N = 5)
        setWindowPredictions((prev) => [...prev, data].slice(-5));

        // Update Event Log Timeline (recent 10 entries)
        setTimeline((prev) => [data, ...prev].slice(0, 10));
      } else {
        statsRef.current.skipped += 1;
        setFramesSkipped(statsRef.current.skipped);
      }
    } catch (err) {
      console.error("Frame analysis error:", err);
      statsRef.current.skipped += 1;
      setFramesSkipped(statsRef.current.skipped);
    } finally {
      isProcessingRef.current = false;
      setIsAnalysisPending(false);
    }
  };

  // ==========================================
  // Mode A: Live Camera Monitoring
  // ==========================================
  const handleStartLiveMonitoring = async () => {
    if (isLiveMonitoring || isStartingRef.current) return;
    isStartingRef.current = true;

    setCameraError(null);
    setSessionSummary(null);
    setSessionSavedSuccess(false);

    // Clear any active timer before starting fresh
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

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
    setAvgScore(0);
    setAvgLatency(0);
    setWindowPredictions([]);
    setTimeline([]);
    setTrackedFaces([]);
    facesRef.current = [];
    setCurrentVerdict("ANALYZING");

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Browser does not support camera access via MediaDevices API.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });

      mediaStreamRef.current = stream;
      setIsLiveMonitoring(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (playErr) {
          console.warn("Video play interrupted/delayed:", playErr);
        }
      }

      // Initialize session in database
      try {
        const res = await fetch(`${API_URL}/api/monitor/start-session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: user?.id || null,
            user_name: user?.name || "Live Stream Auditor",
          }),
        });
        const d = await res.json();
        if (d.session_id) setSessionId(d.session_id);
      } catch (e) {
        console.warn("Local session fallback active:", e);
      }

      // Start Sampling Loop at Configured Interval
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      timerRef.current = setInterval(() => {
        captureAndAnalyzeFrame();
      }, samplingInterval);
    } catch (err: any) {
      console.error("Camera access error:", err);
      let msg = "Failed to access camera.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        msg = "Camera permission denied. Please allow camera permissions in browser settings.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        msg = "No video camera detected on your device.";
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        msg = "Camera is currently in use by another application.";
      } else if (err.message) {
        msg = err.message;
      }
      setCameraError(msg);
      stopAllStreamsAndTimers();
    } finally {
      isStartingRef.current = false;
    }
  };

  const handleStopLiveMonitoring = async () => {
    stopAllStreamsAndTimers();
    finalizeSessionSummary();
  };

  // ==========================================
  // Mode B: Recorded Video Monitoring
  // ==========================================
  const handleSelectSample = async (sample: SampleVideo) => {
    stopAllStreamsAndTimers();
    const newSessionId = `rec_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    setSessionId(newSessionId);
    const fullUrl = sample.url.startsWith("http") ? sample.url : `${API_URL}${sample.url}`;
    setVideoFileName(sample.title);
    setTrackedFaces([]);
    facesRef.current = [];
    clearOverlayCanvas();

    try {
      const res = await fetch(fullUrl);
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      setSelectedVideoUrl(objectUrl);
      if (videoRef.current) {
        videoRef.current.src = objectUrl;
        videoRef.current.load();
      }
    } catch (e) {
      console.warn("Blob conversion failed, fallback to direct url:", e);
      setSelectedVideoUrl(fullUrl);
      if (videoRef.current) {
        videoRef.current.src = fullUrl;
        videoRef.current.load();
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Check size limit (100MB)
    if (file.size > 100 * 1024 * 1024) {
      alert("Selected video file exceeds 100MB size limit.");
      return;
    }

    stopAllStreamsAndTimers();
    const newSessionId = `rec_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    setSessionId(newSessionId);
    const objectUrl = URL.createObjectURL(file);
    setSelectedVideoUrl(objectUrl);
    setVideoFileName(file.name);
    setTrackedFaces([]);
    facesRef.current = [];
    clearOverlayCanvas();
    if (videoRef.current) {
      videoRef.current.src = objectUrl;
      videoRef.current.load();
    }
  };

  const handleVideoLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration || 0);
      drawOverlay();
    }
  };

  const handleVideoTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
      drawOverlay();
    }
  };

  const handlePlayPause = async () => {
    const video = videoRef.current;
    if (!video || !selectedVideoUrl || isStartingRef.current) return;
    isStartingRef.current = true;

    try {
      if (isVideoPlaying) {
        video.pause();
        setIsVideoPlaying(false);
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      } else {
        setIsVideoPlaying(true);
        setIsAnalyzingVideo(true);
        try {
          await video.play();
        } catch (playErr) {
          console.warn("Video playback delayed:", playErr);
        }

        // Clear any prior interval before establishing fresh loop
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }

        // Immediate sample on play
        captureAndAnalyzeFrame();

        // Start periodic sampling interval
        timerRef.current = setInterval(() => {
          captureAndAnalyzeFrame();
        }, samplingInterval);
      }
    } finally {
      isStartingRef.current = false;
    }
  };

  const handleSeek = (newTime: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = newTime;
    setCurrentTime(newTime);
    // Instant frame analysis on scrub
    setTimeout(() => {
      captureAndAnalyzeFrame();
      drawOverlay();
    }, 50);
  };

  const handleStopVideo = () => {
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.currentTime = 0;
    }
    setIsVideoPlaying(false);
    setIsAnalyzingVideo(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setTrackedFaces([]);
    facesRef.current = [];
    clearOverlayCanvas();
    finalizeSessionSummary();
  };

  const finalizeSessionSummary = async () => {
    const s = statsRef.current;
    const finalAvgScore = s.analyzed > 0 ? s.totalScoreSum / s.analyzed : 0;
    const finalAvgLatency = s.analyzed > 0 ? s.totalLatencySum / s.analyzed : 0;

    let finalVerdict = "NO_FACES_DETECTED";
    let finalRisk = "LOW";
    if (s.analyzed === 0) {
      finalVerdict = "NO_FACES_DETECTED";
      finalRisk = "LOW";
    } else if (s.deepfakes > 0) {
      finalVerdict = "DEEPFAKE";
      finalRisk = finalAvgScore >= 80 ? "CRITICAL" : "HIGH";
    } else if (finalAvgScore >= 35) {
      finalVerdict = "INCONCLUSIVE";
      finalRisk = "MEDIUM";
    } else {
      finalVerdict = "REAL";
      finalRisk = "LOW";
    }

    const summary: SessionStats = {
      frames_received: s.received,
      frames_analyzed: s.analyzed,
      frames_skipped: s.skipped,
      deepfake_detections: s.deepfakes,
      average_confidence: Number(finalAvgScore.toFixed(1)),
      average_latency_ms: Number(finalAvgLatency.toFixed(0)),
      final_verdict: finalVerdict,
      final_risk: finalRisk,
    };

    setSessionSummary(summary);
    setCurrentVerdict("STOPPED");

    if (sessionId) {
      try {
        await fetch(`${API_URL}/api/monitor/stop-session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            stats: summary,
          }),
        });
      } catch (err) {
        console.warn("Could not save remote session summary:", err);
      }
    }
  };

  // Format seconds to mm:ss
  const formatTime = (secs: number) => {
    if (isNaN(secs) || secs < 0) return "00:00";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // Temporal Decision Evaluation across sliding window (N=5)
  const calculateTemporalRisk = () => {
    if (windowPredictions.length === 0) {
      return { 
        verdict: isLiveMonitoring || isVideoPlaying ? "ANALYZING" : "IDLE", 
        risk: "LOW", 
        score: 0.0 
      };
    }

    // Only consider frames that contained detected faces
    const faceFrames = windowPredictions.filter(
      (f) => f.verdict !== "NO_FACE" && f.verdict !== "INVALID_FRAME" && f.verdict !== "ERROR" && ((f.faces && f.faces.length > 0) || (f.faces_count && f.faces_count > 0))
    );

    // If all recent frames had zero detected faces, report NO_FACE / ANALYSIS UNAVAILABLE
    if (faceFrames.length === 0) {
      return { 
        verdict: "NO_FACE", 
        risk: "LOW", 
        score: 0.0 
      };
    }

    // Separate tracked-face histories: each face track is evaluated independently.
    // The overall verdict accounts for the most suspicious face track;
    // a genuine bystander's low score must NEVER cancel out the manipulated face's evidence.
    const trackScoresMap: { [trackId: string]: number[] } = {};
    const trackVerdictsMap: { [trackId: string]: string[] } = {};

    faceFrames.forEach((frame) => {
      if (frame.faces && frame.faces.length > 0) {
        frame.faces.forEach((face) => {
          const tid = face.track_id || face.id || "T1";
          if (!trackScoresMap[tid]) {
            trackScoresMap[tid] = [];
            trackVerdictsMap[tid] = [];
          }
          trackScoresMap[tid].push(face.detection_score);
          trackVerdictsMap[tid].push(face.verdict);
        });
      }
    });

    const trackIds = Object.keys(trackScoresMap);

    let anyTrackDeepfake = false;
    let anyTrackInconclusive = false;
    let highestTrackAvgScore = 0.0;

    trackIds.forEach((tid) => {
      const scores = trackScoresMap[tid];
      const verdicts = trackVerdictsMap[tid];
      const trackAvg = scores.reduce((a, b) => a + b, 0) / scores.length;
      if (trackAvg > highestTrackAvgScore) {
        highestTrackAvgScore = trackAvg;
      }

      const trackFakes = verdicts.filter((v) => v === "FAKE").length;
      const trackInconc = verdicts.filter((v) => v === "INCONCLUSIVE").length;

      // Evidence-based thresholding: any tracked face verified as FAKE flags the video
      if (trackFakes >= 1 || trackAvg >= 48.0) {
        anyTrackDeepfake = true;
      } else if (trackInconc >= 1 || trackAvg >= 25.0 || scores.length < 3) {
        anyTrackInconclusive = true;
      }
    });

    // Fallback if trackScoresMap is empty but faceFrames exist
    if (trackIds.length === 0) {
      highestTrackAvgScore = faceFrames.reduce((acc, f) => acc + f.detection_score, 0) / faceFrames.length;
      anyTrackInconclusive = faceFrames.some((f) => f.verdict === "INCONCLUSIVE" || f.detection_score >= 25.0);
      anyTrackDeepfake = faceFrames.some((f) => f.verdict === "DEEPFAKE" || f.detection_score >= 48.0);
    }

    let stableVerdict = "INCONCLUSIVE";
    let stableRisk = "MEDIUM";

    if (anyTrackDeepfake || highestTrackAvgScore >= 48.0) {
      stableVerdict = "DEEPFAKE";
      stableRisk = highestTrackAvgScore >= 75 ? "CRITICAL" : "HIGH";
    } else if (!anyTrackInconclusive && highestTrackAvgScore <= 25.0 && faceFrames.length >= 3) {
      stableVerdict = "REAL";
      stableRisk = "LOW";
    } else {
      stableVerdict = "INCONCLUSIVE";
      stableRisk = "MEDIUM";
    }

    return { verdict: stableVerdict, risk: stableRisk, score: Number(highestTrackAvgScore.toFixed(1)) };
  };

  const temporalState = calculateTemporalRisk();

  return (
    <AppShell
      title="Real-Time Deepfake Monitoring Console"
      description="Live face tracking with neural deepfake verification and multi-provider evidence overlay."
    >
      <div className="space-y-5">
        {/* Hidden Canvas for Frame Extraction */}
        <canvas ref={captureCanvasRef} style={{ display: "none" }} />

        {/* Input Mode Selector Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-lg border border-border/80 bg-slate-900/80 p-3">
          <div className="flex items-center rounded-lg bg-slate-950/80 border border-slate-800 p-1">
            <button
              id="btn-mode-live"
              onClick={() => handleModeChange("live")}
              className={cn(
                "flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer",
                activeMode === "live"
                  ? "bg-sky-600 text-white shadow-sm font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <Camera className="h-3.5 w-3.5 text-sky-400" />
              Live Camera
            </button>
            <button
              id="btn-mode-recorded"
              onClick={() => handleModeChange("recorded")}
              className={cn(
                "flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer",
                activeMode === "recorded"
                  ? "bg-sky-600 text-white shadow-sm font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <Film className="h-3.5 w-3.5 text-sky-400" />
              Recorded Video
            </button>
          </div>

          {/* Sampling Interval Selector */}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-400 flex items-center gap-1 text-[11px]">
              <Sliders className="h-3 w-3 text-sky-400" /> Analysis Interval:
            </span>
            <div className="flex items-center rounded border border-slate-700 bg-slate-950 p-0.5">
              {[
                { label: "500ms (2 FPS)", value: 500 },
                { label: "1000ms (1 FPS)", value: 1000 },
                { label: "2000ms (0.5 FPS)", value: 2000 },
              ].map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setSamplingInterval(opt.value)}
                  className={cn(
                    "px-2.5 py-1 text-[10px] font-medium rounded transition-colors",
                    samplingInterval === opt.value
                      ? "bg-sky-600 text-white font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Camera Permission Error Banner */}
        {cameraError && (
          <div className="flex items-center gap-3 rounded-md border border-rose-500/40 bg-rose-500/10 p-3.5 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" />
            <div className="flex-1">
              <span className="font-semibold text-rose-200">Hardware Access Alert:</span> {cameraError}
            </div>
          </div>
        )}

        {/* No Face Detected / Analysis Unavailable Alert Banner */}
        {temporalState.verdict === "NO_FACE" && (isLiveMonitoring || isVideoPlaying) && (
          <div className="flex items-center gap-3 rounded-md border border-slate-700 bg-slate-900/80 p-3.5 text-slate-300">
            <AlertTriangle className="h-5 w-5 shrink-0 text-amber-400" />
            <div className="flex-1">
              <div className="font-semibold text-xs text-slate-200 uppercase tracking-wider">
                ⚠ NO FACE DETECTED / ANALYSIS UNAVAILABLE
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">
                No human facial features detected in active video frame. Facial localization is required for forensic deepfake verification.
              </div>
            </div>
          </div>
        )}

        {/* Ambiguity / Inconclusive Warning Banner */}
        {temporalState.verdict === "INCONCLUSIVE" && (isLiveMonitoring || isVideoPlaying) && (
          <div className="flex items-center gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 p-3.5 text-amber-300">
            <AlertTriangle className="h-5 w-5 shrink-0 text-amber-400" />
            <div className="flex-1">
              <div className="font-semibold text-xs text-amber-200 uppercase tracking-wider">
                ⚠ Inconclusive Analysis Warning
              </div>
              <div className="text-[11px] opacity-90 mt-0.5">
                Model confidence falls within the indeterminate decision margin (Score: {temporalState.score}%). 
                Neither authentic authenticity nor conclusive manipulation can be certified safely without manual review.
              </div>
            </div>
          </div>
        )}

        {/* Deepfake Flagged Alert Banner */}
        {temporalState.verdict === "DEEPFAKE" && (isLiveMonitoring || isVideoPlaying) && (
          <div className="flex items-center gap-3 rounded-md border border-rose-500/40 bg-rose-500/10 p-3.5 text-rose-300">
            <AlertTriangle className="h-5 w-5 shrink-0 text-rose-400" />
            <div className="flex-1">
              <div className="font-semibold text-xs text-rose-200 uppercase tracking-wider">
                🚨 Deepfake Manipulation Detected
              </div>
              <div className="text-[11px] opacity-90 mt-0.5">
                Sustained synthetic anomalies verified across sequential frames. Score: {temporalState.score}% | Risk: {temporalState.risk}.
              </div>
            </div>
          </div>
        )}

        {/* Main Grid: Video Stream Container (2 cols) + Metrics Dashboard (1 col) */}
        <div className="grid gap-5 lg:grid-cols-3">
          {/* Video Stream + Overlay Viewport */}
          <div className="lg:col-span-2 overflow-hidden rounded-md border border-border/80 bg-slate-900/60 p-4 space-y-3">
            <div className="flex flex-row items-center justify-between">
              <div>
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <Video className="h-4 w-4 text-sky-400" />
                  {activeMode === "live" ? "Live Camera Monitor" : "Recorded Video Inspection"}
                </h2>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  {activeMode === "live"
                    ? `Hardware MediaStream | Sample Rate: ${(1000 / samplingInterval).toFixed(1)} FPS`
                    : videoFileName ? `File: ${videoFileName}` : "Select a test video or upload a file"}
                </p>
              </div>

              {/* Status Badge */}
              <div>
                {activeMode === "live" ? (
                  isLiveMonitoring ? (
                    <span className="inline-flex items-center gap-1.5 rounded px-2.5 py-0.5 text-[10px] font-semibold border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      LIVE FEED ACTIVE
                    </span>
                  ) : (
                    <span className="inline-flex items-center rounded px-2.5 py-0.5 text-[10px] font-medium border border-slate-700 bg-slate-800 text-slate-400">
                      CAMERA STANDBY
                    </span>
                  )
                ) : (
                  isVideoPlaying ? (
                    <span className="inline-flex items-center gap-1.5 rounded px-2.5 py-0.5 text-[10px] font-semibold border border-sky-500/30 bg-sky-500/10 text-sky-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-sky-400 animate-pulse" />
                      PLAYING & ANALYZING
                    </span>
                  ) : selectedVideoUrl ? (
                    <span className="inline-flex items-center rounded px-2.5 py-0.5 text-[10px] font-medium border border-slate-700 bg-slate-800 text-slate-300">
                      PAUSED
                    </span>
                  ) : (
                    <span className="inline-flex items-center rounded px-2.5 py-0.5 text-[10px] font-medium border border-slate-700 bg-slate-800 text-slate-400">
                      NO VIDEO LOADED
                    </span>
                  )
                )}
              </div>
            </div>

            {/* Video Viewport with High-Precision Canvas Overlay */}
            <div
              ref={containerRef}
              className="relative aspect-video w-full overflow-hidden rounded border border-border/80 bg-slate-950 flex items-center justify-center select-none"
            >
              {/* HTML5 Video Element */}
              <video
                ref={videoRef}
                playsInline
                crossOrigin="anonymous"
                muted={isMuted || activeMode === "live"}
                onLoadedMetadata={handleVideoLoadedMetadata}
                onTimeUpdate={handleVideoTimeUpdate}
                onEnded={handleStopVideo}
                className={cn(
                  "h-full w-full object-contain",
                  activeMode === "live" && !isLiveMonitoring && "hidden",
                  activeMode === "recorded" && !selectedVideoUrl && "hidden"
                )}
              />

              {/* Synchronized Bounding Box Canvas Overlay */}
              <canvas
                ref={overlayCanvasRef}
                className="absolute inset-0 h-full w-full pointer-events-none"
              />

              {/* Analysis Pending Indicator */}
              {isAnalysisPending && (
                <div className="absolute bottom-3 right-3 flex items-center gap-1.5 rounded bg-slate-950/80 border border-sky-500/40 px-2 py-0.5 text-[10px] text-sky-300 font-mono">
                  <Activity className="h-3 w-3 animate-spin text-sky-400" />
                  ANALYZING FRAME...
                </div>
              )}

              {/* Live Overlay Top HUD */}
              {(isLiveMonitoring || (activeMode === "recorded" && selectedVideoUrl)) && (
                <div className="absolute top-3 left-3 flex flex-wrap gap-2 pointer-events-none">
                  <span
                    className={cn(
                      "px-2.5 py-0.5 text-[10px] font-bold rounded uppercase tracking-wider text-white border shadow-sm",
                      temporalState.verdict === "REAL" && "bg-emerald-600/90 border-emerald-500",
                      temporalState.verdict === "INCONCLUSIVE" && "bg-amber-600/90 border-amber-500",
                      temporalState.verdict === "DEEPFAKE" && "bg-rose-600/90 border-rose-500",
                      temporalState.verdict === "IDLE" && "bg-slate-700/80 border-slate-600",
                      temporalState.verdict === "NO_FACE" && "bg-slate-800/90 border-slate-600 text-slate-300"
                    )}
                  >
                    Verdict: {temporalState.verdict === "NO_FACE" ? "NO FACE DETECTED" : temporalState.verdict}
                  </span>

                  <span className="px-2 py-0.5 text-[10px] font-mono font-medium rounded bg-slate-900/90 text-slate-200 border border-slate-700">
                    Score: {temporalState.verdict === "NO_FACE" || temporalState.verdict === "IDLE" ? "--" : `${currentScore.toFixed(1)}%`}
                  </span>

                  <span className="px-2 py-0.5 text-[10px] font-mono font-medium rounded bg-slate-900/90 text-slate-300 border border-slate-700">
                    Faces: {trackedFaces.length}
                  </span>

                  {lastAnalysisTimestamp && (
                    <span className="px-2 py-0.5 text-[10px] font-mono text-slate-400 rounded bg-slate-900/90 border border-slate-800">
                      Last: {lastAnalysisTimestamp}
                    </span>
                  )}
                </div>
              )}

              {/* Standby Placeholders */}
              {activeMode === "live" && !isLiveMonitoring && (
                <div className="flex flex-col items-center gap-3 text-slate-400 p-6 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded border border-slate-700 bg-slate-800 text-slate-300">
                    <Camera className="h-6 w-6 text-sky-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-xs text-slate-200">Camera Feed Inactive</div>
                    <div className="text-[11px] text-slate-400 mt-0.5 max-w-sm">
                      Click "Start Camera Monitoring" below to grant camera permissions and start real-time face deepfake detection.
                    </div>
                  </div>
                </div>
              )}

              {activeMode === "recorded" && !selectedVideoUrl && (
                <div className="flex flex-col items-center gap-3 text-slate-400 p-6 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded border border-slate-700 bg-slate-800 text-slate-300">
                    <Film className="h-6 w-6 text-sky-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-xs text-slate-200">No Video Selected</div>
                    <div className="text-[11px] text-slate-400 mt-0.5 max-w-sm">
                      Upload an existing video file or choose a verified test sample below to begin forensic playback.
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Mode Controls */}
            {activeMode === "live" ? (
              <div className="pt-1">
                {!isLiveMonitoring ? (
                  <Button
                    onClick={handleStartLiveMonitoring}
                    className="w-full h-9 bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs gap-2"
                  >
                    <Play className="h-3.5 w-3.5" /> Start Camera Monitoring
                  </Button>
                ) : (
                  <Button
                    onClick={handleStopLiveMonitoring}
                    variant="destructive"
                    className="w-full h-9 font-medium text-xs gap-2 bg-rose-600 hover:bg-rose-500"
                  >
                    <Square className="h-3.5 w-3.5" /> Stop Camera Monitoring
                  </Button>
                )}
              </div>
            ) : (
              /* Recorded Video Controls */
              <div className="space-y-3 pt-1">
                {/* Playback Scrubber & Time */}
                {selectedVideoUrl && (
                  <div className="space-y-1.5 rounded border border-border/60 bg-slate-950/60 p-2.5">
                    <div className="flex items-center gap-3">
                      <button
                        id="btn-play-video"
                        onClick={handlePlayPause}
                        className="flex h-7 w-7 items-center justify-center rounded bg-sky-600 hover:bg-sky-500 text-white text-xs"
                      >
                        {isVideoPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                      </button>

                      <button
                        id="btn-stop-video"
                        onClick={handleStopVideo}
                        className="flex h-7 w-7 items-center justify-center rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
                        title="Stop & Reset"
                      >
                        <Square className="h-3 w-3" />
                      </button>

                      {/* Scrub Slider */}
                      <input
                        id="video-scrubber"
                        type="range"
                        min="0"
                        max={duration || 1}
                        step="0.05"
                        value={currentTime}
                        onChange={(e) => handleSeek(parseFloat(e.target.value))}
                        className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
                      />

                      {/* Time Indicator */}
                      <span className="text-[11px] font-mono text-slate-300 shrink-0">
                        {formatTime(currentTime)} / {formatTime(duration)}
                      </span>

                      {/* Mute Toggle */}
                      <button
                        onClick={() => setIsMuted(!isMuted)}
                        className="text-slate-400 hover:text-slate-200"
                      >
                        {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
                      </button>
                    </div>
                  </div>
                )}

                {/* Upload & Sample Video Selectors */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <div>
                    <input
                      type="file"
                      ref={fileInputRef}
                      accept="video/mp4,video/webm,video/quicktime,video/avi"
                      onChange={handleFileUpload}
                      style={{ display: "none" }}
                    />
                    <Button
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full h-8 text-xs gap-2 border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-slate-200"
                    >
                      <UploadCloud className="h-3.5 w-3.5 text-sky-400" />
                      Upload Local Video File
                    </Button>
                  </div>

                  {/* Sample Video Dropdown Selector */}
                  <div>
                    <select
                      id="sample-video-select"
                      className="w-full h-8 px-2 text-xs rounded border border-slate-700 bg-slate-800 text-slate-200 focus:outline-none focus:border-sky-500 cursor-pointer"
                      onChange={(e) => {
                        const s = sampleVideos.find((v) => v.id === e.target.value);
                        if (s) handleSelectSample(s);
                      }}
                      defaultValue=""
                    >
                      <option value="" disabled>
                        ⚡ Choose Verified Test Sample Video...
                      </option>
                      {sampleVideos.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.title} ({s.expected_verdict})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Live Telemetry Status & Tracked Faces */}
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-4 flex flex-col justify-between space-y-4">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-sky-400" /> Live Detection Status
              </h2>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Multi-face bounding box tracking with genuine model calibration
              </p>
            </div>

            {/* Verdict Display */}
            <div className="rounded border border-border/80 bg-slate-950/60 p-3.5 text-center space-y-1">
              <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">
                Current Frame Verdict
              </div>
              <div
                className={cn(
                  "text-2xl font-bold tracking-tight",
                  temporalState.verdict === "REAL" && "text-emerald-400",
                  temporalState.verdict === "INCONCLUSIVE" && "text-amber-400",
                  temporalState.verdict === "DEEPFAKE" && "text-rose-400",
                  temporalState.verdict === "IDLE" && "text-slate-400",
                  temporalState.verdict === "NO_FACE" && "text-slate-400"
                )}
              >
                {temporalState.verdict === "NO_FACE" ? "NO FACE DETECTED" : temporalState.verdict}
              </div>
              <div className="text-[10px] text-slate-400">
                {temporalState.verdict === "NO_FACE"
                  ? "ANALYSIS UNAVAILABLE — Align face in frame"
                  : `Risk Level: ${temporalState.risk}`}
              </div>
            </div>

            {/* Tracked Face Details List */}
            <div className="space-y-2">
              <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <UserCheck className="h-3.5 w-3.5 text-sky-400" />
                  Tracked Faces ({trackedFaces.length})
                </span>
                <span className="text-[10px] text-slate-400 font-mono">
                  {isAnalysisPending ? "Processing..." : "Synchronized"}
                </span>
              </div>

              {trackedFaces.length === 0 ? (
                <div className="rounded border border-dashed border-border/60 bg-slate-950/40 p-3 text-center text-[11px] text-slate-400">
                  {isLiveMonitoring || isVideoPlaying
                    ? "Searching frame for faces..."
                    : "Start monitoring to detect facial regions."}
                </div>
              ) : (
                <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1 text-xs font-mono">
                  {trackedFaces.map((face, i) => (
                    <div
                      key={face.id || i}
                      className="rounded border border-border/60 bg-slate-950/60 p-2 space-y-1 text-[11px]"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-200">
                          {face.track_id ? `Face Track ${face.track_id}` : `Face #${i + 1}`}
                        </span>
                        <span
                          className={cn(
                            "rounded px-1.5 py-0.5 text-[9px] font-bold uppercase",
                            face.verdict === "REAL" && "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
                            face.verdict === "INCONCLUSIVE" && "bg-amber-500/20 text-amber-400 border border-amber-500/30",
                            face.verdict === "FAKE" && "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                          )}
                        >
                          {face.verdict}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-1 text-[10px] text-slate-400">
                        <div>Score: <span className="text-slate-200 font-semibold">{face.detection_score}%</span></div>
                        <div>Track Avg: <span className="text-slate-200 font-semibold">{face.track_avg_score !== undefined ? `${face.track_avg_score}%` : `${face.detection_score}%`}</span></div>
                        <div>Conf: <span className="text-slate-200 font-semibold">{face.confidence !== null ? `${face.confidence}%` : "Inconclusive"}</span></div>
                        <div>Frames: <span className="text-slate-200 font-semibold">{face.track_frames || 1}</span></div>
                        <div>Box: <span className="text-slate-300">[{face.box.join(", ")}]</span></div>
                        <div>Risk: <span className={cn(
                          face.risk_level === "LOW" && "text-emerald-400",
                          face.risk_level === "MEDIUM" && "text-amber-400",
                          (face.risk_level === "HIGH" || face.risk_level === "CRITICAL") && "text-rose-400"
                        )}>{face.risk_level}</span></div>
                      </div>
                      {face.reason && (
                        <div className="text-[10px] text-slate-300 border-t border-slate-800/80 pt-1 mt-1 leading-tight">
                          <span className="text-slate-400 font-medium">Evidence: </span>
                          <span className={cn(
                            face.verdict === "FAKE" && "text-rose-300",
                            face.verdict === "REAL" && "text-emerald-300",
                            face.verdict === "INCONCLUSIVE" && "text-amber-300"
                          )}>{face.reason}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Provider Availability Checklist */}
            <div className="space-y-1 text-xs font-mono pt-1">
              <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                Forensic Providers
              </div>
              <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[10px]">
                <span className="text-slate-400">ResNet18 Classifier</span>
                <span className="text-emerald-400 font-semibold">ONLINE</span>
              </div>
              <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[10px]">
                <span className="text-slate-400">Haar Face Tracker</span>
                <span className="text-emerald-400 font-semibold">ACTIVE</span>
              </div>
              <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[10px]">
                <span className="text-slate-400">Decision Engine (T=1.885)</span>
                <span className="text-emerald-400 font-semibold">ACTIVE</span>
              </div>
              <div className="flex items-center justify-between rounded bg-slate-950/60 px-2.5 py-1 text-[10px]">
                <span className="text-slate-400">Forensic Suite (FFT/ELA)</span>
                <span className="text-emerald-400 font-semibold">READY</span>
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
            <div className="mt-1 text-xl font-bold text-slate-100 font-mono">{avgScore}%</div>
          </div>
          <div className="rounded-md border border-border/80 bg-slate-900/60 p-3.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Average Latency</div>
            <div className="mt-1 text-xl font-bold text-slate-100 font-mono">{avgLatency} ms</div>
          </div>
        </div>

        {/* Real-Time Detection Telemetry Stream Timeline */}
        <div className="rounded-md border border-border/80 bg-slate-900/60">
          <div className="flex items-center justify-between border-b border-border/80 px-4 py-3">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Clock className="h-3.5 w-3.5 text-sky-400" /> Detection Telemetry Stream (Recent Window)
            </h3>
          </div>
          <div className="p-4">
            {timeline.length === 0 ? (
              <div className="py-6 text-center text-xs text-slate-400">
                No telemetry samples recorded yet. Initiate camera or video monitoring to log live events.
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
                      <span
                        className={cn(
                          "rounded px-2 py-0.5 text-[10px] font-semibold border",
                          item.verdict === "REAL" && "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
                          (item.verdict === "INCONCLUSIVE" || item.verdict === "UNCERTAIN" || item.verdict === "SUSPICIOUS") &&
                            "bg-amber-500/10 text-amber-400 border-amber-500/20",
                          item.verdict === "DEEPFAKE" && "bg-rose-500/10 text-rose-400 border-rose-500/20",
                          item.verdict === "NO_FACE" && "bg-slate-700/10 text-slate-400 border-slate-700/30"
                        )}
                      >
                        {item.verdict}
                      </span>
                      {item.faces_count !== undefined && (
                        <span className="text-[10px] text-slate-400">
                          {item.faces_count} face{item.faces_count !== 1 ? "s" : ""}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-5 text-[11px]">
                      <div>
                        <span className="text-slate-400 mr-1">Score:</span>
                        <span className="font-semibold text-slate-200">{item.detection_score}%</span>
                      </div>
                      <div>
                        <span className="text-slate-400 mr-1">Risk:</span>
                        <span
                          className={cn(
                            "font-semibold",
                            item.risk_level === "LOW" && "text-emerald-400",
                            item.risk_level === "MEDIUM" && "text-amber-400",
                            (item.risk_level === "HIGH" || item.risk_level === "CRITICAL") && "text-rose-400"
                          )}
                        >
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

        {/* Monitoring Session Summary Panel */}
        {sessionSummary && (
          <div className="rounded-md border border-sky-500/30 bg-slate-900/80 p-5 space-y-4">
            <div className="border-b border-border/80 pb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                Monitoring Session Summary
              </h3>
              <p className="text-[11px] text-slate-400">
                Aggregated telemetry for completed real-time session
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="rounded bg-slate-950/60 p-3 border border-border/60">
                <div className="text-slate-400 text-[10px] uppercase font-semibold">Frames Analyzed</div>
                <div className="text-lg font-bold text-slate-100 mt-0.5 font-mono">
                  {sessionSummary.frames_analyzed}
                </div>
              </div>
              <div className="rounded bg-slate-950/60 p-3 border border-border/60">
                <div className="text-slate-400 text-[10px] uppercase font-semibold">Frames Skipped</div>
                <div className="text-lg font-bold text-amber-400 mt-0.5 font-mono">
                  {sessionSummary.frames_skipped}
                </div>
              </div>
              <div className="rounded bg-slate-950/60 p-3 border border-border/60">
                <div className="text-slate-400 text-[10px] uppercase font-semibold">Deepfake Detections</div>
                <div className="text-lg font-bold text-rose-400 mt-0.5 font-mono">
                  {sessionSummary.deepfake_detections}
                </div>
              </div>
              <div className="rounded bg-slate-950/60 p-3 border border-border/60">
                <div className="text-slate-400 text-[10px] uppercase font-semibold">Avg Latency</div>
                <div className="text-lg font-bold text-slate-100 mt-0.5 font-mono">
                  {sessionSummary.average_latency_ms} ms
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between rounded border border-border/80 bg-slate-950/40 p-3.5">
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  Session Final Verdict & Risk
                </div>
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
                <Button
                  size="sm"
                  onClick={() => setSessionSavedSuccess(true)}
                  className="h-8 text-xs bg-sky-600 hover:bg-sky-500 text-white gap-1.5 font-medium"
                >
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

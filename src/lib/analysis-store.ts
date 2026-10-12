import { useSyncExternalStore } from "react";

export type AnalysisVerdict = "REAL" | "DEEPFAKE" | "FAKE" | "LIKELY_REAL" | "UNCERTAIN" | "NOT_SUITABLE";

export interface AgentTrace {
  name: string;
  summary: string;
}

export interface AnalysisResult {
  id: string;
  fileName: string;
  fileType: "image" | "video";
  mediaUrl: string;
  verdict: AnalysisVerdict;
  prediction?: string;
  confidence: number; // 0-100
  confidenceLevel?: string;
  fakeProbability?: number;
  realProbability?: number;
  rawModelScore?: number;
  uncertainty?: number;
  explanation: string;
  reason?: string;
  userFacingStatement?: string;
  evidence?: string[];
  recommendation?: string;
  analysisStatus?: string;
  faceDetected?: boolean;
  mediaCategory?: string;
  agents: AgentTrace[];
  createdAt: number;
  timeline?: { agent: string; status: string; duration: number }[];
  reportUrl?: string;
  naturalLanguageAnswer?: {
    question: string;
    answer: string;
    prediction: string;
    grounded: boolean;
    abstention: boolean;
  };
}

export function formatBackendAnalysis(item: any, apiUrl?: string): AnalysisResult {
  const baseApiUrl = apiUrl || (typeof window !== "undefined" ? (import.meta.env.VITE_API_URL || "http://localhost:5000") : "http://localhost:5000");
  const pred = (item.prediction || "REAL") as AnalysisVerdict;
  const isNotSuitable = pred === "NOT_SUITABLE";
  const conf = isNotSuitable ? 0 : Math.round(item.confidence || 0);
  const isFake = pred === "DEEPFAKE" || pred === "FAKE";
  return {
    id: item.id,
    fileName: item.file_name,
    fileType: item.media_type as "image" | "video",
    mediaUrl: `${baseApiUrl}/api/analysis/uploads/${item.file_name}`,
    verdict: pred,
    prediction: pred,
    confidence: conf,
    confidenceLevel: isNotSuitable ? "Not Performed" : (conf >= 75 ? "High" : conf >= 50 ? "Medium" : "Low"),
    fakeProbability: isNotSuitable ? undefined : (isFake ? conf / 100 : (100 - conf) / 100),
    realProbability: isNotSuitable ? undefined : (isFake ? (100 - conf) / 100 : conf / 100),
    explanation: isNotSuitable
      ? "No detectable human face was found in this image. SatyaDrishti's current image deepfake detector is designed for facial-media authenticity analysis. Upload a clear image containing a visible human face."
      : `Consensus was reached with a ${pred.toLowerCase()} verdict.`,
    reason: isNotSuitable
      ? "No suitable face detected."
      : `Consensus was reached with a ${pred.toLowerCase()} verdict based on multi-agent forensic evaluation.`,
    userFacingStatement: isNotSuitable
      ? "No detectable human face was found in this image. SatyaDrishti's current image deepfake detector is designed for facial-media authenticity analysis. Upload a clear image containing a visible human face."
      : isFake
      ? "Deepfake detected / high likelihood of manipulation."
      : pred === "UNCERTAIN"
      ? "Insufficient evidence to determine authenticity reliably."
      : "No strong evidence of manipulation was detected.",
    evidence: isNotSuitable
      ? ["Eligibility: Unsupported media (No detectable human face found in image)."]
      : [
        `File analyzed: ${item.file_name}`,
        `Consensus verdict: ${pred} (${conf}% confidence)`,
      ],
    recommendation: isNotSuitable
      ? "Upload a clear image containing a visible human face."
      : "Forensic safety audit record permanently archived in verified database.",
    analysisStatus: isNotSuitable ? "Not performed" : "Completed",
    faceDetected: !isNotSuitable,
    mediaCategory: isNotSuitable ? "Document / Non-face image" : "Facial Media",
    agents: [
      { name: "Validation Agent", summary: isNotSuitable ? "Media check: Unsupported (No face)" : "Completed" },
      { name: `${item.media_type === "video" ? "Video" : "Image"} Specialist Agent`, summary: isNotSuitable ? "Skipped (No face detected)" : "Completed" },
      { name: "Confidence Agent", summary: isNotSuitable ? "Not performed" : "Completed" },
      { name: "Fusion Agent", summary: isNotSuitable ? "Verdict: NOT SUITABLE" : "Completed" },
    ],
    createdAt: new Date(item.timestamp).getTime(),
    reportUrl: `${baseApiUrl}/api/analysis/report/${item.id}.pdf`,
  };
}

type State = {
  currentUserId: string | number | null;
  latest: AnalysisResult | null;
  selectedReport: AnalysisResult | null;
  history: AnalysisResult[];
};

let state: State = { currentUserId: null, latest: null, selectedReport: null, history: [] };
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

export const analysisStore = {
  get: () => state,
  subscribe(l: () => void) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
  setUser(userId: string | number | null) {
    if (state.currentUserId !== userId) {
      state = {
        currentUserId: userId,
        latest: null,
        selectedReport: null,
        history: [],
      };
      emit();
    }
  },
  save(result: AnalysisResult, userId?: string | number | null) {
    state = {
      currentUserId: userId !== undefined ? userId : state.currentUserId,
      latest: result,
      selectedReport: result,
      history: [result, ...state.history.filter((h) => h.id !== result.id)].slice(0, 50),
    };
    emit();
  },
  selectReport(result: AnalysisResult | null) {
    state = {
      ...state,
      selectedReport: result,
    };
    emit();
  },
  setHistory(history: AnalysisResult[], userId?: string | number | null) {
    const isDifferentUser = userId !== undefined && state.currentUserId !== userId;

    let effectiveLatest: AnalysisResult | null = null;
    if (history.length > 0) {
      if (!isDifferentUser && state.latest && history.some((h) => h.id === state.latest?.id)) {
        effectiveLatest = state.latest;
      } else {
        effectiveLatest = history[0];
      }
    } else {
      if (!isDifferentUser && state.latest && state.currentUserId === (userId ?? state.currentUserId)) {
        effectiveLatest = state.latest;
      } else {
        effectiveLatest = null;
      }
    }

    state = {
      currentUserId: userId !== undefined ? userId : state.currentUserId,
      latest: effectiveLatest,
      selectedReport: isDifferentUser ? null : state.selectedReport,
      history: history.slice(0, 50),
    };
    emit();
  },
  clear() {
    state = { currentUserId: null, latest: null, selectedReport: null, history: [] };
    emit();
  },
};

export function useAnalysisState() {
  return useSyncExternalStore(
    analysisStore.subscribe,
    analysisStore.get,
    analysisStore.get,
  );
}
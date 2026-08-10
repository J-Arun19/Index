import { useSyncExternalStore } from "react";

export type AnalysisVerdict = "REAL" | "DEEPFAKE";

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
  confidence: number; // 0-100
  explanation: string;
  agents: AgentTrace[];
  createdAt: number;
  confidenceLevel?: string;
  evidence?: string[];
  recommendation?: string;
  timeline?: { agent: string; status: string; duration: number }[];
  reportUrl?: string;
}

type State = {
  latest: AnalysisResult | null;
  history: AnalysisResult[];
};

let state: State = { latest: null, history: [] };
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
  save(result: AnalysisResult) {
    state = {
      latest: result,
      history: [result, ...state.history.filter(h => h.id !== result.id)].slice(0, 20),
    };
    emit();
  },
  setHistory(history: AnalysisResult[]) {
    state = {
      latest: state.latest || history[0] || null,
      history: history.slice(0, 20),
    };
    emit();
  },
  clear() {
    state = { latest: null, history: [] };
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
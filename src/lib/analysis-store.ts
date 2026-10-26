import { useSyncExternalStore } from "react";

export type AnalysisVerdict = "REAL" | "FAKE" | "UNCERTAIN" | "NOT_SUITABLE";

export interface AgentTrace {
  name: string;
  summary: string;
}

export interface ForensicEvidenceCard {
  id: string;
  title: string;
  region_name: string;
  crop_data_url?: string;
  crop_box?: [number, number, number, number];
  measurement_label: string;
  observed_finding: string;
  plain_explanation: string;
  verdict_support: "SUPPORTS_FAKE" | "SUPPORTS_REAL" | string;
  evidence_strength: "STRONG" | "MODERATE" | "CONSISTENT_NATURAL" | string;
  limitations: string;
}

export interface VideoEvidenceSnapshot {
  id: string;
  frame_index: number;
  timestamp: string;
  snapshot_data_url?: string;
  finding_title: string;
  region_name?: string;
  observed_finding: string;
  plain_explanation: string;
  evidence_source: string;
  measurement_label: string;
  evidence_strength: "STRONG" | "MODERATE" | "CONSISTENT_NATURAL" | string;
  verdict_support: "SUPPORTS_FAKE" | "SUPPORTS_REAL" | string;
  limitations: string;
}

export interface TrackedFaceResult {
  track_id: string;
  total_frames: number;
  verdict: "REAL" | "FAKE" | "INCONCLUSIVE" | string;
  fake_probability: number;
  mean_spatial_score: number;
  max_spatial_score: number;
  boundary_flicker: number;
  mouth_texture: number;
  eye_sharpness: number;
  anomalies_detected: number;
  sample_box?: [number, number, number, number];
}

export interface AnalysisResult {
  id: string;
  fileName: string;
  saved_filename?: string;
  token?: string;
  fileType: "image" | "video";
  mediaUrl: string;
  verdict: AnalysisVerdict;
  prediction?: string;
  canonicalVerdict?: AnalysisVerdict;
  primaryVerdict?: AnalysisVerdict | string;
  provisionalSuggestion?: "FAKE" | "REAL" | null;
  originalVerdict?: string;
  isBinaryFallback?: boolean;
  isEstimate?: boolean;
  reliabilityWarning?: string | null;
  abstentionCode?: string | null;
  abstention_code?: string | null;
  confidence: number | null; // 0-100 or null if uncertified
  confidenceLevel?: string;
  fakeProbability?: number;
  realProbability?: number;
  rawModelScore?: number;
  uncertainty?: number;
  explanation: string;
  reason?: string;
  userFacingStatement?: string;
  evidence?: string[];
  influencingEvidence?: string[];
  recommendation?: string;
  analysisStatus?: string;
  faceDetected?: boolean;
  faceBbox?: [number, number, number, number] | null;
  evidenceCards?: ForensicEvidenceCard[];
  evidenceSnapshots?: VideoEvidenceSnapshot[];
  trackedFaces?: TrackedFaceResult[];
  audioVisualSync?: {
    analyzed: boolean;
    reason?: string;
  };
  qualityGate?: {
    blur_score?: number;
    resolution?: [number, number] | string | { width?: number; height?: number; w?: number; h?: number };
    is_suitable?: boolean;
    issues?: string[];
  };
  features?: Record<string, any>;
  mediaCategory?: string;
  models?: Array<{
    name: string;
    task: string;
    predicted_class: string;
    raw_score?: number | null;
    calibrated_prob?: number | null;
    calibration_status?: string;
    status: string;
    limitations?: string;
  }>;
  measurements?: Record<string, any>;
  supportingEvidence?: string[];
  contradictingEvidence?: string[];
  unavailableChecks?: string[];
  decision?: {
    method?: string;
    score?: number | null;
    score_type?: string;
    confidence?: number | null;
    confidence_type?: string;
    criteria_met?: boolean;
    reason?: string;
  };
  limitations?: string[];
  isProvisional?: boolean;
  agents: AgentTrace[];
  createdAt: number;
  timeline?: { agent: string; status: string; duration: number }[];
  reportUrl?: string;
  visualAnalysis?: any;
  evidenceBreakdown?: any;
  approxCostUsd?: number;
  providersRan?: string[];
  naturalLanguageAnswer?: {
    question: string;
    answer: string;
    prediction: string;
    grounded: boolean;
    abstention: boolean;
  };
}

export interface VerdictTheme {
  verdict: AnalysisVerdict;
  label: string;
  badgeLabel: string;
  textColor: string;
  bgColor: string;
  borderColor: string;
  borderClass: string;
  bgClass: string;
  textClass: string;
  badgeClass: string;
  dotClass: string;
  iconName: "AlertTriangle" | "ShieldCheck" | "HelpCircle" | "FileX";
  description: string;
}

export function getCanonicalVerdict(item: any): {
  canonicalVerdict: AnalysisVerdict;
  provisionalSuggestion: "FAKE" | "REAL" | null;
  isBinaryFallback: boolean;
  reliabilityWarning: string | null;
  abstentionCode?: string | null;
} {
  const rawCanonical = String(item.canonical_verdict || item.canonicalVerdict || "").trim().toUpperCase();
  const rawVerdict = String(item.verdict || "").trim().toUpperCase();
  const rawPred = String(item.primary_verdict || item.primaryVerdict || item.prediction || "").trim().toUpperCase();
  const rawOrig = String(item.original_verdict || item.originalVerdict || "").trim().toUpperCase();
  const abstentionCode = item.abstention_code || item.abstentionCode || null;

  if (rawCanonical === "NOT_SUITABLE" || rawVerdict === "NOT_SUITABLE" || rawPred === "NOT_SUITABLE" || rawOrig === "NOT_SUITABLE" || rawPred === "UNSUPPORTED") {
    return {
      canonicalVerdict: "NOT_SUITABLE",
      provisionalSuggestion: null,
      isBinaryFallback: false,
      reliabilityWarning: null,
      abstentionCode: abstentionCode || "OUT_OF_DOMAIN",
    };
  }

  // Canonical verdict REAL or FAKE
  const effectiveVerdict = rawCanonical || rawVerdict || rawPred;
  if (effectiveVerdict === "FAKE" || effectiveVerdict === "DEEPFAKE") {
    return {
      canonicalVerdict: "FAKE",
      provisionalSuggestion: null,
      isBinaryFallback: false,
      reliabilityWarning: null,
    };
  }

  if (effectiveVerdict === "REAL" || effectiveVerdict === "LIKELY_REAL" || effectiveVerdict === "NO MANIPULATION DETECTED") {
    return {
      canonicalVerdict: "REAL",
      provisionalSuggestion: null,
      isBinaryFallback: false,
      reliabilityWarning: null,
    };
  }

  // Truly UNCERTAIN
  const isBinaryFallback = Boolean(item.is_binary_fallback ?? item.isBinaryFallback ?? false);
  const explicitSuggestion = (item.provisional_suggestion || item.provisionalSuggestion) as "FAKE" | "REAL" | null;
  const reliabilityWarning = item.reliability_warning || item.reliabilityWarning || null;

  return {
    canonicalVerdict: "UNCERTAIN",
    provisionalSuggestion: isBinaryFallback ? explicitSuggestion : null,
    isBinaryFallback: isBinaryFallback,
    reliabilityWarning: reliabilityWarning || "Model indicators contain conflicting or ambiguous evidence. Secondary manual verification is advised.",
    abstentionCode: abstentionCode || "INSUFFICIENT_EVIDENCE",
  };
}

export function getVerdictTheme(verdict: AnalysisVerdict, isProvisional = false): VerdictTheme {
  switch (verdict) {
    case "FAKE":
      return {
        verdict: "FAKE",
        label: "FAKE",
        badgeLabel: "FAKE",
        textColor: "#f43f5e",
        bgColor: "#fef2f2",
        borderColor: "#f43f5e",
        borderClass: "border-rose-500/30",
        bgClass: "bg-rose-500/10",
        textClass: "text-rose-400",
        badgeClass: "bg-rose-500/15 text-rose-400 border-rose-500/30",
        dotClass: "bg-rose-400",
        iconName: "AlertTriangle",
        description: "Multi-signal consensus confirms deepfake manipulation patterns in media features.",
      };
    case "REAL":
      return {
        verdict: "REAL",
        label: "REAL",
        badgeLabel: "REAL",
        textColor: "#10b981",
        bgColor: "#ecfdf5",
        borderColor: "#10b981",
        borderClass: "border-emerald-500/30",
        bgClass: "bg-emerald-500/10",
        textClass: "text-emerald-400",
        badgeClass: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
        dotClass: "bg-emerald-400",
        iconName: "ShieldCheck",
        description: "Authentic natural media features observed. Notice: Absence of detected manipulation does not guarantee absolute authenticity.",
      };
    case "NOT_SUITABLE":
      return {
        verdict: "NOT_SUITABLE",
        label: "NOT SUITABLE",
        badgeLabel: "NOT SUITABLE",
        textColor: "#94a3b8",
        bgColor: "#f8fafc",
        borderColor: "#64748b",
        borderClass: "border-slate-700",
        bgClass: "bg-slate-900/90",
        textClass: "text-slate-300",
        badgeClass: "bg-slate-800 text-slate-300 border-slate-700",
        dotClass: "bg-slate-400",
        iconName: "FileX",
        description: "No detectable human face was found in this media. SatyaDrishti is designed for facial-media authenticity analysis.",
      };
    case "UNCERTAIN":
    default:
      return {
        verdict: "UNCERTAIN",
        label: "UNCERTAIN",
        badgeLabel: isProvisional ? "UNCERTAIN • Provisional" : "UNCERTAIN",
        textColor: "#f59e0b",
        bgColor: "#fffbeb",
        borderColor: "#f59e0b",
        borderClass: "border-amber-500/30",
        bgClass: "bg-amber-500/10",
        textClass: "text-amber-400",
        badgeClass: "bg-amber-500/15 text-amber-400 border-amber-500/30",
        dotClass: "bg-amber-400",
        iconName: "HelpCircle",
        description: "Indicators lack consensus or are ambiguous. Authenticity has NOT been established.",
      };
  }
}

export function formatBackendAnalysis(item: any, apiUrl?: string): AnalysisResult {
  const baseApiUrl = apiUrl || (typeof window !== "undefined" ? (import.meta.env.VITE_API_URL || "http://localhost:5000") : "http://localhost:5000");
  const { canonicalVerdict, provisionalSuggestion, isBinaryFallback, reliabilityWarning, abstentionCode } = getCanonicalVerdict(item);
  const isNotSuitable = canonicalVerdict === "NOT_SUITABLE";
  const isUncertain = canonicalVerdict === "UNCERTAIN";
  const isFake = canonicalVerdict === "FAKE";
  const isReal = canonicalVerdict === "REAL";

  const isUncertified = isBinaryFallback || isUncertain || isNotSuitable;
  const rawConf = typeof item.confidence === "number" ? item.confidence : (typeof item.confidence_score === "number" ? item.confidence_score : null);
  const conf = isUncertified ? null : (rawConf !== null ? Math.round(rawConf) : null);

  let confLevel = "Uncertified";
  if (isNotSuitable) {
    confLevel = "Not Performed";
  } else if (isBinaryFallback) {
    confLevel = "Provisional";
  } else if (isUncertain) {
    confLevel = "Uncertified";
  } else if (conf !== null) {
    confLevel = conf >= 80 ? "High" : conf >= 60 ? "Medium" : "Low";
  }

  const effectiveId = item.id || item.analysis_id || item.file_id || "";
  const rawFileName = item.file_name || item.saved_filename || item.fileName || "";
  let cleanDisplayName = rawFileName;
  if (rawFileName.includes("_")) {
    const prefix = rawFileName.split("_")[0];
    if (prefix.length === 32 || prefix.length === 36 || prefix.includes("-")) {
      cleanDisplayName = rawFileName.slice(prefix.length + 1);
    }
  }

  // Determine upload file name with guaranteed {file_id}_ prefix
  let uploadFileName = item.saved_filename || item.file_name || "";
  if (!uploadFileName || !uploadFileName.includes("_")) {
    if (effectiveId && cleanDisplayName) {
      uploadFileName = `${effectiveId}_${cleanDisplayName}`;
    } else {
      uploadFileName = uploadFileName || effectiveId;
    }
  }

  // Preserve server-provided mediaUrl if available, ensuring absolute URL with baseApiUrl and valid file_id
  let resolvedMediaUrl = "";
  const rawMedia = item.media_url || item.mediaUrl;
  if (rawMedia && typeof rawMedia === "string") {
    let fullUrl = rawMedia.startsWith("http")
      ? rawMedia
      : `${baseApiUrl}${rawMedia.startsWith("/") ? "" : "/"}${rawMedia}`;

    // Guarantee that the unique file ID is present in the path or query parameters
    if (effectiveId && !fullUrl.includes(effectiveId)) {
      const sep = fullUrl.includes("?") ? "&" : "?";
      fullUrl = `${fullUrl}${sep}file_id=${encodeURIComponent(effectiveId)}`;
    }
    resolvedMediaUrl = fullUrl;
  } else {
    resolvedMediaUrl = `${baseApiUrl}/api/analysis/uploads/${uploadFileName}`;
  }

  let token = item.token;
  if (!token && rawMedia && typeof rawMedia === "string" && rawMedia.includes("token=")) {
    try {
      const parsedUrl = new URL(rawMedia, "http://localhost");
      token = parsedUrl.searchParams.get("token") || undefined;
    } catch {}
  }
  if (token && !resolvedMediaUrl.includes("token=")) {
    const sep = resolvedMediaUrl.includes("?") ? "&" : "?";
    resolvedMediaUrl = `${resolvedMediaUrl}${sep}token=${encodeURIComponent(token)}`;
  }

  return {
    id: item.id,
    fileName: item.fileName || cleanDisplayName,
    saved_filename: uploadFileName,
    token: token || undefined,
    fileType: (item.media_type || item.fileType) as "image" | "video",
    mediaUrl: resolvedMediaUrl,
    verdict: canonicalVerdict,
    prediction: canonicalVerdict,
    canonicalVerdict: canonicalVerdict,
    primaryVerdict: canonicalVerdict,
    provisionalSuggestion: provisionalSuggestion,
    originalVerdict: item.original_verdict || item.originalVerdict || (isUncertain ? "UNCERTAIN" : canonicalVerdict),
    isBinaryFallback: isBinaryFallback,
    isEstimate: isBinaryFallback || isUncertain,
    reliabilityWarning: reliabilityWarning,
    abstentionCode: abstentionCode || item.abstention_code || item.abstentionCode || null,
    abstention_code: abstentionCode || item.abstention_code || item.abstentionCode || null,
    confidence: conf,
    confidenceLevel: item.confidence_level || item.confidenceLevel || confLevel,
    fakeProbability: isNotSuitable ? undefined : (item.fake_probability ?? item.fakeProbability),
    realProbability: isNotSuitable ? undefined : (item.real_probability ?? item.realProbability),
    rawModelScore: item.raw_model_score ?? item.rawModelScore,
    uncertainty: item.uncertainty,
    explanation: item.explanation || (isNotSuitable
      ? "No detectable human face was found in this media. SatyaDrishti's current deepfake detector is designed for facial-media authenticity analysis. Upload a clear media file containing a visible human face."
      : isFake
      ? "Multi-signal consensus confirms deepfake manipulation patterns in media features."
      : isReal
      ? "Authentic natural media features observed. Note: Absence of detected manipulation does not guarantee absolute authenticity."
      : (isBinaryFallback && provisionalSuggestion
      ? `Indicators lack consensus or contain conflicting specialist predictions. Media leans ${provisionalSuggestion} on marginal probability, but authenticity has NOT been established.`
      : "Insufficient evidence to reliably classify this media. Authenticity has NOT been established.")),
    reason: item.reason || (isNotSuitable
      ? "No suitable face detected."
      : isFake
      ? "Strong evidence of synthetic manipulation identified."
      : isReal
      ? "Visual and frequency features align with authentic media."
      : "Insufficient evidence to determine authenticity reliably."),
    userFacingStatement: item.user_facing_statement || item.userFacingStatement || (isNotSuitable
      ? "No detectable human face was found in this media. SatyaDrishti's current deepfake detector is designed for facial-media authenticity analysis. Upload a clear media file containing a visible human face."
      : isFake
      ? "Deepfake detected / high likelihood of manipulation based on forensic anomaly patterns."
      : isReal
      ? "Authentic natural media features observed. Note: Absence of detected manipulation does not guarantee absolute authenticity."
      : "Insufficient evidence to reliably classify this media."),
    evidence: (item.evidence && item.evidence.length > 0) ? item.evidence : (isNotSuitable
      ? ["Eligibility: Unsupported media (No detectable human face found in image)."]
      : [
        `File analyzed: ${item.file_name || cleanDisplayName}`,
        `Consensus verdict: ${canonicalVerdict} (${confLevel} status)`,
      ]),
    recommendation: item.recommendation || (isNotSuitable
      ? "Upload a clear image containing a visible human face."
      : isUncertain
      ? "Insufficient evidence or conflicting detector indicators. Independent manual verification recommended."
      : "Forensic safety audit record permanently archived in verified database."),
    analysisStatus: item.analysis_status || item.analysisStatus || (isNotSuitable ? "Not performed" : "Completed"),
    faceDetected: item.face_detected !== undefined ? item.face_detected : (item.faceDetected !== undefined ? item.faceDetected : !isNotSuitable),
    faceBbox: item.face_bbox || item.faceBbox || item.fusion_result?.face_bbox || null,
    evidenceCards: item.evidence_cards || item.evidenceCards || item.fusion_result?.evidence_cards || [],
    evidenceSnapshots: item.evidence_snapshots || item.evidenceSnapshots || item.fusion_result?.evidence_snapshots || [],
    trackedFaces: item.tracked_faces || item.trackedFaces || item.fusion_result?.tracked_faces || [],
    audioVisualSync: item.audio_visual_sync || item.audioVisualSync || item.fusion_result?.audio_visual_sync || undefined,
    qualityGate: item.quality_gate || item.qualityGate || item.fusion_result?.quality_gate || undefined,
    features: item.features || item.fusion_result?.features || undefined,
    mediaCategory: item.media_category || item.mediaCategory || (isNotSuitable ? "Document / Non-face image" : "Facial Media"),
    models: item.models || item.fusion_result?.models || item.image_agent_result?.models || item.video_agent_result?.models || [],
    measurements: item.measurements || item.fusion_result?.measurements || item.image_agent_result?.measurements || item.video_agent_result?.measurements || {},
    supportingEvidence: item.supporting_evidence || item.supportingEvidence || item.fusion_result?.supporting_evidence || [],
    contradictingEvidence: item.contradicting_evidence || item.contradictingEvidence || item.fusion_result?.contradicting_evidence || [],
    unavailableChecks: item.unavailable_checks || item.unavailableChecks || item.fusion_result?.unavailable_checks || [],
    decision: item.decision || item.fusion_result?.decision || item.image_agent_result?.decision || item.video_agent_result?.decision,
    limitations: item.limitations || item.fusion_result?.limitations || item.image_agent_result?.limitations || [],
    isProvisional: item.is_provisional ?? item.isProvisional ?? isBinaryFallback,
    agents: item.agents || [
      { name: "Validation Agent", summary: isNotSuitable ? "Media check: Unsupported (No face)" : "Completed" },
      { name: `${item.media_type === "video" ? "Video" : "Image"} Specialist Agent`, summary: isNotSuitable ? "Skipped (No face detected)" : "Completed" },
      { name: "Confidence Agent", summary: isNotSuitable ? "Not performed" : "Completed" },
      { name: "Fusion Agent", summary: isNotSuitable ? "Verdict: NOT SUITABLE" : `Verdict: ${canonicalVerdict}` },
    ],
    createdAt: new Date(item.timestamp || Date.now()).getTime(),
    reportUrl: `${baseApiUrl}/api/analysis/report/${item.id}.pdf`,
    visualAnalysis: item.visual_analysis || item.visualAnalysis,
    evidenceBreakdown: item.evidence_breakdown || item.evidenceBreakdown,
    influencingEvidence: item.influencing_evidence || item.influencingEvidence || [],
    approxCostUsd: item.approx_cost_usd ?? item.approxCostUsd ?? 0.0,
    providersRan: item.providers_ran || item.providersRan || [],
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
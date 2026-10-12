import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Eye,
  Fingerprint,
  LockKeyhole,
  Network,
  Radio,
  ScanFace,
  ShieldCheck,
  Sparkles,
  Waves,
} from "lucide-react";

type BiometricScannerHUDProps = {
  title?: string;
  subtitle?: string;
};

type Telemetry = {
  frame: number;
  cycle: number;
  signal: number;
};

const LANDMARKS = [
  { x: 155, y: 168, label: "L_EYE" },
  { x: 245, y: 168, label: "R_EYE" },
  { x: 200, y: 212, label: "NASION" },
  { x: 168, y: 257, label: "MOUTH_L" },
  { x: 232, y: 257, label: "MOUTH_R" },
  { x: 200, y: 286, label: "MENTON" },
  { x: 130, y: 208, label: "ZYG_L" },
  { x: 270, y: 208, label: "ZYG_R" },
];

export function BiometricScannerHUD({
  title = "Multimodal Forensic Scanner",
  subtitle = "SatyaDrishti analyzes visual artifacts, spatial inconsistencies and forensic signals.",
}: BiometricScannerHUDProps) {
  const [telemetry, setTelemetry] = useState<Telemetry>({
    frame: 1,
    cycle: 1,
    signal: 97,
  });

  const [activeAgent, setActiveAgent] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setTelemetry((previous) => ({
        frame: previous.frame >= 9999 ? 1 : previous.frame + 1,
        cycle: previous.cycle >= 999 ? 1 : previous.cycle + 1,
        signal: 96 + Math.floor(Math.random() * 4),
      }));

      setActiveAgent((previous) => (previous + 1) % 3);
    }, 1800);

    return () => window.clearInterval(timer);
  }, []);

  const signalBars = useMemo(
    () =>
      Array.from({ length: 24 }, (_, index) => ({
        id: index,
        height: 5 + ((index * 13) % 17),
      })),
    [],
  );

  return (
    <div className="relative mx-auto w-full max-w-[680px] select-none text-white">
      {/* Ambient background */}
      <div className="pointer-events-none absolute -inset-16 overflow-hidden">
        <div className="absolute left-1/2 top-1/2 h-[520px] w-[520px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-500/[0.035] blur-3xl" />
        <div className="absolute left-[15%] top-[20%] h-32 w-32 rounded-full bg-blue-500/[0.04] blur-3xl" />
        <div className="absolute bottom-[10%] right-[10%] h-40 w-40 rounded-full bg-cyan-400/[0.03] blur-3xl" />
      </div>

      {/* Main shell */}
      <div className="relative overflow-hidden rounded-2xl border border-cyan-400/20 bg-[#020914]/90 shadow-[0_30px_100px_-35px_rgba(0,200,255,0.35)] backdrop-blur-xl">
        {/* Fine grid */}
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.045]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(56,189,248,.8) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,.8) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
          }}
        />

        {/* Top status bar */}
        <div className="relative flex items-center justify-between border-b border-cyan-400/10 bg-[#06111e]/80 px-5 py-3">
          <div className="flex items-center gap-3">
            <div className="relative flex h-2.5 w-2.5">
              <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400/70" />
              <span className="relative h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,.8)]" />
            </div>

            <div>
              <div className="font-mono text-[10px] font-semibold tracking-[0.2em] text-cyan-300">
                FORENSIC ENGINE
              </div>
              <div className="mt-0.5 text-[9px] tracking-wider text-slate-500">
                SATYADRISHTI / ANALYSIS CORE
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 font-mono text-[9px] text-slate-500">
            <span>
              FRAME{" "}
              <strong className="text-cyan-300">
                {String(telemetry.frame).padStart(4, "0")}
              </strong>
            </span>

            <span className="hidden sm:inline">
              CYCLE{" "}
              <strong className="text-cyan-300">
                #{String(telemetry.cycle).padStart(3, "0")}
              </strong>
            </span>

            <span className="flex items-center gap-1.5 text-emerald-400">
              <Activity className="h-3 w-3" />
              READY
            </span>
          </div>
        </div>

        {/* Scanner area */}
        <div className="relative h-[430px] overflow-hidden sm:h-[470px]">
          {/* Radial illumination */}
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(14,165,233,0.12),transparent_48%,rgba(2,9,20,0.75)_80%)]" />

          {/* Outer targeting frame */}
          <div className="absolute inset-5 rounded-xl border border-cyan-400/[0.08]" />

          {/* Corner brackets */}
          <div className="absolute left-5 top-5 h-10 w-10 border-l border-t border-cyan-300/70" />
          <div className="absolute right-5 top-5 h-10 w-10 border-r border-t border-cyan-300/70" />
          <div className="absolute bottom-5 left-5 h-10 w-10 border-b border-l border-cyan-300/70" />
          <div className="absolute bottom-5 right-5 h-10 w-10 border-b border-r border-cyan-300/70" />

          {/* Small corner markers */}
          <div className="absolute left-10 top-10 h-1 w-1 rounded-full bg-cyan-300 shadow-[0_0_8px_#22d3ee]" />
          <div className="absolute right-10 top-10 h-1 w-1 rounded-full bg-cyan-300 shadow-[0_0_8px_#22d3ee]" />
          <div className="absolute bottom-10 left-10 h-1 w-1 rounded-full bg-cyan-300 shadow-[0_0_8px_#22d3ee]" />
          <div className="absolute bottom-10 right-10 h-1 w-1 rounded-full bg-cyan-300 shadow-[0_0_8px_#22d3ee]" />

          {/* Radar rings */}
          <div className="absolute left-1/2 top-1/2 h-[350px] w-[350px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-400/[0.08]" />

          <div className="absolute left-1/2 top-1/2 h-[290px] w-[290px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-dashed border-cyan-400/[0.13]" />

          <div className="absolute left-1/2 top-1/2 h-[220px] w-[220px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-400/[0.12]" />

          {/* Radar sweep */}
          <div className="absolute left-1/2 top-1/2 h-[310px] w-[310px] -translate-x-1/2 -translate-y-1/2 animate-[spin_7s_linear_infinite] rounded-full">
            <div className="absolute left-1/2 top-0 h-1/2 w-px origin-bottom bg-gradient-to-t from-cyan-400/50 to-transparent" />

            <div className="absolute left-1/2 top-0 h-1/2 w-20 -translate-x-1/2 origin-bottom bg-gradient-to-t from-cyan-400/[0.10] to-transparent blur-sm" />
          </div>

          {/* Face scanner */}
          <div className="absolute left-1/2 top-1/2 h-[350px] w-[350px] -translate-x-1/2 -translate-y-1/2 sm:h-[390px] sm:w-[390px]">
            <svg
              viewBox="0 0 400 400"
              className="h-full w-full overflow-visible"
              fill="none"
            >
              <defs>
                <linearGradient id="faceLine" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#67e8f9" stopOpacity="0.9" />
                  <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.45" />
                  <stop offset="100%" stopColor="#0284c7" stopOpacity="0.15" />
                </linearGradient>

                <filter id="scannerGlow">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>

                <radialGradient id="faceFill">
                  <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.10" />
                  <stop offset="70%" stopColor="#0284c7" stopOpacity="0.035" />
                  <stop offset="100%" stopColor="#020617" stopOpacity="0" />
                </radialGradient>
              </defs>

              {/* Face silhouette */}
              <ellipse
                cx="200"
                cy="200"
                rx="112"
                ry="145"
                fill="url(#faceFill)"
                stroke="url(#faceLine)"
                strokeWidth="1"
                strokeDasharray="5 4"
                filter="url(#scannerGlow)"
              />

              {/* Horizontal mesh */}
              {Array.from({ length: 15 }).map((_, index) => {
                const y = 68 + index * 19;
                const dy = Math.abs(y - 200);
                const rx = Math.sqrt(
                  Math.max(
                    0,
                    112 * 112 * (1 - (dy * dy) / (145 * 145)),
                  ),
                );

                return (
                  <ellipse
                    key={`horizontal-${index}`}
                    cx="200"
                    cy={y}
                    rx={rx}
                    ry="4"
                    stroke="rgba(56,189,248,0.25)"
                    strokeWidth="0.65"
                  />
                );
              })}

              {/* Vertical mesh */}
              {Array.from({ length: 15 }).map((_, index) => {
                const x = 88 + index * 16;
                const dx = Math.abs(x - 200);

                const ry = Math.sqrt(
                  Math.max(
                    0,
                    145 * 145 * (1 - (dx * dx) / (112 * 112)),
                  ),
                );

                return (
                  <ellipse
                    key={`vertical-${index}`}
                    cx={x}
                    cy="200"
                    rx="4"
                    ry={ry}
                    stroke="rgba(56,189,248,0.18)"
                    strokeWidth="0.65"
                  />
                );
              })}

              {/* Facial guide lines */}
              <path
                d="M145 169 Q200 150 255 169"
                stroke="rgba(103,232,249,0.4)"
              />

              <path
                d="M150 220 Q200 235 250 220"
                stroke="rgba(103,232,249,0.3)"
              />

              <path
                d="M170 264 Q200 278 230 264"
                stroke="rgba(103,232,249,0.25)"
              />

              {/* Central reticle */}
              <circle
                cx="200"
                cy="200"
                r="24"
                stroke="rgba(34,211,238,0.25)"
              />

              <circle
                cx="200"
                cy="200"
                r="8"
                stroke="#67e8f9"
                strokeOpacity="0.8"
              />

              <circle
                cx="200"
                cy="200"
                r="2.5"
                fill="#67e8f9"
                filter="url(#scannerGlow)"
              />

              <path
                d="M200 169V187 M200 213V231 M169 200H187 M213 200H231"
                stroke="#67e8f9"
                strokeWidth="1"
              />

              {/* Landmark nodes */}
              {LANDMARKS.map((node, index) => (
                <g key={node.label}>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r="8"
                    stroke="#22d3ee"
                    strokeOpacity="0.16"
                  />

                  <circle
                    cx={node.x}
                    cy={node.y}
                    r="3"
                    fill="#67e8f9"
                    filter="url(#scannerGlow)"
                  />

                  {index % 2 === 0 && (
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r="7"
                      stroke="#22d3ee"
                      strokeOpacity="0.35"
                      className="animate-ping"
                    />
                  )}
                </g>
              ))}
            </svg>
          </div>

          {/* Scan beam */}
          <div className="pointer-events-none absolute inset-x-14 top-0 h-full overflow-hidden">
            <div className="absolute left-0 right-0 top-[10%] h-px animate-[scan_4s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-cyan-300 to-transparent shadow-[0_0_18px_rgba(34,211,238,0.9)]" />

            <div className="absolute left-0 right-0 top-[10%] h-20 animate-[scan_4s_ease-in-out_infinite] bg-gradient-to-b from-cyan-400/[0.08] to-transparent blur-sm" />
          </div>

          {/* Left information panel */}
          <div className="absolute left-9 top-20 hidden w-32 space-y-2 md:block">
            <TelemetryItem
              icon={<ScanFace className="h-3 w-3" />}
              label="FACE MAP"
              value="ACTIVE"
            />

            <TelemetryItem
              icon={<Fingerprint className="h-3 w-3" />}
              label="LANDMARKS"
              value="68 / 68"
            />

            <TelemetryItem
              icon={<Waves className="h-3 w-3" />}
              label="SIGNAL"
              value={`${telemetry.signal}%`}
            />
          </div>

          {/* Right information panel */}
          <div className="absolute right-9 top-20 hidden w-36 space-y-2 md:block">
            <AgentItem
              icon={<Eye className="h-3 w-3" />}
              name="VISION AGENT"
              active={activeAgent === 0}
            />

            <AgentItem
              icon={<Waves className="h-3 w-3" />}
              name="FORENSIC AGENT"
              active={activeAgent === 1}
            />

            <AgentItem
              icon={<Network className="h-3 w-3" />}
              name="FUSION AGENT"
              active={activeAgent === 2}
            />
          </div>

          {/* Bottom diagnostic strip */}
          <div className="absolute bottom-7 left-1/2 flex w-[calc(100%-64px)] max-w-[500px] -translate-x-1/2 items-center justify-between rounded-lg border border-cyan-400/10 bg-[#020914]/80 px-3 py-2 backdrop-blur-xl">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-md border border-emerald-400/20 bg-emerald-400/[0.06]">
                <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              </div>

              <div>
                <div className="font-mono text-[8px] tracking-[0.18em] text-slate-500">
                  SYSTEM STATUS
                </div>
                <div className="font-mono text-[10px] text-emerald-400">
                  FORENSIC CORE READY
                </div>
              </div>
            </div>

            <div className="flex h-6 items-end gap-[2px]">
              {signalBars.map((bar) => (
                <span
                  key={bar.id}
                  className="w-[2px] rounded-full bg-cyan-400/60 transition-all"
                  style={{
                    height: `${bar.height}px`,
                    opacity: 0.25 + (bar.id % 5) * 0.12,
                  }}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Agent architecture footer */}
        <div className="relative border-t border-cyan-400/10 bg-[#040c17]/80 px-5 py-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/[0.06]">
                <BrainCircuit className="h-4 w-4 text-cyan-300" />
              </div>

              <div>
                <div className="text-xs font-medium text-slate-200">
                  {title}
                </div>

                <div className="mt-0.5 max-w-md text-[10px] leading-relaxed text-slate-500">
                  {subtitle}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <StatusPill
                icon={<LockKeyhole className="h-3 w-3" />}
                text="SECURE"
              />

              <StatusPill
                icon={<Cpu className="h-3 w-3" />}
                text="AI CORE"
              />
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between border-t border-white/[0.04] pt-3">
            <div className="flex items-center gap-2 text-[9px] font-mono text-slate-600">
              <Radio className="h-3 w-3 text-cyan-500/60" />
              MULTI-AGENT ORCHESTRATION ONLINE
            </div>

            <div className="flex items-center gap-1.5 text-[9px] font-mono text-emerald-500/70">
              <CheckCircle2 className="h-3 w-3" />
              READY
            </div>
          </div>
        </div>
      </div>

      {/* Bottom ambient label */}
      <div className="mt-4 flex items-center justify-center gap-2 text-[9px] font-mono tracking-[0.18em] text-slate-600">
        <Sparkles className="h-3 w-3 text-cyan-500/50" />
        TRUSTED MEDIA AUTHENTICITY PIPELINE
      </div>

      {/* Required animation */}
      <style>{`
        @keyframes scan {
          0% {
            transform: translateY(0);
            opacity: 0;
          }
          10% {
            opacity: 1;
          }
          50% {
            opacity: 1;
          }
          90% {
            opacity: 0.8;
          }
          100% {
            transform: translateY(390px);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
}

function TelemetryItem({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-md border border-cyan-400/10 bg-[#020914]/70 px-2.5 py-2 backdrop-blur-md">
      <div className="flex items-center gap-1.5 text-[8px] font-mono tracking-wider text-slate-500">
        {icon}
        {label}
      </div>

      <div className="mt-1 font-mono text-[10px] text-cyan-300">
        {value}
      </div>
    </div>
  );
}

function AgentItem({
  icon,
  name,
  active,
}: {
  icon: React.ReactNode;
  name: string;
  active: boolean;
}) {
  return (
    <div
      className={`flex items-center gap-2 rounded-md border px-2.5 py-2 font-mono text-[8px] transition-all duration-500 ${active
          ? "border-cyan-400/25 bg-cyan-400/[0.07] text-cyan-300 shadow-[0_0_20px_rgba(34,211,238,0.05)]"
          : "border-white/[0.05] bg-[#020914]/60 text-slate-600"
        }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${active
            ? "bg-cyan-300 shadow-[0_0_8px_rgba(103,232,249,0.9)]"
            : "bg-slate-700"
          }`}
      />

      {icon}

      <span>{name}</span>
    </div>
  );
}

function StatusPill({
  icon,
  text,
}: {
  icon: React.ReactNode;
  text: string;
}) {
  return (
    <div className="flex items-center gap-1.5 rounded-full border border-cyan-400/10 bg-cyan-400/[0.04] px-2.5 py-1 text-[8px] font-mono text-slate-500">
      <span className="text-cyan-400/70">{icon}</span>
      {text}
    </div>
  );
}
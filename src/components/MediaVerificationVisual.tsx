import { ShieldCheck, Image, Film, Cpu, Layers } from "lucide-react";

interface MediaVerificationVisualProps {
  className?: string;
}

export function MediaVerificationVisual({ className = "" }: MediaVerificationVisualProps) {
  return (
    <div className={`relative w-full max-w-[620px] select-none text-slate-100 ${className}`}>
      {/* Subtle radial ambient background glow behind illustration */}
      <div className="pointer-events-none absolute -inset-10 overflow-hidden" aria-hidden="true">
        <div className="absolute left-1/2 top-1/2 h-[450px] w-[450px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-600/[0.035] blur-[120px]" />
        <div className="absolute left-[15%] top-[20%] h-36 w-36 rounded-full bg-teal-500/[0.03] blur-[90px]" />
      </div>

      {/* Editorial Heading directly on page background (NO outer card) */}
      <div className="mb-4 sm:mb-6 animate-fade-slide-up [animation-delay:100ms]">
        <div className="inline-flex items-center gap-2 text-[10px] sm:text-[11px] font-mono uppercase tracking-wider text-teal-400">
          <span className="relative flex h-1.5 w-1.5">
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-teal-400" />
          </span>
          <span>Autonomous Multi-Agent Forensics</span>
        </div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white mt-1.5">
          Digital Media Verification
        </h2>
        <p className="text-xs sm:text-[13px] text-slate-400 mt-1 max-w-lg leading-relaxed">
          Cohesive forensic inspection across spatial sensor residuals, temporal video continuity, and spectral frequency analysis.
        </p>
      </div>

      {/* Central Illustration Area (Directly on canvas, no enclosing card) */}
      <div className="relative my-2 flex min-h-[420px] sm:min-h-[440px] items-center justify-center animate-fade-slide-up [animation-delay:180ms]">
        {/* Subtle slow horizontal scanning beam across illustration area */}
        <div
          className="pointer-events-none absolute left-0 right-0 h-[1.5px] bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent animate-scan-beam"
          aria-hidden="true"
        />

        {/* Concentric Orbital Rings & Fine Curved Connecting Lines (SVG) */}
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox="0 0 560 420"
          fill="none"
          aria-hidden="true"
        >
          <defs>
            <linearGradient id="flowLineLeft" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.45" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.08" />
            </linearGradient>
            <linearGradient id="flowLineRight" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#14b8a6" stopOpacity="0.45" />
              <stop offset="100%" stopColor="#14b8a6" stopOpacity="0.08" />
            </linearGradient>
            <linearGradient id="flowLineBottom" x1="0%" y1="100%" x2="0%" y2="0%">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#14b8a6" stopOpacity="0.08" />
            </linearGradient>
          </defs>

          {/* Outermost delicate orbital ring */}
          <circle
            cx="280"
            cy="195"
            r="165"
            stroke="rgba(255, 255, 255, 0.03)"
            strokeWidth="1"
          />

          {/* Middle rotating dashed orbital ring */}
          <circle
            cx="280"
            cy="195"
            r="120"
            stroke="rgba(6, 182, 212, 0.16)"
            strokeWidth="1.25"
            strokeDasharray="4 6"
            className="animate-orbital-slow origin-[280px_195px]"
          />

          {/* Inner rotating dashed orbital ring (Reverse) */}
          <circle
            cx="280"
            cy="195"
            r="82"
            stroke="rgba(20, 184, 166, 0.18)"
            strokeWidth="1"
            strokeDasharray="3 5"
            className="animate-orbital-slow-reverse origin-[280px_195px]"
          />

          {/* Fine Curved Connecting Line: Image Stream -> Central Core */}
          <path
            d="M 155 105 C 205 105, 230 165, 270 185"
            stroke="url(#flowLineLeft)"
            strokeWidth="1.25"
            strokeDasharray="4 4"
          />

          {/* Fine Curved Connecting Line: Video Stream -> Central Core */}
          <path
            d="M 405 105 C 355 105, 330 165, 290 185"
            stroke="url(#flowLineRight)"
            strokeWidth="1.25"
            strokeDasharray="4 4"
          />

          {/* Fine Curved Connecting Line: Spectral Motif -> Central Core */}
          <path
            d="M 280 325 C 280 280, 280 240, 280 215"
            stroke="url(#flowLineBottom)"
            strokeWidth="1.25"
            strokeDasharray="3 3"
          />
        </svg>

        {/* Central Floating Verification Shield Core */}
        <div className="relative z-20 flex flex-col items-center">
          {/* Outermost clay glow ring pedestal */}
          <div className="clay-card relative flex h-32 w-32 items-center justify-center rounded-full p-2 transition-transform duration-500 hover:scale-105">
            {/* Subtle rotating reticle border */}
            <div
              className="pointer-events-none absolute inset-1.5 rounded-full border border-dashed border-cyan-500/25 animate-reticle"
              aria-hidden="true"
            />

            {/* Recessed intermediate clay well */}
            <div className="clay-sunken flex h-24 w-24 items-center justify-center rounded-full">
              {/* Central tactile core emblem with gentle floating motion */}
              <div className="clay-button-primary animate-float-gentle flex h-16 w-16 flex-col items-center justify-center rounded-full text-white shadow-lg">
                <ShieldCheck className="h-7 w-7 text-white drop-shadow-[0_2px_4px_rgba(0,0,0,0.5)]" />
              </div>
            </div>
          </div>

          {/* Core Identification Label with Abundant Vertical Breathing Room */}
          <div className="mt-4 text-center">
            <span className="text-[10px] font-semibold tracking-widest uppercase text-cyan-300">
              Forensic Verification Core
            </span>
            <p className="text-[10px] text-slate-400 mt-0.5">
              Multi-Agent Neural Consensus
            </p>
          </div>
        </div>

        {/* Floating Media Element 1: Spatial Image Tile (Left) */}
        <div className="clay-tile animate-float-gentle absolute left-1 sm:left-4 top-4 z-10 w-44 rounded-2xl p-3 sm:w-48 sm:p-3.5">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
            <div className="flex items-center gap-1.5">
              <Image className="h-3.5 w-3.5 text-cyan-400" />
              <span className="text-[11px] font-semibold text-slate-200">Image Stream</span>
            </div>
            <span className="rounded bg-cyan-500/10 px-1.5 py-0.5 text-[9px] font-mono text-cyan-300">
              SPATIAL
            </span>
          </div>
          <div className="mt-2.5 space-y-1.5">
            <div className="text-[10px] text-slate-400">
              Sensor Pattern Analysis
            </div>
            {/* Illustrative mini pixel residual lattice */}
            <div className="clay-sunken flex h-5 items-center justify-around rounded-md px-2">
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400/90" />
              <span className="h-1 w-1 rounded-full bg-slate-600" />
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400/70" />
              <span className="h-1 w-1 rounded-full bg-slate-600" />
              <span className="h-1.5 w-1.5 rounded-full bg-teal-400/90" />
            </div>
            <p className="text-[9px] text-slate-400 leading-tight">
              Pixel-domain residual validation
            </p>
          </div>
        </div>

        {/* Floating Media Element 2: Video-Frame Sequence (Right) */}
        <div className="clay-tile animate-float-gentle-alt absolute right-1 sm:right-4 top-4 z-10 w-44 rounded-2xl p-3 sm:w-48 sm:p-3.5">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
            <div className="flex items-center gap-1.5">
              <Film className="h-3.5 w-3.5 text-teal-400" />
              <span className="text-[11px] font-semibold text-slate-200">Video Sequence</span>
            </div>
            <span className="rounded bg-teal-500/10 px-1.5 py-0.5 text-[9px] font-mono text-teal-300">
              TEMPORAL
            </span>
          </div>
          <div className="mt-2.5 space-y-1.5">
            <div className="text-[10px] text-slate-400">
              Inter-Frame Coherence
            </div>
            {/* Illustrative cascading 3-frame sequence strip */}
            <div className="clay-sunken flex h-5 items-center justify-between rounded-md px-2.5">
              <div className="h-2.5 w-3.5 rounded-xs border border-teal-500/40 bg-teal-500/20" />
              <div className="h-0.5 w-2 bg-teal-500/40" />
              <div className="h-2.5 w-3.5 rounded-xs border border-teal-500/40 bg-teal-500/20" />
              <div className="h-0.5 w-2 bg-teal-500/40" />
              <div className="h-2.5 w-3.5 rounded-xs border border-teal-500/40 bg-teal-500/20" />
            </div>
            <p className="text-[9px] text-slate-400 leading-tight">
              Temporal optical flow tracking
            </p>
          </div>
        </div>

        {/* Floating Media Element 3: Spectral Analysis Motif (Bottom Center) */}
        <div className="clay-tile animate-float-gentle absolute bottom-2 left-1/2 -translate-x-1/2 z-10 flex items-center gap-3 rounded-2xl px-4 py-2.5 w-auto max-w-[280px]">
          <div className="clay-sunken flex h-7 w-7 items-center justify-center rounded-xl text-cyan-400 shrink-0">
            <Cpu className="h-3.5 w-3.5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-semibold text-slate-200 truncate">Spectral Forensics</span>
              <span className="rounded bg-cyan-500/10 px-1.5 py-0.2 text-[8px] font-mono text-cyan-300">
                FREQUENCY
              </span>
            </div>
            {/* Mini frequency indicator */}
            <div className="mt-1 flex items-end gap-1">
              <span className="h-1.5 w-1 rounded-t bg-cyan-500/60" />
              <span className="h-2.5 w-1 rounded-t bg-cyan-400" />
              <span className="h-3 w-1 rounded-t bg-teal-400" />
              <span className="h-2 w-1 rounded-t bg-cyan-400" />
              <span className="h-1 w-1 rounded-t bg-teal-500/60" />
              <span className="text-[9px] text-slate-400 ml-1.5 truncate">Fourier domain scan</span>
            </div>
          </div>
        </div>
      </div>

      {/* Minimal Architecture Ribbon directly on page background */}
      <div className="relative z-10 mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] pt-3 text-[11px] animate-fade-slide-up [animation-delay:240ms]">
        <span className="font-medium text-slate-300">
          Autonomous Pipeline Architecture
        </span>
        <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
          <span className="rounded-md border border-white/[0.06] bg-white/[0.02] px-2 py-0.5 text-[10px] text-slate-400">
            Spatial Residuals
          </span>
          <span className="rounded-md border border-white/[0.06] bg-white/[0.02] px-2 py-0.5 text-[10px] text-slate-400">
            Temporal Continuity
          </span>
          <span className="rounded-md border border-white/[0.06] bg-white/[0.02] px-2 py-0.5 text-[10px] text-slate-400">
            Spectral Forensics
          </span>
          <span className="rounded-md border border-white/[0.06] bg-white/[0.02] px-2 py-0.5 text-[10px] text-slate-400">
            Decision Fusion
          </span>
        </div>
      </div>
    </div>
  );
}

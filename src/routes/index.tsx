import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import {
  ShieldCheck,
  Eye,
  Video,
  AudioWaveform,
  Cpu,
  ArrowRight,
  Sparkles,
  Activity,
  Layers,
  FileCheck,
  CheckCircle2,
  Lock,
  ChevronRight,
} from "lucide-react";
import { Logo } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { BiometricScannerHUD } from "@/components/BiometricScannerHUD";
import { CyberBackground } from "@/components/CyberBackground";
import { authStore } from "@/lib/auth-store";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "SatyaDrishti | Agentic AI Deepfake Detection" },
      {
        name: "description",
        content:
          "Next-generation multi-agent deepfake detection and synthetic media forensics.",
      },
    ],
  }),
  component: StartingLandingPage,
});

function StartingLandingPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<"vision" | "video" | "audio">("vision");
  const [metricCount, setMetricCount] = useState(99.4);

  useEffect(() => {
    const timer = setInterval(() => {
      setMetricCount((prev) => +(99.3 + Math.random() * 0.5).toFixed(1));
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="relative min-h-screen bg-background text-foreground overflow-x-hidden selection:bg-sky-500/30 selection:text-sky-200">
      {/* Dynamic Cyber Particle Canvas Background */}
      <CyberBackground className="opacity-50" />

      {/* Ambient Radial Gradients */}
      <div className="pointer-events-none absolute -top-40 left-1/2 -translate-x-1/2 h-[500px] w-[900px] rounded-full bg-gradient-to-b from-sky-500/15 via-cyan-500/5 to-transparent blur-3xl" />
      <div className="pointer-events-none absolute top-1/3 -left-40 h-[400px] w-[500px] rounded-full bg-sky-600/10 blur-[100px]" />
      <div className="pointer-events-none absolute top-2/3 -right-40 h-[450px] w-[550px] rounded-full bg-cyan-600/10 blur-[100px]" />

      {/* Top Navigation Bar */}
      <header className="relative z-20 border-b border-border/70 bg-background/75 backdrop-blur-md sticky top-0">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3.5">
          <Logo size="md" />

          {/* Center telemetry pill */}
          <div className="hidden md:flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-500/10 px-3.5 py-1 text-xs font-mono text-sky-400">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span>NEURAL CONSENSUS: ONLINE • LATENCY 42ms</span>
          </div>

          {/* Action links */}
          <div className="flex items-center gap-3">
            <Link
              to="/login"
              className="text-xs font-medium text-slate-300 hover:text-white px-3 py-1.5 rounded-md hover:bg-slate-800/60 transition-colors"
            >
              Sign In
            </Link>
            <Link
              to="/signup"
              className="hidden sm:inline-flex text-xs font-medium text-sky-400 hover:text-sky-300 border border-sky-500/30 hover:border-sky-500/60 px-3.5 py-1.5 rounded-md transition-all bg-sky-500/5"
            >
              Create Account
            </Link>
            <Link to="/dashboard">
              <Button
                size="sm"
                className="relative bg-gradient-to-r from-sky-600 to-cyan-600 hover:from-sky-500 hover:to-cyan-500 text-white text-xs font-semibold px-4 shadow-md shadow-sky-600/20 overflow-hidden group"
              >
                <div className="absolute inset-0 animate-shimmer pointer-events-none" />
                <span>Launch Console</span>
                <ArrowRight className="ml-1.5 h-3.5 w-3.5 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Hero Section */}
      <main className="relative z-10 mx-auto max-w-7xl px-6 pt-12 pb-20 lg:pt-16">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Column: Hero Text & CTAs */}
          <div className="lg:col-span-7 space-y-6 animate-fade-slide-up">
            {/* Animated Announcement Badge */}
            <div className="inline-flex items-center gap-2 rounded-full border border-sky-500/40 bg-sky-500/10 px-3.5 py-1 text-xs font-mono text-sky-300 shadow-[0_0_20px_rgba(56,189,248,0.2)]">
              <Sparkles className="h-3.5 w-3.5 text-sky-400 animate-pulse" />
              <span>AGENTIC MULTI-MODAL DEEPFAKE DETECTION v2.4</span>
            </div>

            {/* Main Headline with Gradient Glow */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-100 leading-[1.12]">
              Verify Truth in the{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-400 via-cyan-300 to-blue-500 animate-pulse-glow">
                Synthetic Media
              </span>{" "}
              Era.
            </h1>

            {/* Subtitle */}
            <p className="text-sm sm:text-base text-slate-400 max-w-xl leading-relaxed">
              SatyaDrishti deploys autonomous specialized neural agents to cross-examine
              facial landmarks, micro-temporal kinetics, diffusion artifacts, and acoustic
              signatures to deliver explainable, audit-ready authenticity verdicts.
            </p>

            {/* CTA Group */}
            <div className="flex flex-wrap items-center gap-3.5 pt-2">
              <Link to="/login">
                <Button
                  size="lg"
                  className="relative h-11 px-6 bg-gradient-to-r from-sky-600 to-cyan-600 hover:from-sky-500 hover:to-cyan-500 text-white text-xs font-semibold shadow-lg shadow-sky-600/30 overflow-hidden group"
                >
                  <div className="absolute inset-0 animate-shimmer pointer-events-none" />
                  <ShieldCheck className="mr-2 h-4 w-4 text-sky-200" />
                  <span>Enter Forensic Workspace</span>
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Button>
              </Link>

              <Link to="/signup">
                <Button
                  variant="outline"
                  size="lg"
                  className="h-11 px-5 border-slate-700 bg-slate-900/60 hover:bg-slate-800 text-slate-200 text-xs font-medium hover:border-sky-500/50 transition-all"
                >
                  Register Account
                </Button>
              </Link>

              <button
                type="button"
                onClick={() => {
                  authStore.signInGuest();
                  navigate({ to: "/dashboard" });
                }}
                className="text-xs text-slate-400 hover:text-sky-400 underline underline-offset-4 cursor-pointer pl-1 transition-colors bg-transparent border-0 p-0"
              >
                Guest Analyst Demo →
              </button>
            </div>

            {/* Quick Live Telemetry Bar */}
            <div className="grid grid-cols-3 gap-3 pt-6 border-t border-border/60 max-w-lg">
              <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3 backdrop-blur-sm">
                <div className="text-lg font-bold text-sky-400 font-mono flex items-center gap-1">
                  <span>{metricCount}%</span>
                </div>
                <div className="text-[11px] text-slate-400">Benchmark Precision</div>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3 backdrop-blur-sm">
                <div className="text-lg font-bold text-emerald-400 font-mono">
                  &lt; 1.2s
                </div>
                <div className="text-[11px] text-slate-400">Multi-Agent Latency</div>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3 backdrop-blur-sm">
                <div className="text-lg font-bold text-cyan-400 font-mono">
                  4 Agents
                </div>
                <div className="text-[11px] text-slate-400">Consensus Engine</div>
              </div>
            </div>
          </div>

          {/* Right Column: High-Tech Animated Biometric Scanner HUD */}
          <div className="lg:col-span-5 flex justify-center animate-fade-slide-up [animation-delay:200ms]">
            <div className="w-full max-w-md">
              <BiometricScannerHUD
                title="Live Holographic Forensic HUD"
                subtitle="Analyzing face topology, frequency distortion, and temporal frame drift in real time."
              />
            </div>
          </div>
        </div>

        {/* Feature Cards Showcase */}
        <section className="mt-24 space-y-8">
          <div className="text-center max-w-2xl mx-auto space-y-2 animate-fade-slide-up">
            <div className="inline-flex items-center gap-1.5 rounded-full border border-sky-500/30 bg-sky-500/10 px-3 py-0.5 text-xs font-mono text-sky-400">
              <Layers className="h-3.5 w-3.5" />
              <span>SPECIALIZED AGENT ARCHITECTURE</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100">
              Four Independent Forensic Neural Agents
            </h2>
            <p className="text-xs sm:text-sm text-slate-400">
              Each agent operates an isolated forensic layer before converging into an explainable consensus matrix.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Agent 1: Image Agent */}
            <div className="group relative rounded-xl border border-slate-800 bg-slate-900/50 p-5 backdrop-blur-sm hover:border-sky-500/60 hover:bg-slate-900/80 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_10px_30px_-10px_rgba(56,189,248,0.2)]">
              <div className="h-10 w-10 rounded-lg border border-sky-500/30 bg-sky-500/10 flex items-center justify-center text-sky-400 mb-4 group-hover:scale-110 transition-transform">
                <Eye className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 group-hover:text-sky-300 transition-colors">
                Visual Spatial Agent
              </h3>
              <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">
                Dissects pixel residuals, boundary blending inconsistencies, GAN checkerboard patterns, and warping artifacts.
              </p>
              <div className="mt-4 flex items-center gap-1.5 text-[11px] font-mono text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                <span>Spatial ResNet-50</span>
              </div>
            </div>

            {/* Agent 2: Video Agent */}
            <div className="group relative rounded-xl border border-slate-800 bg-slate-900/50 p-5 backdrop-blur-sm hover:border-sky-500/60 hover:bg-slate-900/80 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_10px_30px_-10px_rgba(56,189,248,0.2)]">
              <div className="h-10 w-10 rounded-lg border border-sky-500/30 bg-sky-500/10 flex items-center justify-center text-sky-400 mb-4 group-hover:scale-110 transition-transform">
                <Video className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 group-hover:text-sky-300 transition-colors">
                Temporal Kinetics Agent
              </h3>
              <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">
                Tracks optical flow, eyelid flutter frequencies, head rotational vectors, and inter-frame coherence anomalies.
              </p>
              <div className="mt-4 flex items-center gap-1.5 text-[11px] font-mono text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                <span>3D-CNN / LSTM Flow</span>
              </div>
            </div>

            {/* Agent 3: Audio Agent */}
            <div className="group relative rounded-xl border border-slate-800 bg-slate-900/50 p-5 backdrop-blur-sm hover:border-sky-500/60 hover:bg-slate-900/80 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_10px_30px_-10px_rgba(56,189,248,0.2)]">
              <div className="h-10 w-10 rounded-lg border border-sky-500/30 bg-sky-500/10 flex items-center justify-center text-sky-400 mb-4 group-hover:scale-110 transition-transform">
                <AudioWaveform className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 group-hover:text-sky-300 transition-colors">
                Spectral Audio Agent
              </h3>
              <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">
                Analyzes mel-spectrogram harmonics, vocal tract resonance deviations, and neural vocoder synthesis fingerprints.
              </p>
              <div className="mt-4 flex items-center gap-1.5 text-[11px] font-mono text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                <span>Acoustic Wavelet Engine</span>
              </div>
            </div>

            {/* Agent 4: Consensus Engine */}
            <div className="group relative rounded-xl border border-slate-800 bg-slate-900/50 p-5 backdrop-blur-sm hover:border-sky-500/60 hover:bg-slate-900/80 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_10px_30px_-10px_rgba(56,189,248,0.2)]">
              <div className="h-10 w-10 rounded-lg border border-sky-500/30 bg-sky-500/10 flex items-center justify-center text-sky-400 mb-4 group-hover:scale-110 transition-transform">
                <Cpu className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 group-hover:text-sky-300 transition-colors">
                Consensus Engine
              </h3>
              <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">
                Fuses probability outputs with weighted cross-attention to produce transparent, auditable forensic reports.
              </p>
              <div className="mt-4 flex items-center gap-1.5 text-[11px] font-mono text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                <span>Bayesian Fusion</span>
              </div>
            </div>
          </div>
        </section>

        {/* Bottom CTA Banner */}
        <section className="mt-20 relative overflow-hidden rounded-2xl border border-sky-500/30 bg-gradient-to-r from-slate-950 via-sky-950/30 to-slate-950 p-8 sm:p-12 text-center shadow-2xl">
          <div className="absolute inset-0 bg-grid opacity-20 pointer-events-none" />
          <div className="relative z-10 max-w-2xl mx-auto space-y-4">
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100">
              Ready to Inspect Media Authenticity?
            </h2>
            <p className="text-xs sm:text-sm text-slate-400">
              Sign in with your analyst credentials or explore our instant live forensic demo console.
            </p>
            <div className="flex flex-wrap justify-center gap-3 pt-2">
              <Link to="/login">
                <Button className="h-10 px-6 bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold">
                  Sign In Now
                  <ArrowRight className="ml-2 h-3.5 w-3.5" />
                </Button>
              </Link>
              <Link to="/signup">
                <Button variant="outline" className="h-10 px-5 text-xs border-slate-700 bg-slate-900/60 hover:bg-slate-800 text-slate-200">
                  Create Analyst Account
                </Button>
              </Link>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-border/60 bg-slate-950/60 py-6 text-center text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-sky-400" />
            <span className="font-semibold text-slate-400">SatyaDrishti</span>
            <span>— Explainable Agentic AI Media Authenticity Platform</span>
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <Link to="/login" className="hover:text-sky-400 transition-colors">Sign In</Link>
            <Link to="/signup" className="hover:text-sky-400 transition-colors">Sign Up</Link>
            <Link to="/dashboard" className="hover:text-sky-400 transition-colors">Console</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}

import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, useEffect, type FormEvent } from "react";
import { Eye, EyeOff, Mail, Lock, ArrowRight } from "lucide-react";
import { Logo } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import { authStore, useAuthState } from "@/lib/auth-store";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "SatyaDrishti | Sign In" },
      {
        name: "description",
        content: "Sign in to SatyaDrishti to run agentic deepfake analysis.",
      },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const navigate = useNavigate();
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const { user } = useAuthState();

  useEffect(() => {
    if (user) {
      navigate({ to: "/dashboard" });
    }
  }, [user, navigate]);


  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Invalid email or password.");
      }

      authStore.signIn(data.user);
      toast.success("Welcome back! Sign in successful.");
      navigate({ to: "/dashboard" });
    } catch (err: any) {
      console.error("Login error:", err);
      toast.error(err.message || "An error occurred during sign in.");
    } finally {
      setLoading(false);
    }
  }


  return (
    <div className="grid min-h-screen grid-cols-1 bg-background text-foreground lg:grid-cols-2">
      {/* Form side */}
      <div className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          <Logo size="lg" />

          <div className="mt-10">
            <h1 className="text-2xl font-semibold tracking-tight">Welcome back</h1>
            <p className="mt-1.5 text-sm text-muted-foreground">
              Sign in to your workspace to run new deepfake analyses.
            </p>
          </div>

          <form onSubmit={onSubmit} className="mt-8 space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="email">Email</Label>
              <div className="relative">
                <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  required
                  autoComplete="email"
                  placeholder="you@company.com"
                  className="pl-9"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label htmlFor="password">Password</Label>
                <button
                  type="button"
                  className="text-xs text-muted-foreground hover:text-primary"
                >
                  Forgot password?
                </button>
              </div>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPw ? "text" : "password"}
                  required
                  autoComplete="current-password"
                  placeholder="••••••••"
                  className="pl-9 pr-9"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:text-foreground"
                  aria-label={showPw ? "Hide password" : "Show password"}
                >
                  {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              <Checkbox id="remember" defaultChecked />
              Remember me on this device
            </label>

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Signing in…" : "Sign in"}
              {!loading && <ArrowRight className="ml-1.5 h-4 w-4" />}
            </Button>

            <div className="text-center text-sm">
              <span className="text-muted-foreground">Don't have an account? </span>
              <Link to="/signup" className="font-medium text-primary hover:text-primary/90 transition-colors">
                Sign Up
              </Link>
            </div>
          </form>

          <p className="mt-8 text-center text-xs text-muted-foreground">
            Protected workspace. Access is limited to authorized researchers.
          </p>
          <p className="mt-2 text-center text-xs text-muted-foreground">
            <Link to="/dashboard" className="hover:text-primary">
              Continue as guest →
            </Link>
          </p>
        </div>
      </div>

      {/* Illustration side */}
      <div className="relative hidden overflow-hidden border-l border-border/70 bg-sidebar lg:block">
        <div className="absolute inset-0 bg-grid opacity-60" />
        <div className="absolute inset-0 bg-gradient-to-br from-transparent via-background/40 to-background" />

        <div className="relative flex h-full flex-col justify-between p-12">
          <FaceMesh />

          <div className="max-w-md">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs text-primary">
              <span className="h-1.5 w-1.5 rounded-full bg-primary" />
              Multi-agent detection engine
            </div>
            <h2 className="mt-4 text-2xl font-semibold tracking-tight text-gradient-cyan">
              Detect synthetic media with explainable, agentic AI.
            </h2>
            <p className="mt-3 text-sm text-muted-foreground">
              SatyaDrishti orchestrates specialized image, video, and confidence
              agents to produce transparent verdicts you can audit.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function FaceMesh() {
  // Simple, elegant face-mesh illustration in SVG — no glow, no cyberpunk.
  return (
    <div className="relative mx-auto flex aspect-square w-full max-w-md items-center justify-center">
      <svg
        viewBox="0 0 400 400"
        className="h-full w-full text-primary/70"
        fill="none"
        stroke="currentColor"
        strokeWidth="0.8"
      >
        <defs>
          <radialGradient id="fade" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.9" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
          </radialGradient>
          <mask id="mask">
            <rect width="400" height="400" fill="url(#fade)" />
          </mask>
        </defs>
        <g mask="url(#mask)">
          {/* Face oval */}
          <ellipse cx="200" cy="205" rx="110" ry="140" />
          {/* Horizontal contour lines */}
          {Array.from({ length: 14 }).map((_, i) => {
            const y = 80 + i * 20;
            const dy = Math.abs(y - 205);
            const rx = Math.sqrt(Math.max(0, 110 * 110 * (1 - (dy * dy) / (140 * 140))));
            return <ellipse key={`h${i}`} cx="200" cy={y} rx={rx} ry={4} />;
          })}
          {/* Vertical mesh */}
          {Array.from({ length: 11 }).map((_, i) => {
            const x = 100 + i * 20;
            const dx = Math.abs(x - 200);
            const ry = Math.sqrt(Math.max(0, 140 * 140 * (1 - (dx * dx) / (110 * 110))));
            return <ellipse key={`v${i}`} cx={x} cy="205" rx={4} ry={ry} />;
          })}
          {/* Landmark dots */}
          {[
            [160, 180],
            [240, 180],
            [200, 220],
            [175, 265],
            [225, 265],
            [200, 260],
          ].map(([x, y], i) => (
            <circle key={i} cx={x} cy={y} r="2.2" fill="currentColor" stroke="none" />
          ))}
        </g>
      </svg>
    </div>
  );
}
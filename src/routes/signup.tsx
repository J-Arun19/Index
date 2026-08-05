import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, useEffect, type FormEvent } from "react";
import { Eye, EyeOff, Mail, Lock, User, ArrowRight } from "lucide-react";
import { Logo } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { useAuthState } from "@/lib/auth-store";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export const Route = createFileRoute("/signup")({
  head: () => ({
    meta: [
      { title: "SatyaDrishti | Create Account" },
      {
        name: "description",
        content: "Create a SatyaDrishti account to run agentic deepfake analysis.",
      },
    ],
  }),
  component: SignupPage,
});

function checkPasswordStrength(pw: string) {
  if (!pw) return { score: 0, label: "Empty" };
  let score = 0;
  if (pw.length >= 8) score += 1;
  if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) score += 1;
  if (/\d/.test(pw)) score += 1;
  if (/[^A-Za-z0-9]/.test(pw)) score += 1;

  if (score <= 1) {
    return { score, label: "Weak" };
  } else if (score <= 3) {
    return { score, label: "Medium" };
  } else {
    return { score, label: "Strong" };
  }
}

function SignupPage() {
  const navigate = useNavigate();
  const [showPw, setShowPw] = useState(false);
  const [showConfirmPw, setShowConfirmPw] = useState(false);
  const [loading, setLoading] = useState(false);

  // Form states
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const { user } = useAuthState();

  useEffect(() => {
    if (user) {
      navigate({ to: "/dashboard" });
    }
  }, [user, navigate]);

  const strength = checkPasswordStrength(password);


  async function onSubmit(e: FormEvent) {
    e.preventDefault();

    // 1. Validations
    if (!name.trim() || !email.trim() || !password || !confirmPassword) {
      toast.error("Please fill in all fields.");
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      toast.error("Please enter a valid email address.");
      return;
    }

    if (password !== confirmPassword) {
      toast.error("Passwords do not match.");
      return;
    }

    setLoading(true);

    try {
      // 2. Call backend register API
      const response = await fetch(`${API_URL}/api/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: name.trim(),
          email: email.trim(),
          password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to create account.");
      }

      toast.success("Account created successfully. Please sign in.");
      
      // Redirect to Sign In page
      navigate({ to: "/login" });
    } catch (err: any) {
      console.error("Signup error:", err);
      toast.error(err.message || "An error occurred during sign up.");
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

          <div className="mt-8">
            <h1 className="text-2xl font-semibold tracking-tight">Create your account</h1>
            <p className="mt-1.5 text-sm text-muted-foreground">
              Sign up to start running multi-agent deepfake scans.
            </p>
          </div>

          <form onSubmit={onSubmit} className="mt-6 space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="name">Full Name</Label>
              <div className="relative">
                <User className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="name"
                  type="text"
                  required
                  placeholder="John Doe"
                  className="pl-9"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="email">Email Address</Label>
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
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPw ? "text" : "password"}
                  required
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

              {/* Password strength meter */}
              {password && (
                <div className="mt-1 space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Strength:</span>
                    <span className={
                      strength.score <= 1 
                        ? "text-red-400 font-medium" 
                        : strength.score <= 3 
                          ? "text-amber-400 font-medium" 
                          : "text-emerald-400 font-medium"
                    }>
                      {strength.label}
                    </span>
                  </div>
                  <div className="h-1 w-full bg-secondary rounded-full overflow-hidden flex gap-0.5">
                    <div className={`h-full flex-1 rounded-l-full transition-all duration-300 ${
                      strength.score >= 1 
                        ? (strength.score === 1 ? 'bg-red-500' : strength.score <= 3 ? 'bg-amber-500' : 'bg-emerald-500') 
                        : 'bg-border'
                    }`} />
                    <div className={`h-full flex-1 transition-all duration-300 ${
                      strength.score >= 3 
                        ? (strength.score <= 3 ? 'bg-amber-500' : 'bg-emerald-500') 
                        : 'bg-border'
                    }`} />
                    <div className={`h-full flex-1 rounded-r-full transition-all duration-300 ${
                      strength.score >= 4 ? 'bg-emerald-500' : 'bg-border'
                    }`} />
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="confirmPassword">Confirm Password</Label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="confirmPassword"
                  type={showConfirmPw ? "text" : "password"}
                  required
                  placeholder="••••••••"
                  className="pl-9 pr-9"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPw((v) => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:text-foreground"
                  aria-label={showConfirmPw ? "Hide password" : "Show password"}
                >
                  {showConfirmPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <Button type="submit" className="w-full mt-2" disabled={loading}>
              {loading ? "Creating account…" : "Create Account"}
              {!loading && <ArrowRight className="ml-1.5 h-4 w-4" />}
            </Button>

            <div className="text-center text-sm">
              <span className="text-muted-foreground">Already have an account? </span>
              <Link to="/login" className="font-medium text-primary hover:text-primary/90 transition-colors">
                Sign In
              </Link>
            </div>
          </form>

          <p className="mt-8 text-center text-xs text-muted-foreground">
            Protected workspace. Access is limited to authorized researchers.
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

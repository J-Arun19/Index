import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, useEffect, type FormEvent } from "react";
import { Eye, EyeOff, Mail, Lock, User, ArrowRight, ShieldCheck, Sparkles, Loader2, CheckCircle } from "lucide-react";
import { Logo } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { authStore, useAuthState } from "@/lib/auth-store";
import { BiometricScannerHUD } from "@/components/BiometricScannerHUD";
import { CyberBackground } from "@/components/CyberBackground";
import { supabase } from "@/lib/supabase";

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

    if (loading) return;

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

    if (password.length < 6) {
      toast.error("Password must be at least 6 characters long.");
      return;
    }

    if (password !== confirmPassword) {
      toast.error("Passwords do not match.");
      return;
    }

    setLoading(true);

    try {
      let createdUser: any = null;
      const cleanEmail = email.trim();
      const cleanName = name.trim();

      // 1. Create the user account in Supabase via certified API
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
      try {
        const response = await fetch(`${API_URL}/api/auth/register`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            name: cleanName,
            email: cleanEmail,
            password,
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          if (data.error && data.error.toLowerCase().includes("already registered")) {
            toast.error("An account with this email already exists. Please sign in.");
          } else {
            toast.error(data.error || "Failed to create account in Supabase.");
          }
          return;
        }

        createdUser = data.user;
      } catch (backendErr) {
        console.warn("Backend registration failed, attempting direct Supabase signUp fallback:", backendErr);
        const { data, error } = await supabase.auth.signUp({
          email: cleanEmail,
          password,
          options: {
            data: { full_name: cleanName },
          },
        });

        if (error) {
          if (
            error.message.toLowerCase().includes("already registered") ||
            (error as any).status === 422 ||
            (error as any).code === "user_already_exists"
          ) {
            toast.error("An account with this email already exists. Please sign in.");
            return;
          }
          toast.error(error.message || "Registration failed. Please check your details.");
          return;
        }
        createdUser = data?.user;
      }

      // 2. Establish authenticated Supabase session with the registered credentials
      if (createdUser) {
        try {
          const { data: authData, error: authError } = await supabase.auth.signInWithPassword({
            email: cleanEmail,
            password,
          });

          if (!authError && authData?.session) {
            authStore.signIn({
              id: authData.user.id,
              name: authData.user.user_metadata?.full_name || cleanName,
              email: authData.user.email || cleanEmail,
            });
            toast.success("Account created successfully! Entering forensic workspace...");
            navigate({ to: "/dashboard" });
            return;
          }
        } catch (signInErr) {
          console.warn("Automatic sign-in after registration deferred to manual login:", signInErr);
        }

        // If auto-session deferred, notify user to sign in
        toast.success("Account created successfully! Please sign in with your credentials.");
        navigate({ to: "/login" });
      }
    } catch (err: any) {
      console.error("Unexpected registration error:", err);
      toast.error(err.message || "An unexpected error occurred during sign up.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative grid min-h-screen grid-cols-1 bg-background text-foreground lg:grid-cols-2 overflow-hidden">
      {/* Background Cyber Animation Canvas */}
      <CyberBackground className="opacity-40" />

      {/* Form side */}
      <div className="relative z-10 flex items-center justify-center px-6 py-10">
        <div className="w-full max-w-sm">
          {/* Header & Logo with Staggered Entrance */}
          <div className="animate-fade-slide-up [animation-delay:100ms]">
            <div className="flex items-center justify-between">
              <Logo size="lg" />
              <div className="inline-flex items-center gap-1.5 rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-0.5 text-[10px] font-mono text-sky-400">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
                </span>
                REGISTRATION NODE
              </div>
            </div>

            <div className="mt-6">
              <h1 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
                <span>Create Analyst Account</span>
                <Sparkles className="h-4 w-4 text-sky-400 animate-pulse" />
              </h1>
              <p className="mt-1 text-xs text-slate-400">
                Register for certified access to the multi-agent forensic platform.
              </p>
            </div>
          </div>

          {/* Form Fields with Staggered Entrance */}
          <form
            onSubmit={onSubmit}
            className="mt-6 space-y-3.5 animate-fade-slide-up [animation-delay:200ms]"
          >
            {/* Full Name */}
            <div className="space-y-1.5">
              <Label htmlFor="name" className="text-xs text-slate-300 font-medium">
                Full Name
              </Label>
              <div className="relative group">
                <User className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400 group-focus-within:text-sky-400 transition-colors" />
                <Input
                  id="name"
                  type="text"
                  required
                  placeholder="Dr. Alex Vance"
                  className="pl-9 h-9 text-xs border-slate-700 bg-slate-900/70 text-slate-200 placeholder:text-slate-500 focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 transition-all shadow-inner"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
            </div>

            {/* Email */}
            <div className="space-y-1.5">
              <Label htmlFor="email" className="text-xs text-slate-300 font-medium">
                Email Address
              </Label>
              <div className="relative group">
                <Mail className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400 group-focus-within:text-sky-400 transition-colors" />
                <Input
                  id="email"
                  type="email"
                  required
                  autoComplete="email"
                  placeholder="analyst@satyadrishti.org"
                  className="pl-9 h-9 text-xs border-slate-700 bg-slate-900/70 text-slate-200 placeholder:text-slate-500 focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 transition-all shadow-inner"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            {/* Password */}
            <div className="space-y-1.5">
              <Label htmlFor="password" className="text-xs text-slate-300 font-medium">
                Password
              </Label>
              <div className="relative group">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400 group-focus-within:text-sky-400 transition-colors" />
                <Input
                  id="password"
                  type={showPw ? "text" : "password"}
                  required
                  placeholder="••••••••"
                  className="pl-9 pr-9 h-9 text-xs border-slate-700 bg-slate-900/70 text-slate-200 placeholder:text-slate-500 focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 transition-all shadow-inner"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-400 hover:text-slate-200 transition-colors"
                  aria-label={showPw ? "Hide password" : "Show password"}
                >
                  {showPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                </button>
              </div>

              {/* Password strength animated meter */}
              {password && (
                <div className="mt-1.5 space-y-1 animate-fade-slide-up">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-slate-400">Security Index:</span>
                    <span
                      className={`font-semibold transition-colors duration-300 ${
                        strength.score <= 1
                          ? "text-rose-400"
                          : strength.score <= 3
                            ? "text-amber-400"
                            : "text-emerald-400"
                      }`}
                    >
                      {strength.label}
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800/80 rounded-full overflow-hidden flex gap-1 p-0.5 border border-slate-700/50">
                    <div
                      className={`h-full flex-1 rounded-full transition-all duration-500 ${
                        strength.score >= 1
                          ? strength.score === 1
                            ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]"
                            : strength.score <= 3
                              ? "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]"
                              : "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"
                          : "bg-transparent"
                      }`}
                    />
                    <div
                      className={`h-full flex-1 rounded-full transition-all duration-500 ${
                        strength.score >= 2
                          ? strength.score <= 3
                            ? "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]"
                            : "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"
                          : "bg-transparent"
                      }`}
                    />
                    <div
                      className={`h-full flex-1 rounded-full transition-all duration-500 ${
                        strength.score >= 4
                          ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"
                          : "bg-transparent"
                      }`}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div className="space-y-1.5">
              <Label htmlFor="confirmPassword" className="text-xs text-slate-300 font-medium">
                Confirm Password
              </Label>
              <div className="relative group">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400 group-focus-within:text-sky-400 transition-colors" />
                <Input
                  id="confirmPassword"
                  type={showConfirmPw ? "text" : "password"}
                  required
                  placeholder="••••••••"
                  className="pl-9 pr-9 h-9 text-xs border-slate-700 bg-slate-900/70 text-slate-200 placeholder:text-slate-500 focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 transition-all shadow-inner"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPw((v) => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-400 hover:text-slate-200 transition-colors"
                  aria-label={showConfirmPw ? "Hide password" : "Show password"}
                >
                  {showConfirmPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                </button>
              </div>
            </div>

            {/* Submit Button with Shimmer */}
            <Button
              type="submit"
              className="relative w-full h-10 bg-gradient-to-r from-sky-600 to-cyan-600 hover:from-sky-500 hover:to-cyan-500 text-white text-xs font-semibold overflow-hidden transition-all shadow-lg shadow-sky-600/20 hover:shadow-sky-500/30 group mt-1"
              disabled={loading}
            >
              <div className="absolute inset-0 animate-shimmer pointer-events-none" />
              {loading ? (
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Registering Credential…</span>
                </div>
              ) : (
                <div className="flex items-center justify-center gap-2">
                  <span>Register Analyst Account</span>
                  <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-1" />
                </div>
              )}
            </Button>

            <div className="text-center text-xs pt-1">
              <span className="text-slate-400">Already registered? </span>
              <Link to="/login" className="font-semibold text-sky-400 hover:text-sky-300 transition-colors underline-offset-4 hover:underline">
                Sign In
              </Link>
            </div>
          </form>

          {/* Footer badge */}
          <div className="mt-6 border-t border-border/60 pt-4 text-center text-[11px] text-slate-400 animate-fade-slide-up [animation-delay:300ms]">
            <div className="flex items-center justify-center gap-1.5 text-slate-500">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              <span>Multi-Tenant Enterprise Forensic Security</span>
            </div>
          </div>
        </div>
      </div>

      {/* High-Tech Animated Biometric HUD Graphic Side */}
      <div className="relative hidden overflow-hidden border-l border-border/80 bg-slate-950/90 lg:flex items-center justify-center p-12">
        <div className="absolute inset-0 bg-grid opacity-25 animate-cyber-grid pointer-events-none" />
        <div className="absolute inset-0 bg-gradient-to-br from-sky-500/5 via-slate-950/70 to-slate-950 pointer-events-none" />

        <div className="relative z-10 w-full flex flex-col items-center justify-center animate-fade-slide-up [animation-delay:150ms]">
          <BiometricScannerHUD
            title="Forensic Node Enrollment & Verification"
            subtitle="Deploy multi-agent neural pipelines for deepfake forensic tracking with cryptographically verifiable proofs."
          />
        </div>
      </div>
    </div>
  );
}

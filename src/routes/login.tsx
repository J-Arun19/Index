import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, useEffect, type FormEvent } from "react";
import { Eye, EyeOff, Mail, Lock, ArrowRight, ShieldCheck, Loader2 } from "lucide-react";
import { Logo } from "@/components/Logo";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import { authStore, useAuthState } from "@/lib/auth-store";
import { MediaVerificationVisual } from "@/components/MediaVerificationVisual";
import { CyberBackground } from "@/components/CyberBackground";
import { supabase } from "@/lib/supabase";

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
  const [rememberSession, setRememberSession] = useState(false);
  const { user } = useAuthState();

  useEffect(() => {
    if (user) {
      navigate({ to: "/dashboard" });
    }
  }, [user, navigate]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();

    if (loading) return;

    if (!email.trim() || !password) {
      toast.error("Please enter both email and password.");
      return;
    }

    setLoading(true);
    const cleanEmail = email.trim();
    const maskedEmail = cleanEmail.replace(/^(.)(.*)(@.*)$/, (_, first, _middle, domain) => `${first}***${domain}`);

    try {
      // 1. Direct Supabase Auth signInWithPassword
      const { data, error } = await supabase.auth.signInWithPassword({
        email: cleanEmail,
        password,
      });

      if (error) {
        console.error("AUTH LOGIN ERROR", {
          email: maskedEmail,
          message: error.message,
          code: (error as any).code,
          status: error.status,
        });

        if (
          error.message?.toLowerCase().includes("email not confirmed") ||
          (error as any).code === "email_not_confirmed"
        ) {
          toast.error("Please verify your email address before signing in.");
          return;
        }

        if (
          (error as any).code === "invalid_credentials" ||
          error.message?.toLowerCase().includes("invalid login credentials")
        ) {
          toast.error("Invalid email or password.");
          return;
        }

        toast.error(error.message || "Failed to sign in. Please check your credentials.");
        return;
      }

      if (data?.session && data?.user) {
        console.log("AUTH LOGIN SUCCESS", {
          email: maskedEmail,
          hasSession: !!data.session,
          hasUser: !!data.user,
          confirmed: !!data.user.email_confirmed_at,
          redirectDestination: "/dashboard",
        });

        const authUser = data.user;
        let userProfile = {
          id: authUser.id,
          name: authUser.user_metadata?.full_name || authUser.email?.split("@")[0] || "Forensic Analyst",
          email: authUser.email || cleanEmail,
        };

        // 2. Safe, idempotent profile check with public.users
        try {
          const { data: profileRow } = await supabase
            .from("users")
            .select("id, name, email")
            .eq("email", userProfile.email)
            .maybeSingle();

          if (profileRow) {
            userProfile = {
              id: authUser.id,
              name: profileRow.name || userProfile.name,
              email: profileRow.email,
            };
          }
        } catch (profileErr) {
          console.warn("Optional profile lookup bypassed:", profileErr);
        }

        // 3. Establish active session in application store and navigate
        authStore.signIn(userProfile);
        toast.success("Welcome back! Sign in successful.");
        navigate({ to: "/dashboard" });
        return;
      }

      // Edge case: No error, but no session returned
      toast.error("Sign in failed to establish a session. Please try again.");
    } catch (err: any) {
      console.error("Login unexpected error:", err);
      toast.error(err.message || "An unexpected error occurred during sign in.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-screen w-full bg-[#060a13] text-slate-100 flex flex-col justify-center overflow-x-hidden selection:bg-cyan-500/25 selection:text-cyan-200">
      {/* Background Cyber Ambient Particles */}
      <CyberBackground className="opacity-20" />

      {/* Subtle radial ambient background illumination */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden="true">
        <div className="absolute -left-48 top-1/4 h-[550px] w-[550px] rounded-full bg-cyan-600/[0.035] blur-[140px]" />
        <div className="absolute -right-48 bottom-1/4 h-[650px] w-[650px] rounded-full bg-teal-600/[0.03] blur-[150px]" />
      </div>

      {/* Split-Screen Composition: ~42% Auth / ~58% Product Visualization (Seamless, NO outer cards) */}
      <main className="relative z-10 mx-auto flex w-full max-w-[1400px] min-h-screen flex-col lg:flex-row items-center justify-between px-6 sm:px-10 lg:px-14 xl:px-20 py-10 lg:py-0 gap-12 lg:gap-0">
        {/* LEFT PANEL — AUTHENTICATION AREA (~42% desktop width, unboxed) */}
        <section
          aria-labelledby="auth-heading"
          className="flex w-full lg:w-[42%] justify-center lg:justify-start lg:pr-10 xl:pr-16 py-4 lg:py-8"
        >
          {/* Direct Form Container — NO outer card, NO heavy borders */}
          <div className="w-full max-w-[400px]">
            {/* Logo and Brand Header */}
            <div className="animate-fade-slide-up [animation-delay:80ms]">
              <div className="flex items-center justify-between">
                <Logo size="md" />
                <div className="inline-flex items-center gap-1.5 rounded-full border border-cyan-500/20 bg-cyan-500/[0.06] px-2.5 py-0.5 text-[10px] font-mono text-cyan-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                  <span>v2.4 FORENSIC</span>
                </div>
              </div>

              <div className="mt-8">
                <h1
                  id="auth-heading"
                  className="text-2xl sm:text-[28px] font-bold tracking-tight text-white"
                >
                  Sign in to SatyaDrishti
                </h1>
                <p className="mt-2 text-xs sm:text-[13px] text-slate-400 leading-relaxed">
                  Verify digital media with AI-powered forensic analysis.
                </p>
              </div>
            </div>

            {/* Login Form with Recessed Inputs */}
            <form
              onSubmit={onSubmit}
              noValidate
              className="mt-7 space-y-4 animate-fade-slide-up [animation-delay:160ms]"
            >
              {/* Email Input Field */}
              <div className="space-y-1.5">
                <label
                  htmlFor="email"
                  className="block text-xs font-medium text-slate-300"
                >
                  Email Address
                </label>
                <div className="clay-sunken relative flex items-center rounded-xl transition-all duration-200">
                  <Mail
                    className="pointer-events-none absolute left-3.5 h-4 w-4 text-slate-400"
                    aria-hidden="true"
                  />
                  <input
                    id="email"
                    name="email"
                    type="email"
                    required
                    autoComplete="email"
                    placeholder="analyst@satyadrishti.org"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full bg-transparent py-2.5 pl-10 pr-4 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none"
                    aria-required="true"
                  />
                </div>
              </div>

              {/* Password Input Field */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label
                    htmlFor="password"
                    className="block text-xs font-medium text-slate-300"
                  >
                    Password
                  </label>
                  <Link
                    to="/forgot-password"
                    className="text-[11px] font-medium text-cyan-400/90 transition-colors hover:text-cyan-300 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-cyan-500 rounded px-1 cursor-pointer"
                  >
                    Forgot password?
                  </Link>
                </div>
                <div className="clay-sunken relative flex items-center rounded-xl transition-all duration-200">
                  <Lock
                    className="pointer-events-none absolute left-3.5 h-4 w-4 text-slate-400"
                    aria-hidden="true"
                  />
                  <input
                    id="password"
                    name="password"
                    type={showPw ? "text" : "password"}
                    required
                    autoComplete="current-password"
                    placeholder="••••••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full bg-transparent py-2.5 pl-10 pr-10 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none"
                    aria-required="true"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw((v) => !v)}
                    className="absolute right-2.5 flex h-7 w-7 items-center justify-center rounded-lg text-slate-400 transition-colors hover:text-slate-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-cyan-500 cursor-pointer"
                    aria-label={showPw ? "Hide password" : "Show password"}
                  >
                    {showPw ? (
                      <EyeOff className="h-4 w-4" aria-hidden="true" />
                    ) : (
                      <Eye className="h-4 w-4" aria-hidden="true" />
                    )}
                  </button>
                </div>
              </div>

              {/* Remember Session Option */}
              <div className="flex items-center gap-2 pt-0.5">
                <Checkbox
                  id="remember"
                  checked={rememberSession}
                  onCheckedChange={(checked) => setRememberSession(Boolean(checked))}
                  className="border-slate-700 data-[state=checked]:bg-cyan-600 data-[state=checked]:border-cyan-600"
                />
                <label
                  htmlFor="remember"
                  className="cursor-pointer select-none text-xs text-slate-400 transition-colors hover:text-slate-300"
                >
                  Remember session on this workstation
                </label>
              </div>

              {/* Tactile Clay Sign-In Button */}
              <button
                type="submit"
                disabled={loading}
                className="clay-button-primary group relative flex w-full items-center justify-center gap-2 rounded-xl py-2.5 px-4 text-xs font-semibold text-white cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin text-white" aria-hidden="true" />
                    <span>Signing in…</span>
                  </>
                ) : (
                  <>
                    <span>Sign in to SatyaDrishti</span>
                    <ArrowRight
                      className="h-3.5 w-3.5 transition-transform duration-200 group-hover:translate-x-1"
                      aria-hidden="true"
                    />
                  </>
                )}
              </button>

              {/* Sign-Up Link */}
              <div className="pt-1 text-center text-xs">
                <span className="text-slate-400">Don't have an analyst account? </span>
                <Link
                  to="/signup"
                  className="font-semibold text-cyan-400 underline-offset-4 transition-colors hover:text-cyan-300 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-cyan-500 rounded"
                >
                  Sign Up
                </Link>
              </div>
            </form>

            {/* Understated Guest Analyst Entry & Security Note */}
            <div className="mt-8 space-y-3.5 border-t border-white/[0.06] pt-5 animate-fade-slide-up [animation-delay:240ms]">
              <div className="flex items-center gap-2 text-[11px] text-slate-400">
                <ShieldCheck className="h-3.5 w-3.5 text-teal-400 shrink-0" aria-hidden="true" />
                <span>Zero-Trust Forensic Pipeline Active</span>
              </div>

              <div>
                <button
                  type="button"
                  onClick={() => {
                    authStore.signInGuest();
                    toast.success("Signed in as Guest Forensic Analyst.");
                    navigate({ to: "/dashboard" });
                  }}
                  className="inline-flex items-center gap-1.5 text-xs text-slate-400 transition-colors hover:text-cyan-300 focus-visible:outline-none focus-visible:underline cursor-pointer"
                >
                  <span>Continue as guest analyst</span>
                  <ArrowRight className="h-3 w-3" aria-hidden="true" />
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Faint Vertical Divider between sections on desktop */}
        <div
          className="hidden lg:block h-[560px] w-px bg-gradient-to-b from-transparent via-white/[0.08] to-transparent shrink-0"
          aria-hidden="true"
        />

        {/* RIGHT PANEL — FORENSIC VISUALIZATION (~58% desktop width, unboxed) */}
        <section
          aria-label="Digital Media Verification Architecture"
          className="w-full lg:w-[58%] flex items-center justify-center lg:justify-end lg:pl-10 xl:pl-16 py-4 lg:py-8"
        >
          <MediaVerificationVisual />
        </section>
      </main>
    </div>
  );
}
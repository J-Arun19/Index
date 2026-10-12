import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { Mail, ArrowLeft, ArrowRight, ShieldCheck, Loader2, CheckCircle2, Copy, Check, KeyRound } from "lucide-react";
import { Logo } from "@/components/Logo";
import { toast } from "sonner";
import { MediaVerificationVisual } from "@/components/MediaVerificationVisual";
import { CyberBackground } from "@/components/CyberBackground";
import { supabase } from "@/lib/supabase";

export const Route = createFileRoute("/forgot-password")({
  head: () => ({
    meta: [
      { title: "SatyaDrishti | Password Recovery" },
      {
        name: "description",
        content: "Reset your SatyaDrishti account password.",
      },
    ],
  }),
  component: ForgotPasswordPage,
});

function ForgotPasswordPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [emailError, setEmailError] = useState("");
  const [recoveryLink, setRecoveryLink] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  function validateEmail(val: string): boolean {
    if (!val.trim()) {
      setEmailError("Please enter your registered email address.");
      return false;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(val.trim())) {
      setEmailError("Please enter a valid email address (e.g. analyst@satyadrishti.org).");
      return false;
    }
    setEmailError("");
    return true;
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();

    if (loading) return;

    if (!validateEmail(email)) {
      return;
    }

    setLoading(true);
    setEmailError("");
    setRecoveryLink(null);
    setCopied(false);
    const cleanEmail = email.trim();

    try {
      // Dynamic origin to support any local, staging or production domain
      const redirectOrigin = typeof window !== "undefined" ? window.location.origin : "http://localhost:8080";
      const redirectUrl = `${redirectOrigin}/reset-password`;

      let generatedLink: string | null = null;
      let reqSuccess = false;

      // 1. Try backend certified recovery endpoint (uses Supabase service-role admin.generate_link, bypassing cloud SMTP rate limits)
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
      try {
        const resp = await fetch(`${API_URL}/api/auth/forgot-password`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: cleanEmail, redirectTo: redirectUrl }),
        });
        const data = await resp.json();
        if (resp.ok && data.status === "success") {
          reqSuccess = true;
          if (data.recovery_link) {
            generatedLink = data.recovery_link;
            setRecoveryLink(data.recovery_link);
          }
        } else if (!resp.ok && data.error && !data.error.includes("500") && !data.error.toLowerCase().includes("failed to generate")) {
          setEmailError(data.error);
          toast.error(data.error);
          return;
        }
      } catch (apiErr) {
        console.warn("Backend auth recovery service unreachable, falling back to direct Supabase auth:", apiErr);
      }

      // 2. If backend was unreachable or returned an error, fallback to direct Supabase auth
      if (!reqSuccess) {
        const { error } = await supabase.auth.resetPasswordForEmail(cleanEmail, {
          redirectTo: redirectUrl,
        });

        if (error) {
          console.error("Direct Supabase recovery failed:", error.message);
          const msgLower = error.message.toLowerCase();
          if (msgLower.includes("rate limit") || (error as any).code === "over_email_send_rate_limit") {
            setEmailError("Too many reset attempts. Please wait a few minutes before trying again.");
            toast.error("Too many reset requests. Please wait a moment.");
            return;
          }
          if (msgLower.includes("network") || msgLower.includes("fetch")) {
            setEmailError("Network connection error. Please check your connection and try again.");
            toast.error("Unable to connect to authentication server.");
            return;
          }
          toast.error(error.message || "Failed to send reset email. Please try again.");
          return;
        }
      }

      setSubmitted(true);
      toast.success(generatedLink ? "Password recovery link ready." : "Recovery instructions dispatched.");
    } catch (err: any) {
      console.error("Unexpected error in password recovery:", err);
      toast.error(err.message || "An unexpected error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  // Masked email for display privacy (e.g., a***t@domain.org)
  const maskedEmail = email
    .trim()
    .replace(/^(.)(.*)(@.*)$/, (_, first, _middle, domain) => `${first}***${domain}`);

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
        {/* LEFT PANEL — RECOVERY AREA (~42% desktop width, unboxed) */}
        <section
          aria-labelledby="recovery-heading"
          className="flex w-full lg:w-[42%] justify-center lg:justify-start lg:pr-10 xl:pr-16 py-4 lg:py-8"
        >
          <div className="w-full max-w-[400px]">
            {/* Header & Logo with Staggered Entrance */}
            <div className="animate-fade-slide-up [animation-delay:80ms]">
              <div className="flex items-center justify-between">
                <Logo size="md" />
                <div className="inline-flex items-center gap-1.5 rounded-full border border-cyan-500/20 bg-cyan-500/[0.06] px-2.5 py-0.5 text-[10px] font-mono text-cyan-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                  <span>IDENTITY RECOVERY</span>
                </div>
              </div>

              <div className="mt-8">
                <h1
                  id="recovery-heading"
                  className="text-2xl sm:text-[28px] font-bold tracking-tight text-white"
                >
                  {submitted ? "Check your email" : "Reset your password"}
                </h1>
                <p className="mt-2 text-xs sm:text-[13px] text-slate-400 leading-relaxed">
                  {submitted
                    ? "We have processed a secure password recovery link for your registered email."
                    : "Enter your registered email address to receive secure password recovery instructions."}
                </p>
              </div>
            </div>

            {submitted ? (
              /* Success Confirmation View */
              <div className="mt-7 space-y-5 animate-fade-slide-up [animation-delay:160ms]">
                <div className="clay-sunken rounded-2xl p-5 border border-teal-500/20 bg-teal-500/[0.03]">
                  <div className="flex items-start gap-3.5">
                    <CheckCircle2 className="h-5 w-5 text-teal-400 shrink-0 mt-0.5" aria-hidden="true" />
                    <div className="space-y-1.5 text-xs text-slate-300 leading-relaxed">
                      <p className="font-semibold text-slate-100">
                        {recoveryLink ? "Recovery token generated" : "Recovery link dispatched"}
                      </p>
                      <p className="text-slate-400">
                        Instructions prepared for <span className="font-mono text-cyan-300">{maskedEmail}</span>.
                      </p>
                      <p className="text-[11px] text-slate-400 pt-1">
                        {recoveryLink
                          ? "A verified single-use cryptographic reset link has been created for your account. Click below to proceed to the password reset screen."
                          : "If an account exists under this address, the link will expire in 1 hour. Please check your spam or quarantine folder if it does not arrive promptly."}
                      </p>
                    </div>
                  </div>
                </div>

                {recoveryLink && (
                  <div className="space-y-2 pt-1">
                    <a
                      href={recoveryLink}
                      className="clay-button-primary flex w-full items-center justify-center gap-2 rounded-xl py-2.5 px-4 text-xs font-semibold text-white cursor-pointer"
                    >
                      <KeyRound className="h-3.5 w-3.5" aria-hidden="true" />
                      <span>Proceed to Set New Password</span>
                      <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
                    </a>

                    <button
                      type="button"
                      onClick={async () => {
                        try {
                          await navigator.clipboard.writeText(recoveryLink);
                          setCopied(true);
                          toast.success("Recovery link copied to clipboard!");
                          setTimeout(() => setCopied(false), 2500);
                        } catch {
                          toast.error("Failed to copy link.");
                        }
                      }}
                      className="clay-sunken flex w-full items-center justify-center gap-2 rounded-xl py-2 px-4 text-[11px] font-medium text-slate-300 hover:text-white hover:border-cyan-500/40 transition-colors cursor-pointer"
                    >
                      {copied ? (
                        <>
                          <Check className="h-3.5 w-3.5 text-teal-400" />
                          <span className="text-teal-300">Link copied to clipboard</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-3.5 w-3.5 text-slate-400" />
                          <span>Copy recovery link</span>
                        </>
                      )}
                    </button>
                  </div>
                )}

                <div className="space-y-3 pt-2">
                  <button
                    type="button"
                    onClick={() => {
                      setSubmitted(false);
                      setEmail("");
                      setRecoveryLink(null);
                      setCopied(false);
                    }}
                    className="clay-sunken w-full rounded-xl py-2.5 px-4 text-xs font-medium text-slate-300 hover:text-white hover:border-cyan-500/40 transition-colors cursor-pointer"
                  >
                    Resend to a different email
                  </button>

                  <Link
                    to="/login"
                    className="clay-sunken flex w-full items-center justify-center gap-2 rounded-xl py-2.5 px-4 text-xs font-semibold text-slate-300 hover:text-white cursor-pointer"
                  >
                    <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>Back to sign in</span>
                  </Link>
                </div>
              </div>
            ) : (
              /* Email Input Form View */
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
                    Registered Email Address
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
                      onChange={(e) => {
                        setEmail(e.target.value);
                        if (emailError) setEmailError("");
                      }}
                      className="w-full bg-transparent py-2.5 pl-10 pr-4 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none"
                      aria-required="true"
                      aria-invalid={Boolean(emailError)}
                      aria-describedby={emailError ? "email-error" : undefined}
                    />
                  </div>
                  {emailError && (
                    <p id="email-error" className="text-[11px] text-rose-400 pt-0.5">
                      {emailError}
                    </p>
                  )}
                </div>

                {/* Submit Reset Button */}
                <button
                  type="submit"
                  disabled={loading}
                  className="clay-button-primary group relative flex w-full items-center justify-center gap-2 rounded-xl py-2.5 px-4 text-xs font-semibold text-white cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed mt-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin text-white" aria-hidden="true" />
                      <span>Dispatching reset link…</span>
                    </>
                  ) : (
                    <>
                      <span>Send reset link</span>
                      <ArrowRight
                        className="h-3.5 w-3.5 transition-transform duration-200 group-hover:translate-x-1"
                        aria-hidden="true"
                      />
                    </>
                  )}
                </button>

                {/* Back to Sign In Link */}
                <div className="pt-2 text-center text-xs">
                  <Link
                    to="/login"
                    className="inline-flex items-center gap-1.5 font-medium text-slate-400 hover:text-cyan-300 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-cyan-500 rounded px-1"
                  >
                    <ArrowLeft className="h-3 w-3" aria-hidden="true" />
                    <span>Back to sign in</span>
                  </Link>
                </div>
              </form>
            )}

            {/* Zero-Trust Security Footer */}
            <div className="mt-8 space-y-3.5 border-t border-white/[0.06] pt-5 animate-fade-slide-up [animation-delay:240ms]">
              <div className="flex items-center gap-2 text-[11px] text-slate-400">
                <ShieldCheck className="h-3.5 w-3.5 text-teal-400 shrink-0" aria-hidden="true" />
                <span>Encrypted cryptographic recovery tokens only</span>
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

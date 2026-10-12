import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, useEffect, type FormEvent } from "react";
import { Eye, EyeOff, Lock, ArrowRight, ShieldCheck, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { Logo } from "@/components/Logo";
import { toast } from "sonner";
import { MediaVerificationVisual } from "@/components/MediaVerificationVisual";
import { CyberBackground } from "@/components/CyberBackground";
import { supabase } from "@/lib/supabase";

export const Route = createFileRoute("/reset-password")({
  head: () => ({
    meta: [
      { title: "SatyaDrishti | Set New Password" },
      {
        name: "description",
        content: "Set a new secure password for your SatyaDrishti account.",
      },
    ],
  }),
  component: ResetPasswordPage,
});

function ResetPasswordPage() {
  const navigate = useNavigate();
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [showConfirmPw, setShowConfirmPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [checkingSession, setCheckingSession] = useState(true);
  const [hasValidSession, setHasValidSession] = useState(false);

  useEffect(() => {
    let isMounted = true;

    // 1. Initial check for active session, hash tokens or recovery params
    async function initRecovery() {
      try {
        // A. Extract access_token / refresh_token from window.location.hash
        if (typeof window !== "undefined" && window.location.hash) {
          const hashStr = window.location.hash.replace(/^#/, "");
          const hashParams = new URLSearchParams(hashStr);
          const accessToken = hashParams.get("access_token");
          const refreshToken = hashParams.get("refresh_token");

          if (accessToken) {
            const { data, error } = await supabase.auth.setSession({
              access_token: accessToken,
              refresh_token: refreshToken || "",
            });
            if (data?.session && isMounted) {
              setHasValidSession(true);
              setCheckingSession(false);
              return;
            }
          }
        }

        // B. Extract token_hash or code from window.location.search (PKCE/OTP flow)
        if (typeof window !== "undefined" && window.location.search) {
          const searchParams = new URLSearchParams(window.location.search);
          const tokenHash = searchParams.get("token_hash");
          const code = searchParams.get("code");

          if (tokenHash) {
            const { data, error } = await supabase.auth.verifyOtp({
              token_hash: tokenHash,
              type: "recovery",
            });
            if (data?.session && isMounted) {
              setHasValidSession(true);
              setCheckingSession(false);
              return;
            }
          } else if (code) {
            const { data, error } = await supabase.auth.exchangeCodeForSession(code);
            if (data?.session && isMounted) {
              setHasValidSession(true);
              setCheckingSession(false);
              return;
            }
          }
        }

        // C. Standard check for existing active session
        const { data: { session } } = await supabase.auth.getSession();
        if (isMounted) {
          if (session) {
            setHasValidSession(true);
          }
          setCheckingSession(false);
        }
      } catch (err) {
        console.warn("Session retrieval check:", err);
        if (isMounted) setCheckingSession(false);
      }
    }

    initRecovery();

    // 2. Listen for PASSWORD_RECOVERY auth state event
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (!isMounted) return;

      if (event === "PASSWORD_RECOVERY" || session) {
        setHasValidSession(true);
        setCheckingSession(false);
      } else if (event === "SIGNED_OUT") {
        setHasValidSession(false);
      }
    });

    return () => {
      isMounted = false;
      subscription.unsubscribe();
    };
  }, []);

  // Password validation rules
  const hasMinLength = password.length >= 8;
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasNumber = /[0-9]/.test(password);
  const passwordsMatch = password.length > 0 && password === confirmPassword;
  const isFormValid = hasMinLength && hasUpper && hasLower && hasNumber && passwordsMatch;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();

    if (loading) return;

    if (!hasMinLength) {
      setErrorMsg("Password must be at least 8 characters long.");
      return;
    }
    if (!hasUpper || !hasLower) {
      setErrorMsg("Password must include both uppercase and lowercase letters.");
      return;
    }
    if (!hasNumber) {
      setErrorMsg("Password must include at least one number.");
      return;
    }
    if (!passwordsMatch) {
      setErrorMsg("Passwords do not match. Please verify both fields.");
      return;
    }

    setLoading(true);
    setErrorMsg("");

    try {
      // Supabase Auth updateUser updates the current authenticated recovery user's password
      const { data, error } = await supabase.auth.updateUser({
        password: password,
      });

      if (error) {
        console.error("Password update error:", error.message);

        const msgLower = error.message.toLowerCase();
        if (msgLower.includes("same as") || msgLower.includes("different")) {
          setErrorMsg("New password must be different from your previous password.");
        } else if (msgLower.includes("session") || msgLower.includes("expired") || msgLower.includes("jwt")) {
          setErrorMsg("Your password recovery session has expired. Please request a new reset link.");
        } else {
          setErrorMsg(error.message || "Failed to update password. Please try again.");
        }

        toast.error(error.message || "Password update failed.");
        return;
      }

      if (data?.user) {
        setSuccess(true);
        toast.success("Password successfully updated!");
        // Clear active session to enforce explicit sign-in with the new credentials
        try {
          await supabase.auth.signOut();
        } catch {
          // ignore signout fallback
        }
      }
    } catch (err: any) {
      console.error("Unexpected error during password reset:", err);
      setErrorMsg(err.message || "An unexpected error occurred. Please try again.");
      toast.error("Failed to update password.");
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
        {/* LEFT PANEL — PASSWORD RESET AREA (~42% desktop width, unboxed) */}
        <section
          aria-labelledby="reset-heading"
          className="flex w-full lg:w-[42%] justify-center lg:justify-start lg:pr-10 xl:pr-16 py-4 lg:py-8"
        >
          <div className="w-full max-w-[400px]">
            {/* Header & Logo with Staggered Entrance */}
            <div className="animate-fade-slide-up [animation-delay:80ms]">
              <div className="flex items-center justify-between">
                <Logo size="md" />
                <div className="inline-flex items-center gap-1.5 rounded-full border border-cyan-500/20 bg-cyan-500/[0.06] px-2.5 py-0.5 text-[10px] font-mono text-cyan-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                  <span>SECURE CREDENTIALS</span>
                </div>
              </div>

              <div className="mt-8">
                <h1
                  id="reset-heading"
                  className="text-2xl sm:text-[28px] font-bold tracking-tight text-white"
                >
                  {success ? "Password updated" : "Set new password"}
                </h1>
                <p className="mt-2 text-xs sm:text-[13px] text-slate-400 leading-relaxed">
                  {success
                    ? "Your SatyaDrishti credentials have been securely updated. You can now sign in."
                    : "Create a strong, unique password for your forensic analyst account."}
                </p>
              </div>
            </div>

            {checkingSession ? (
              /* Loading Session Verification */
              <div className="mt-8 flex items-center gap-3 rounded-2xl p-5 clay-sunken animate-fade-slide-up">
                <Loader2 className="h-5 w-5 animate-spin text-cyan-400 shrink-0" />
                <span className="text-xs text-slate-400">Verifying secure recovery token…</span>
              </div>
            ) : success ? (
              /* Success View */
              <div className="mt-7 space-y-5 animate-fade-slide-up [animation-delay:160ms]">
                <div className="clay-sunken rounded-2xl p-5 border border-teal-500/25 bg-teal-500/[0.04]">
                  <div className="flex items-start gap-3.5">
                    <CheckCircle2 className="h-5 w-5 text-teal-400 shrink-0 mt-0.5" aria-hidden="true" />
                    <div className="space-y-1.5 text-xs text-slate-300 leading-relaxed">
                      <p className="font-semibold text-slate-100">
                        Password updated successfully
                      </p>
                      <p className="text-slate-400">
                        All active sessions have been secured. Use your new credentials on your next sign in.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="pt-2">
                  <Link
                    to="/login"
                    className="clay-button-primary flex w-full items-center justify-center gap-2 rounded-xl py-2.5 px-4 text-xs font-semibold text-white cursor-pointer"
                  >
                    <span>Sign in to SatyaDrishti</span>
                    <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
                  </Link>
                </div>
              </div>
            ) : !hasValidSession ? (
              /* No Active Recovery Session Notice */
              <div className="mt-7 space-y-5 animate-fade-slide-up [animation-delay:160ms]">
                <div className="clay-sunken rounded-2xl p-5 border border-amber-500/25 bg-amber-500/[0.04]">
                  <div className="flex items-start gap-3.5">
                    <AlertCircle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" aria-hidden="true" />
                    <div className="space-y-1 text-xs text-slate-300 leading-relaxed">
                      <p className="font-semibold text-slate-100">
                        Invalid or expired link
                      </p>
                      <p className="text-slate-400">
                        No active password recovery session was detected. Password reset links are single-use and expire after 1 hour.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3 pt-1">
                  <Link
                    to="/forgot-password"
                    className="clay-button-primary flex w-full items-center justify-center gap-2 rounded-xl py-2.5 px-4 text-xs font-semibold text-white cursor-pointer"
                  >
                    <span>Request new recovery link</span>
                    <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
                  </Link>

                  <div className="text-center pt-1">
                    <Link
                      to="/login"
                      className="text-xs font-medium text-slate-400 hover:text-cyan-300 transition-colors"
                    >
                      Return to sign in
                    </Link>
                  </div>
                </div>
              </div>
            ) : (
              /* New Password Entry Form */
              <form
                onSubmit={onSubmit}
                noValidate
                className="mt-7 space-y-4 animate-fade-slide-up [animation-delay:160ms]"
              >
                {/* New Password Field */}
                <div className="space-y-1.5">
                  <label
                    htmlFor="password"
                    className="block text-xs font-medium text-slate-300"
                  >
                    New Password
                  </label>
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
                      autoComplete="new-password"
                      placeholder="Minimum 8 characters"
                      value={password}
                      onChange={(e) => {
                        setPassword(e.target.value);
                        if (errorMsg) setErrorMsg("");
                      }}
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

                {/* Confirm Password Field */}
                <div className="space-y-1.5">
                  <label
                    htmlFor="confirmPassword"
                    className="block text-xs font-medium text-slate-300"
                  >
                    Confirm New Password
                  </label>
                  <div className="clay-sunken relative flex items-center rounded-xl transition-all duration-200">
                    <Lock
                      className="pointer-events-none absolute left-3.5 h-4 w-4 text-slate-400"
                      aria-hidden="true"
                    />
                    <input
                      id="confirmPassword"
                      name="confirmPassword"
                      type={showConfirmPw ? "text" : "password"}
                      required
                      autoComplete="new-password"
                      placeholder="Repeat new password"
                      value={confirmPassword}
                      onChange={(e) => {
                        setConfirmPassword(e.target.value);
                        if (errorMsg) setErrorMsg("");
                      }}
                      className="w-full bg-transparent py-2.5 pl-10 pr-10 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none"
                      aria-required="true"
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPw((v) => !v)}
                      className="absolute right-2.5 flex h-7 w-7 items-center justify-center rounded-lg text-slate-400 transition-colors hover:text-slate-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-cyan-500 cursor-pointer"
                      aria-label={showConfirmPw ? "Hide password" : "Show password"}
                    >
                      {showConfirmPw ? (
                        <EyeOff className="h-4 w-4" aria-hidden="true" />
                      ) : (
                        <Eye className="h-4 w-4" aria-hidden="true" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Password Requirements Guide */}
                <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3 text-[11px] space-y-1 text-slate-400">
                  <div className="flex items-center gap-1.5">
                    <span className={`h-1.5 w-1.5 rounded-full ${hasMinLength ? "bg-teal-400" : "bg-slate-600"}`} />
                    <span className={hasMinLength ? "text-slate-200" : ""}>At least 8 characters</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className={`h-1.5 w-1.5 rounded-full ${hasUpper && hasLower ? "bg-teal-400" : "bg-slate-600"}`} />
                    <span className={hasUpper && hasLower ? "text-slate-200" : ""}>Uppercase & lowercase letters</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className={`h-1.5 w-1.5 rounded-full ${hasNumber ? "bg-teal-400" : "bg-slate-600"}`} />
                    <span className={hasNumber ? "text-slate-200" : ""}>At least one number</span>
                  </div>
                  {confirmPassword.length > 0 && (
                    <div className="flex items-center gap-1.5">
                      <span className={`h-1.5 w-1.5 rounded-full ${passwordsMatch ? "bg-teal-400" : "bg-rose-500"}`} />
                      <span className={passwordsMatch ? "text-slate-200" : "text-rose-400"}>
                        {passwordsMatch ? "Passwords match" : "Passwords do not match"}
                      </span>
                    </div>
                  )}
                </div>

                {errorMsg && (
                  <p className="text-[11px] text-rose-400 pt-0.5">
                    {errorMsg}
                  </p>
                )}

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={loading || !isFormValid}
                  className="clay-button-primary group relative flex w-full items-center justify-center gap-2 rounded-xl py-2.5 px-4 text-xs font-semibold text-white cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed mt-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin text-white" aria-hidden="true" />
                      <span>Updating password…</span>
                    </>
                  ) : (
                    <>
                      <span>Set new password</span>
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
                    className="font-medium text-slate-400 hover:text-cyan-300 transition-colors"
                  >
                    Cancel and return to sign in
                  </Link>
                </div>
              </form>
            )}

            {/* Zero-Trust Security Footer */}
            <div className="mt-8 space-y-3.5 border-t border-white/[0.06] pt-5 animate-fade-slide-up [animation-delay:240ms]">
              <div className="flex items-center gap-2 text-[11px] text-slate-400">
                <ShieldCheck className="h-3.5 w-3.5 text-teal-400 shrink-0" aria-hidden="true" />
                <span>Zero-Trust Cryptographic Pipeline Active</span>
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

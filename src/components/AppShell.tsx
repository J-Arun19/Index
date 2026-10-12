import { Link, useLocation, useNavigate } from "@tanstack/react-router";
import { LayoutDashboard, ScanSearch, Activity, FileCheck2, LogOut } from "lucide-react";
import type { ReactNode } from "react";
import { Logo } from "./Logo";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { authStore, useAuthState } from "@/lib/auth-store";
import { analysisStore } from "@/lib/analysis-store";
import { supabase } from "@/lib/supabase";

const nav = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analysis", label: "New Analysis", icon: ScanSearch },
  { to: "/monitoring", label: "Real-Time Monitoring", icon: Activity },
  { to: "/result", label: "Latest Result", icon: FileCheck2 },
] as const;


export function AppShell({
  children,
  title,
  description,
  actions,
}: {
  children: ReactNode;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { user } = useAuthState();

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut();
    } catch (e) {
      console.error("Supabase signOut error:", e);
    }
    authStore.signOut();
    navigate({ to: "/login" });
  };

  return (
    <div className="flex min-h-screen bg-background text-foreground font-sans antialiased">
      {/* Sidebar */}
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-border/80 bg-sidebar px-3 py-5 lg:flex">
        <Link to="/dashboard" className="px-2 py-1">
          <Logo />
        </Link>

        <div className="mt-6 px-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          Navigation
        </div>

        <nav className="mt-2 flex flex-col gap-0.5">
          {nav.map((item) => {
            const active = pathname === item.to;
            const Icon = item.icon;
            return (
              <Link
                key={item.to}
                to={item.to}
                onClick={() => {
                  if (item.to === "/result") {
                    analysisStore.selectReport(null);
                  }
                }}
                className={cn(
                  "group flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs transition-colors",
                  active
                    ? "bg-slate-800/90 text-slate-100 font-medium border-l-2 border-sky-500 pl-2"
                    : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200",
                )}
              >
                <Icon className={cn("h-4 w-4 shrink-0", active ? "text-sky-400" : "text-slate-400 group-hover:text-slate-200")} />
                <span className="truncate">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto space-y-3 pt-4 border-t border-border/60">
          {/* System Status Block */}
          <div className="rounded-md border border-border/60 bg-slate-900/60 p-2.5 text-[11px] text-slate-400">
            <div className="flex items-center gap-2 font-medium text-slate-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              <span>System Status</span>
            </div>
            <p className="mt-1 text-[10px] text-slate-400">6 Analysis Modules Operational</p>
          </div>

          {/* User Account */}
          {user && (
            <div className="flex items-center gap-2.5 px-1 py-1">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded bg-slate-800 text-xs font-semibold text-sky-400 border border-slate-700">
                {user.name.charAt(0).toUpperCase()}
              </div>
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs font-medium text-slate-200">{user.name}</div>
                <div className="truncate text-[11px] text-slate-400">{user.email}</div>
              </div>
            </div>
          )}

          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start h-8 text-xs text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
            onClick={handleSignOut}
          >
            <LogOut className="mr-2 h-3.5 w-3.5" /> Sign out
          </Button>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 border-b border-border/80 bg-background/90 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6 lg:px-8">
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold tracking-tight text-slate-100 sm:text-lg">
                {title}
              </h1>
              {description && (
                <p className="truncate text-xs text-slate-400">
                  {description}
                </p>
              )}
            </div>
            {actions && <div className="shrink-0">{actions}</div>}
          </div>
          {/* Mobile Nav */}
          <div className="mx-auto flex max-w-6xl gap-1 overflow-x-auto px-4 pb-2 sm:px-6 lg:hidden">
            {nav.map((item) => {
              const active = pathname === item.to;
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  onClick={() => {
                    if (item.to === "/result") {
                      analysisStore.selectReport(null);
                    }
                  }}
                  className={cn(
                    "shrink-0 rounded-md px-2.5 py-1 text-xs",
                    active
                      ? "bg-slate-800 text-slate-100 font-medium"
                      : "text-slate-400 hover:text-slate-200",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </header>

        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 lg:px-8">
          {children}
        </main>
      </div>
    </div>
  );
}
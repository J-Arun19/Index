import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import {
  Activity,
  ShieldCheck,
  AlertTriangle,
  Gauge,
  Plus,
  ArrowUpRight,
  ImageIcon,
  Video,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useAnalysisState, analysisStore, AnalysisVerdict, formatBackendAnalysis } from "@/lib/analysis-store";
import { cn } from "@/lib/utils";
import { useAuthState } from "@/lib/auth-store";

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "SatyaDrishti | Dashboard" },
      { name: "description", content: "Overview of your recent deepfake analyses." },
    ],
  }),
  component: DashboardPage,
});

function DashboardPage() {
  const navigate = useNavigate();
  const { history } = useAnalysisState();
  const { user, isInitialized } = useAuthState();

  useEffect(() => {
    if (isInitialized && !user) {
      navigate({ to: "/login" });
    }
  }, [user, isInitialized, navigate]);

  useEffect(() => {
    if (!isInitialized) return;
    if (!user) {
      analysisStore.clear();
      return;
    }

    analysisStore.setUser(user.id);

    const fetchHistory = async () => {
      try {
        const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
        const res = await fetch(`${API_URL}/api/history?user_id=${encodeURIComponent(String(user.id))}`);
        if (res.ok) {
          const data = await res.json();
          if (data.status === "success" && Array.isArray(data.history)) {
            const formatted = data.history.map((item: any) => formatBackendAnalysis(item, API_URL));
            analysisStore.setHistory(formatted, user.id);
          } else {
            analysisStore.setHistory([], user.id);
          }
        } else {
          analysisStore.setHistory([], user.id);
        }
      } catch (err) {
        console.error("Failed to fetch analysis history:", err);
        analysisStore.setHistory([], user.id);
      }
    };
    fetchHistory();
  }, [user, isInitialized]);

  const firstName = user?.name ? user.name.split(" ")[0] : "Researcher";

  const total = history.length;
  const real = history.filter((h) => h.verdict === "REAL" || h.verdict === "LIKELY_REAL").length;
  const fake = history.filter((h) => h.verdict === "DEEPFAKE" || h.verdict === "FAKE").length;
  const uncertain = history.filter((h) => h.verdict === "UNCERTAIN").length;
  const accuracy =
    total === 0
      ? "—"
      : `${Math.round(
          history.reduce((s, h) => s + h.confidence, 0) / total,
        )}%`;

  const cards = [
    {
      label: "Total analyses",
      value: total,
      icon: Activity,
      tone: "text-foreground",
    },
    {
      label: "Real media",
      value: real,
      icon: ShieldCheck,
      tone: "text-emerald-400",
    },
    {
      label: "Deepfakes detected",
      value: fake,
      icon: AlertTriangle,
      tone: "text-rose-400",
    },
    {
      label: "Avg. confidence",
      value: accuracy,
      icon: Gauge,
      tone: "text-primary",
    },
  ];

  return (
    <AppShell
      title="Dashboard"
      description="Overview of media authenticity operations and analysis history."
      actions={
        <Button asChild size="sm" className="bg-sky-600 text-white hover:bg-sky-500 font-medium">
          <Link to="/analysis">
            <Plus className="mr-1.5 h-3.5 w-3.5" /> Execute New Analysis
          </Link>
        </Button>
      }
    >
      {/* Workspace Header Panel */}
      <section className="mb-6 rounded-md border border-border/80 bg-slate-900/80 p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-sky-400">
              Operational Workspace
            </div>
            <h2 className="mt-1 text-base font-semibold tracking-tight text-slate-100 sm:text-lg">
              {user ? `Auditor Workspace — ${firstName}` : "Media Authenticity Forensics"}
            </h2>
            <p className="mt-0.5 text-xs text-slate-400">
              Submit digital images or video streams for multi-agent forensic verification.
            </p>
          </div>
          <div className="shrink-0">
            <Button asChild variant="outline" size="sm" className="h-8 text-xs border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-slate-100">
              <Link to="/analysis">
                Launch Pipeline <ArrowUpRight className="ml-1 h-3.5 w-3.5 text-slate-400" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Metric Modules Row */}
      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((c) => {
          const Icon = c.icon;
          return (
            <div
              key={c.label}
              className="rounded-md border border-border/80 bg-slate-900/60 p-4 transition-colors hover:border-slate-700"
            >
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-medium uppercase tracking-wider text-[10px]">
                  {c.label}
                </span>
                <Icon className={cn("h-4 w-4", c.tone)} />
              </div>
              <div className="mt-2 text-2xl font-bold tracking-tight text-slate-100 font-mono">
                {c.value}
              </div>
            </div>
          );
        })}
      </section>

      {/* Recent Analyses Data Table */}
      <section className="mt-6 rounded-md border border-border/80 bg-slate-900/60">
        <div className="flex items-center justify-between border-b border-border/80 px-4 py-3">
          <div>
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">Recent Analyses</h3>
            <p className="text-[11px] text-slate-400">
              Audit log of completed media verification tasks.
            </p>
          </div>
          <Button asChild size="sm" variant="ghost" className="h-7 text-xs text-slate-400 hover:text-slate-200">
            <Link to="/analysis">New Analysis</Link>
          </Button>
        </div>

        {history.length === 0 ? (
          <div className="px-4 py-12 text-center">
            <div className="mx-auto grid h-10 w-10 place-items-center rounded border border-border/80 bg-slate-800/60 text-slate-400">
              <Activity className="h-4 w-4" />
            </div>
            <p className="mt-3 text-xs font-medium text-slate-300">No media analyses logged</p>
            <p className="mt-1 text-[11px] text-slate-400">
              Execute your first media analysis to populate the audit record.
            </p>
            <Button asChild size="sm" className="mt-3 h-8 text-xs bg-sky-600 hover:bg-sky-500 text-white">
              <Link to="/analysis">Execute Analysis</Link>
            </Button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="border-b border-border/80 bg-slate-950/40 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-4 py-2.5">File Name</th>
                  <th className="px-4 py-2.5">Media Type</th>
                  <th className="px-4 py-2.5">Verdict</th>
                  <th className="px-4 py-2.5">Confidence</th>
                  <th className="px-4 py-2.5">Timestamp</th>
                  <th className="px-4 py-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60">
                {history.map((h) => (
                  <tr
                    key={h.id}
                    className="hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="max-w-[240px] truncate px-4 py-2.5 font-medium text-slate-200 font-mono">
                      {h.fileName}
                    </td>
                    <td className="px-4 py-2.5 text-slate-400">
                      <span className="inline-flex items-center gap-1.5 uppercase text-[10px] tracking-wide">
                        {h.fileType === "image" ? (
                          <ImageIcon className="h-3 w-3 text-slate-400" />
                        ) : (
                          <Video className="h-3 w-3 text-slate-400" />
                        )}
                        {h.fileType}
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-semibold border",
                          h.verdict === "NOT_SUITABLE"
                            ? "bg-slate-800 text-slate-300 border-slate-700"
                            : h.verdict === "REAL" || h.verdict === "LIKELY_REAL"
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                            : h.verdict === "UNCERTAIN"
                            ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                            : "bg-rose-500/10 text-rose-400 border-rose-500/20",
                        )}
                      >
                        <span className={cn(
                          "h-1.5 w-1.5 rounded-full",
                          h.verdict === "NOT_SUITABLE"
                            ? "bg-slate-400"
                            : h.verdict === "REAL" || h.verdict === "LIKELY_REAL"
                            ? "bg-emerald-400"
                            : h.verdict === "UNCERTAIN"
                            ? "bg-amber-400"
                            : "bg-rose-400"
                        )} />
                        {h.verdict === "NOT_SUITABLE" ? "NOT SUITABLE" : h.verdict}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-slate-300 font-mono">
                      {h.verdict === "NOT_SUITABLE" ? "N/A" : `${h.confidence}%`}
                    </td>
                    <td className="px-4 py-2.5 text-slate-400">
                      {formatRelative(h.createdAt)}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <Button
                        asChild
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-xs text-sky-400 hover:text-sky-300 hover:bg-slate-800"
                        onClick={() => {
                          analysisStore.selectReport(h);
                        }}
                      >
                        <Link to="/result" search={{ id: h.id }}>
                          View Report
                        </Link>
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </AppShell>
  );
}

function formatRelative(ts: number) {
  const diff = Date.now() - ts;
  const m = Math.floor(diff / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}
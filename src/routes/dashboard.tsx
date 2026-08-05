import { createFileRoute, Link } from "@tanstack/react-router";
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
import { useAnalysisState } from "@/lib/analysis-store";
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
  const { history } = useAnalysisState();
  const { user } = useAuthState();

  const firstName = user?.name ? user.name.split(" ")[0] : "Researcher";

  const total = history.length;
  const real = history.filter((h) => h.verdict === "REAL").length;
  const fake = history.filter((h) => h.verdict === "DEEPFAKE").length;
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
      description={`Welcome back, ${firstName}! Here is an overview of your analyses.`}
      actions={
        <Button asChild>
          <Link to="/analysis">
            <Plus className="mr-1.5 h-4 w-4" /> Start new analysis
          </Link>
        </Button>
      }
    >
      {/* Welcome */}
      <section className="mb-6 rounded-2xl border border-border/70 bg-card p-6">
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4 sm:flex sm:justify-between">
          <div className="min-w-0">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">
              Workspace
            </div>
            <h2 className="mt-1 truncate text-xl font-semibold tracking-tight">
              {user ? `Welcome back, ${firstName}!` : "Ready to inspect new media."}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Upload an image or video and let the agents run a full detection
              pipeline.
            </p>
          </div>
          <div className="hidden shrink-0 sm:block">
            <Button asChild variant="secondary">
              <Link to="/analysis">
                Open analysis <ArrowUpRight className="ml-1 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Summary cards */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((c) => {
          const Icon = c.icon;
          return (
            <div
              key={c.label}
              className="rounded-2xl border border-border/70 bg-card p-5"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  {c.label}
                </span>
                <Icon className={cn("h-4 w-4", c.tone)} />
              </div>
              <div className="mt-3 text-2xl font-semibold tracking-tight">
                {c.value}
              </div>
            </div>
          );
        })}
      </section>

      {/* Recent table */}
      <section className="mt-8 rounded-2xl border border-border/70 bg-card">
        <div className="flex items-center justify-between border-b border-border/70 px-5 py-4">
          <div>
            <h3 className="text-sm font-semibold">Recent analyses</h3>
            <p className="text-xs text-muted-foreground">
              Your latest 20 detection runs.
            </p>
          </div>
          <Button asChild size="sm" variant="ghost">
            <Link to="/analysis">New</Link>
          </Button>
        </div>

        {history.length === 0 ? (
          <div className="px-5 py-12 text-center">
            <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-border/70 bg-secondary/40 text-muted-foreground">
              <Activity className="h-5 w-5" />
            </div>
            <p className="mt-3 text-sm font-medium">No analyses yet</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Run your first analysis to populate this table.
            </p>
            <Button asChild size="sm" className="mt-4">
              <Link to="/analysis">Start new analysis</Link>
            </Button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                <tr className="border-b border-border/70">
                  <th className="px-5 py-3 font-medium">File</th>
                  <th className="px-5 py-3 font-medium">Type</th>
                  <th className="px-5 py-3 font-medium">Verdict</th>
                  <th className="px-5 py-3 font-medium">Confidence</th>
                  <th className="px-5 py-3 font-medium">When</th>
                  <th className="px-5 py-3" />
                </tr>
              </thead>
              <tbody>
                {history.map((h) => (
                  <tr
                    key={h.id}
                    className="border-b border-border/60 last:border-0 hover:bg-secondary/30"
                  >
                    <td className="max-w-[220px] truncate px-5 py-3 font-medium">
                      {h.fileName}
                    </td>
                    <td className="px-5 py-3 text-muted-foreground">
                      <span className="inline-flex items-center gap-1.5">
                        {h.fileType === "image" ? (
                          <ImageIcon className="h-3.5 w-3.5" />
                        ) : (
                          <Video className="h-3.5 w-3.5" />
                        )}
                        {h.fileType}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <Badge
                        variant="outline"
                        className={cn(
                          "border-transparent",
                          h.verdict === "REAL"
                            ? "bg-emerald-500/10 text-emerald-400"
                            : "bg-rose-500/10 text-rose-400",
                        )}
                      >
                        {h.verdict}
                      </Badge>
                    </td>
                    <td className="px-5 py-3 text-muted-foreground">
                      {h.confidence}%
                    </td>
                    <td className="px-5 py-3 text-muted-foreground">
                      {formatRelative(h.createdAt)}
                    </td>
                    <td className="px-5 py-3 text-right">
                      <Button asChild size="sm" variant="ghost">
                        <Link to="/result">View</Link>
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
import { createFileRoute, Link } from "@tanstack/react-router";
import {
  Download,
  ShieldCheck,
  AlertTriangle,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { useAnalysisState } from "@/lib/analysis-store";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/result")({
  head: () => ({
    meta: [
      { title: "SatyaDrishti | Detection Results" },
      {
        name: "description",
        content: "Detailed deepfake detection result with confidence and agent trace.",
      },
    ],
  }),
  component: ResultPage,
});

function ResultPage() {
  const { latest } = useAnalysisState();

  if (!latest) {
    return (
      <AppShell title="Analysis result">
        <div className="mx-auto max-w-md rounded-2xl border border-border/70 bg-card p-10 text-center">
          <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-border/70 bg-secondary/40 text-primary">
            <Sparkles className="h-5 w-5" />
          </div>
          <h2 className="mt-4 text-lg font-semibold">No result yet</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Run an analysis to see the report here.
          </p>
          <Button asChild className="mt-5">
            <Link to="/analysis">Start new analysis</Link>
          </Button>
        </div>
      </AppShell>
    );
  }

  const isFake = latest.verdict === "DEEPFAKE";

  return (
    <AppShell
      title="Analysis result"
      description={latest.fileName}
      actions={
        <div className="flex gap-2">
          <Button asChild variant="secondary" size="sm">
            <Link to="/analysis">
              <RotateCcw className="mr-1.5 h-4 w-4" /> New analysis
            </Link>
          </Button>
          <Button size="sm" onClick={() => window.print()}>
            <Download className="mr-1.5 h-4 w-4" /> Download PDF
          </Button>
        </div>
      }
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        {/* Media preview */}
        <div className="rounded-2xl border border-border/70 bg-card p-4">
          <div className="overflow-hidden rounded-xl border border-border/70 bg-black/40">
            {latest.fileType === "image" ? (
              <img
                src={latest.mediaUrl}
                alt={latest.fileName}
                className="mx-auto max-h-[520px] w-full object-contain"
              />
            ) : (
              <video
                src={latest.mediaUrl}
                controls
                className="mx-auto max-h-[520px] w-full"
              />
            )}
          </div>
          <div className="mt-3 flex items-center justify-between px-1 text-xs text-muted-foreground">
            <span className="truncate">{latest.fileName}</span>
            <span className="uppercase tracking-wider">{latest.fileType}</span>
          </div>
        </div>

        {/* Verdict + confidence + explanation */}
        <div className="space-y-6">
          <div
            className={cn(
              "rounded-2xl border p-6",
              isFake
                ? "border-rose-500/30 bg-rose-500/5"
                : "border-emerald-500/30 bg-emerald-500/5",
            )}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  Verdict
                </div>
                <div
                  className={cn(
                    "mt-1 inline-flex items-center gap-2 text-3xl font-semibold tracking-tight sm:text-4xl",
                    isFake ? "text-rose-400" : "text-emerald-400",
                  )}
                >
                  {isFake ? (
                    <AlertTriangle className="h-7 w-7" />
                  ) : (
                    <ShieldCheck className="h-7 w-7" />
                  )}
                  {latest.verdict}
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  {isFake
                    ? "The pipeline flagged this media as synthetically generated."
                    : "The pipeline found this media to be authentic."}
                </p>
              </div>
              <ConfidenceRing value={latest.confidence} tone={isFake ? "danger" : "ok"} />
            </div>
          </div>

          <div className="rounded-2xl border border-border/70 bg-card p-6">
            <h3 className="text-sm font-semibold">Explanation</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              {latest.explanation}
            </p>
          </div>

          <div className="rounded-2xl border border-border/70 bg-card p-6">
            <h3 className="text-sm font-semibold">Agent summary</h3>
            <ul className="mt-3 divide-y divide-border/60">
              {latest.agents.map((a) => (
                <li key={a.name} className="grid grid-cols-[140px_minmax(0,1fr)] gap-4 py-3 text-sm">
                  <span className="font-medium">{a.name}</span>
                  <span className="min-w-0 text-muted-foreground">{a.summary}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

function ConfidenceRing({
  value,
  tone,
}: {
  value: number;
  tone: "ok" | "danger";
}) {
  const r = 34;
  const c = 2 * Math.PI * r;
  const offset = c - (value / 100) * c;
  const color = tone === "ok" ? "oklch(0.72 0.16 155)" : "oklch(0.68 0.2 20)";
  return (
    <div className="relative h-24 w-24 shrink-0">
      <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90">
        <circle
          cx="40"
          cy="40"
          r={r}
          stroke="oklch(1 0 0 / 10%)"
          strokeWidth="6"
          fill="none"
        />
        <circle
          cx="40"
          cy="40"
          r={r}
          stroke={color}
          strokeWidth="6"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="text-lg font-semibold">{value}%</div>
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            confidence
          </div>
        </div>
      </div>
    </div>
  );
}
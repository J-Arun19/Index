import { ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";

export function Logo({
  className,
  showText = true,
  size = "md",
}: {
  className?: string;
  showText?: boolean;
  size?: "sm" | "md" | "lg";
}) {
  const box = size === "lg" ? "h-11 w-11" : size === "sm" ? "h-8 w-8" : "h-9 w-9";
  const icon = size === "lg" ? "h-6 w-6" : size === "sm" ? "h-4 w-4" : "h-5 w-5";
  const title = size === "lg" ? "text-xl" : "text-base";

  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <div
        className={cn(
          "grid shrink-0 place-items-center rounded-xl border border-primary/30 bg-primary/10 text-primary",
          box,
        )}
      >
        <ShieldCheck className={icon} strokeWidth={2.2} />
      </div>
      {showText && (
        <div className="min-w-0 leading-tight">
          <div className={cn("truncate font-semibold tracking-tight", title)}>
            SatyaDrishti
          </div>
          <div className="truncate text-[11px] text-muted-foreground">
            Agentic AI Deepfake Detection
          </div>
        </div>
      )}
    </div>
  );
}
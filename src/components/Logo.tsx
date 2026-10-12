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
  const box = size === "lg" ? "h-10 w-10" : size === "sm" ? "h-7 w-7" : "h-8 w-8";
  const icon = size === "lg" ? "h-5 w-5" : size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4";
  const title = size === "lg" ? "text-lg" : "text-sm";

  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <div
        className={cn(
          "grid shrink-0 place-items-center rounded-md border border-sky-500/30 bg-sky-500/10 text-sky-400",
          box,
        )}
      >
        <ShieldCheck className={icon} strokeWidth={2} />
      </div>
      {showText && (
        <div className="min-w-0 leading-tight">
          <div className={cn("truncate font-semibold tracking-tight text-slate-100", title)}>
            SatyaDrishti
          </div>
          <div className="truncate text-[11px] font-medium text-slate-400">
            Media Authenticity Forensics
          </div>
        </div>
      )}
    </div>
  );
}
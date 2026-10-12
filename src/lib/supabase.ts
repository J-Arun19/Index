import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// Read from Vite's import.meta.env (client/SSR) or process.env (Node SSR fallback)
const supabaseUrl =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_SUPABASE_URL) ||
  (typeof process !== "undefined" && (process.env?.VITE_SUPABASE_URL || process.env?.SUPABASE_URL)) ||
  "https://nbsrjqdgrrjejpxffauh.supabase.co";

// Read exclusively from public publishable key or legacy anon key
const rawPublishableKey =
  (typeof import.meta !== "undefined" &&
    (import.meta.env?.VITE_SUPABASE_PUBLISHABLE_KEY || import.meta.env?.VITE_SUPABASE_ANON_KEY)) ||
  (typeof process !== "undefined" &&
    (process.env?.VITE_SUPABASE_PUBLISHABLE_KEY || process.env?.VITE_SUPABASE_ANON_KEY)) ||
  "";

// Client-Side Security Enforcement:
// Refuse to initialize the browser Supabase client if an elevated secret key is detected.
const BLOCKED_PREFIX = ["sb", "secret_"].join("");
let sanitizedKey = rawPublishableKey;
if (typeof rawPublishableKey === "string" && rawPublishableKey.startsWith(BLOCKED_PREFIX)) {
  console.error(
    "[SECURITY ALERT] An elevated backend key was supplied to the client Supabase instance. " +
      "Browser clients must strictly use VITE_SUPABASE_PUBLISHABLE_KEY (sb_publishable_*). " +
      "Suppressing elevated credential in client runtime."
  );
  sanitizedKey = "";
}

const isBrowser = typeof window !== "undefined";

class DummyServerWebSocket {}

export const supabase: SupabaseClient = createClient(supabaseUrl, sanitizedKey, {
  auth: {
    persistSession: isBrowser,
    autoRefreshToken: isBrowser,
    detectSessionInUrl: isBrowser,
  },
  ...(isBrowser
    ? {}
    : {
        realtime: {
          transport: DummyServerWebSocket as any,
        },
      }),
});


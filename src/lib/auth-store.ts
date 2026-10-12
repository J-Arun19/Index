import { useSyncExternalStore } from "react";
import { supabase } from "./supabase";
import { analysisStore } from "./analysis-store";

export interface User {
  id: string | number;
  name: string;
  email: string;
}

type AuthState = {
  user: User | null;
  isAuthenticated: boolean;
  isInitialized: boolean;
};

const IS_SERVER = typeof window === "undefined";

function parseStoredUser(): User | null {
  if (IS_SERVER) return null;
  try {
    const saved = localStorage.getItem("satyadrishti_user");
    if (saved) {
      return JSON.parse(saved);
    }
  } catch (e) {
    console.error("Failed to parse cached auth state:", e);
  }
  return null;
}

const initialCachedUser = parseStoredUser();
if (initialCachedUser?.id) {
  analysisStore.setUser(initialCachedUser.id);
}

let state: AuthState = {
  user: initialCachedUser,
  isAuthenticated: !!initialCachedUser,
  isInitialized: false,
};

const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

// Client-side session synchronization with Supabase Auth
if (!IS_SERVER) {
  // 1. Listen for Supabase auth state transitions
  supabase.auth.onAuthStateChange((event, session) => {
    if (session?.user) {
      const userObj: User = {
        id: session.user.id,
        name: session.user.user_metadata?.full_name || session.user.email?.split("@")[0] || "Forensic Analyst",
        email: session.user.email || "",
      };
      state = { user: userObj, isAuthenticated: true, isInitialized: true };
      localStorage.setItem("satyadrishti_user", JSON.stringify(userObj));
      analysisStore.setUser(userObj.id);
      emit();
    } else if (event === "SIGNED_OUT") {
      state = { user: null, isAuthenticated: false, isInitialized: true };
      localStorage.removeItem("satyadrishti_user");
      analysisStore.clear();
      emit();
    }
  });

  // 2. Hydrate session from Supabase on application load
  supabase.auth.getSession().then(({ data: { session }, error }) => {
    if (error) {
      console.warn("Error hydrating Supabase session:", error);
    }
    if (session?.user) {
      const userObj: User = {
        id: session.user.id,
        name: session.user.user_metadata?.full_name || session.user.email?.split("@")[0] || "Forensic Analyst",
        email: session.user.email || "",
      };
      state = { user: userObj, isAuthenticated: true, isInitialized: true };
      localStorage.setItem("satyadrishti_user", JSON.stringify(userObj));
      analysisStore.setUser(userObj.id);
    } else {
      // If Supabase confirms there is no session, preserve guest analyst sessions
      const cached = parseStoredUser();
      if (cached && (cached.id === "guest" || String(cached.email).includes("guest"))) {
        state = { user: cached, isAuthenticated: true, isInitialized: true };
        analysisStore.setUser("guest");
      } else {
        state = { user: null, isAuthenticated: false, isInitialized: true };
        localStorage.removeItem("satyadrishti_user");
        analysisStore.clear();
      }
    }
    emit();
  }).catch(() => {
    state = { ...state, isInitialized: true };
    emit();
  });
}

export const authStore = {
  get: () => state,
  subscribe(l: () => void) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
  signIn(user: User) {
    state = { user, isAuthenticated: true, isInitialized: true };
    analysisStore.setUser(user.id);
    if (!IS_SERVER) {
      localStorage.setItem("satyadrishti_user", JSON.stringify(user));
    }
    emit();
  },
  signInGuest() {
    const guestUser: User = {
      id: "guest",
      name: "Guest Analyst",
      email: "guest@satyadrishti.internal",
    };
    state = { user: guestUser, isAuthenticated: true, isInitialized: true };
    analysisStore.setUser("guest");
    if (!IS_SERVER) {
      localStorage.setItem("satyadrishti_user", JSON.stringify(guestUser));
    }
    emit();
  },
  signOut() {
    state = { user: null, isAuthenticated: false, isInitialized: true };
    analysisStore.clear();
    if (!IS_SERVER) {
      localStorage.removeItem("satyadrishti_user");
      supabase.auth.signOut().catch((err) => {
        console.warn("Supabase signOut error:", err);
      });
    }
    emit();
  },
};

export function useAuthState() {
  return useSyncExternalStore(
    authStore.subscribe,
    authStore.get,
    authStore.get,
  );
}


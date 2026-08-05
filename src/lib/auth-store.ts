import { useSyncExternalStore } from "react";

export interface User {
  id: number;
  name: string;
  email: string;
}

type AuthState = {
  user: User | null;
  isAuthenticated: boolean;
};

const IS_SERVER = typeof window === "undefined";

function loadInitialState(): AuthState {
  if (IS_SERVER) {
    return { user: null, isAuthenticated: false };
  }
  try {
    const saved = localStorage.getItem("satyadrishti_user");
    if (saved) {
      const user = JSON.parse(saved);
      return { user, isAuthenticated: true };
    }
  } catch (e) {
    console.error("Failed to parse auth state:", e);
  }
  return { user: null, isAuthenticated: false };
}

let state: AuthState = loadInitialState();
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

export const authStore = {
  get: () => state,
  subscribe(l: () => void) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
  signIn(user: User) {
    state = { user, isAuthenticated: true };
    if (!IS_SERVER) {
      localStorage.setItem("satyadrishti_user", JSON.stringify(user));
    }
    emit();
  },
  signOut() {
    state = { user: null, isAuthenticated: false };
    if (!IS_SERVER) {
      localStorage.removeItem("satyadrishti_user");
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

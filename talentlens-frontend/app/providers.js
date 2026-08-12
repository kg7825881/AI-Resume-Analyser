"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "talentlens-app-state";
const DEFAULT_STATE = { currentRole: null, batch: null, resultsCache: {} };

const AppStateContext = createContext(null);

export function AppStateProvider({ children }) {
  // Always start from the same default on both server and client — reading
  // localStorage here (even guarded by typeof window) makes the very first
  // client render differ from the server-rendered HTML, which is what was
  // causing the hydration error. Real saved state gets loaded below, in an
  // effect that only ever runs on the client, after hydration is done.
  const [state, setState] = useState(DEFAULT_STATE);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) setState(JSON.parse(raw));
    } catch {
      // ignore corrupt storage
    } finally {
      setHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!hydrated) return; // don't clobber saved state with the default before it's loaded
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // storage full/unavailable — non-fatal
    }
  }, [state, hydrated]);

  const api = useMemo(
    () => ({
      state,
      setCurrentRole: (role) => setState((s) => ({ ...s, currentRole: role })),
      setBatch: (batch) => setState((s) => ({ ...s, batch })),
      setResultsForRole: (roleId, payload) =>
        setState((s) => ({ ...s, resultsCache: { ...s.resultsCache, [roleId]: payload } })),
      startNewScreening: () => setState((s) => ({ ...s, currentRole: null, batch: null })),
    }),
    [state]
  );

  return <AppStateContext.Provider value={api}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used inside AppStateProvider");
  return ctx;
}

// --- Toast ---

const ToastContext = createContext(null);

export function ToastProvider({ children }) {
  const [toast, setToast] = useState(null); // { message, kind }

  const show = useMemo(
    () => (message, kind = "info") => {
      setToast({ message, kind, id: Date.now() });
    },
    []
  );

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3200);
    return () => clearTimeout(t);
  }, [toast]);

  return (
    <ToastContext.Provider value={show}>
      {children}
      <div className={`toast ${toast ? "show" : ""} ${toast?.kind === "error" ? "error" : ""}`}>
        {toast?.message}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
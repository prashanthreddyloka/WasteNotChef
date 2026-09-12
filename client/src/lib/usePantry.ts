import { useCallback, useEffect, useRef, useState } from "react";
import type { PantryItem, SessionUser } from "../types";
import { loadInventory, saveInventory } from "./api";

const guestKey = "wastenotchef:items";
export function readGuestPantry(): PantryItem[] {
  try { const value = JSON.parse(localStorage.getItem(guestKey) ?? "[]"); return Array.isArray(value) ? value.filter(item => typeof item?.id === "string" && typeof item?.name === "string") : []; }
  catch { return []; }
}
export function usePantry(session: SessionUser | null) {
  const key = session?.mode === "account" ? session.id : session?.mode;
  const [owner, setOwner] = useState<string | undefined>();
  const [items, setItems] = useState<PantryItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const entries = useRef<PantryItem[]>([]);
  const version = useRef<number | null>(null);
  const generation = useRef(0);
  const lock = useRef(false);
  const activeKey = useRef(key);

  const refresh = useCallback(async () => {
    if (lock.current || session?.mode !== "account") return;
    const current = generation.current;
    lock.current = true; setBusy(true); setError(""); setStatus("Syncing pantry…");
    try {
      const snapshot = await loadInventory();
      if (current !== generation.current) return;
      entries.current = snapshot.items; version.current = snapshot.version; setItems(snapshot.items); setStatus("Pantry synced with your account");
    } catch (err) { if (current === generation.current) { setError(err instanceof Error ? err.message : "Could not load your pantry."); setStatus("Sync unavailable"); } }
    finally { if (current === generation.current) { lock.current = false; setBusy(false); } }
  }, [key]);

  useEffect(() => {
    generation.current += 1; activeKey.current = key; lock.current = false; version.current = null; entries.current = []; setItems([]); setOwner(key); setError(""); setBusy(false);
    if (session?.mode === "account") void refresh();
    else if (session?.mode === "guest") { entries.current = readGuestPantry(); setItems(entries.current); setStatus("Saved on this device · Guest mode"); }
    return () => { generation.current += 1; };
  }, [key, refresh]);

  useEffect(() => {
    if (session?.mode !== "account") return;
    const focus = () => { if (!document.querySelector(".pantry-card.is-editing") && !document.activeElement?.closest("input,textarea,form")) void refresh(); };
    window.addEventListener("focus", focus);
    return () => window.removeEventListener("focus", focus);
  }, [refresh, key]);

  async function update(updater: (current: PantryItem[]) => PantryItem[]) {
    if (activeKey.current !== key || !session) throw new Error("Your account changed. Please retry in the current account.");
    if (lock.current) throw new Error("Please wait for the current save to finish.");
    const current = generation.current;
    const next = updater(entries.current);
    lock.current = true; setBusy(true); setError("");
    try {
      if (session?.mode === "account") {
        if (version.current === null) throw new Error("Refresh your pantry before editing.");
        setStatus("Saving to your account…");
        const snapshot = await saveInventory(next, version.current);
        if (current !== generation.current) return;
        version.current = snapshot.version; entries.current = snapshot.items; setItems(snapshot.items); setStatus("Pantry synced with your account");
      } else {
        localStorage.setItem(guestKey, JSON.stringify(next)); entries.current = next; setItems(next); setStatus("Saved on this device · Guest mode");
      }
    } catch (err) { if (current === generation.current) { setError(err instanceof Error ? err.message : "Save failed."); setStatus("Changes not saved"); } throw err; }
    finally { if (current === generation.current) { lock.current = false; setBusy(false); } }
  }
  return { items: owner === key ? items : [], busy, error, status, refresh, update };
}

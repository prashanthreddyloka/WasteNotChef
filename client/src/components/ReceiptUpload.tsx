import { useEffect, useRef, useState } from "react";
import { uploadReceipt } from "../lib/api";
import type { PantryItem } from "../types";

export function ReceiptUpload({ onItemsAdded }: { onItemsAdded: (items: PantryItem[]) => Promise<void> | void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const inFlight = useRef(false);
  const active = useRef(true);
  useEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function scan(file?: File) {
    if (!file || inFlight.current) return;
    setError("");
    setMessage("");
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size > 10 * 1024 * 1024) {
      setError("Choose a JPEG, PNG, or WebP image under 10 MB.");
      return;
    }
    inFlight.current = true;
    setBusy(true);
    try {
      const result = await uploadReceipt(file);
      if (!active.current) return;
      await onItemsAdded(result.items);
      setMessage(result.items.length
        ? `${result.items.length} food item${result.items.length === 1 ? "" : "s"} from this receipt are in your inventory. Skipped ${result.skippedLines} other lines. Uploading the same image again won't add duplicates.`
        : "No food purchases recognized. Try a clearer photo with item names and prices visible. Your inventory has not changed.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not scan the receipt. Please try again.");
    } finally {
      setBusy(false);
      inFlight.current = false;
    }
  }

  return (
    <section className="rounded-[1.8rem] border border-teal-100 bg-white/85 p-6 shadow-float" aria-busy={busy}>
      <p className="text-sm font-semibold uppercase tracking-[0.3em] text-teal-700">Receipt scanning</p>
      <h2 className="mt-2 font-display text-2xl text-ink">Groceries to inventory</h2>
      <p className="mt-2 text-slate-600">Upload a grocery receipt photo or screenshot. Recognized food purchases are added to your list; household goods, totals, and payment details are skipped.</p>
      <p className="mt-2 text-sm text-slate-500">JPEG, PNG, or WebP · Up to 10 MB. We enhance contrast and read prices on following lines. Very blurry text may still be missed; review your list after scanning.</p>
      <button type="button" disabled={busy} onClick={() => inputRef.current?.click()} className="mt-4 rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white disabled:opacity-50">
        {busy ? "Scanning receipt…" : "Upload receipt"}
      </button>
      <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={event => {
        const file = event.target.files?.[0];
        event.target.value = "";
        void scan(file);
      }} />
      <p role="status" className="mt-3 text-sm text-teal-800">{busy ? "Reading purchases and filtering your receipt. This can take a moment." : message}</p>
      {error && <p role="alert" className="mt-3 text-sm text-red-700">{error}</p>}
    </section>
  );
}

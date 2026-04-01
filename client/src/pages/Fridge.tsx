import { motion } from "framer-motion";
import { useState } from "react";
import type { PantryItem } from "../types";

type FridgeProps = {
  items: PantryItem[];
  onUpdateItem: (id: string, updates: Partial<PantryItem>) => void;
  onAddItem: (item: PantryItem) => void;
  onGenerateRecipes: () => Promise<void> | void;
  generatingRecipes?: boolean;
};

export function Fridge({ items, onUpdateItem, onAddItem, onGenerateRecipes, generatingRecipes }: FridgeProps) {
  const [draftName, setDraftName] = useState("");
  const [draftQuantity, setDraftQuantity] = useState("");
  const [draftExpiry, setDraftExpiry] = useState("");

  function sourceLabel(item: PantryItem) {
    if (item.detectionSource === "merged") {
      return "Gemini / visual + OCR";
    }

    if (item.detectionSource === "manual") {
      return "manual entry";
    }

    if (item.detectionSource === "gemini") {
      return "Gemini vision";
    }

    if (item.detectionSource === "visual") {
      return "visual estimate";
    }

    return "OCR label";
  }

  function expiryLabel(item: PantryItem) {
    if (item.expirySource === "ocr") {
      return "date from OCR";
    }

    if (item.expirySource === "rule") {
      return "date from shelf-life rule";
    }

    return "no date found";
  }

  function addManualItem() {
    const normalizedName = draftName.trim();
    if (!normalizedName) {
      return;
    }

    onAddItem({
      id: `manual-${normalizedName.toLowerCase().replace(/\s+/g, "-")}-${Date.now()}`,
      name: normalizedName.toLowerCase(),
      quantity: draftQuantity.trim() || undefined,
      detectedExpiry: draftExpiry || null,
      inferredExpiry: null,
      confidence: 1,
      detectionSource: "manual",
      expirySource: draftExpiry ? "ocr" : "none",
      notes: "Added manually after review."
    });

    setDraftName("");
    setDraftQuantity("");
    setDraftExpiry("");
  }

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-teal-700">Fridge review</p>
          <h1 className="mt-2 font-display text-4xl text-ink">Detected ingredients</h1>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void onGenerateRecipes()}
            disabled={generatingRecipes || items.length === 0}
            className="rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-900 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {generatingRecipes ? "Generating..." : "Get Recipes"}
          </button>
          <div className="rounded-full bg-white/80 px-4 py-2 text-sm text-slate-500 shadow-sm">Edit dates or add missing items before generating the week quest</div>
        </div>
      </div>

      <div className="rounded-[2rem] border border-white/70 bg-white/85 p-5 shadow-float">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-coral">Missing something?</p>
            <h2 className="mt-2 font-display text-3xl text-ink">Add an item manually</h2>
            <p className="mt-2 text-sm text-slate-600">
              If the upload missed an ingredient, add it here and it will flow into recipes and planning like any detected item.
            </p>
          </div>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-[1.2fr_0.9fr_0.9fr_auto]">
          <input
            type="text"
            value={draftName}
            onChange={(event) => setDraftName(event.target.value)}
            placeholder="Item name, for example yogurt or strawberries"
            className="rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
          />
          <input
            type="text"
            value={draftQuantity}
            onChange={(event) => setDraftQuantity(event.target.value)}
            placeholder="Quantity"
            className="rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
          />
          <input
            type="date"
            value={draftExpiry}
            onChange={(event) => setDraftExpiry(event.target.value)}
            className="rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
          />
          <button
            type="button"
            onClick={addManualItem}
            className="rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-900"
          >
            Add item
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="rounded-[2rem] border border-dashed border-slate-300 bg-white/70 p-10 text-center shadow-float">
          <h2 className="font-display text-3xl text-ink">No confident detections yet</h2>
          <p className="mx-auto mt-3 max-w-2xl text-sm leading-7 text-slate-600">
            This usually means the photo had glare, distant packaging, or no readable labels. Try a closer image with better lighting,
            or use the demo fridge from the onboarding modal.
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {items.map((item, index) => (
            <motion.article
              key={item.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.04 }}
              className="rounded-[1.8rem] border border-white/70 bg-white/85 p-5 shadow-float"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="font-display text-2xl text-ink">{item.name}</h2>
                  <p className="mt-1 text-sm text-slate-500">{item.notes ?? "OCR-derived item"}</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className="rounded-full bg-oat px-3 py-1 text-xs font-semibold text-slate-700">
                      {sourceLabel(item)}
                    </span>
                    <span className="rounded-full bg-mist px-3 py-1 text-xs font-semibold text-slate-600">
                      {expiryLabel(item)}
                    </span>
                  </div>
                </div>
                <span className="rounded-full bg-mist px-3 py-1 text-xs font-semibold text-slate-600">
                  {Math.round(item.confidence * 100)}% confident
                </span>
              </div>

              <label className="mt-5 block text-sm font-medium text-slate-600">
                Expiry date
                <input
                  type="date"
                  value={item.detectedExpiry ?? item.inferredExpiry ?? ""}
                  onChange={(event) => onUpdateItem(item.id, { detectedExpiry: event.target.value })}
                  className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 outline-none ring-0 focus:border-teal-400"
                />
              </label>

              <div className="mt-4 flex gap-3">
                <button type="button" className="rounded-full bg-ink px-4 py-2 text-sm font-semibold text-white">
                  Add to pantry
                </button>
                <button type="button" className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-600">
                  Ignore
                </button>
              </div>
            </motion.article>
          ))}
        </div>
      )}
    </div>
  );
}

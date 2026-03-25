import { motion } from "framer-motion";
import type { PantryItem } from "../types";

type FridgeProps = {
  items: PantryItem[];
  onUpdateItem: (id: string, updates: Partial<PantryItem>) => void;
};

export function Fridge({ items, onUpdateItem }: FridgeProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-teal-700">Fridge review</p>
          <h1 className="mt-2 font-display text-4xl text-ink">Detected ingredients</h1>
        </div>
        <div className="rounded-full bg-white/80 px-4 py-2 text-sm text-slate-500 shadow-sm">Edit dates before generating the week quest</div>
      </div>

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
    </div>
  );
}

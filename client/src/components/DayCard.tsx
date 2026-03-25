import { motion } from "framer-motion";
import type { DayPlan } from "../types";

type DayCardProps = {
  dayPlan: DayPlan;
  index: number;
  onDragStart: (index: number) => void;
  onDrop: (index: number) => void;
};

export function DayCard({ dayPlan, index, onDragStart, onDrop }: DayCardProps) {
  return (
    <motion.article
      layout
      draggable
      onDragStart={() => onDragStart(index)}
      onDragOver={(event) => event.preventDefault()}
      onDrop={() => onDrop(index)}
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative rounded-[1.8rem] border border-white/70 bg-white/85 p-5 shadow-float backdrop-blur"
    >
      <div className="absolute left-6 top-0 h-6 w-px -translate-y-full bg-gradient-to-b from-transparent to-teal-300" />
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-teal-600">Day {index + 1}</p>
          <h3 className="font-display text-2xl text-ink">{dayPlan.recipe.title}</h3>
        </div>
        <div className="rounded-full bg-glow px-3 py-2 text-xs font-semibold text-teal-900">Priority {Math.round(dayPlan.priority)}</div>
      </div>

      <p className="text-sm text-slate-500">{dayPlan.scheduledDate}</p>
      <p className="mt-4 text-sm leading-6 text-slate-600">{dayPlan.reasoning}</p>

      <div className="mt-4 flex flex-wrap gap-2">
        {dayPlan.itemsConsumed.map((item) => (
          <span key={item} className="rounded-full border border-slate-200 px-3 py-1 text-xs font-medium text-slate-600">
            {item}
          </span>
        ))}
      </div>

      <div className="mt-5 rounded-2xl bg-oat p-3 text-sm text-slate-600">
        Waste score after this cook: <span className="font-semibold text-ink">{dayPlan.wasteScore}</span>
      </div>
    </motion.article>
  );
}

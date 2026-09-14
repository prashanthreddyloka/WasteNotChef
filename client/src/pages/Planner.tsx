import { ArrowDownTrayIcon } from "@heroicons/react/24/outline";
import { useState } from "react";
import { DayCard } from "../components/DayCard";
import type { DayPlan } from "../types";

type PlannerProps = {
  dayPlans: DayPlan[];
  onReorder: (plans: DayPlan[]) => void;
};

export function Planner({ dayPlans, onReorder }: PlannerProps) {
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  function handleDrop(index: number) {
    if (dragIndex === null || dragIndex === index) {
      return;
    }

    const next = [...dayPlans];
    const [moved] = next.splice(dragIndex, 1);
    next.splice(index, 0, moved);
    onReorder(next.map((plan, position) => ({ ...plan, scheduledDate: dayPlans[position].scheduledDate })));
    setDragIndex(null);
  }

  function exportPlan() {
    const blob = new Blob([JSON.stringify(dayPlans, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "wastenotchef-plan.json";
    anchor.click();
    URL.revokeObjectURL(url);
    window.print();
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-teal-700">Week as quest</p>
          <h1 className="mt-2 font-display text-4xl text-ink">Animated cook order</h1>
          <p className="mt-3 max-w-2xl text-slate-600">
            Drag day cards to adjust the plan. Each card shows the reasoning that put it there, so edits stay understandable.
          </p>
        </div>
        <button type="button" onClick={exportPlan} className="inline-flex items-center gap-2 rounded-full bg-ink px-4 py-3 text-sm font-semibold text-white">
          <ArrowDownTrayIcon className="h-4 w-4" />
          Export + print
        </button>
      </div>

      <div className="relative rounded-[2.2rem] border border-slate-200 bg-gradient-to-b from-white/90 to-mist/40 p-5 sm:p-8">
        <div className="absolute left-8 top-8 h-[calc(100%-4rem)] w-px bg-gradient-to-b from-teal-200 via-coral/30 to-transparent" />
        {dayPlans.length === 0 ? (
          <div className="rounded-[1.8rem] border border-dashed border-slate-300 bg-white/70 p-10 text-center">
            <h2 className="font-display text-3xl text-ink">No recipes in the planner yet</h2>
            <p className="mt-3 text-sm leading-7 text-slate-600">
              Add a recipe from the Recipes page and it will appear here with steps, reasoning, and a related video link.
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {dayPlans.map((dayPlan, index) => (
              <div key={`${dayPlan.scheduledDate}-${dayPlan.recipe.id}`} className="relative pl-8">
                <div className="absolute left-[1.2rem] top-10 h-3 w-3 rounded-full bg-coral ring-4 ring-white" />
                <DayCard dayPlan={dayPlan} index={index} onDragStart={setDragIndex} onDrop={handleDrop} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type DashboardProps = {
  timeseries: Array<{ date: string; wasteScore: number; recipeTitle?: string }>;
};

export function Dashboard({ timeseries }: DashboardProps) {
  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.3em] text-coral">Waste dashboard</p>
        <h1 className="mt-2 font-display text-4xl text-ink">Waste score over time</h1>
        <p className="mt-3 max-w-2xl text-slate-600">
          Daily and weekly scores make it easier to spot when unused ingredients are stacking up before they become actual waste.
        </p>
      </div>

      <div className="rounded-[2rem] border border-white/70 bg-white/85 p-5 shadow-float">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timeseries}>
              <defs>
                <linearGradient id="waste" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#0f766e" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#0f766e" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#475467", fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fill: "#475467", fontSize: 12 }} />
              <Tooltip />
              <Area type="monotone" dataKey="wasteScore" stroke="#0f766e" fill="url(#waste)" strokeWidth={3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {timeseries.slice(-4).map((point) => (
          <article key={point.date} className="rounded-[1.75rem] border border-slate-200 bg-white/80 p-5">
            <div className="text-sm text-slate-500">{point.date}</div>
            <div className="mt-2 font-display text-3xl text-ink">{point.wasteScore}</div>
            <div className="mt-2 text-sm text-slate-600">{point.recipeTitle ?? "Plan snapshot"}</div>
          </article>
        ))}
      </div>
    </div>
  );
}

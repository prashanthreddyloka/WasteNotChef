export function Settings() {
  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.3em] text-teal-700">Settings</p>
        <h1 className="mt-2 font-display text-4xl text-ink">Notifications, export, and privacy</h1>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="rounded-[1.75rem] border border-white/70 bg-white/85 p-6 shadow-float">
          <h2 className="font-display text-2xl text-ink">Reminder preferences</h2>
          <div className="mt-4 space-y-3 text-sm text-slate-600">
            <label className="flex items-center justify-between rounded-2xl bg-mist p-4">
              <span>Web push reminders</span>
              <span className="font-semibold text-teal-700">Stubbed until configured</span>
            </label>
            <label className="flex items-center justify-between rounded-2xl bg-oat p-4">
              <span>Email reminders</span>
              <span className="font-semibold text-coral">Console fallback active</span>
            </label>
          </div>
        </section>

        <section className="rounded-[1.75rem] border border-white/70 bg-white/85 p-6 shadow-float">
          <h2 className="font-display text-2xl text-ink">Privacy note</h2>
          <p className="mt-4 text-sm leading-7 text-slate-600">
            Analytics are console-only in this demo build. No external analytics provider is wired by default, and export is available as local JSON plus a printable plan flow.
          </p>
          <div className="mt-4 rounded-2xl bg-mist p-4 text-sm text-slate-600">
            Integrations placeholder: grocery sync, shared household calendars, and smart device inventory feeds.
          </div>
        </section>
      </div>
    </div>
  );
}

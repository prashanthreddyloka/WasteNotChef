import type { NotificationPrefs, SessionUser } from "../types";

type SettingsProps = {
  session: SessionUser;
  notificationPrefs: NotificationPrefs;
  onUpdateNotificationPrefs: (updates: Partial<NotificationPrefs>) => void;
  onEnableBrowserNotifications: () => Promise<void> | void;
};

export function Settings({
  session,
  notificationPrefs,
  onUpdateNotificationPrefs,
  onEnableBrowserNotifications
}: SettingsProps) {
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
            <div className="rounded-2xl bg-mist p-4">
              <div className="flex items-center justify-between gap-4">
                <span>Browser notifications</span>
                <button
                  type="button"
                  onClick={() => void onEnableBrowserNotifications()}
                  className="rounded-full bg-ink px-4 py-2 text-xs font-semibold text-white"
                >
                  {notificationPrefs.browserPermission === "granted" ? "Enabled" : "Enable"}
                </button>
              </div>
              <div className="mt-2 text-xs text-slate-500">
                Permission status: <span className="font-semibold">{notificationPrefs.browserPermission}</span>
              </div>
            </div>
            <label className="flex items-center justify-between rounded-2xl bg-oat p-4">
              <span>Email reminders</span>
              <input
                type="checkbox"
                checked={notificationPrefs.emailEnabled}
                onChange={(event) => onUpdateNotificationPrefs({ emailEnabled: event.target.checked })}
              />
            </label>
            <input
              type="email"
              value={notificationPrefs.email ?? ""}
              onChange={(event) => onUpdateNotificationPrefs({ email: event.target.value })}
              placeholder="Reminder email"
              className="w-full rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
            />
            <label className="block rounded-2xl bg-white p-4">
              <span className="text-sm font-medium text-slate-700">Remind me before expiry</span>
              <select
                value={notificationPrefs.reminderDays}
                onChange={(event) => onUpdateNotificationPrefs({ reminderDays: Number(event.target.value) })}
                className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
              >
                {[1, 2, 3, 5, 7].map((days) => (
                  <option key={days} value={days}>
                    {days} day{days > 1 ? "s" : ""} before
                  </option>
                ))}
              </select>
            </label>
          </div>
        </section>

        <section className="rounded-[1.75rem] border border-white/70 bg-white/85 p-6 shadow-float">
          <h2 className="font-display text-2xl text-ink">Profile and privacy</h2>
          <div className="mt-4 rounded-2xl bg-mist p-4 text-sm text-slate-600">
            Signed in as <span className="font-semibold text-ink">{session.name}</span> via{" "}
            <span className="font-semibold text-ink">{session.mode === "guest" ? "guest mode" : "local profile"}</span>.
          </div>
          <p className="mt-4 text-sm leading-7 text-slate-600">
            Analytics are console-only in this demo build. No external analytics provider is wired by default, and export is available as local JSON plus a printable plan flow.
          </p>
          <div className="mt-4 rounded-2xl bg-mist p-4 text-sm text-slate-600">
            Integrations placeholder: grocery sync, shared household calendars, and smart device inventory feeds.
          </div>
        </section>
      </div>

      <section className="rounded-[1.75rem] border border-white/70 bg-white/85 p-6 shadow-float">
        <p className="text-sm font-semibold uppercase tracking-[0.3em] text-coral">Suggested improvements</p>
        <h2 className="mt-2 font-display text-3xl text-ink">What I would add next</h2>
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          {[
            ["Household mode", "Shared fridge edits, roles, and a common weekly plan for families or roommates."],
            ["Smart substitutions", "Suggest nearby ingredient swaps and pantry-first replacements per recipe."],
            ["Shopping bridge", "Turn missing ingredients into a quick grocery list after plan generation."]
          ].map(([title, copy]) => (
            <article key={title} className="rounded-2xl bg-oat p-4">
              <div className="font-semibold text-ink">{title}</div>
              <div className="mt-2 text-sm leading-6 text-slate-600">{copy}</div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

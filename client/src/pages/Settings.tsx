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
          <h2 className="font-display text-2xl text-ink">Reminder availability</h2>
          <p className="mt-3 text-sm text-slate-600">Automatic browser and email reminders are not connected yet. Review dates in your pantry; granting browser permission alone does not schedule notifications.</p>
          <div className="mt-4 space-y-3 text-sm text-slate-600">
            <div className="rounded-2xl bg-mist p-4">
              <div className="flex items-center justify-between gap-4">
                <span>Browser notifications</span>
                <button
                  type="button"
                  disabled
                  onClick={() => void onEnableBrowserNotifications()}
                  className="rounded-full bg-ink px-4 py-2 text-xs font-semibold text-white"
                >
                  Not available yet
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
                disabled
                checked={false}
                onChange={(event) => onUpdateNotificationPrefs({ emailEnabled: event.target.checked })}
              />
            </label>
            <input
              type="email"
              disabled
              aria-label="Reminder email (not available yet)"
              value={notificationPrefs.email ?? ""}
              onChange={(event) => onUpdateNotificationPrefs({ email: event.target.value })}
              placeholder="Reminder email"
              className="w-full rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
            />
            <label className="block rounded-2xl bg-white p-4">
              <span className="text-sm font-medium text-slate-700">Remind me before expiry</span>
              <select
                disabled
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
            <span className="font-semibold text-ink">{session.mode === "guest" ? "guest mode" : "your account"}</span>.
          </div>
          <p className="mt-4 text-sm leading-7 text-slate-600">
            Your pantry is separate from other accounts. Photo recognition may send an uploaded image to the configured image-recognition provider. Temporary server uploads are removed after processing. Plans can be exported from the Planner.
          </p>
          <div className="mt-4 rounded-2xl bg-mist p-4 text-sm text-slate-600">
            {session.mode === "account" ? "Your inventory syncs across devices when you sign in. Use Refresh pantry to load changes made elsewhere." : "Guest inventory stays on this device. Sign in to import it into a personal account."}
          </div>
        </section>
      </div>

    </div>
  );
}

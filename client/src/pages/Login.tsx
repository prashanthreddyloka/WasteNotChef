import { motion } from "framer-motion";
import { useState } from "react";

type LoginProps = {
  onGuestLogin: () => void;
  onLocalLogin: (payload: { name: string; email: string }) => void;
};

export function Login({ onGuestLogin, onLocalLogin }: LoginProps) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");

  return (
    <div className="mx-auto grid min-h-[70vh] max-w-6xl items-center gap-8 lg:grid-cols-[1.1fr_0.9fr]">
      <div className="rounded-[2.5rem] bg-hero p-8 shadow-float">
        <p className="text-sm font-semibold uppercase tracking-[0.35em] text-teal-700">Welcome back</p>
        <h1 className="mt-4 font-display text-5xl leading-tight text-ink">
          Sign in or continue as a guest to keep your fridge, recipes, and plans in sync.
        </h1>
        <p className="mt-5 max-w-xl text-lg leading-8 text-slate-600">
          Guest mode works instantly and stores your progress on this device. Local sign-in keeps a named profile for reminders and a more personal experience.
        </p>
        <div className="mt-8 grid gap-3 sm:grid-cols-3">
          {[
            ["Guest mode", "Instant access"],
            ["Recipes", "Fresh from your latest fridge"],
            ["Reminders", "Browser and email preferences"]
          ].map(([title, copy]) => (
            <div key={title} className="rounded-[1.5rem] border border-white/70 bg-white/70 p-4">
              <div className="font-semibold text-ink">{title}</div>
              <div className="mt-1 text-sm text-slate-500">{copy}</div>
            </div>
          ))}
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-[2rem] border border-white/70 bg-white/85 p-6 shadow-float"
      >
        <h2 className="font-display text-3xl text-ink">Start cooking smarter</h2>
        <p className="mt-3 text-sm leading-7 text-slate-600">
          Use guest mode for a fast demo, or create a light local profile for named reminder preferences.
        </p>

        <button
          type="button"
          onClick={onGuestLogin}
          className="mt-6 w-full rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-900"
        >
          Continue as Guest
        </button>

        <div className="my-6 flex items-center gap-3 text-xs uppercase tracking-[0.3em] text-slate-400">
          <div className="h-px flex-1 bg-slate-200" />
          or
          <div className="h-px flex-1 bg-slate-200" />
        </div>

        <div className="space-y-3">
          <input
            type="text"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Your name"
            className="w-full rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
          />
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="Email for reminders"
            className="w-full rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
          />
          <button
            type="button"
            onClick={() => onLocalLogin({ name: name.trim() || "Home cook", email: email.trim() })}
            disabled={!email.trim()}
            className="w-full rounded-full bg-coral px-5 py-3 text-sm font-semibold text-white transition hover:bg-[#f46e49] disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            Create local profile
          </button>
        </div>
      </motion.div>
    </div>
  );
}

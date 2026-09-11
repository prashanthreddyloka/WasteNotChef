import { motion } from "framer-motion";
import { useState } from "react";
import { KitchenScene } from "../components/KitchenScene";

type LoginProps = {
  onGuestLogin: () => void;
  onLocalLogin: (payload: { name: string; email: string }) => void;
};

export function Login({ onGuestLogin, onLocalLogin }: LoginProps) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");

  return (
    <div className="mx-auto grid min-h-[70vh] max-w-6xl items-center gap-8 lg:grid-cols-[1.1fr_0.9fr]">
      <div className="login-intro">
        <p className="text-sm font-semibold uppercase tracking-[0.35em] text-teal-700">Welcome back</p>
        <h1 className="mt-4 font-display text-5xl leading-tight text-ink">
          A fresh start for<br />the food you love.
        </h1>
        <p className="mt-5 max-w-xl text-lg leading-8 text-slate-600">
          Your groceries have good things ahead. Scan, plan, and make more of what’s already in your kitchen.
        </p>
        <KitchenScene />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-[2rem] border border-white/70 bg-white/85 p-6 shadow-float"
      >
        <h2 className="font-display text-3xl text-ink">Start cooking smarter</h2>
        <p className="mt-3 text-sm leading-7 text-slate-600">
          Jump in as a guest or create a local profile. Your kitchen is saved on this device.
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

import { useState } from "react";
import { KitchenScene } from "../components/KitchenScene";
import { authenticate } from "../lib/api";
import type { SessionUser } from "../types";

type Props = { onGuestLogin: () => void; onAuthenticated: (result: { user: SessionUser; token: string; recoveryCode?: string }) => void };
export function Login({ onGuestLogin, onAuthenticated }: Props) {
  const [mode, setMode] = useState<"login" | "register" | "recover">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [recoveryCode, setRecoveryCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const label = mode === "login" ? "Sign in" : mode === "register" ? "Create account" : "Reset password";
  return <div className="mx-auto grid min-h-[70vh] max-w-6xl items-center gap-8 lg:grid-cols-[1.1fr_0.9fr]">
    <div className="login-intro"><p className="eyebrow">YOUR KITCHEN, WHEREVER YOU ARE</p><h1 className="mt-4 font-display text-5xl leading-tight text-ink">A fresh start for<br />the food you love.</h1><p className="mt-5 max-w-xl text-lg leading-8 text-slate-600">Sign in to keep your pantry across devices. Your ingredients, corrections, and expiry dates belong to your account.</p><KitchenScene /></div>
    <section className="rounded-[2rem] border border-white/70 bg-white/85 p-6 shadow-float">
      <h2 className="font-display text-3xl text-ink">{label}</h2><p className="mt-3 text-sm leading-7 text-slate-600">{mode === "recover" ? "Use the recovery code saved when you created your account. Resetting your password signs out all other sessions." : "A personal pantry, ready when you are. Guest mode stays on this device."}</p>
      <form className="pantry-edit mt-5" onSubmit={async event => {
        event.preventDefault(); if (busy) return; setBusy(true); setError("");
        try { onAuthenticated(await authenticate(mode, { name, email, password, recoveryCode })); }
        catch (err) { setError(err instanceof Error ? err.message : "Could not sign in."); }
        finally { setBusy(false); }
      }}><fieldset disabled={busy} className="space-y-4">
        {mode === "register" && <label>Your name<input required maxLength={80} autoComplete="name" value={name} onChange={event => setName(event.target.value)} /></label>}
        <label>Email<input required type="email" maxLength={254} autoComplete="username" value={email} onChange={event => setEmail(event.target.value)} /></label>
        {mode === "recover" && <label>Recovery code<input required autoComplete="off" value={recoveryCode} onChange={event => setRecoveryCode(event.target.value)} /></label>}
        <label>{mode === "recover" ? "New password" : "Password"}<input required type="password" minLength={mode === "login" ? 1 : 12} maxLength={128} autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} onChange={event => setPassword(event.target.value)} /></label>
        {mode !== "login" && <p className="text-xs text-slate-500">Use at least 12 characters. You’ll receive a recovery code to save securely; password recovery does not use email.</p>}
        {error && <p role="alert" className="text-sm text-red-700">{error}</p>}
        <button className="primary-action w-full" type="submit">{busy ? "Please wait…" : label}</button>
      </fieldset></form>
      <div className="mt-5 flex flex-wrap gap-4 text-xs"><button disabled={busy} onClick={() => { setMode(mode === "register" ? "login" : "register"); setError(""); setPassword(""); }}>{mode === "register" ? "Already have an account? Sign in" : "Create an account"}</button><button disabled={busy} onClick={() => { setMode(mode === "recover" ? "login" : "recover"); setError(""); setPassword(""); }}>{mode === "recover" ? "Back to sign in" : "Forgot password?"}</button></div>
      <button type="button" disabled={busy} onClick={onGuestLogin} className="secondary-action mt-5">Continue as Guest →</button>
    </section>
  </div>;
}

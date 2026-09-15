import { useEffect, useRef, useState } from "react";
import { Link, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { demoItems, demoPlan, demoRecipes } from "./data/demo";
import { ApiError, currentAccount, endSession, fetchRecipes, fetchWasteSeries, fetchWeekPlan, getSessionToken, setSessionToken, uploadPhoto } from "./lib/api";
import { readGuestPantry, usePantry } from "./lib/usePantry";
import { ChatWidget } from "./components/ChatWidget";
import { Dashboard } from "./pages/Dashboard";
import { Fridge } from "./pages/Fridge";
import { Landing } from "./pages/Landing";
import { Login } from "./pages/Login";
import { PublicHome } from "./pages/PublicHome";
import { Planner } from "./pages/Planner";
import { Recipes } from "./pages/Recipes";
import { Settings } from "./pages/Settings";
import type { DayPlan, NotificationPrefs, PantryItem, Recipe, SessionUser } from "./types";

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [session, setSession] = useState<SessionUser | null>(null);
  const [restoring, setRestoring] = useState(true);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [dayPlans, setDayPlans] = useState<DayPlan[]>([]);
  const [busy, setBusy] = useState(false);
  const [recipesBusy, setRecipesBusy] = useState(false);
  const [appError, setAppError] = useState("");
  const [recoveryCode, setRecoveryCode] = useState("");
  const [timeseries, setTimeseries] = useState<Array<{ date: string; wasteScore: number; recipeTitle?: string }>>([]);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [notificationPrefs, setNotificationPrefs] = useState<NotificationPrefs>({ webPushEnabled: false, emailEnabled: false, email: "", reminderDays: 2, browserPermission: "Notification" in window ? Notification.permission : "default" });
  const pantry = usePantry(session);
  const epoch = useRef(0);
  const [viewOwner, setViewOwner] = useState<string | null>(null);
  const viewKey = session?.mode === "account" ? session.id ?? null : session?.mode ?? null;
  const items = pantry.items;
  useEffect(() => { if (location.hash === "#scan") document.getElementById("scan")?.scrollIntoView({ behavior: "smooth" }); }, [location.pathname, location.hash, restoring, session]);
  useEffect(() => {
    let cancelled = false;
    async function restore() {
      try {
        if (getSessionToken()) { const user = await currentAccount(); if (!cancelled) setSession(user); }
        else {
          const local = JSON.parse(localStorage.getItem("wastenotchef:session") ?? "null");
          if (!cancelled && local && ["guest", "local"].includes(local.mode)) setSession({ mode: "guest", name: "Guest cook" });
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) setSessionToken(null);
        if (!cancelled) setAppError(error instanceof Error ? error.message : "Could not restore your session.");
      } finally { if (!cancelled) setRestoring(false); }
    }
    void restore(); return () => { cancelled = true; };
  }, []);
  useEffect(() => {
    epoch.current += 1; setRecipes([]); setDayPlans([]); setTimeseries([]); setViewOwner(viewKey);
    if (session) {
      const storage = session.mode === "guest" ? localStorage : sessionStorage;
      const prefix = session.mode === "guest" ? "wastenotchef:" : `wastenotchef:${session.id}:`;
      try {
        const savedRecipes = JSON.parse(storage.getItem(`${prefix}recipes`) ?? "[]");
        const savedPlans = JSON.parse(storage.getItem(`${prefix}dayPlans`) ?? "[]");
        if (Array.isArray(savedRecipes)) setRecipes(savedRecipes);
        if (Array.isArray(savedPlans)) setDayPlans(savedPlans);
      } catch { /* Invalid browser caches do not prevent opening the pantry. */ }
    }
    setNotificationPrefs(current => ({ ...current, email: session?.email ?? "", emailEnabled: false }));
    setShowOnboarding(session?.mode === "guest" && !localStorage.getItem("wastenotchef:onboarded"));
  }, [session?.id, session?.mode]);
  useEffect(() => {
    if (!session || viewOwner !== viewKey) return;
    const storage = session.mode === "guest" ? localStorage : sessionStorage;
    const prefix = session.mode === "guest" ? "wastenotchef:" : `wastenotchef:${session.id}:`;
    try { storage.setItem(`${prefix}recipes`, JSON.stringify(recipes)); storage.setItem(`${prefix}dayPlans`, JSON.stringify(dayPlans)); } catch { /* Inventory saving reports its own storage errors. */ }
  }, [recipes, dayPlans, viewOwner, viewKey]);
  useEffect(() => {
    let cancelled = false;
    if (session?.mode === "account") {
      const today = new Date();
      void fetchWasteSeries(`${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,"0")}-01`, today.toISOString().slice(0,10)).then(result => { if (!cancelled) setTimeseries(result.timeseries ?? []); }).catch(() => {});
    }
    return () => { cancelled = true; };
  }, [session?.id]);

  async function updateItems(updater: (current: PantryItem[]) => PantryItem[]) { await pantry.update(updater); setRecipes([]); }
  async function addItems(added: PantryItem[]) {
    if (!added.length) return;
    await updateItems(current => { const existing = new Set(current.map(item => item.id)); return [...current, ...added.filter(item => !existing.has(item.id))]; });
  }
  async function handleFile(file: File) {
    setBusy(true); setAppError(""); const current = epoch.current;
    try {
      const uploaded = await uploadPhoto(file);
      if (current !== epoch.current) return;
      const added = uploaded.items;
      if (!added.length) { setAppError("No confident ingredients found. Try a clearer photo or add items manually."); return; }
      await addItems(added); navigate("/fridge");
      if (uploaded.recognition === "labels") setAppError("This scan used readable labels. Unlabeled ingredients may be missing; add them manually and review the estimated dates.");
    } catch (error) { setAppError(error instanceof Error ? error.message : "Could not analyze this photo."); }
    finally { setBusy(false); }
  }
  async function generateRecipes() {
    setRecipesBusy(true); setAppError(""); const current = epoch.current;
    try { const result = await fetchRecipes(items); if (current === epoch.current) { setRecipes(result); navigate("/recipes"); } }
    catch (error) { setAppError(error instanceof Error ? error.message : "Could not find recipes."); }
    finally { setRecipesBusy(false); }
  }
  async function planCurrentPantry() {
    setBusy(true); setAppError(""); const current = epoch.current;
    try {
      const result = await fetchWeekPlan(items, { mealsPerDay: 1, skipDays: [], preferCuisineTags: [], maxLeftovers: 2 });
      if (current === epoch.current) { setDayPlans(result); setTimeseries(result.map(day => ({ date: day.scheduledDate, wasteScore: day.wasteScore, recipeTitle: day.recipe.title }))); }
    } catch (error) { setAppError(error instanceof Error ? error.message : "Could not plan your week."); }
    finally { setBusy(false); }
  }
  function guestLogin() { setSessionToken(null); setSession({ mode: "guest", name: "Guest cook" }); localStorage.setItem("wastenotchef:session", JSON.stringify({ mode: "guest" })); setAppError(""); }
  async function logout() {
    try { if (session?.mode === "account") await endSession(); else setSessionToken(null); epoch.current += 1; setSession(null); localStorage.removeItem("wastenotchef:session"); setRecoveryCode(""); setAppError(""); }
    catch (error) { setAppError(error instanceof Error ? error.message : "Could not sign out."); }
  }
  async function enableNotifications() {
    if (!("Notification" in window)) return;
    const permission = await Notification.requestPermission(); setNotificationPrefs(current => ({ ...current, browserPermission: permission, webPushEnabled: permission === "granted" }));
  }
  if (!session && location.pathname === "/") return <PublicHome />;
  if (restoring) return <div className="app-shell min-h-screen p-10" role="status">Opening your kitchen…</div>;
  if (!session) return <div className="app-shell min-h-screen px-4 py-8 sm:px-6 lg:px-8">{appError && <p role="alert" className="mx-auto max-w-6xl text-red-700">{appError}</p>}<Login onGuestLogin={guestLogin} onAuthenticated={result => { setSessionToken(result.token); localStorage.removeItem("wastenotchef:session"); setSession(result.user); setRecoveryCode(result.recoveryCode ?? ""); setAppError(""); navigate("/fridge"); }} /></div>;
  const pending = pantry.busy || busy;
  return <div className="app-shell min-h-screen text-ink">
    {recoveryCode && <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-5"><section role="dialog" aria-modal="true" aria-labelledby="recovery-heading" className="w-full max-w-xl rounded-3xl bg-white p-7"><h2 id="recovery-heading" className="font-display text-3xl">Save your recovery code</h2><p className="my-4 text-sm leading-7">Keep this code in a password manager. It is shown only once and lets you reset your password. We don’t send password-reset emails.</p><code className="block break-all rounded-xl bg-oat p-4">{recoveryCode}</code><button autoFocus className="primary-action mt-5" onClick={() => setRecoveryCode("")}>I’ve saved my code</button></section></div>}
    {showOnboarding && <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-5"><section role="dialog" aria-modal="true" aria-labelledby="demo-heading" className="max-w-xl rounded-3xl bg-white p-7"><h2 id="demo-heading" className="font-display text-3xl">Try a sample kitchen?</h2><p className="my-4">Explore a demo pantry, or start with your own ingredients.</p><div className="flex gap-4"><button className="primary-action" onClick={async () => { try { await updateItems(() => demoItems); setRecipes(demoRecipes); setDayPlans(demoPlan); localStorage.setItem("wastenotchef:onboarded","true"); setShowOnboarding(false); } catch {} }}>Load demo fridge</button><button autoFocus onClick={() => { localStorage.setItem("wastenotchef:onboarded","true"); setShowOnboarding(false); }}>Use my pantry</button></div></section></div>}
    <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8"><header className="kitchen-nav sticky top-4 z-40 px-4 py-3 backdrop-blur"><div className="flex flex-wrap items-center justify-between gap-3"><Link to="/" className="font-display text-2xl"><span className="brand-leaf" aria-hidden="true">✳</span> WasteNotChef</Link><div className="flex flex-wrap items-center gap-3"><nav className="flex flex-wrap gap-2">{[["/","Home"],["/fridge","Fridge"],["/recipes","Recipes"],["/planner","Planner"],["/dashboard","Dashboard"],["/settings","Settings"]].map(([href,label]) => <Link key={href} to={href} aria-current={location.pathname === href ? "page" : undefined} className={`rounded-full px-4 py-2 text-sm font-semibold ${location.pathname === href ? "bg-ink text-white" : "text-slate-600"}`}>{label}</Link>)}</nav><span className="rounded-full bg-mist px-3 py-2 text-xs">{session.name}</span><button disabled={pending || recipesBusy} onClick={() => void logout()} className="text-xs">Log out</button>{session.mode === "guest" && <button disabled={pending} onClick={() => { setSession(null); localStorage.removeItem("wastenotchef:session"); }} className="text-xs font-semibold">Sign in / Join</button>}</div></div></header>
      <main className="py-8">{(appError || pantry.error) && <div role="alert" className="mb-5 rounded-2xl bg-amber-50 p-4 text-sm text-amber-900">{appError || pantry.error}{pantry.error && session.mode === "account" && <button className="ml-3 underline" onClick={() => void pantry.refresh()}>Refresh pantry</button>}</div>}
        <Routes><Route path="/" element={<Landing onDemoUpload={handleFile} busy={pending} />} />
          <Route path="/fridge" element={<><div className="mb-4 flex flex-wrap items-center justify-between gap-3 text-xs"><p role="status">{pantry.status}</p>{session.mode === "account" && <div className="flex gap-4"><button disabled={pending} onClick={() => void pantry.refresh()}>Refresh pantry</button>{readGuestPantry().length > 0 && <button disabled={pending} onClick={() => void addItems(readGuestPantry()).catch(() => {})}>Import pantry from this device</button>}</div>}</div><fieldset disabled={pending} className="min-w-0"><Fridge items={items} synced={session.mode === "account"} onItemsAdded={addItems} onRemoveItem={id => updateItems(current => current.filter(item => item.id !== id))} onAddItem={item => addItems([item])} onUpdateItem={(id, changes) => updateItems(current => current.map(item => item.id === id ? { ...item, ...changes } : item))} onGenerateRecipes={generateRecipes} generatingRecipes={recipesBusy} /></fieldset></>} />
          <Route path="/recipes" element={<Recipes recipes={recipes} onAddToPlan={recipe => { setDayPlans(current => [...current, { scheduledDate: new Date(Date.now()+current.length*86400000).toISOString().slice(0,10), recipe, itemsConsumed: recipe.ingredients.map(item => item.name), priority: recipe.score ?? 70, reasoning: "Added from your recipes.", leftovers: [], wasteScore: 88 }]); navigate("/planner"); }} />} />
          <Route path="/planner" element={<><button className="primary-action mb-5" disabled={pending || !items.length} onClick={() => void planCurrentPantry()}>{busy ? "Planning…" : "Plan from my pantry"}</button><Planner dayPlans={dayPlans} onReorder={setDayPlans} /></>} />
          <Route path="/dashboard" element={<Dashboard timeseries={timeseries} />} />
          <Route path="/settings" element={<Settings session={session} notificationPrefs={notificationPrefs} onUpdateNotificationPrefs={changes => setNotificationPrefs(current => ({ ...current, ...changes }))} onEnableBrowserNotifications={enableNotifications} />} />
        </Routes>
      </main></div><ChatWidget currentPage={location.pathname} pantryItems={items} />
  </div>;
}
export default App;

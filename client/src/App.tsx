import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { Link, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { demoItems, demoPlan, demoRecipes } from "./data/demo";
import { fetchRecipes, fetchWasteSeries, fetchWeekPlan, uploadPhoto } from "./lib/api";
import { Dashboard } from "./pages/Dashboard";
import { Fridge } from "./pages/Fridge";
import { Landing } from "./pages/Landing";
import { Planner } from "./pages/Planner";
import { Recipes } from "./pages/Recipes";
import { Settings } from "./pages/Settings";
import type { DayPlan, PantryItem, Recipe } from "./types";

function OnboardingModal({ open, onClose, onUseDemo }: { open: boolean; onClose: () => void; onUseDemo: () => void }) {
  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-5 backdrop-blur">
      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        className="w-full max-w-2xl rounded-[2rem] bg-white p-8 shadow-float"
      >
        <p className="text-sm font-semibold uppercase tracking-[0.35em] text-teal-700">Welcome</p>
        <h2 className="mt-3 font-display text-4xl text-ink">Try WasteNotChef with a sample fridge first.</h2>
        <p className="mt-4 text-slate-600">
          You'll land with a preloaded pantry, recipes, and an example quest so the interaction flow makes sense immediately.
        </p>
        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          <button type="button" onClick={onUseDemo} className="rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white">
            Load demo fridge
          </button>
          <button type="button" onClick={onClose} className="rounded-full border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700">
            Start empty
          </button>
        </div>
      </motion.div>
    </div>
  );
}

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [items, setItems] = useState<PantryItem[]>([]);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [dayPlans, setDayPlans] = useState<DayPlan[]>([]);
  const [busy, setBusy] = useState(false);
  const [timeseries, setTimeseries] = useState<Array<{ date: string; wasteScore: number; recipeTitle?: string }>>(
    demoPlan.map((day) => ({ date: day.scheduledDate, wasteScore: day.wasteScore, recipeTitle: day.recipe.title }))
  );
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [appError, setAppError] = useState<string | null>(null);

  useEffect(() => {
    const visited = window.localStorage.getItem("wastenotchef:onboarded");
    if (!visited) {
      setShowOnboarding(true);
    }
  }, []);

  useEffect(() => {
    console.log("[analytics-stub]", { path: location.pathname, timestamp: new Date().toISOString() });
  }, [location.pathname]);

  const navItems = useMemo(
    () => [
      ["/", "Home"],
      ["/fridge", "Fridge"],
      ["/recipes", "Recipes"],
      ["/planner", "Planner"],
      ["/dashboard", "Dashboard"],
      ["/settings", "Settings"]
    ],
    []
  );

  async function handleFile(file: File) {
    setBusy(true);
    setAppError(null);
    try {
      const uploadedItems = await uploadPhoto(file);
      if (uploadedItems.length === 0) {
        setItems([]);
        setRecipes([]);
        setDayPlans([]);
        setAppError("No confident ingredients were detected from this photo. Try a closer, better-lit image or use the demo fridge.");
        navigate("/fridge");
        return;
      }

      setItems(uploadedItems);
      const nextRecipes = await fetchRecipes(uploadedItems);
      setRecipes(nextRecipes);
      const planned = await fetchWeekPlan(uploadedItems, {
        mealsPerDay: 1,
        skipDays: [],
        preferCuisineTags: ["quick", "comfort"],
        maxLeftovers: 2
      });
      setDayPlans(planned);
      setTimeseries(planned.map((day) => ({ date: day.scheduledDate, wasteScore: day.wasteScore, recipeTitle: day.recipe.title })));
      navigate("/fridge");
    } catch (error) {
      console.warn("API unavailable during upload.", error);
      setAppError("We could not analyze this upload right now. Check the live API connection or try again in a moment.");
      navigate("/fridge");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    async function refreshWasteSeries() {
      try {
        const today = new Date();
        const from = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-01`;
        const to = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
        const result = await fetchWasteSeries(from, to);
        if (result.timeseries?.length) {
          setTimeseries(result.timeseries);
        }
      } catch (error) {
        console.warn("Waste API unavailable, staying on local series.", error);
      }
    }

    void refreshWasteSeries();
  }, []);

  function closeOnboarding(loadDemo = false) {
    window.localStorage.setItem("wastenotchef:onboarded", "true");
    setShowOnboarding(false);
    if (loadDemo) {
      setItems(demoItems);
      setRecipes(demoRecipes);
      setDayPlans(demoPlan);
      navigate("/planner");
    }
  }

  return (
    <div className="min-h-screen bg-[#f4f8fb] text-ink">
      <OnboardingModal open={showOnboarding} onClose={() => closeOnboarding(false)} onUseDemo={() => closeOnboarding(true)} />
      <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
        <header className="sticky top-4 z-40 rounded-full border border-white/70 bg-white/80 px-4 py-3 shadow-lg backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Link to="/" className="font-display text-2xl text-ink">
              WasteNotChef
            </Link>
            <nav className="flex flex-wrap gap-2">
              {navItems.map(([href, label]) => (
                <Link
                  key={href}
                  to={href}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                    location.pathname === href ? "bg-ink text-white" : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {label}
                </Link>
              ))}
            </nav>
          </div>
        </header>

        <main className="py-8">
          {appError ? (
            <div className="mb-6 rounded-[1.5rem] border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900">
              {appError}
            </div>
          ) : null}
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.3 }}
            >
              <Routes>
                <Route path="/" element={<Landing onDemoUpload={handleFile} busy={busy} />} />
                <Route
                  path="/fridge"
                  element={
                    <Fridge
                      items={items}
                      onUpdateItem={(id, updates) =>
                        setItems((current) => current.map((item) => (item.id === id ? { ...item, ...updates } : item)))
                      }
                    />
                  }
                />
                <Route
                  path="/recipes"
                  element={
                    <Recipes
                      recipes={recipes}
                      onAddToPlan={(recipe) =>
                        setDayPlans((current) => [
                          ...current,
                          {
                            scheduledDate: new Date().toISOString().slice(0, 10),
                            recipe,
                            itemsConsumed: recipe.ingredients.map((ingredient) => ingredient.name),
                            priority: recipe.score ?? 70,
                            reasoning: "Added manually from recipe suggestions.",
                            leftovers: [],
                            wasteScore: 88
                          }
                        ])
                      }
                    />
                  }
                />
                <Route path="/planner" element={<Planner dayPlans={dayPlans} onReorder={setDayPlans} />} />
                <Route path="/dashboard" element={<Dashboard timeseries={timeseries} />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;

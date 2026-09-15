import { Link } from "react-router-dom";
import { KitchenScene } from "../components/KitchenScene";

export function PublicHome() {
  return <div className="app-shell min-h-screen text-ink"><div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
    <header className="kitchen-nav flex items-center justify-between gap-4 px-5 py-4"><Link to="/" className="font-display text-2xl">✳ WasteNotChef</Link><Link to="/login" className="primary-action">Open your kitchen →</Link></header>
    <main className="landing-page py-8">
      <section className="kitchen-hero"><div className="hero-copy"><p className="eyebrow">YOUR FOOD. MORE POSSIBILITIES.</p><h1>Waste less.<br />Cook more.<br /><em>Use what you have.</em></h1><p className="hero-description">WasteNotChef turns grocery receipts and fridge photos into an organized pantry. Track ingredients, review estimated expiry dates, and find recipes for the food you already have.</p><div className="hero-actions"><Link to="/login" className="primary-action">Start your pantry →</Link><a href="#how-it-works" className="secondary-action">See how it works</a></div><p className="hero-footnote">Try guest mode on this device, or create an account to sync your pantry.</p></div><KitchenScene /></section>
      <section id="how-it-works" aria-label="How WasteNotChef works" className="feature-grid">
        {[ ["01", "Scan your groceries", "Upload a grocery receipt or a fridge photo. Review detected food items, correct quantities, and remove anything you don't need."], ["02", "Keep track of your food", "See your ingredients together with editable expiry dates. Missing dates receive food-specific estimates with storage assumptions."], ["03", "Find your next meal", "Discover recipes using your pantry and build a meal plan that prioritizes ingredients with earlier dates."] ].map(([step, title, copy]) => <article key={step} className="rounded-3xl bg-white/80 p-7"><p className="eyebrow">{step}</p><h2 className="mt-3 font-display text-3xl">{title}</h2><p className="mt-4 leading-7 text-slate-600">{copy}</p></article>)}
      </section>
      <section className="my-10 rounded-3xl bg-white/80 p-8"><h2 className="font-display text-3xl">A pantry that fits your everyday cooking.</h2><p className="my-4 max-w-3xl leading-7 text-slate-600">Use WasteNotChef to organize groceries, review what needs attention, and explore meal ideas. Guest inventory stays on your device. With an account, your pantry syncs when you sign in on another device.</p><p className="mb-6 max-w-3xl text-sm leading-6 text-slate-600">Scans can miss items, and estimated dates are planning guides. Review results and follow the food's packaging and storage instructions.</p><Link to="/login" className="primary-action">Get started →</Link></section>
    </main><footer className="py-6 text-sm text-slate-600">WasteNotChef · A little care for the food you bring home.</footer>
  </div></div>;
}

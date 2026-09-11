import { ArrowRightIcon, CameraIcon, CalendarDaysIcon, ReceiptPercentIcon } from "@heroicons/react/24/outline";
import { Link } from "react-router-dom";
import { CameraUpload } from "../components/CameraUpload";
import { KitchenScene } from "../components/KitchenScene";

type LandingProps = { onDemoUpload: (file: File) => Promise<void>; busy: boolean };
export function Landing({ onDemoUpload, busy }: LandingProps) {
  return (
    <div className="landing-page">
      <section className="kitchen-hero">
        <div className="hero-copy">
          <p className="eyebrow"><span className="fresh-dot" /> YOUR INGREDIENTS. THEIR NEXT CHAPTER.</p>
          <h1>Good food.<br />Great possibilities.<br /><em>Less waste.</em></h1>
          <p className="hero-description">That spinach in the back? Tomorrow’s favorite meal. Turn your groceries into a little inspiration, and a plan you’ll actually want to cook.</p>
          <div className="hero-actions"><Link className="primary-action" to="/fridge">Open my kitchen <ArrowRightIcon className="h-4 w-4" /></Link><a className="secondary-action" href="#scan">Scan a fridge photo ↗</a></div>
          <div className="hero-footnote"><span>01 / Scan</span><span>02 / Review</span><span>03 / Make something good</span></div>
        </div>
        <KitchenScene />
      </section>
      <div className="kitchen-divider"><span>A FRESH TAKE ON EVERYDAY COOKING</span><span>Less guessing. More using what’s there.</span></div>
      <section className="feature-grid" aria-label="Ways to use your kitchen">
        {[
          { icon: ReceiptPercentIcon, n: "01", title: "The shop, sorted.", copy: "Scan your receipt. Bring food purchases into your pantry, then review the details.", href: "/fridge", label: "Scan a receipt" },
          { icon: CameraIcon, n: "02", title: "Meet your ingredients.", copy: "A fridge photo is a starting point. Keep the good finds and correct anything we miss.", href: "/fridge", label: "Explore my pantry" },
          { icon: CalendarDaysIcon, n: "03", title: "Make a meal of it.", copy: "Find recipes for what you have and give your week a delicious little head start.", href: "/planner", label: "See my meal plan" }
        ].map(({ icon: Icon, n, title, copy, href, label }) => <Link to={href} className="feature-tile" key={n}><div className="flex items-center justify-between"><span className="feature-icon"><Icon className="h-6 w-6" /></span><span className="feature-number">{n}</span></div><h2>{title}</h2><p>{copy}</p><span className="feature-link">{label}<ArrowRightIcon className="h-4 w-4" /></span></Link>)}
      </section>
      <section id="scan" className="scan-section"><div><p className="eyebrow">A SMALL HABIT, A FRESH START</p><h2>What’s in your fridge<br /><em>could be dinner.</em></h2><p>Take a clear photo of your ingredients or their labels. We’ll help you find a place to start.</p><Link to="/fridge" className="secondary-action">Just got groceries? Scan your receipt →</Link></div><CameraUpload onFileSelected={onDemoUpload} busy={busy} /></section>
      <footer className="kitchen-footer"><span>WasteNotChef <span aria-hidden="true">✳</span></span><span>A little care for the food you bring home.</span></footer>
    </div>
  );
}

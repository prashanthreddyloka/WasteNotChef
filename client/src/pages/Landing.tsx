import { ArrowRightIcon } from "@heroicons/react/24/outline";
import { motion, useScroll, useTransform } from "framer-motion";
import { Link } from "react-router-dom";
import { CameraUpload } from "../components/CameraUpload";

type LandingProps = {
  onDemoUpload: (file: File) => Promise<void>;
  busy: boolean;
};

export function Landing({ onDemoUpload, busy }: LandingProps) {
  const { scrollYProgress } = useScroll();
  const y = useTransform(scrollYProgress, [0, 1], [0, -90]);

  return (
    <div className="overflow-hidden">
      <section className="relative rounded-[2.5rem] bg-hero px-6 py-10 sm:px-10 lg:px-14 lg:py-14">
        <motion.div style={{ y }} className="pointer-events-none absolute -right-16 top-8 h-48 w-48 rounded-full bg-coral/20 blur-3xl" />
        <motion.div style={{ y }} className="pointer-events-none absolute -left-10 bottom-10 h-56 w-56 rounded-full bg-teal-200/30 blur-3xl" />
        <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="max-w-2xl">
            <motion.p initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} className="text-sm font-semibold uppercase tracking-[0.4em] text-teal-700">
              Waste less, cook smarter
            </motion.p>
            <motion.h1 initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }} className="mt-4 font-display text-5xl leading-tight text-ink sm:text-6xl">
              Snap your fridge. Get a week-long cooking quest.
            </motion.h1>
            <motion.p initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.16 }} className="mt-5 max-w-xl text-lg leading-8 text-slate-600">
              WasteNotChef turns a messy fridge photo into detected ingredients, ranked recipes, and an animated day-by-day plan that explains exactly why each meal should happen when it does.
            </motion.p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/planner" className="rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-900">
                See planner flow
              </Link>
              <a href="#features" className="rounded-full border border-slate-300 bg-white/80 px-5 py-3 text-sm font-semibold text-slate-700">
                Explore features
              </a>
            </div>
            <div className="mt-8 grid gap-3 sm:grid-cols-3">
              {[
                ["12 sec", "From photo to first plan"],
                ["EDF", "At-risk ingredients scheduled first"],
                ["PWA-ready", "Camera + responsive mobile flow"]
              ].map(([value, label]) => (
                <div key={label} className="rounded-[1.5rem] border border-white/70 bg-white/70 p-4 backdrop-blur">
                  <div className="text-2xl font-semibold text-ink">{value}</div>
                  <div className="mt-1 text-sm text-slate-500">{label}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-5">
            <CameraUpload onFileSelected={onDemoUpload} busy={busy} />
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="rounded-[2rem] border border-white/70 bg-ink p-5 text-white shadow-float"
            >
              <p className="text-sm uppercase tracking-[0.35em] text-teal-200">How it feels</p>
              <div className="mt-4 space-y-4">
                {[
                  "1. Capture a fresh photo from the fridge door.",
                  "2. Review detected ingredients and fix expiry guesses in seconds.",
                  "3. Drop the resulting quest into your week with clear reasoning."
                ].map((step) => (
                  <div key={step} className="flex items-start gap-3 rounded-2xl bg-white/10 p-3">
                    <ArrowRightIcon className="mt-1 h-4 w-4 flex-none text-teal-200" />
                    <span className="text-sm text-white/85">{step}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <section id="features" className="mt-12 grid gap-5 lg:grid-cols-3">
        {[
          ["Camera-first upload", "Mobile capture with graceful file fallback, OCR extraction, and editable expiry controls."],
          ["Quest planner", "Animated day cards, drag-to-reorder timeline, and explainable scheduling logic."],
          ["Waste intelligence", "Track score trends, export plans, and nudge yourself before ingredients slip."]
        ].map(([title, copy], index) => (
          <motion.article
            key={title}
            initial={{ opacity: 0, y: 22 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.08 }}
            className="rounded-[2rem] border border-slate-200 bg-white/75 p-6 shadow-float backdrop-blur"
          >
            <div className="mb-4 h-12 w-12 rounded-2xl bg-mist" />
            <h2 className="font-display text-2xl text-ink">{title}</h2>
            <p className="mt-3 text-sm leading-7 text-slate-600">{copy}</p>
          </motion.article>
        ))}
      </section>

      <section className="mt-12 grid gap-6 rounded-[2.5rem] border border-slate-200 bg-white/80 p-8 shadow-float lg:grid-cols-[1fr_0.8fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-coral">Pricing</p>
          <h2 className="mt-3 font-display text-4xl text-ink">Start free, upgrade when you want shared planning.</h2>
          <p className="mt-4 max-w-xl text-slate-600">
            The demo experience is fully open. A future paid tier can cover collaboration, advanced notifications, and household insights.
          </p>
        </div>
        <div className="rounded-[2rem] bg-oat p-6">
          <div className="text-sm uppercase tracking-[0.3em] text-slate-500">Placeholder pricing</div>
          <div className="mt-4 text-5xl font-display text-ink">$0</div>
          <div className="mt-2 text-slate-500">Solo pantry planning</div>
          <div className="mt-6 rounded-full bg-ink px-5 py-3 text-center text-sm font-semibold text-white">Household tier coming soon</div>
        </div>
      </section>
    </div>
  );
}

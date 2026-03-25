import { AnimatePresence } from "framer-motion";
import { RecipeCard } from "../components/RecipeCard";
import type { Recipe } from "../types";

type RecipesProps = {
  recipes: Recipe[];
  onAddToPlan: (recipe: Recipe) => void;
};

export function Recipes({ recipes, onAddToPlan }: RecipesProps) {
  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.3em] text-coral">Recipe engine</p>
        <h1 className="mt-2 font-display text-4xl text-ink">Prioritized suggestions</h1>
        <p className="mt-3 max-w-2xl text-slate-600">
          Matches are ranked by pantry coverage first, then optional substitutions keep momentum when you are one ingredient short.
        </p>
      </div>

      <AnimatePresence>
        <div className="grid gap-4 lg:grid-cols-2">
          {recipes.map((recipe) => (
            <RecipeCard key={recipe.id} recipe={recipe} onAdd={onAddToPlan} />
          ))}
        </div>
      </AnimatePresence>
    </div>
  );
}

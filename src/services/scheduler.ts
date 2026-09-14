import { addDays } from "date-fns";
import { daysUntil, formatDate, toDate } from "../utils/dates";
import { canonicalFoodName } from "../../client/src/lib/shelfLife";

export type PlannerItem = {
  id: string;
  name: string;
  quantity?: string;
  detectedExpiry?: string | null;
  inferredExpiry?: string | null;
  importanceWeight?: number;
};

export type PlannerRecipe = {
  id: string;
  title: string;
  tags: string[];
  cookTime: number;
  ingredients: Array<{ name: string; qty: string; optional?: boolean }>;
  steps: string[];
};

export type PlannerPreferences = {
  mealsPerDay: number;
  skipDays: number[];
  preferCuisineTags: string[];
  maxLeftovers: number;
};

export type PlannedDay = {
  scheduledDate: string;
  recipe: PlannerRecipe;
  itemsConsumed: string[];
  priority: number;
  reasoning: string;
  leftovers: string[];
  wasteScore: number;
};

export type WeekPlanResult = {
  dayPlans: PlannedDay[];
  wasteProjection: { weeklyWasteScore: number; atRiskItems: string[] };
  reasoning: string;
  metadata: { scheduler: string; complexity: string };
};

function expiryOf(item: PlannerItem): Date | null {
  return toDate(item.detectedExpiry ?? item.inferredExpiry);
}

function recipeScore(recipe: PlannerRecipe, items: PlannerItem[], preferences: PlannerPreferences, dayOffset: number, referenceDate: Date): number {
  const available = new Set(items.map((item) => canonicalFoodName(item.name)));
  const matched = recipe.ingredients.filter((ingredient) => available.has(canonicalFoodName(ingredient.name)));
  const expiringSoonBonus = matched.reduce((sum, ingredient) => {
    const item = items.find((entry) => canonicalFoodName(entry.name) === canonicalFoodName(ingredient.name));
    const expiry = item ? expiryOf(item) : null;
    return sum + (expiry ? Math.max(0, 8 - daysUntil(expiry, referenceDate)) : 0);
  }, 0);
  const preferenceBonus = recipe.tags.some((tag) => preferences.preferCuisineTags.includes(tag)) ? 3 : 0;
  const leftoverPenalty = Math.max(0, matched.length - preferences.maxLeftovers);
  return matched.length * 10 + expiringSoonBonus + preferenceBonus - leftoverPenalty - dayOffset * 0.15;
}

export function calculateWasteScore(items: PlannerItem[], plannedItems: string[], referenceDate = new Date()): number {
  const planned = new Set(plannedItems.map(canonicalFoodName));
  const penalty = items.reduce((sum, item) => {
    const expiry = expiryOf(item);
    const expired = expiry ? daysUntil(expiry, referenceDate) < 0 : false;
    const unused = !planned.has(canonicalFoodName(item.name));
    return sum + (expired || unused ? 12 * (item.importanceWeight ?? 1) : 0);
  }, 0);

  return Math.max(0, Math.min(100, Number((100 - penalty).toFixed(2))));
}

export function planWeek(
  items: PlannerItem[],
  recipes: PlannerRecipe[],
  preferences: PlannerPreferences,
  startDate = new Date()
): WeekPlanResult {
  const activeItems = [...items].sort((a, b) => {
    const aExpiry = expiryOf(a);
    const bExpiry = expiryOf(b);
    return (aExpiry?.getTime() ?? Number.MAX_SAFE_INTEGER) - (bExpiry?.getTime() ?? Number.MAX_SAFE_INTEGER);
  });

  const usedItemIds = new Set<string>();
  const dayPlans: PlannedDay[] = [];

  for (let dayOffset = 0; dayOffset < 7; dayOffset += 1) {
    if (preferences.skipDays.includes(dayOffset)) {
      continue;
    }

    const scheduledDate = addDays(startDate, dayOffset);
    for (let meal = 0; meal < preferences.mealsPerDay; meal += 1) {
    const remaining = activeItems.filter(item => !usedItemIds.has(item.id) && (!expiryOf(item) || daysUntil(expiryOf(item)!, scheduledDate) >= 0));
    const earliest = remaining.find(item => recipes.some(recipe => recipe.ingredients.some(ingredient => canonicalFoodName(ingredient.name) === canonicalFoodName(item.name))));
    if (!earliest) break;
    const rankedRecipes = recipes.filter(recipe => recipe.ingredients.some(ingredient => canonicalFoodName(ingredient.name) === canonicalFoodName(earliest.name)))
      .map((recipe) => ({ recipe, score: recipeScore(recipe, remaining, preferences, dayOffset, scheduledDate) }))
      .sort((a, b) => b.score - a.score);
    const chosen = rankedRecipes.find(({ recipe }) =>
      recipe.ingredients.some((ingredient) =>
        remaining.some((item) => canonicalFoodName(item.name) === canonicalFoodName(ingredient.name))
      )
    ) ?? rankedRecipes[0];

    if (!chosen) {
      continue;
    }

    const itemsConsumed = chosen.recipe.ingredients
      .filter((ingredient) => remaining.some((item) => canonicalFoodName(item.name) === canonicalFoodName(ingredient.name)))
      .map((ingredient) => ingredient.name);

    itemsConsumed.forEach((name) => {
      const match = remaining.find((item) => canonicalFoodName(item.name) === canonicalFoodName(name) && !usedItemIds.has(item.id));
      if (match) {
        usedItemIds.add(match.id);
      }
    });

    const firstAtRisk = earliest;
    const expiry = firstAtRisk ? expiryOf(firstAtRisk) : null;
    const days = expiry ? daysUntil(expiry, scheduledDate) : null;
    const missing = chosen.recipe.ingredients
      .filter((ingredient) => !itemsConsumed.includes(ingredient.name))
      .map((ingredient) => ingredient.name);

    dayPlans.push({
      scheduledDate: formatDate(scheduledDate) as string,
      recipe: chosen.recipe,
      itemsConsumed,
      leftovers: [],
      priority: Number((chosen.score + (days !== null ? Math.max(0, 5 - days) : 0)).toFixed(2)),
      reasoning: expiry
        ? `Uses ${firstAtRisk?.name} due in ${days} day(s).${missing.length ? ` You may need: ${missing.join(", ")}.` : ""}`
        : "Best coverage match for current pantry and preferences.",
      wasteScore: calculateWasteScore(
        activeItems,
        Array.from(usedItemIds).map((id) => activeItems.find((item) => item.id === id)?.name ?? ""),
        scheduledDate
      )
    });
    }
  }

  return {
    dayPlans,
    wasteProjection: {
      weeklyWasteScore: calculateWasteScore(
        activeItems,
        Array.from(usedItemIds).map((id) => activeItems.find((item) => item.id === id)?.name ?? ""), startDate
      ),
      atRiskItems: activeItems
        .filter((item) => {
          const expiry = expiryOf(item);
          return expiry ? daysUntil(expiry, startDate) <= 3 : false;
        })
        .map((item) => item.name)
    },
    reasoning: "Deterministic EDF scheduler with utilization and preference-aware tie-breaks. Complexity is O(7 * R * I).",
    metadata: {
      scheduler: "earliest-deadline-first-with-utilization-tiebreak",
      complexity: "O(D * R * I) where D=7 days, R=recipes, I=items"
    }
  };
}

import { addDays } from "date-fns";
import { daysUntil, formatDate, toDate } from "../utils/dates";

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

function recipeScore(recipe: PlannerRecipe, items: PlannerItem[], preferences: PlannerPreferences, dayOffset: number): number {
  const available = new Set(items.map((item) => item.name.toLowerCase()));
  const matched = recipe.ingredients.filter((ingredient) => available.has(ingredient.name.toLowerCase()));
  const expiringSoonBonus = matched.reduce((sum, ingredient) => {
    const item = items.find((entry) => entry.name.toLowerCase() === ingredient.name.toLowerCase());
    const expiry = item ? expiryOf(item) : null;
    return sum + (expiry ? Math.max(0, 8 - daysUntil(expiry)) : 0);
  }, 0);
  const preferenceBonus = recipe.tags.some((tag) => preferences.preferCuisineTags.includes(tag)) ? 3 : 0;
  const leftoverPenalty = Math.max(0, matched.length - preferences.maxLeftovers);
  return matched.length * 10 + expiringSoonBonus + preferenceBonus - leftoverPenalty - dayOffset * 0.15;
}

export function calculateWasteScore(items: PlannerItem[], plannedItems: string[], referenceDate = new Date()): number {
  const penalty = items.reduce((sum, item) => {
    const expiry = expiryOf(item);
    const expired = expiry ? daysUntil(expiry, referenceDate) < 0 : false;
    const unused = !plannedItems.includes(item.name);
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
    const rankedRecipes = recipes
      .map((recipe) => ({ recipe, score: recipeScore(recipe, activeItems, preferences, dayOffset) }))
      .sort((a, b) => b.score - a.score);
    const chosen = rankedRecipes.find(({ recipe }) =>
      recipe.ingredients.some((ingredient) =>
        activeItems.some((item) => item.name.toLowerCase() === ingredient.name.toLowerCase() && !usedItemIds.has(item.id))
      )
    ) ?? rankedRecipes[0];

    if (!chosen) {
      continue;
    }

    const itemsConsumed = chosen.recipe.ingredients
      .filter((ingredient) => activeItems.some((item) => item.name.toLowerCase() === ingredient.name.toLowerCase()))
      .map((ingredient) => ingredient.name);

    itemsConsumed.forEach((name) => {
      const match = activeItems.find((item) => item.name.toLowerCase() === name.toLowerCase() && !usedItemIds.has(item.id));
      if (match) {
        usedItemIds.add(match.id);
      }
    });

    const firstAtRisk = itemsConsumed
      .map((name) => activeItems.find((item) => item.name.toLowerCase() === name.toLowerCase()))
      .find(Boolean);
    const expiry = firstAtRisk ? expiryOf(firstAtRisk) : null;
    const days = expiry ? daysUntil(expiry, scheduledDate) : null;
    const leftovers = chosen.recipe.ingredients
      .filter((ingredient) => !itemsConsumed.includes(ingredient.name))
      .map((ingredient) => ingredient.name);

    dayPlans.push({
      scheduledDate: formatDate(scheduledDate) as string,
      recipe: chosen.recipe,
      itemsConsumed,
      leftovers,
      priority: Number((chosen.score + (days !== null ? Math.max(0, 5 - days) : 0)).toFixed(2)),
      reasoning: expiry
        ? `Uses ${firstAtRisk?.name} expiring in ${days} day(s); EDF prioritized with utilization tie-breakers.`
        : "Best coverage match for current pantry and preferences.",
      wasteScore: calculateWasteScore(
        activeItems,
        Array.from(usedItemIds).map((id) => activeItems.find((item) => item.id === id)?.name ?? ""),
        scheduledDate
      )
    });
  }

  return {
    dayPlans,
    wasteProjection: {
      weeklyWasteScore: calculateWasteScore(
        activeItems,
        Array.from(usedItemIds).map((id) => activeItems.find((item) => item.id === id)?.name ?? "")
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

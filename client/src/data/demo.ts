import type { DayPlan, PantryItem, Recipe } from "../types";

export const demoItems: PantryItem[] = [
  { id: "milk-1", name: "milk", confidence: 0.91, detectedExpiry: "2026-03-27", notes: "Door shelf carton" },
  { id: "spinach-2", name: "spinach", confidence: 0.88, inferredExpiry: "2026-03-28", notes: "Crisper drawer bunch" },
  { id: "eggs-3", name: "eggs", confidence: 0.95, detectedExpiry: "2026-03-30", notes: "Half-dozen remaining" },
  { id: "cheese-4", name: "cheese", confidence: 0.86, inferredExpiry: "2026-04-03", notes: "Sharp cheddar block" },
  { id: "broccoli-5", name: "broccoli", confidence: 0.84, inferredExpiry: "2026-03-29", notes: "Top shelf" },
  { id: "tortillas-6", name: "tortillas", confidence: 0.8, inferredExpiry: "2026-04-01", notes: "Opened package" }
];

export const demoRecipes: Recipe[] = [
  {
    id: "garden-frittata",
    title: "Garden Frittata",
    tags: ["breakfast", "vegetarian"],
    cookTime: 20,
    steps: ["Whisk eggs.", "Saute spinach.", "Cook gently until just set."],
    ingredients: [
      { name: "eggs", qty: "6" },
      { name: "spinach", qty: "1 cup" },
      { name: "cheese", qty: "1/2 cup", optional: true }
    ],
    score: 28,
    coverage: "3/3",
    substitutionSuggestions: []
  },
  {
    id: "tortilla-melts",
    title: "Loaded Tortilla Melts",
    tags: ["quick", "comfort"],
    cookTime: 15,
    steps: ["Fill tortillas.", "Toast until crisp.", "Serve with greens."],
    ingredients: [
      { name: "tortillas", qty: "4" },
      { name: "cheese", qty: "1 cup" },
      { name: "spinach", qty: "1 cup", optional: true }
    ],
    score: 24,
    coverage: "3/3",
    substitutionSuggestions: []
  }
];

export const demoPlan: DayPlan[] = [
  {
    scheduledDate: "2026-03-25",
    recipe: demoRecipes[0],
    itemsConsumed: ["milk", "eggs", "spinach"],
    priority: 92,
    reasoning: "Uses milk expiring in 2 days; EDF keeps the most fragile items in play first.",
    leftovers: ["cheese"],
    wasteScore: 90
  },
  {
    scheduledDate: "2026-03-26",
    recipe: demoRecipes[1],
    itemsConsumed: ["tortillas", "cheese", "spinach"],
    priority: 84,
    reasoning: "Batches remaining greens with tortillas to avoid a second fridge linger day.",
    leftovers: [],
    wasteScore: 94
  }
];

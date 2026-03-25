export type PantryItem = {
  id: string;
  name: string;
  quantity?: string;
  detectedExpiry?: string | null;
  inferredExpiry?: string | null;
  confidence: number;
  bbox?: { x: number; y: number; width: number; height: number };
  notes?: string;
};

export type Recipe = {
  id: string;
  title: string;
  tags: string[];
  cookTime: number;
  steps: string[];
  ingredients: Array<{ name: string; qty: string; optional?: boolean }>;
  score?: number;
  coverage?: string;
  substitutionSuggestions?: string[];
};

export type DayPlan = {
  scheduledDate: string;
  recipe: Recipe;
  itemsConsumed: string[];
  priority: number;
  reasoning: string;
  leftovers: string[];
  wasteScore: number;
};

export type PlannerPreferences = {
  mealsPerDay: number;
  skipDays: number[];
  preferCuisineTags: string[];
  maxLeftovers: number;
};

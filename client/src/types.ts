export type PantryItem = {
  id: string;
  name: string;
  quantity?: string;
  detectedExpiry?: string | null;
  inferredExpiry?: string | null;
  addedAt?: string;
  expiryBasis?: string;
  expiryKind?: "estimate" | "review";
  confidence: number;
  detectionSource?: "visual" | "ocr" | "gemini" | "merged" | "manual";
  expirySource?: "ocr" | "rule" | "manual" | "none";
  reviewed?: boolean;
  bbox?: { x: number; y: number; width: number; height: number };
  notes?: string;
};

export type Recipe = {
  id: string;
  title: string;
  tags: string[];
  country?: string;
  continent?: string;
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

export type SessionUser = {
  mode: "guest" | "account";
  id?: string;
  name: string;
  email?: string;
};

export type NotificationPrefs = {
  webPushEnabled: boolean;
  emailEnabled: boolean;
  email?: string;
  reminderDays: number;
  browserPermission: "default" | "granted" | "denied";
};

import type { DayPlan, PantryItem, PlannerPreferences, Recipe } from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "http://localhost:4000/api";

export async function uploadPhoto(file: File): Promise<PantryItem[]> {
  const formData = new FormData();
  formData.append("image", file);
  const response = await fetch(`${API_BASE}/upload-photo`, { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error("Upload failed");
  }
  const data = await response.json();
  return data.items;
}

export async function fetchRecipes(items: PantryItem[]): Promise<Recipe[]> {
  const response = await fetch(`${API_BASE}/recipes/from-items`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items: items.map(({ name }) => ({ name })) })
  });
  if (!response.ok) {
    throw new Error("Recipe fetch failed");
  }
  const data = await response.json();
  return data.recipes;
}

export async function fetchWeekPlan(items: PantryItem[], preferences: PlannerPreferences): Promise<DayPlan[]> {
  const response = await fetch(`${API_BASE}/plan-week`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      items: items.map((item) => ({
        id: item.id,
        name: item.name,
        quantity: item.quantity,
        detectedExpiry: item.detectedExpiry,
        inferredExpiry: item.inferredExpiry
      })),
      preferences
    })
  });
  if (!response.ok) {
    throw new Error("Planning failed");
  }
  const data = await response.json();
  return data.weekPlan;
}

export async function fetchWasteSeries(from: string, to: string) {
  const response = await fetch(`${API_BASE}/waste-score?from=${from}&to=${to}`);
  if (!response.ok) {
    throw new Error("Waste fetch failed");
  }
  return response.json();
}

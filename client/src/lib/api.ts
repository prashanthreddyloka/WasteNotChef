import type { DayPlan, PantryItem, PlannerPreferences, Recipe, SessionUser } from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "http://localhost:4000/api";

export const getSessionToken = () => window.sessionStorage.getItem("wastenotchef:token");
export function setSessionToken(token: string | null) {
  if (token) window.sessionStorage.setItem("wastenotchef:token", token);
  else window.sessionStorage.removeItem("wastenotchef:token");
}
export class ApiError extends Error { constructor(message: string, public status: number) { super(message); } }
export async function apiFetch(url: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers);
  const token = getSessionToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(url, { ...options, headers });
}
async function jsonResult(response: Response) {
  const data = await response.json();
  if (!response.ok) throw new ApiError(data.error ?? "Request failed. Please try again.", response.status);
  return data;
}
export async function authenticate(mode: "login" | "register" | "recover", input: { email: string; password: string; name?: string; recoveryCode?: string }): Promise<{ user: SessionUser; token: string; recoveryCode?: string }> {
  return jsonResult(await fetch(`${API_BASE}/auth/${mode}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input) }));
}
export async function currentAccount(): Promise<SessionUser> { return (await jsonResult(await apiFetch(`${API_BASE}/auth/me`))).user; }
export async function endSession() {
  const response = await apiFetch(`${API_BASE}/auth/logout`, { method: "POST" });
  if (!response.ok && response.status !== 401) throw new Error("Could not sign out. Please try again.");
  setSessionToken(null);
}
export type InventorySnapshot = { items: PantryItem[]; version: number };
export async function loadInventory(): Promise<InventorySnapshot> { return jsonResult(await apiFetch(`${API_BASE}/inventory`)); }
export async function saveInventory(items: PantryItem[], version: number): Promise<InventorySnapshot> {
  return jsonResult(await apiFetch(`${API_BASE}/inventory`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ items, version }) }));
}

export async function uploadReceipt(file: File): Promise<{ items: PantryItem[]; skippedLines: number }> {
  const formData = new FormData();
  formData.append("image", file);
  const response = await apiFetch(`${API_BASE}/upload-receipt`, { method: "POST", body: formData });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error ?? "Receipt scan failed. Please try again.");
  }
  return response.json();
}

export async function uploadPhoto(file: File): Promise<{ items: PantryItem[]; recognition: "vision" | "labels" }> {
  const formData = new FormData();
  formData.append("image", file);
  const response = await apiFetch(`${API_BASE}/upload-photo`, { method: "POST", body: formData });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error ?? "Photo scan failed. Please try again.");
  }
  const data = await response.json();
  return data;
}

export async function fetchRecipes(items: PantryItem[]): Promise<Recipe[]> {
  const response = await apiFetch(`${API_BASE}/recipes/from-items`, {
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
  const response = await apiFetch(`${API_BASE}/plan-week`, {
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
  const response = await apiFetch(`${API_BASE}/waste-score?from=${from}&to=${to}`);
  if (!response.ok) {
    throw new Error("Waste fetch failed");
  }
  return response.json();
}

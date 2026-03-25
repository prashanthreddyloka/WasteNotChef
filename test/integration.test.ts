import { describe, expect, it } from "vitest";
import { planWeek, type PlannerItem, type PlannerRecipe } from "../src/services/scheduler";

describe("EDF integration", () => {
  it("schedules at-risk pantry items earliest across a week", () => {
    const items: PlannerItem[] = [
      { id: "1", name: "milk", detectedExpiry: "2026-03-26" },
      { id: "2", name: "spinach", detectedExpiry: "2026-03-27" },
      { id: "3", name: "eggs", detectedExpiry: "2026-03-28" },
      { id: "4", name: "chicken", detectedExpiry: "2026-03-29" },
      { id: "5", name: "broccoli", detectedExpiry: "2026-03-30" },
      { id: "6", name: "rice", inferredExpiry: "2026-04-15" }
    ];

    const recipes: PlannerRecipe[] = [
      {
        id: "r1",
        title: "Breakfast Bowl",
        tags: ["breakfast"],
        cookTime: 10,
        ingredients: [{ name: "milk", qty: "1 cup" }, { name: "eggs", qty: "2" }],
        steps: []
      },
      {
        id: "r2",
        title: "Chicken Rice",
        tags: ["dinner"],
        cookTime: 25,
        ingredients: [{ name: "chicken", qty: "1 lb" }, { name: "broccoli", qty: "2 cups" }, { name: "rice", qty: "2 cups" }],
        steps: []
      },
      {
        id: "r3",
        title: "Green Scramble",
        tags: ["quick"],
        cookTime: 15,
        ingredients: [{ name: "spinach", qty: "1 cup" }, { name: "eggs", qty: "2" }],
        steps: []
      }
    ];

    const result = planWeek(
      items,
      recipes,
      {
        mealsPerDay: 1,
        skipDays: [],
        preferCuisineTags: [],
        maxLeftovers: 2
      },
      new Date("2026-03-25")
    );

    expect(result.dayPlans[0].itemsConsumed).toContain("milk");
    expect(result.dayPlans[1].itemsConsumed.join(",")).toMatch(/spinach|chicken/);
  });
});

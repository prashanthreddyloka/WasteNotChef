import { describe, expect, it } from "vitest";
import { calculateWasteScore, planWeek, type PlannerItem, type PlannerRecipe } from "../src/services/scheduler";

const recipes: PlannerRecipe[] = [
  {
    id: "omelette",
    title: "Omelette",
    tags: ["breakfast"],
    cookTime: 15,
    ingredients: [{ name: "eggs", qty: "2" }, { name: "spinach", qty: "1 cup", optional: true }],
    steps: []
  },
  {
    id: "salad",
    title: "Crunchy Salad",
    tags: ["light"],
    cookTime: 10,
    ingredients: [{ name: "lettuce", qty: "1 head" }, { name: "tomato", qty: "1", optional: true }],
    steps: []
  }
];

describe("scheduler", () => {
  it("prioritizes earliest expiry ingredients first", () => {
    const items: PlannerItem[] = [
      { id: "1", name: "lettuce", detectedExpiry: "2026-03-26" },
      { id: "2", name: "eggs", detectedExpiry: "2026-03-29" },
      { id: "3", name: "spinach", detectedExpiry: "2026-03-27" }
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

    expect(result.dayPlans[0].recipe.title).toBe("Crunchy Salad");
    expect(result.dayPlans[0].itemsConsumed).toContain("lettuce");
  });

  it("calculates lower waste scores when items are unused or expired", () => {
    const items: PlannerItem[] = [
      { id: "1", name: "milk", detectedExpiry: "2026-03-24", importanceWeight: 1.5 },
      { id: "2", name: "bread", detectedExpiry: "2026-03-30", importanceWeight: 1 }
    ];

    expect(calculateWasteScore(items, ["bread"], new Date("2026-03-25"))).toBeLessThan(100);
  });
});

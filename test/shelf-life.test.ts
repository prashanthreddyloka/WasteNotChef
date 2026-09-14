import { describe, expect, it } from "vitest";
import { shelfLifeFor, withDefaultExpiry } from "../client/src/lib/shelfLife";

describe("default expiry dates", () => {
  const start = new Date(2026, 8, 12, 12);
  it("assigns different dates to tomatoes and eggs and resolves common names", () => {
    expect(withDefaultExpiry({ name: "cherry tomatoes" }, start).inferredExpiry).toBe("2026-09-14");
    expect(withDefaultExpiry({ name: "eggs" }, start).inferredExpiry).toBe("2026-10-03");
    expect(shelfLifeFor("organic tomatoes").days).toBe(2);
    expect(shelfLifeFor("capsicum").days).toBe(4);
  });
  it("preserves entered dates and never extends an estimate on refresh", () => {
    const entered = { name: "eggs", detectedExpiry: "2026-09-15" };
    expect(withDefaultExpiry(entered, start)).toEqual(entered);
    const estimated = withDefaultExpiry({ name: "eggs" }, start);
    expect(withDefaultExpiry(estimated, new Date(2026, 9, 1))).toEqual(estimated);
    expect(withDefaultExpiry({ ...estimated, name: "tomatoes", inferredExpiry: null }, new Date(2026, 9, 1)).inferredExpiry).toBe("2026-09-14");
  });
  it("distinguishes preparation and gives unknown items a review reminder", () => {
    expect(shelfLifeFor("hard boiled eggs").days).toBe(7);
    expect(shelfLifeFor("cooked chicken").days).toBe(3);
    expect(shelfLifeFor("raw chicken").days).toBe(2);
    expect(shelfLifeFor("dry rice").days).toBe(180);
    expect(shelfLifeFor("cooked rice").days).toBe(3);
    expect(withDefaultExpiry({ name: "mystery container" }, start)).toMatchObject({ inferredExpiry: "2026-09-13", expiryKind: "review" });
  });
  it("handles calendar month and year boundaries", () => {
    expect(withDefaultExpiry({ name: "tomatoes" }, new Date(2026, 11, 31, 12)).inferredExpiry).toBe("2027-01-02");
  });
});

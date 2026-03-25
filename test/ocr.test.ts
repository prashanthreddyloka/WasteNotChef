import fs from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { analyzeFridgePhoto } from "../src/services/analyzeFridgePhoto";
import { normalizeCandidateDate } from "../src/utils/dates";

const fixturePath = path.resolve("test", "fixture-image.txt");

describe("analyzeFridgePhoto", () => {
  beforeEach(() => {
    fs.writeFileSync(fixturePath, "fake-image");
    process.env.MOCK_OCR_TEXT = "Milk best by 04/02/2026 spinach eggs use by 04/01/2026";
  });

  afterEach(() => {
    delete process.env.MOCK_OCR_TEXT;
    if (fs.existsSync(fixturePath)) {
      fs.unlinkSync(fixturePath);
    }
  });

  it("extracts food items and nearby expiry candidates deterministically", async () => {
    const items = await analyzeFridgePhoto(
      fixturePath,
      [
        { name: "milk", shelfLifeDays: 7 },
        { name: "spinach", shelfLifeDays: 5 },
        { name: "eggs", shelfLifeDays: 21 }
      ],
      new Date("2026-03-25")
    );

    expect(items.map((item) => item.name)).toEqual(expect.arrayContaining(["milk", "spinach", "eggs"]));
    expect(items.find((item) => item.name === "milk")?.detectedExpiry).toBe("2026-04-02");
  });

  it("parses common date formats", () => {
    expect(normalizeCandidateDate("04/02/2026")?.toISOString()).toContain("2026-04-02");
    expect(normalizeCandidateDate("Apr 2 2026")?.toISOString()).toContain("2026-04-02");
  });
});

import { describe, expect, it } from "vitest";
import { mkdir, unlink } from "node:fs/promises";
import sharp from "sharp";
import { analyzeReceipt } from "../src/services/analyzeReceipt";

// Opt-in because the first OCR run downloads Tesseract's English language model.
describe.runIf(process.env.REAL_OCR_TEST === "1")("receipt image processing", () => {
  it("reads a mildly blurred receipt with abbreviations and separate price lines", async () => {
    await mkdir("tmp", { recursive: true });
    const fixture = `tmp/blurred-receipt-${process.pid}.png`;
    const lines = ["GROCERY RECEIPT", "ORG MLK", "3.99", "BNLS CHKN 7.49", "LEMON SOAP 2.99", "TOTAL 14.47"];
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="750"><rect width="100%" height="100%" fill="white"/>${lines.map((line, index) => `<text x="60" y="${90 + index * 100}" fill="black" font-family="Arial" font-size="52">${line}</text>`).join("")}</svg>`;
    await sharp(Buffer.from(svg)).resize(700).blur(0.65).png().toFile(fixture);
    try {
      const result = await analyzeReceipt(fixture, [{ name: "milk", shelfLifeDays: 7 }]);
      expect(result.items.map(item => item.name)).toEqual(["milk", "chicken"]);
    } finally { await unlink(fixture); }
  }, 90000);
});

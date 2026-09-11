import { describe, expect, it } from "vitest";
import { parseReceipt } from "../src/services/analyzeReceipt";

const rules = [{ name: "milk", shelfLifeDays: 7 }];
const parse = (text: string, id = "scan") => parseReceipt(text, rules, id, new Date("2026-09-11T12:00:00Z"));

describe("receipt parsing", () => {
  it("keeps food purchases and excludes household goods, pet food, totals and discounts", () => {
    const result = parse("FRESH MARKET\n09/10/2026\n2 x WHL MILK 6.00\nBANANA 1.20\nALMOND MILK 4.50\nLEMON DISH SOAP 2.99\nCHICKEN DOG FOOD 7.99\nPAPER TOWELS 3.50\nMILK COUPON -1.00\nSUBTOTAL 25.19\nTAX 1.00\nVISA 26.19");
    expect(result.items.map(item => item.name)).toEqual(["milk", "bananas", "almond milk"]);
    expect(result.items[0].quantity).toBe("2");
    expect(result.items[0].inferredExpiry).toBe("2026-09-18");
    expect(result.items.every(item => item.detectedExpiry === null)).toBe(true);
    expect(result.skippedLines).toBe(9);
  });
  it("does not manufacture items for blank or unrecognized receipts", () => {
    expect(parse("").items).toEqual([]);
    expect(parse("THANK YOU\nTOTAL 9.50\nMYSTERY SKU 9.50\nsoap 2.99").items).toEqual([]);
    expect(parse("MILK MARKET 12.99\nMILK COUPON 1.00\nTEA TREE SHAMPOO 5.00").items).toEqual([]);
  });
  it("preserves repeated purchases with stable IDs for image retries", () => {
    const text = "MILK 3.00\nMILK 3.00";
    const first = parse(text).items;
    expect(first).toHaveLength(2);
    expect(first[0].id).not.toBe(first[1].id);
    expect(parse(text).items).toEqual(first);
    expect(parse(text, "different-image").items[0].id).not.toBe(first[0].id);
  });
  it("reads explicit weights without confusing prices or SKUs with quantities", () => {
    expect(parse("123456 BANANAS 1.5 lb 2.00").items[0].quantity).toBe("1.5 lb");
    expect(parse("123456 MILK 2.99").items[0].quantity).toBe("1");
  });
});

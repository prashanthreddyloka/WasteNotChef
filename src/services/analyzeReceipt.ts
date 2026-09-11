import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { FOOD_KEYWORDS, type DetectedItem, type IngredientRuleLike } from "./analyzeFridgePhoto";
import { runOcr } from "./ocr";
import { formatDate, inferExpiryDate } from "../utils/dates";

const aliases: Record<string, string> = {
  banana: "bananas", apple: "apples", egg: "eggs", carrot: "carrots",
  tomato: "tomato", tomatoes: "tomato", potatoes: "potato", avocado: "avocado",
  strawberry: "strawberries", blueberry: "blueberries", mushroom: "mushrooms",
  "grnd beef": "ground beef", "chkn brst": "chicken", "chicken breast": "chicken",
  "org milk": "milk", "whl milk": "milk", "alm mlk": "almond milk",
  "bell peppers": "bell pepper", "brn rice": "rice"
};
const extraFoods = ["oats", "flour", "sugar", "salt", "olive oil", "cereal", "coffee", "tea", "peanut butter", "honey", "pasta sauce", "tuna", "juice", "crackers", "nuts", "soup"];
const excluded = /\b(soap|detergent|cleaner|shampoo|conditioner|lotion|candle|scent|scented|towels?|tissues?|napkins?|diapers?|wipes?|toothpaste|toothbrush|bleach|dishwashing|dishwasher|sanitizer|trash|garbage|pet|dog|cat|litter|coupon|discount|savings|refund|void|return)\b/i;
const metadata = /\b(subtotal|total|tax|balance|change|cash|credit|debit|visa|mastercard|payment|tender|cashier|receipt|register|transaction|loyalty|rewards|store|market|supermarket|grocery|address|phone|thank|welcome)\b/i;

export function parseReceipt(text: string, rules: IngredientRuleLike[], receiptId: string, referenceDate = new Date()): { items: DetectedItem[]; skippedLines: number } {
  const vocabulary = [...new Set([...FOOD_KEYWORDS, ...extraFoods, ...rules.map(rule => rule.name.toLowerCase()), ...Object.keys(aliases)])]
    .sort((a, b) => b.length - a.length);
  const items: DetectedItem[] = [];
  let skippedLines = 0;
  for (const [index, raw] of text.split(/\r?\n/).entries()) {
    if (!raw.trim()) continue;
    // Require a priced purchase line; receipt headings and arbitrary OCR text are not inventory.
    const price = raw.match(/\s+[$£€]?(-?\d+[.,]\d{2})\s*[A-Z*]?\s*$/i);
    const description = price ? raw.slice(0, price.index) : "";
    const normalized = description.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    const match = vocabulary.find(word => ` ${normalized} `.includes(` ${word} `));
    if (!price || Number(price[1].replace(",", ".")) <= 0 || excluded.test(raw) || metadata.test(raw) || !match) {
      skippedLines += 1;
      continue;
    }
    const name = aliases[match] ?? match;
    const rule = rules.find(entry => entry.name.toLowerCase() === name);
    // Only explicit counts/weights become quantities; prices and SKU numbers never do.
    const count = description.match(/^\s*(\d{1,3})\s*[xX@]\s*/);
    const weight = description.match(/\b(\d+(?:\.\d+)?)\s*(kg|lb|lbs|oz|g)\b/i);
    items.push({
      id: `receipt-${receiptId}-${index}`, name,
      quantity: count ? count[1] : weight ? `${weight[1]} ${weight[2].toLowerCase()}` : "1",
      confidence: 0.85, detectedExpiry: null,
      inferredExpiry: rule ? formatDate(inferExpiryDate(referenceDate, rule.shelfLifeDays)) : null,
      notes: rule ? "Added from receipt. Expiry estimated from upload date; check packaging." : "Added from receipt. Check packaging for expiry."
    });
  }
  return { items, skippedLines };
}

export async function analyzeReceipt(imagePath: string, rules: IngredientRuleLike[]) {
  const receiptId = createHash("sha256").update(await readFile(imagePath)).digest("hex");
  const ocr = await runOcr(imagePath);
  return parseReceipt(ocr.text, rules, receiptId);
}

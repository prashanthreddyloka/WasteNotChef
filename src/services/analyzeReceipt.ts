import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { FOOD_KEYWORDS, type DetectedItem, type IngredientRuleLike } from "./analyzeFridgePhoto";
import { fuzzyMatchToken, runOcr } from "./ocr";
import sharp from "sharp";
import { withDefaultExpiry } from "../../client/src/lib/shelfLife";
import { formatDate, inferExpiryDate } from "../utils/dates";

const aliases: Record<string, string> = {
  banana: "bananas", apple: "apples", egg: "eggs", carrot: "carrots",
  tomato: "tomato", tomatoes: "tomato", potatoes: "potato", avocado: "avocado",
  strawberry: "strawberries", blueberry: "blueberries", mushroom: "mushrooms",
  "grnd beef": "ground beef", "chkn brst": "chicken", "chicken breast": "chicken",
  "org milk": "milk", "whl milk": "milk", "alm mlk": "almond milk",
  "bell peppers": "bell pepper", "brn rice": "rice",
  "bnls chkn": "chicken", "chkn": "chicken", "chk breast": "chicken", "gr beef": "ground beef",
  "grnd trky": "turkey", "grk yog": "yogurt", "grk yogurt": "yogurt", "yoghurt": "yogurt",
  "strwbry": "strawberries", "strawb": "strawberries", "bluebry": "blueberries", "bby spin": "spinach",
  "romaine": "lettuce", "rom hrt": "lettuce", "brocc": "broccoli", "caulif": "cauliflower",
  "avoc": "avocado", "avo": "avocado", "ban": "bananas", "tom": "tomato", "pot": "potato",
  "moz cheese": "mozzarella", "mozz": "mozzarella", "ched cheese": "cheese", "swr cream": "sour cream",
  "org mlk": "milk", "whl mlk": "milk", "2pct mlk": "milk", "mlk": "milk",
  "pnut btr": "peanut butter", "pb butter": "peanut butter", "evoo": "olive oil", "oj": "orange juice",
  "ww bread": "bread", "sourdgh": "bread", "tort": "tortillas", "psta": "pasta"
};
const extraFoods = ["oats", "flour", "sugar", "salt", "olive oil", "cereal", "coffee", "tea", "peanut butter", "honey", "pasta sauce", "tuna", "juice", "crackers", "nuts", "soup"];
const excluded = /\b(soap|detergent|cleaner|shampoo|conditioner|lotion|candle|scent|scented|towels?|tissues?|napkins?|diapers?|wipes?|toothpaste|toothbrush|bleach|dishwashing|dishwasher|sanitizer|trash|garbage|pet|dog|cat|litter|coupon|discount|savings|refund|void|return)\b/i;
const metadata = /\b(subtotal|total|tax|balance|change|cash|credit|debit|visa|mastercard|payment|tender|cashier|receipt|register|transaction|loyalty|rewards|store|market|supermarket|grocery|address|phone|thank|welcome)\b/i;

export function parseReceipt(text: string, rules: IngredientRuleLike[], receiptId: string, referenceDate = new Date()): { items: DetectedItem[]; skippedLines: number } {
  const vocabulary = [...new Set([...FOOD_KEYWORDS, ...extraFoods, ...rules.map(rule => rule.name.toLowerCase()), ...Object.keys(aliases)])]
    .sort((a, b) => b.length - a.length);
  const items: DetectedItem[] = [];
  let skippedLines = 0;
  const lines = text.split(/\r?\n/).filter(line => line.trim());
  for (let index = 0; index < lines.length; index += 1) {
    let raw = lines[index];
    if (!raw.trim()) continue;
    // Pair a product with a following price/weight line, but never cross another product or a total.
    const followingPrice = /^\s*[$£€]?-?\d+[.,]\d{2}\s*[A-Z*]?\s*$/i;
    const quantityLine = /^\s*\d+(?:[.,]\d+)?\s*(?:x|@|kg|lb|lbs|oz)\s*[$£€]?\d+(?:[.,]\d+)?(?:\s+[$£€]?\d+[.,]\d{2})?\s*$/i;
    if (!/\d+[.,]\d{2}\s*[A-Z*]?\s*$/i.test(raw) && !excluded.test(raw) && !metadata.test(raw)) {
      const following = lines[index + 1] ?? "";
      if (followingPrice.test(following)) { raw += ` ${following.trim()}`; index += 1; }
      else if (quantityLine.test(following)) {
        const last = lines[index + 2] ?? "";
        raw += ` ${following.trim()}`; index += 1;
        if (followingPrice.test(last)) { raw += ` ${last.trim()}`; index += 1; }
      }
    }
    // Require a priced purchase line; receipt headings and arbitrary OCR text are not inventory.
    const price = raw.match(/\s+[$£€]?(-?\d+[.,]\d{2})\s*[A-Z*]?\s*$/i);
    const description = price ? raw.slice(0, price.index) : "";
    const normalized = description.toLowerCase().replace(/(?<=[a-z])[01](?=[a-z])|(?<=[a-z])[01]\b/g, digit => digit === "0" ? "o" : "i").replace(/[^a-z0-9]+/g, " ").trim();
    let match = vocabulary.find(word => ` ${normalized} `.includes(` ${word} `));
    let fuzzy = false;
    if (!match) {
      for (const token of normalized.split(" ").filter(token => token.length >= 5 && /^[a-z]+$/.test(token))) {
        const candidate = fuzzyMatchToken(token, vocabulary.filter(word => word.length >= 5 && !word.includes(" ")));
        if (candidate.match && candidate.score >= 0.84) { match = candidate.match; fuzzy = true; break; }
      }
    }
    if (!price || Number(price[1].replace(",", ".")) <= 0 || excluded.test(raw) || excluded.test(normalized) || metadata.test(normalized) || !match) {
      skippedLines += 1;
      continue;
    }
    const name = aliases[match] ?? match;
    const rule = rules.find(entry => entry.name.toLowerCase() === name);
    // Only explicit counts/weights become quantities; prices and SKU numbers never do.
    const count = description.match(/(?:^|\s)(\d{1,3})\s*[xX@]\s*/);
    const weight = description.match(/\b(\d+(?:\.\d+)?)\s*(kg|lb|lbs|oz|g)\b/i);
    items.push({
      id: `receipt-${receiptId}-${index}`, name,
      quantity: count ? count[1] : weight ? `${weight[1]} ${weight[2].toLowerCase()}` : "1",
      confidence: fuzzy ? 0.7 : 0.85, detectedExpiry: null,
      inferredExpiry: rule ? formatDate(inferExpiryDate(referenceDate, rule.shelfLifeDays)) : null,
      notes: `${fuzzy ? "Approximate OCR match; review the name. " : ""}${rule ? "Added from receipt. Expiry estimated from upload date; check packaging." : "Added from receipt. Check packaging for expiry."}`
    });
  }
  return { items: items.map(item => withDefaultExpiry({ ...item, inferredExpiry: null }, referenceDate)), skippedLines };
}

export async function analyzeReceipt(imagePath: string, rules: IngredientRuleLike[]) {
  const bytes = await readFile(imagePath);
  const receiptId = createHash("sha256").update(bytes).digest("hex");
  // Bounded decoding prevents huge compressed images from exhausting the OCR worker.
  const image = sharp(bytes, { limitInputPixels: 24000000 });
  const metadata = await image.metadata();
  const cleaned = await image.rotate().resize({ width: Math.min(2600, Math.max(1600, metadata.width ?? 1600)), height: 5000, fit: "inside" }).flatten({ background: "white" }).grayscale().normalize().sharpen({ sigma: 1 }).png().toBuffer();
  const cleanedOcr = await runOcr(cleaned);
  const parsed = parseReceipt(cleanedOcr.text, rules, receiptId);
  // If cleanup didn't produce useful results, retry the original pixels once.
  if (!parsed.items.length && cleanedOcr.engine !== "stub") {
    const original = await runOcr(bytes);
    const fallback = parseReceipt(original.text, rules, receiptId);
    if (fallback.items.length) return fallback;
  }
  return parsed;
}

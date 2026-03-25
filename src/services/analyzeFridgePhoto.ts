import fs from "node:fs";
import path from "node:path";
import { fuzzyMatchToken, runOcr, type OcrWord } from "./ocr";
import { extractDates, formatDate, inferExpiryDate } from "../utils/dates";

export type DetectedItem = {
  id: string;
  name: string;
  quantity?: string;
  detectedExpiry?: string | null;
  inferredExpiry?: string | null;
  confidence: number;
  bbox?: { x: number; y: number; width: number; height: number };
  notes?: string;
};

export type IngredientRuleLike = {
  name: string;
  shelfLifeDays: number;
};

export const FOOD_KEYWORDS = [
  "milk", "eggs", "spinach", "lettuce", "yogurt", "cheese", "butter", "chicken",
  "salmon", "tofu", "broccoli", "carrots", "bell pepper", "tomato", "cucumber",
  "strawberries", "blueberries", "apples", "bananas", "grapes", "mushrooms",
  "rice", "pasta", "tortillas", "bread", "bagels", "cream", "sour cream",
  "avocado", "lime", "lemon", "garlic", "onion", "kale", "zucchini", "cauliflower",
  "ground beef", "turkey", "bacon", "ham", "shrimp", "pork", "beans", "chickpeas",
  "lentils", "cilantro", "parsley", "basil", "ginger", "scallions", "corn",
  "peas", "orange juice", "almond milk", "cottage cheese", "feta", "mozzarella",
  "potato", "sweet potato", "sausage", "sausage links", "kimchi"
];

function buildDeterministicId(name: string, index: number): string {
  return `${name.replace(/\s+/g, "-")}-${index + 1}`;
}

function nearestExpiryForWord(word: OcrWord, expiryWords: Array<{ date: string; bbox?: OcrWord["bbox"] }>): string | null {
  if (expiryWords.length === 0) {
    return null;
  }

  if (!word.bbox) {
    return expiryWords[0]?.date ?? null;
  }

  const ranked = expiryWords.map((entry) => {
    if (!entry.bbox) {
      return { date: entry.date, distance: Number.MAX_SAFE_INTEGER };
    }
    const dx = entry.bbox.x - word.bbox.x;
    const dy = entry.bbox.y - word.bbox.y;
    return { date: entry.date, distance: Math.sqrt(dx * dx + dy * dy) };
  });

  ranked.sort((a, b) => a.distance - b.distance);
  return ranked[0]?.date ?? null;
}

export async function analyzeFridgePhoto(
  imagePath: string,
  ingredientRules: IngredientRuleLike[],
  referenceDate = new Date()
): Promise<DetectedItem[]> {
  const absolutePath = path.resolve(imagePath);
  if (!fs.existsSync(absolutePath)) {
    throw new Error(`Image not found: ${absolutePath}`);
  }

  // TODO: Plug in a real object detector after the deterministic OCR pipeline.
  const ocr = await runOcr(absolutePath);
  const tokens = ocr.words.length > 0
    ? ocr.words
    : ocr.text.split(/\s+/).filter(Boolean).map((word) => ({ text: word, confidence: 0.65 }));
  const dates = extractDates(ocr.text).map((date, index) => ({
    date: formatDate(date) as string,
    bbox: tokens[index]?.bbox
  }));

  const matchedTokens = tokens
    .map((word, index) => ({ index, word, ...fuzzyMatchToken(word.text, FOOD_KEYWORDS) }))
    .filter((entry) => entry.match && entry.score >= 0.53);

  const deduped = new Map<string, DetectedItem>();
  for (const [index, entry] of matchedTokens.entries()) {
    const name = entry.match as string;
    const rule = ingredientRules.find((item) => item.name.toLowerCase() === name.toLowerCase());
    const detectedExpiry = nearestExpiryForWord(entry.word, dates);
    const inferredExpiry = !detectedExpiry && rule
      ? formatDate(inferExpiryDate(referenceDate, rule.shelfLifeDays))
      : null;

    if (!deduped.has(name)) {
      deduped.set(name, {
        id: buildDeterministicId(name, index),
        name,
        quantity: /\d/.test(entry.word.text) ? entry.word.text : undefined,
        detectedExpiry,
        inferredExpiry,
        confidence: Number((0.5 + entry.score * 0.45).toFixed(2)),
        bbox: entry.word.bbox,
        notes: detectedExpiry
          ? "Expiry inferred from nearby OCR text."
          : inferredExpiry
            ? "No explicit date found; shelf-life rule applied."
            : "Detected from keyword match."
      });
    }
  }

  if (deduped.size === 0) {
    return ingredientRules.slice(0, 4).map((rule, index) => ({
      id: buildDeterministicId(rule.name, index),
      name: rule.name,
      confidence: 0.42,
      inferredExpiry: formatDate(inferExpiryDate(referenceDate, rule.shelfLifeDays)),
      notes: "Fallback demo items used because OCR did not confidently detect a product."
    }));
  }

  return Array.from(deduped.values());
}

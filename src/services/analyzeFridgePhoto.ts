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
  detectionSource?: "ocr" | "gemini" | "merged";
  expirySource?: "ocr" | "rule" | "none";
  bbox?: { x: number; y: number; width: number; height: number };
  notes?: string;
};

export type IngredientRuleLike = {
  name: string;
  shelfLifeDays: number;
};

type TokenWithOptionalBbox = {
  text: string;
  confidence: number;
  bbox?: { x: number; y: number; width: number; height: number };
};

type GeminiCandidate = {
  name: string;
  confidence: number;
  quantityEstimate?: string;
  visiblePackaging?: string;
  freshnessClue?: string;
  readableDate?: string;
  notes?: string;
};

const FOOD_KEYWORDS = [
  "milk", "eggs", "spinach", "lettuce", "yogurt", "cheese", "butter", "chicken",
  "salmon", "tofu", "broccoli", "carrots", "bell pepper", "tomato", "cucumber",
  "strawberries", "blueberries", "apples", "bananas", "grapes", "mushrooms",
  "rice", "pasta", "tortillas", "bread", "bagels", "cream", "sour cream",
  "avocado", "lime", "lemon", "garlic", "onion", "kale", "zucchini", "cauliflower",
  "ground beef", "turkey", "bacon", "ham", "shrimp", "pork", "beans", "chickpeas",
  "lentils", "cilantro", "parsley", "basil", "ginger", "scallions", "corn",
  "peas", "orange juice", "almond milk", "cottage cheese", "feta", "mozzarella",
  "potato", "sweet potato", "sausage", "sausage links", "kimchi", "goat",
  "curd", "paneer", "coriander", "capsicum", "cauliflower", "okra", "brinjal",
  "apple", "banana", "water", "juice", "soda"
];

const MIN_FUZZY_SCORE = 0.82;
const MIN_OCR_CONFIDENCE = 0.72;
const MIN_INFERRED_EXPIRY_SCORE = 0.9;
const MIN_GEMINI_CONFIDENCE = 0.46;
const OCR_STOPWORDS = new Set([
  "istock",
  "credit",
  "andrey",
  "popov",
  "getty",
  "images",
  "photo",
  "stock",
  "tm"
]);

const GEMINI_PROMPT = `Analyze this fridge photo and return only raw JSON with this exact shape:
{
  "items": [
    {
      "name": "string",
      "confidence": "number from 0 to 1",
      "quantityEstimate": "short phrase or empty string",
      "visiblePackaging": "short phrase or empty string",
      "freshnessClue": "short phrase or empty string",
      "readableDate": "date string only if actually visible, otherwise empty string",
      "notes": "short phrase explaining the visual clue"
    }
  ]
}

Rules:
- Identify visible foods, fruits, vegetables, drinks, dairy, proteins, condiments, and common packaged fridge items.
- Be conservative and do not invent hidden ingredients.
- Do not invent expiry dates. Only include readableDate if a date is clearly visible in the photo.
- Prefer common kitchen ingredient names like "eggs", "milk", "tomato", "orange juice", "bell pepper".
- Return at most 18 items.`;

function normalizeFoodToken(token: string): string {
  return token.toLowerCase().replace(/[^a-z]/g, "").trim();
}

function normalizeName(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function isUsableToken(token: TokenWithOptionalBbox): boolean {
  const normalized = normalizeFoodToken(token.text);
  if (normalized.length < 3) {
    return false;
  }

  if (OCR_STOPWORDS.has(normalized)) {
    return false;
  }

  return token.confidence >= MIN_OCR_CONFIDENCE;
}

function buildDeterministicId(name: string, index: number): string {
  return `${name.replace(/\s+/g, "-")}-${index + 1}`;
}

function nearestExpiryForWord(word: OcrWord, expiryWords: Array<{ date: string; bbox?: OcrWord["bbox"] }>): string | null {
  if (expiryWords.length === 0) {
    return null;
  }

  const wordBbox = word.bbox;
  if (!wordBbox) {
    return expiryWords[0]?.date ?? null;
  }

  const ranked = expiryWords.map((entry) => {
    if (!entry.bbox) {
      return { date: entry.date, distance: Number.MAX_SAFE_INTEGER };
    }
    const dx = entry.bbox.x - wordBbox.x;
    const dy = entry.bbox.y - wordBbox.y;
    return { date: entry.date, distance: Math.sqrt(dx * dx + dy * dy) };
  });

  ranked.sort((a, b) => a.distance - b.distance);
  return ranked[0]?.date ?? null;
}

function sanitizeJsonText(text: string): string {
  return text.replace(/```json|```/gi, "").trim();
}

function parseGeminiDate(value?: string): string | null {
  if (!value) {
    return null;
  }

  const dates = extractDates(value);
  return formatDate(dates[0] ?? null);
}

function buildOcrItems(
  tokens: TokenWithOptionalBbox[],
  ingredientRules: IngredientRuleLike[],
  expiryWords: Array<{ date: string; bbox?: OcrWord["bbox"] }>,
  referenceDate: Date
): DetectedItem[] {
  const matchedTokens = tokens
    .filter(isUsableToken)
    .map((word, index) => ({ index, word, ...fuzzyMatchToken(word.text, FOOD_KEYWORDS) }))
    .filter((entry) => entry.match && entry.score >= MIN_FUZZY_SCORE);

  const deduped = new Map<string, DetectedItem>();
  for (const [index, entry] of matchedTokens.entries()) {
    const name = entry.match as string;
    const rule = ingredientRules.find((item) => item.name.toLowerCase() === name.toLowerCase());
    const detectedExpiry = nearestExpiryForWord(entry.word, expiryWords);
    const inferredExpiry =
      !detectedExpiry && rule && entry.score >= MIN_INFERRED_EXPIRY_SCORE
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
        detectionSource: "ocr",
        expirySource: detectedExpiry ? "ocr" : inferredExpiry ? "rule" : "none",
        bbox: entry.word.bbox,
        notes: detectedExpiry
          ? "Expiry inferred from nearby OCR text."
          : inferredExpiry
            ? "No explicit date found; shelf-life rule applied."
            : "Detected from a high-confidence OCR keyword match."
      });
    }
  }

  return Array.from(deduped.values());
}

async function callGeminiForIngredients(imagePath: string): Promise<GeminiCandidate[]> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return [];
  }

  if (process.env.MOCK_GEMINI_INGREDIENTS) {
    const parsed = JSON.parse(process.env.MOCK_GEMINI_INGREDIENTS) as { items?: GeminiCandidate[] };
    return Array.isArray(parsed.items) ? parsed.items : [];
  }

  const buffer = await fs.promises.readFile(imagePath);
  const mimeType = path.extname(imagePath).toLowerCase() === ".png" ? "image/png" : "image/jpeg";
  const model = process.env.GEMINI_MODEL || process.env.AI_MODEL || "gemini-2.5-flash";

  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [
        {
          role: "user",
          parts: [
            { text: GEMINI_PROMPT },
            { inlineData: { mimeType, data: buffer.toString("base64") } }
          ]
        }
      ],
      generationConfig: {
        temperature: 0.2,
        responseMimeType: "application/json"
      }
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini fridge analysis failed: ${errorText}`);
  }

  const data = (await response.json()) as {
    candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
  };

  const text = data.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join("").trim();
  if (!text) {
    return [];
  }

  const parsed = JSON.parse(sanitizeJsonText(text)) as { items?: GeminiCandidate[] };
  return Array.isArray(parsed.items) ? parsed.items : [];
}

function buildGeminiItems(
  candidates: GeminiCandidate[],
  ingredientRules: IngredientRuleLike[],
  ocrItems: DetectedItem[],
  referenceDate: Date
): DetectedItem[] {
  const deduped = new Map<string, DetectedItem>();

  for (const [index, candidate] of candidates.entries()) {
    const normalized = normalizeName(candidate.name);
    if (!normalized) {
      continue;
    }

    const match = fuzzyMatchToken(normalized, FOOD_KEYWORDS);
    const resolvedName = match.match && match.score >= 0.74 ? match.match : normalized;
    const confidence = Number(candidate.confidence || 0);
    if (confidence < MIN_GEMINI_CONFIDENCE) {
      continue;
    }

    const key = normalizeName(resolvedName);
    if (deduped.has(key)) {
      continue;
    }

    const rule = ingredientRules.find((item) => item.name.toLowerCase() === resolvedName.toLowerCase());
    const ocrMatch = ocrItems.find((item) => normalizeName(item.name) === key);
    const detectedExpiry = parseGeminiDate(candidate.readableDate) ?? ocrMatch?.detectedExpiry ?? null;
    const inferredExpiry = !detectedExpiry && rule ? formatDate(inferExpiryDate(referenceDate, rule.shelfLifeDays)) : null;
    const mergedNoteParts = [
      candidate.notes,
      candidate.visiblePackaging ? `Packaging: ${candidate.visiblePackaging}.` : "",
      candidate.freshnessClue ? `Freshness clue: ${candidate.freshnessClue}.` : "",
      ocrMatch?.notes && ocrMatch.detectedExpiry ? "OCR also found nearby date text." : ""
    ].filter(Boolean);

    deduped.set(key, {
      id: ocrMatch?.id ?? buildDeterministicId(resolvedName, index),
      name: resolvedName,
      quantity: candidate.quantityEstimate || ocrMatch?.quantity,
      detectedExpiry,
      inferredExpiry: detectedExpiry ? null : inferredExpiry,
      confidence: Number(Math.max(confidence, ocrMatch?.confidence ?? 0).toFixed(2)),
      detectionSource: ocrMatch ? "merged" : "gemini",
      expirySource: detectedExpiry ? "ocr" : inferredExpiry ? "rule" : "none",
      bbox: ocrMatch?.bbox,
      notes:
        mergedNoteParts.join(" ").trim() ||
        (ocrMatch ? "Gemini vision and OCR both supported this item." : "Detected from Gemini image analysis.")
    });
  }

  return Array.from(deduped.values());
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

  const ocr = await runOcr(absolutePath);
  const tokens: TokenWithOptionalBbox[] =
    ocr.words.length > 0
      ? ocr.words
      : ocr.text.split(/\s+/).filter(Boolean).map((word) => ({ text: word, confidence: 0.65 }));

  const expiryWords = extractDates(ocr.text).map((date, index) => ({
    date: formatDate(date) as string,
    bbox: tokens[index]?.bbox
  }));

  const ocrItems = buildOcrItems(tokens, ingredientRules, expiryWords, referenceDate);

  try {
    const geminiCandidates = await callGeminiForIngredients(absolutePath);
    const geminiItems = buildGeminiItems(geminiCandidates, ingredientRules, ocrItems, referenceDate);

    if (geminiItems.length > 0) {
      const merged = new Map<string, DetectedItem>();
      for (const item of [...geminiItems, ...ocrItems]) {
        const key = normalizeName(item.name);
        const existing = merged.get(key);
        if (!existing) {
          merged.set(key, item);
          continue;
        }

        merged.set(key, {
          ...existing,
          ...item,
          confidence: Math.max(existing.confidence, item.confidence),
          detectionSource:
            existing.detectionSource !== item.detectionSource && existing.detectionSource && item.detectionSource
              ? "merged"
              : item.detectionSource ?? existing.detectionSource,
          notes:
            existing.notes && item.notes && existing.notes !== item.notes
              ? `${existing.notes} ${item.notes}`
              : item.notes ?? existing.notes
        });
      }

      return Array.from(merged.values()).sort((left, right) => right.confidence - left.confidence);
    }
  } catch (error) {
    console.warn("Gemini fridge analysis unavailable, falling back to OCR-only detection.", error);
  }

  return ocrItems;
}

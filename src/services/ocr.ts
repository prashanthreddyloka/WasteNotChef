import { createWorker } from "tesseract.js";

export type OcrWord = {
  text: string;
  confidence: number;
  bbox?: { x: number; y: number; width: number; height: number };
};

export type OcrResult = {
  text: string;
  words: OcrWord[];
  engine: "tesseract" | "stub";
};

function normalizeToken(token: string): string {
  return token.toLowerCase().replace(/[^a-z0-9/.-]/g, "").trim();
}

function similarity(left: string, right: string): number {
  if (left === right) {
    return 1;
  }

  if (left.length === 0 || right.length === 0) {
    return 0;
  }

  const matrix = Array.from({ length: left.length + 1 }, () => Array<number>(right.length + 1).fill(0));

  for (let i = 0; i <= left.length; i += 1) {
    matrix[i][0] = i;
  }

  for (let j = 0; j <= right.length; j += 1) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= left.length; i += 1) {
    for (let j = 1; j <= right.length; j += 1) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,
        matrix[i][j - 1] + 1,
        matrix[i - 1][j - 1] + cost
      );
    }
  }

  const distance = matrix[left.length][right.length];
  return 1 - distance / Math.max(left.length, right.length);
}

export function fuzzyMatchToken(token: string, keywords: string[]): { match: string | null; score: number } {
  const normalized = normalizeToken(token);
  let best: { match: string | null; score: number } = { match: null, score: 0 };

  for (const keyword of keywords) {
    const score = similarity(normalized, normalizeToken(keyword));
    if (score > best.score) {
      best = { match: keyword, score };
    }
  }

  return best;
}

export async function runOcr(imagePath: string): Promise<OcrResult> {
  if (process.env.MOCK_OCR_TEXT) {
    const tokens = process.env.MOCK_OCR_TEXT.split(/\s+/).filter(Boolean);
    return {
      text: process.env.MOCK_OCR_TEXT,
      words: tokens.map((token, index) => ({
        text: token,
        confidence: 0.9,
        bbox: { x: 16 * index, y: 20 + index * 3, width: 64, height: 18 }
      })),
      engine: "stub"
    };
  }

  const worker = await createWorker("eng");
  try {
    const result = await worker.recognize(imagePath);
    return {
      text: result.data.text,
      words: result.data.words.map((word) => ({
        text: word.text,
        confidence: word.confidence / 100,
        bbox: {
          x: word.bbox.x0,
          y: word.bbox.y0,
          width: word.bbox.x1 - word.bbox.x0,
          height: word.bbox.y1 - word.bbox.y0
        }
      })),
      engine: "tesseract"
    };
  } finally {
    await worker.terminate();
  }
}

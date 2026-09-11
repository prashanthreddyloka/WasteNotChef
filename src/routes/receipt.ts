import { Router } from "express";
import { unlink } from "node:fs/promises";
import multer from "multer";
import { prisma } from "../db/prismaClient";
import { analyzeReceipt } from "../services/analyzeReceipt";

const upload = multer({
  dest: "tmp/", limits: { fileSize: 10 * 1024 * 1024, files: 1 },
  fileFilter: (_req, file, callback) => callback(null, ["image/jpeg", "image/png", "image/webp"].includes(file.mimetype))
});
export const receiptRouter = Router();
receiptRouter.post("/", (req, res, next) => {
  upload.single("image")(req, res, error => {
    if (error) {
      res.status(400).json({ error: "Upload one JPEG, PNG, or WebP image under 10 MB." });
      return;
    }
    next();
  });
}, async (req, res, next) => {
  if (!req.file) return res.status(400).json({ error: "Choose a JPEG, PNG, or WebP receipt image under 10 MB." });
  try {
    const result = await analyzeReceipt(req.file.path, await prisma.ingredientRule.findMany());
    // Stable IDs make retrying the same image safe; save the whole scan atomically.
    await prisma.$transaction(result.items.map(item => prisma.pantryItem.upsert({
      where: { id: item.id }, update: {},
      create: { ...item, inferredExpiry: item.inferredExpiry ? new Date(item.inferredExpiry) : null, detectedExpiry: null, source: "receipt" }
    })));
    const saved = await prisma.pantryItem.findMany({ where: { id: { in: result.items.map(item => item.id) } } });
    return res.json({ ...result, items: saved.map(item => ({ ...item, detectionSource: "ocr", expirySource: item.detectedExpiry ? "ocr" : item.inferredExpiry ? "rule" : "none", detectedExpiry: item.detectedExpiry?.toISOString().slice(0, 10) ?? null, inferredExpiry: item.inferredExpiry?.toISOString().slice(0, 10) ?? null })) });
  } catch (error) {
    return next(error);
  } finally {
    await unlink(req.file.path).catch(() => undefined);
  }
});

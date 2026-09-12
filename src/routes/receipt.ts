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
    return res.json(result);
  } catch (error) {
    return next(error);
  } finally {
    await unlink(req.file.path).catch(() => undefined);
  }
});

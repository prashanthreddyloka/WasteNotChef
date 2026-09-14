import { unlink } from "node:fs/promises";
import { Router } from "express";
import multer from "multer";
import sharp from "sharp";
import { prisma } from "../db/prismaClient";
import { analyzeFridgePhoto } from "../services/analyzeFridgePhoto";

const upload = multer({ dest: "tmp/", limits: { fileSize: 10 * 1024 * 1024, files: 1 }, fileFilter: (_req, file, callback) => callback(null, ["image/jpeg", "image/png", "image/webp"].includes(file.mimetype)) });

export const uploadRouter = Router();

uploadRouter.post("/", upload.single("image"), async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Image file is required." });
    }

    try { await sharp(req.file.path, { limitInputPixels: 24000000 }).metadata(); }
    catch { return res.status(400).json({ error: "This photo could not be read. Choose a JPEG, PNG, or WebP image under 24 megapixels." }); }
    const rules = await prisma.ingredientRule.findMany();
    const items = await analyzeFridgePhoto(req.file.path, rules, new Date());

    return res.json({ items, recognition: items.some(item => item.detectionSource === "gemini" || item.detectionSource === "merged") ? "vision" : "labels" });
  } catch (error) {
    return next(error);
  } finally {
    if (req.file) await unlink(req.file.path).catch(() => undefined);
  }
});

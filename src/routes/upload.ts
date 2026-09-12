import { unlink } from "node:fs/promises";
import { Router } from "express";
import multer from "multer";
import { prisma } from "../db/prismaClient";
import { analyzeFridgePhoto } from "../services/analyzeFridgePhoto";

const upload = multer({ dest: "tmp/", limits: { fileSize: 10 * 1024 * 1024, files: 1 }, fileFilter: (_req, file, callback) => callback(null, ["image/jpeg", "image/png", "image/webp"].includes(file.mimetype)) });

export const uploadRouter = Router();

uploadRouter.post("/", upload.single("image"), async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Image file is required." });
    }

    const rules = await prisma.ingredientRule.findMany();
    const items = await analyzeFridgePhoto(req.file.path, rules, new Date());

    return res.json({ items });
  } catch (error) {
    return next(error);
  } finally {
    if (req.file) await unlink(req.file.path).catch(() => undefined);
  }
});

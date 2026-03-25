import { Router } from "express";
import multer from "multer";
import { prisma } from "../db/prismaClient";
import { analyzeFridgePhoto } from "../services/analyzeFridgePhoto";

const upload = multer({ dest: "tmp/" });

export const uploadRouter = Router();

uploadRouter.post("/", upload.single("image"), async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Image file is required." });
    }

    const rules = await prisma.ingredientRule.findMany();
    const items = await analyzeFridgePhoto(req.file.path, rules, new Date());

    for (const item of items) {
      await prisma.pantryItem.upsert({
        where: { id: item.id },
        update: {
          name: item.name,
          quantity: item.quantity,
          detectedExpiry: item.detectedExpiry ? new Date(item.detectedExpiry) : null,
          inferredExpiry: item.inferredExpiry ? new Date(item.inferredExpiry) : null,
          confidence: item.confidence,
          notes: item.notes
        },
        create: {
          id: item.id,
          name: item.name,
          quantity: item.quantity,
          detectedExpiry: item.detectedExpiry ? new Date(item.detectedExpiry) : null,
          inferredExpiry: item.inferredExpiry ? new Date(item.inferredExpiry) : null,
          confidence: item.confidence,
          notes: item.notes
        }
      });
    }

    return res.json({ items });
  } catch (error) {
    return next(error);
  }
});

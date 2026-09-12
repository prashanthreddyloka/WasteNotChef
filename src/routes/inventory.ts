import { Router } from "express";
import { z } from "zod";
import { prisma } from "../db/prismaClient";
import { requireAccount } from "../services/auth";

const date = z.string().regex(/^\d{4}-\d{2}-\d{2}$/).refine(value => Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value, "Invalid expiry date.");
const itemSchema = z.object({
  id: z.string().min(1).max(200), name: z.string().trim().min(1).max(80), quantity: z.string().max(60).optional(),
  detectedExpiry: date.nullable().optional(), inferredExpiry: date.nullable().optional(), confidence: z.number().min(0).max(1),
  detectionSource: z.enum(["visual", "ocr", "gemini", "merged", "manual"]).optional(), expirySource: z.enum(["ocr", "rule", "manual", "none"]).optional(),
  reviewed: z.boolean().optional(), notes: z.string().max(2000).optional(),
  bbox: z.object({ x: z.number(), y: z.number(), width: z.number(), height: z.number() }).optional()
});
const body = z.object({ version: z.number().int().nonnegative(), items: z.array(itemSchema).max(500).refine(items => new Set(items.map(item => item.id)).size === items.length, "Duplicate item IDs.") });
export const inventoryRouter = Router();
inventoryRouter.use(requireAccount, (_req, res, next) => { res.setHeader("Cache-Control", "no-store"); next(); });
inventoryRouter.get("/", async (_req, res, next) => {
  try {
    const snapshot = await prisma.$transaction(async tx => {
      const user = await tx.user.findUniqueOrThrow({ where: { id: res.locals.user.id } });
      const rows = await tx.inventoryEntry.findMany({ where: { userId: user.id }, orderBy: { id: "asc" } });
      return { version: user.inventoryVersion, items: rows.map(row => JSON.parse(row.itemJson)) };
    });
    res.json(snapshot);
  } catch (error) { next(error); }
});
inventoryRouter.put("/", async (req, res, next) => {
  try {
    const input = body.parse(req.body);
    const userId = res.locals.user.id as string;
    const saved = await prisma.$transaction(async tx => {
      // Compare-and-swap protects edits from another tab or device being overwritten.
      const changed = await tx.user.updateMany({ where: { id: userId, inventoryVersion: input.version }, data: { inventoryVersion: { increment: 1 } } });
      if (!changed.count) return false;
      await tx.inventoryEntry.deleteMany({ where: { userId } });
      if (input.items.length) await tx.inventoryEntry.createMany({ data: input.items.map(item => ({ userId, clientId: item.id, itemJson: JSON.stringify(item) })) });
      return true;
    });
    if (!saved) { res.status(409).json({ error: "Your pantry changed on another device. Refresh the pantry and apply your edit again." }); return; }
    res.json({ items: input.items, version: input.version + 1 });
  } catch (error) { next(error); }
});

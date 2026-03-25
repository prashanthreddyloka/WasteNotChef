import { type NextFunction, type Request, type Response, Router } from "express";
import { z } from "zod";
import { prisma } from "../db/prismaClient";
import { formatDate } from "../utils/dates";

const querySchema = z.object({
  from: z.string(),
  to: z.string()
});

export const wasteRouter = Router();

async function handleWasteScore(req: Request, res: Response, next: NextFunction) {
  try {
    const { from, to } = querySchema.parse(req.query);
    const plans = await prisma.weekPlan.findMany({
      where: {
        startDate: { gte: new Date(from) },
        endDate: { lte: new Date(to) }
      },
      include: { dayPlans: true },
      orderBy: { startDate: "asc" }
    });

    const timeseries = plans.flatMap((plan) =>
      plan.dayPlans.map((day) => ({
        date: formatDate(day.scheduledDate),
        wasteScore: day.wasteScore,
        recipeTitle: day.recipeTitle
      }))
    );

    return res.json({ timeseries });
  } catch (error) {
    return next(error);
  }
}

wasteRouter.get("/", handleWasteScore);
wasteRouter.get("/score", handleWasteScore);

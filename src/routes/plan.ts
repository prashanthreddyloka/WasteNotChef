import { type NextFunction, type Request, type Response, Router } from "express";
import { z } from "zod";
import { prisma } from "../db/prismaClient";
import { planWeek } from "../services/scheduler";

const bodySchema = z.object({
  items: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      quantity: z.string().optional(),
      detectedExpiry: z.string().nullable().optional(),
      inferredExpiry: z.string().nullable().optional(),
      importanceWeight: z.number().optional()
    })
  ),
  preferences: z.object({
    mealsPerDay: z.number().min(1).max(3),
    skipDays: z.array(z.number().min(0).max(6)),
    preferCuisineTags: z.array(z.string()),
    maxLeftovers: z.number().min(0).max(10)
  })
});

export const planRouter = Router();

async function handlePlanWeek(req: Request, res: Response, next: NextFunction) {
  try {
    const { items, preferences } = bodySchema.parse(req.body);
    const recipeRows = await prisma.recipe.findMany();
    const recipes = recipeRows.map((recipe) => ({
      id: recipe.id,
      title: recipe.title,
      tags: recipe.tags.split(","),
      cookTime: recipe.cookTime,
      steps: JSON.parse(recipe.stepsJson),
      ingredients: JSON.parse(recipe.ingredientsJson)
    }));

    const result = planWeek(items, recipes, preferences, new Date());
    const saved = await prisma.weekPlan.create({
      data: {
        startDate: new Date(result.dayPlans[0]?.scheduledDate ?? new Date()),
        endDate: new Date(result.dayPlans.at(-1)?.scheduledDate ?? new Date()),
        wasteScore: result.wasteProjection.weeklyWasteScore,
        reasoning: result.reasoning,
        metadataJson: JSON.stringify(result.metadata),
        dayPlans: {
          create: result.dayPlans.map((day) => ({
            scheduledDate: new Date(day.scheduledDate),
            recipeId: day.recipe.id,
            recipeTitle: day.recipe.title,
            reasoning: day.reasoning,
            priority: day.priority,
            itemsConsumedJson: JSON.stringify(day.itemsConsumed),
            leftoversJson: JSON.stringify(day.leftovers),
            wasteScore: day.wasteScore
          }))
        }
      }
    });

    return res.json({
      weekPlan: result.dayPlans,
      wasteProjection: result.wasteProjection,
      reasoning: result.reasoning,
      metadata: { ...result.metadata, savedPlanId: saved.id }
    });
  } catch (error) {
    return next(error);
  }
}

planRouter.post("/", handlePlanWeek);
planRouter.post("/week", handlePlanWeek);

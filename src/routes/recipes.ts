import { Router } from "express";
import { z } from "zod";
import { prisma } from "../db/prismaClient";

const bodySchema = z.object({
  items: z.array(z.object({ name: z.string() }))
});

function parseRecipeTags(tagString: string) {
  const rawTags = tagString.split(",").map((tag) => tag.trim()).filter(Boolean);
  const country = rawTags.find((tag) => tag.startsWith("country:"))?.replace("country:", "");
  const continent = rawTags.find((tag) => tag.startsWith("continent:"))?.replace("continent:", "");
  const tags = rawTags.filter((tag) => !tag.startsWith("country:") && !tag.startsWith("continent:"));
  return { tags, country, continent };
}

function ingredientMatchScore(
  ingredients: Array<{ name: string; qty: string; optional?: boolean }>,
  pantry: Set<string>,
  country?: string,
  continent?: string
) {
  const required = ingredients.filter((ingredient) => !ingredient.optional);
  const optional = ingredients.filter((ingredient) => ingredient.optional);
  const matchedRequired = required.filter((ingredient) => pantry.has(ingredient.name.toLowerCase()));
  const matchedOptional = optional.filter((ingredient) => pantry.has(ingredient.name.toLowerCase()));
  const missingRequired = required.filter((ingredient) => !pantry.has(ingredient.name.toLowerCase()));
  const missingOptional = optional.filter((ingredient) => !pantry.has(ingredient.name.toLowerCase()));

  const requiredCoverage = required.length > 0 ? matchedRequired.length / required.length : 0;
  const optionalCoverage = optional.length > 0 ? matchedOptional.length / optional.length : 0;
  const indianPriorityBonus = country?.toLowerCase() === "india" ? 2 : continent?.toLowerCase() === "asia" ? 0.5 : 0;

  const score =
    matchedRequired.length * 18 +
    matchedOptional.length * 6 +
    requiredCoverage * 20 +
    optionalCoverage * 4 -
    missingRequired.length * 12 -
    missingOptional.length * 1.5 +
    indianPriorityBonus;

  return {
    score,
    matchedRequired,
    matchedOptional,
    missingRequired,
    missingOptional,
    requiredCoverage
  };
}

export const recipesRouter = Router();

recipesRouter.post("/from-items", async (req, res, next) => {
  try {
    const { items } = bodySchema.parse(req.body);
    const pantry = new Set(items.map((item) => item.name.toLowerCase()));
    const recipes = await prisma.recipe.findMany();

    const mapped = recipes
      .map((recipe) => {
        const ingredients = JSON.parse(recipe.ingredientsJson) as Array<{ name: string; qty: string; optional?: boolean }>;
        const { tags, country, continent } = parseRecipeTags(recipe.tags);
        const ranking = ingredientMatchScore(ingredients, pantry, country, continent);
        return {
          id: recipe.id,
          title: recipe.title,
          tags,
          country,
          continent,
          cookTime: recipe.cookTime,
          steps: JSON.parse(recipe.stepsJson),
          ingredients,
          score: Number(ranking.score.toFixed(2)),
          coverage: `${ranking.matchedRequired.length + ranking.matchedOptional.length}/${ingredients.length}`,
          substitutionSuggestions: ranking.missingOptional.map(
            (ingredient) => `Optional swap for ${ingredient.name}: pantry-friendly herb, greens, or a neutral staple.`
          ),
          requiredCoverage: ranking.requiredCoverage
        };
      })
      .filter((recipe) => recipe.requiredCoverage > 0 || recipe.score > 8)
      .sort((a, b) => b.score - a.score);

    return res.json({
      recipes: mapped,
      matches: Object.fromEntries(mapped.map((recipe) => [recipe.id, recipe.coverage]))
    });
  } catch (error) {
    return next(error);
  }
});

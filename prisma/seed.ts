import { prisma } from "../src/db/prismaClient";

const ingredientRules = [
  ["milk", "dairy", 7, 5, 1.5], ["eggs", "protein", 21, 14, 1.4], ["spinach", "greens", 5, 3, 1.2],
  ["lettuce", "greens", 7, 5, 1], ["yogurt", "dairy", 10, 7, 1.1], ["cheese", "dairy", 21, 14, 1.3],
  ["butter", "dairy", 30, 21, 0.9], ["chicken", "protein", 3, 2, 1.6], ["salmon", "protein", 2, 2, 1.7],
  ["tofu", "protein", 7, 4, 1.1], ["broccoli", "vegetable", 7, 5, 1], ["carrots", "vegetable", 14, 10, 0.8],
  ["bell pepper", "vegetable", 7, 5, 0.9], ["tomato", "vegetable", 6, 4, 0.8], ["cucumber", "vegetable", 7, 5, 0.8],
  ["strawberries", "fruit", 4, 3, 1.2], ["blueberries", "fruit", 7, 5, 1], ["apples", "fruit", 21, 14, 0.7],
  ["bananas", "fruit", 5, 4, 1], ["grapes", "fruit", 7, 5, 0.9], ["mushrooms", "vegetable", 5, 3, 1.1],
  ["rice", "pantry", 180, 120, 0.3], ["pasta", "pantry", 180, 120, 0.3], ["tortillas", "bread", 10, 7, 0.6],
  ["bread", "bread", 7, 5, 0.9], ["bagels", "bread", 6, 4, 0.8], ["cream", "dairy", 7, 5, 1.3],
  ["sour cream", "dairy", 10, 7, 1.1], ["avocado", "fruit", 4, 3, 1.2], ["lime", "fruit", 10, 7, 0.6],
  ["lemon", "fruit", 14, 10, 0.6], ["garlic", "aromatic", 30, 21, 0.4], ["onion", "aromatic", 30, 21, 0.5],
  ["kale", "greens", 7, 5, 1], ["zucchini", "vegetable", 5, 4, 0.8], ["cauliflower", "vegetable", 7, 5, 0.9],
  ["ground beef", "protein", 3, 2, 1.5], ["turkey", "protein", 3, 2, 1.4], ["bacon", "protein", 7, 5, 1.2],
  ["ham", "protein", 7, 5, 1.1], ["shrimp", "protein", 2, 2, 1.6], ["pork", "protein", 4, 3, 1.5],
  ["beans", "pantry", 180, 120, 0.4], ["chickpeas", "pantry", 180, 120, 0.4], ["lentils", "pantry", 180, 120, 0.4],
  ["cilantro", "herb", 5, 3, 0.8], ["parsley", "herb", 7, 5, 0.7], ["basil", "herb", 5, 3, 0.8],
  ["ginger", "aromatic", 21, 14, 0.4], ["scallions", "aromatic", 7, 5, 0.7], ["corn", "vegetable", 5, 4, 0.8],
  ["peas", "vegetable", 7, 5, 0.7], ["orange juice", "beverage", 7, 5, 0.6], ["almond milk", "beverage", 10, 7, 0.8],
  ["cottage cheese", "dairy", 7, 5, 1], ["feta", "dairy", 14, 10, 1], ["mozzarella", "dairy", 10, 7, 1],
  ["potato", "vegetable", 21, 14, 0.7], ["sweet potato", "vegetable", 21, 14, 0.7], ["sausage", "protein", 5, 4, 1.2],
  ["sausage links", "protein", 5, 4, 1.2], ["kimchi", "ferment", 30, 21, 0.6]
] as const;

const recipes = [
  {
    id: "recipe-creamy-spinach-pasta",
    title: "Creamy Spinach Pasta",
    tags: "comfort,italian,quick",
    cookTime: 25,
    ingredients: [
      { name: "pasta", qty: "8 oz" },
      { name: "spinach", qty: "2 cups" },
      { name: "cream", qty: "1/2 cup", optional: true },
      { name: "cheese", qty: "1/3 cup" },
      { name: "garlic", qty: "2 cloves", optional: true }
    ],
    steps: ["Boil pasta.", "Saute garlic and spinach.", "Add cream and cheese.", "Toss together and finish."]
  },
  {
    id: "recipe-garden-frittata",
    title: "Garden Frittata",
    tags: "breakfast,vegetarian,protein",
    cookTime: 20,
    ingredients: [
      { name: "eggs", qty: "6" },
      { name: "spinach", qty: "1 cup", optional: true },
      { name: "tomato", qty: "1" },
      { name: "cheese", qty: "1/2 cup", optional: true },
      { name: "onion", qty: "1/2", optional: true }
    ],
    steps: ["Whisk eggs.", "Cook vegetables.", "Pour eggs into pan.", "Bake or finish covered until set."]
  },
  {
    id: "recipe-sheet-pan-chicken",
    title: "Sheet Pan Chicken and Veg",
    tags: "protein,weeknight,american",
    cookTime: 35,
    ingredients: [
      { name: "chicken", qty: "1 lb" },
      { name: "broccoli", qty: "2 cups" },
      { name: "carrots", qty: "2" },
      { name: "bell pepper", qty: "1", optional: true }
    ],
    steps: ["Heat oven.", "Season everything.", "Roast until chicken is done."]
  },
  {
    id: "recipe-salmon-rice-bowl",
    title: "Salmon Rice Bowl",
    tags: "protein,asian,balanced",
    cookTime: 30,
    ingredients: [
      { name: "salmon", qty: "2 fillets" },
      { name: "rice", qty: "2 cups cooked" },
      { name: "cucumber", qty: "1", optional: true },
      { name: "avocado", qty: "1", optional: true },
      { name: "scallions", qty: "2", optional: true }
    ],
    steps: ["Cook rice.", "Roast or pan-sear salmon.", "Assemble bowls with toppings."]
  },
  {
    id: "recipe-tofu-stir-fry",
    title: "Crispy Tofu Stir Fry",
    tags: "vegetarian,asian,quick",
    cookTime: 25,
    ingredients: [
      { name: "tofu", qty: "14 oz" },
      { name: "broccoli", qty: "2 cups" },
      { name: "bell pepper", qty: "1" },
      { name: "rice", qty: "2 cups cooked", optional: true },
      { name: "ginger", qty: "1 tbsp", optional: true }
    ],
    steps: ["Press tofu.", "Stir-fry vegetables.", "Crisp tofu and toss with sauce."]
  },
  {
    id: "recipe-berry-yogurt-parfait",
    title: "Berry Yogurt Parfait",
    tags: "breakfast,quick,sweet",
    cookTime: 10,
    ingredients: [
      { name: "yogurt", qty: "1 cup" },
      { name: "strawberries", qty: "1 cup", optional: true },
      { name: "blueberries", qty: "1 cup", optional: true },
      { name: "bananas", qty: "1", optional: true }
    ],
    steps: ["Layer yogurt and fruit.", "Finish with crunchy topping if available."]
  },
  {
    id: "recipe-loaded-tortilla-melts",
    title: "Loaded Tortilla Melts",
    tags: "quick,mexican,comfort",
    cookTime: 15,
    ingredients: [
      { name: "tortillas", qty: "4" },
      { name: "cheese", qty: "1 cup" },
      { name: "beans", qty: "1 cup", optional: true },
      { name: "tomato", qty: "1", optional: true },
      { name: "spinach", qty: "1 cup", optional: true }
    ],
    steps: ["Fill tortillas.", "Toast until crisp and melty.", "Slice and serve."]
  },
  {
    id: "recipe-kimchi-fried-rice",
    title: "Kimchi Fried Rice",
    tags: "asian,quick,fermented",
    cookTime: 20,
    ingredients: [
      { name: "rice", qty: "2 cups cooked" },
      { name: "kimchi", qty: "1 cup" },
      { name: "eggs", qty: "2", optional: true },
      { name: "scallions", qty: "2", optional: true }
    ],
    steps: ["Saute kimchi.", "Add rice.", "Top with eggs and scallions."]
  }
];

async function main() {
  for (const [name, category, shelfLifeDays, openedShelfLifeDays, importanceWeight] of ingredientRules) {
    await prisma.ingredientRule.upsert({
      where: { name },
      update: { category, shelfLifeDays, openedShelfLifeDays, importanceWeight },
      create: { name, category, shelfLifeDays, openedShelfLifeDays, importanceWeight }
    });
  }

  for (const recipe of recipes) {
    await prisma.recipe.upsert({
      where: { id: recipe.id },
      update: {
        title: recipe.title,
        tags: recipe.tags,
        cookTime: recipe.cookTime,
        stepsJson: JSON.stringify(recipe.steps),
        ingredientsJson: JSON.stringify(recipe.ingredients)
      },
      create: {
        id: recipe.id,
        title: recipe.title,
        tags: recipe.tags,
        cookTime: recipe.cookTime,
        stepsJson: JSON.stringify(recipe.steps),
        ingredientsJson: JSON.stringify(recipe.ingredients)
      }
    });
  }
}

main()
  .then(async () => prisma.$disconnect())
  .catch(async (error) => {
    console.error(error);
    await prisma.$disconnect();
    process.exit(1);
  });

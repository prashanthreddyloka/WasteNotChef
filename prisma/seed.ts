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
  ["ground beef", "protein", 3, 2, 1.5], ["goat", "protein", 3, 2, 1.6], ["turkey", "protein", 3, 2, 1.4], ["bacon", "protein", 7, 5, 1.2],
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
    tags: "comfort,italian,quick,country:Italy,continent:Europe",
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
    tags: "breakfast,vegetarian,protein,country:Italy,continent:Europe",
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
    tags: "protein,weeknight,american,country:United States,continent:North America",
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
    tags: "protein,asian,balanced,country:Japan,continent:Asia",
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
    tags: "vegetarian,asian,quick,country:China,continent:Asia",
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
    tags: "breakfast,quick,sweet,country:France,continent:Europe",
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
    tags: "quick,mexican,comfort,country:Mexico,continent:North America",
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
    tags: "asian,quick,fermented,country:Korea,continent:Asia",
    cookTime: 20,
    ingredients: [
      { name: "rice", qty: "2 cups cooked" },
      { name: "kimchi", qty: "1 cup" },
      { name: "eggs", qty: "2", optional: true },
      { name: "scallions", qty: "2", optional: true }
    ],
    steps: ["Saute kimchi.", "Add rice.", "Top with eggs and scallions."]
  },
  {
    id: "recipe-masala-omelette",
    title: "Masala Omelette",
    tags: "breakfast,quick,indian,country:India,continent:Asia",
    cookTime: 15,
    ingredients: [
      { name: "eggs", qty: "3" },
      { name: "onion", qty: "1/2", optional: true },
      { name: "tomato", qty: "1", optional: true },
      { name: "cilantro", qty: "2 tbsp", optional: true }
    ],
    steps: ["Whisk eggs with chopped vegetables.", "Cook in a skillet until set.", "Fold and serve hot."]
  },
  {
    id: "recipe-egg-bhurji",
    title: "Egg Bhurji",
    tags: "breakfast,indian,quick,protein,country:India,continent:Asia",
    cookTime: 18,
    ingredients: [
      { name: "eggs", qty: "4" },
      { name: "onion", qty: "1", optional: true },
      { name: "tomato", qty: "1", optional: true },
      { name: "scallions", qty: "2", optional: true }
    ],
    steps: ["Saute aromatics.", "Add tomatoes and cook down.", "Scramble eggs into the masala and serve."]
  },
  {
    id: "recipe-aloo-paratha-skillet",
    title: "Aloo Paratha Skillet",
    tags: "indian,comfort,flatbread,country:India,continent:Asia",
    cookTime: 25,
    ingredients: [
      { name: "potato", qty: "2" },
      { name: "onion", qty: "1/2", optional: true },
      { name: "cilantro", qty: "2 tbsp", optional: true },
      { name: "butter", qty: "1 tbsp", optional: true }
    ],
    steps: ["Mash cooked potatoes with aromatics.", "Spread over flatbread or toast in a skillet.", "Serve with yogurt or pickle."]
  },
  {
    id: "recipe-palak-paneer-style",
    title: "Palak Paneer Style Greens",
    tags: "indian,vegetarian,curry,country:India,continent:Asia",
    cookTime: 30,
    ingredients: [
      { name: "spinach", qty: "4 cups" },
      { name: "cheese", qty: "1 cup" },
      { name: "cream", qty: "2 tbsp", optional: true },
      { name: "garlic", qty: "2 cloves", optional: true }
    ],
    steps: ["Cook spinach until wilted.", "Blend or mash into a coarse puree.", "Simmer with cheese cubes and finish with cream."]
  },
  {
    id: "recipe-veg-pulao",
    title: "Vegetable Pulao",
    tags: "indian,rice,one-pot,country:India,continent:Asia",
    cookTime: 30,
    ingredients: [
      { name: "rice", qty: "1 cup" },
      { name: "carrots", qty: "2", optional: true },
      { name: "peas", qty: "1/2 cup", optional: true },
      { name: "onion", qty: "1/2", optional: true }
    ],
    steps: ["Saute vegetables and aromatics.", "Add rice and water.", "Cook until fluffy and fragrant."]
  },
  {
    id: "recipe-tomato-rice",
    title: "Tomato Rice",
    tags: "indian,rice,quick,country:India,continent:Asia",
    cookTime: 25,
    ingredients: [
      { name: "rice", qty: "2 cups cooked" },
      { name: "tomato", qty: "2" },
      { name: "onion", qty: "1/2", optional: true },
      { name: "peas", qty: "1/2 cup", optional: true }
    ],
    steps: ["Cook tomatoes into a spiced base.", "Fold in rice.", "Steam briefly and serve."]
  },
  {
    id: "recipe-lemon-rice",
    title: "Lemon Rice",
    tags: "indian,rice,quick,country:India,continent:Asia",
    cookTime: 15,
    ingredients: [
      { name: "rice", qty: "2 cups cooked" },
      { name: "lemon", qty: "1" },
      { name: "peas", qty: "1/2 cup", optional: true },
      { name: "scallions", qty: "2", optional: true }
    ],
    steps: ["Warm rice with tempering ingredients.", "Finish with fresh lemon juice.", "Serve as a quick meal or side."]
  },
  {
    id: "recipe-moong-dal-style-lentils",
    title: "Simple Dal Tadka",
    tags: "indian,dal,comfort,country:India,continent:Asia",
    cookTime: 35,
    ingredients: [
      { name: "lentils", qty: "1 cup" },
      { name: "onion", qty: "1/2", optional: true },
      { name: "tomato", qty: "1", optional: true },
      { name: "garlic", qty: "2 cloves", optional: true }
    ],
    steps: ["Cook lentils until soft.", "Prepare a quick tempering with aromatics.", "Fold together and simmer."]
  },
  {
    id: "recipe-chana-masala-style",
    title: "Chana Masala Style Chickpeas",
    tags: "indian,protein,curry,country:India,continent:Asia",
    cookTime: 30,
    ingredients: [
      { name: "chickpeas", qty: "2 cups" },
      { name: "tomato", qty: "2" },
      { name: "onion", qty: "1", optional: true },
      { name: "cilantro", qty: "2 tbsp", optional: true }
    ],
    steps: ["Cook onions and tomatoes into a masala.", "Simmer chickpeas in the sauce.", "Finish with herbs."]
  },
  {
    id: "recipe-rajma-style-beans",
    title: "Rajma Style Beans",
    tags: "indian,beans,comfort,country:India,continent:Asia",
    cookTime: 35,
    ingredients: [
      { name: "beans", qty: "2 cups" },
      { name: "tomato", qty: "2" },
      { name: "onion", qty: "1", optional: true },
      { name: "rice", qty: "2 cups cooked", optional: true }
    ],
    steps: ["Build a tomato-onion masala.", "Simmer beans until coated and rich.", "Serve with rice."]
  },
  {
    id: "recipe-aloo-gobi-style",
    title: "Aloo Gobi Style Saute",
    tags: "indian,sabzi,vegetarian,country:India,continent:Asia",
    cookTime: 28,
    ingredients: [
      { name: "potato", qty: "2" },
      { name: "cauliflower", qty: "2 cups" },
      { name: "onion", qty: "1/2", optional: true },
      { name: "cilantro", qty: "2 tbsp", optional: true }
    ],
    steps: ["Saute potatoes until they start to color.", "Add cauliflower and cook covered.", "Finish with herbs."]
  },
  {
    id: "recipe-bhindi-style-peppers",
    title: "Masala Pepper Saute",
    tags: "indian,sabzi,quick,country:India,continent:Asia",
    cookTime: 20,
    ingredients: [
      { name: "bell pepper", qty: "2" },
      { name: "onion", qty: "1/2", optional: true },
      { name: "tomato", qty: "1", optional: true }
    ],
    steps: ["Saute peppers with onions.", "Add tomato and cook until lightly jammy.", "Serve as a quick sabzi."]
  },
  {
    id: "recipe-paneer-bhurji-style",
    title: "Cheese Bhurji Toast Filling",
    tags: "indian,quick,breakfast,country:India,continent:Asia",
    cookTime: 15,
    ingredients: [
      { name: "cheese", qty: "1 cup" },
      { name: "tomato", qty: "1", optional: true },
      { name: "onion", qty: "1/2", optional: true },
      { name: "cilantro", qty: "2 tbsp", optional: true }
    ],
    steps: ["Crumble or grate cheese.", "Cook with quick masala aromatics.", "Serve with toast or flatbread."]
  },
  {
    id: "recipe-curd-rice-style",
    title: "Yogurt Rice Bowl",
    tags: "indian,comfort,quick,country:India,continent:Asia",
    cookTime: 10,
    ingredients: [
      { name: "rice", qty: "2 cups cooked" },
      { name: "yogurt", qty: "1 cup" },
      { name: "cucumber", qty: "1", optional: true },
      { name: "carrots", qty: "1", optional: true }
    ],
    steps: ["Mix yogurt into cooled rice.", "Add chopped vegetables.", "Serve chilled or at room temperature."]
  },
  {
    id: "recipe-upma-style-bread-skillet",
    title: "Savory Bread Upma Style Skillet",
    tags: "indian,breakfast,snack,country:India,continent:Asia",
    cookTime: 15,
    ingredients: [
      { name: "bread", qty: "4 slices" },
      { name: "onion", qty: "1/2", optional: true },
      { name: "tomato", qty: "1", optional: true },
      { name: "peas", qty: "1/2 cup", optional: true }
    ],
    steps: ["Toast bread pieces lightly.", "Cook vegetables quickly.", "Toss together and serve hot."]
  },
  {
    id: "recipe-masala-toastie",
    title: "Masala Cheese Toastie",
    tags: "indian,snack,quick,country:India,continent:Asia",
    cookTime: 12,
    ingredients: [
      { name: "bread", qty: "4 slices" },
      { name: "cheese", qty: "1 cup" },
      { name: "tomato", qty: "1", optional: true },
      { name: "onion", qty: "1/4", optional: true }
    ],
    steps: ["Layer bread with cheese and masala vegetables.", "Toast until crisp and melty.", "Slice and serve."]
  },
  {
    id: "recipe-vegetable-uttapam-style",
    title: "Savory Veg Pancake Style Breakfast",
    tags: "indian,breakfast,south-indian,country:India,continent:Asia",
    cookTime: 20,
    ingredients: [
      { name: "onion", qty: "1/2", optional: true },
      { name: "tomato", qty: "1", optional: true },
      { name: "bell pepper", qty: "1", optional: true },
      { name: "cilantro", qty: "2 tbsp", optional: true }
    ],
    steps: ["Spread batter or savory base in a pan.", "Top with chopped vegetables.", "Cook until golden on both sides."]
  },
  {
    id: "recipe-vegetable-korma-style",
    title: "Vegetable Korma Style Curry",
    tags: "indian,curry,vegetarian,country:India,continent:Asia",
    cookTime: 35,
    ingredients: [
      { name: "carrots", qty: "2", optional: true },
      { name: "peas", qty: "1/2 cup", optional: true },
      { name: "potato", qty: "2", optional: true },
      { name: "cream", qty: "1/4 cup", optional: true }
    ],
    steps: ["Cook vegetables until tender.", "Simmer in a creamy spiced sauce.", "Serve with rice or bread."]
  },
  {
    id: "recipe-chicken-curry-style",
    title: "Chicken Curry Style Masala",
    tags: "indian,curry,protein,country:India,continent:Asia",
    cookTime: 40,
    ingredients: [
      { name: "chicken", qty: "1 lb" },
      { name: "onion", qty: "1", optional: true },
      { name: "tomato", qty: "2", optional: true },
      { name: "garlic", qty: "3 cloves", optional: true }
    ],
    steps: ["Brown the chicken lightly.", "Cook aromatics into a masala base.", "Simmer chicken until tender and coated."]
  },
  {
    id: "recipe-chicken-pulao-style",
    title: "Chicken Pulao",
    tags: "indian,rice,protein,country:India,continent:Asia",
    cookTime: 35,
    ingredients: [
      { name: "chicken", qty: "1 lb" },
      { name: "rice", qty: "1 cup" },
      { name: "onion", qty: "1/2", optional: true },
      { name: "peas", qty: "1/2 cup", optional: true }
    ],
    steps: ["Saute chicken with aromatics.", "Add rice and liquid.", "Cook until the chicken is tender and rice is fluffy."]
  },
  {
    id: "recipe-goat-curry-style",
    title: "Goat Curry Style Masala",
    tags: "indian,curry,protein,country:India,continent:Asia",
    cookTime: 55,
    ingredients: [
      { name: "goat", qty: "1 lb" },
      { name: "onion", qty: "1", optional: true },
      { name: "tomato", qty: "2", optional: true },
      { name: "garlic", qty: "3 cloves", optional: true }
    ],
    steps: ["Sear goat pieces lightly.", "Cook onion-tomato masala.", "Simmer until the meat is tender and flavorful."]
  },
  {
    id: "recipe-goat-keema-style",
    title: "Goat Keema Style Saute",
    tags: "indian,protein,quick,country:India,continent:Asia",
    cookTime: 30,
    ingredients: [
      { name: "goat", qty: "1 lb" },
      { name: "peas", qty: "1/2 cup", optional: true },
      { name: "onion", qty: "1/2", optional: true },
      { name: "tomato", qty: "1", optional: true }
    ],
    steps: ["Cook goat mince or finely chopped goat with onions.", "Add peas and tomato.", "Reduce until richly coated."]
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

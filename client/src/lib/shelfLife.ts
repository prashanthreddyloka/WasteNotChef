// Planning estimates, not a guarantee of safety. Sources and assumptions are documented in README.
const days: Record<string, number> = {
  tomato: 2, eggs: 21, milk: 7, spinach: 3, lettuce: 7, yogurt: 7, cheese: 7,
  butter: 30, chicken: 2, turkey: 2, "ground beef": 2, sausage: 2, salmon: 2,
  fish: 2, tuna: 2, shrimp: 3, pork: 3, beef: 3, goat: 3, bacon: 7, ham: 3,
  tofu: 3, paneer: 3, broccoli: 3, carrots: 14, "bell pepper": 4, cucumber: 4,
  strawberries: 2, blueberries: 7, apples: 21, bananas: 2, grapes: 7, mushrooms: 3,
  rice: 3, pasta: 3, beans: 3, chickpeas: 3, lentils: 3, tortillas: 7, bread: 5,
  bagels: 5, cream: 7, "sour cream": 7, avocado: 3, lime: 14, lemon: 14,
  garlic: 30, onion: 30, kale: 3, zucchini: 4, cauliflower: 3, cilantro: 3,
  parsley: 3, basil: 3, ginger: 14, scallions: 7, corn: 2, peas: 3,
  "orange juice": 7, "almond milk": 7, "cottage cheese": 7, feta: 7, mozzarella: 7,
  potato: 14, "sweet potato": 14, kimchi: 7, okra: 3, eggplant: 4,
  soup: 3, leftovers: 3
};
const aliases: Record<string, string> = {
  tomatoes: "tomato", "cherry tomatoes": "tomato", "grape tomatoes": "tomato", egg: "eggs",
  apple: "apples", banana: "bananas", carrot: "carrots", strawberry: "strawberries",
  blueberry: "blueberries", mushroom: "mushrooms", capsicum: "bell pepper", coriander: "cilantro",
  brinjal: "eggplant", curd: "yogurt", yoghurt: "yogurt", potatoes: "potato",
  "sausage links": "sausage", "chicken breast": "chicken", "chicken breasts": "chicken"
};
export function localDate(date = new Date()): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}
export function canonicalFoodName(name: string): string {
  const normalized = name.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().replace(/^(fresh|organic|raw) /, "");
  return aliases[normalized] ?? normalized;
}
export function shelfLifeFor(name: string): { days: number; basis: string; kind: "estimate" | "review" } {
  const normalized = name.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const fridge = "Assumes recently purchased food kept refrigerated at 40°F / 4°C or below.";
  if (/\b(hard boiled|hard cooked)\b/.test(normalized) && /\beggs?\b/.test(normalized)) return { days: 7, basis: `Hard-cooked eggs. ${fridge}`, kind: "estimate" };
  if (/\b(cooked|leftover|leftovers)\b/.test(normalized) && !/\b(uncooked|dry|dried)\b/.test(normalized)) return { days: 3, basis: `Assumes freshly cooked leftovers, promptly refrigerated.`, kind: "estimate" };
  if (/\b(dry|dried|uncooked)\b/.test(normalized) && /\b(rice|pasta|beans|chickpeas|lentils)\b/.test(normalized)) return { days: 180, basis: "Assumes dry, uncooked food in a cool, dry pantry. Follow the package instructions.", kind: "estimate" };
  const resolved = canonicalFoodName(normalized);
  if (days[resolved]) return { days: days[resolved], basis: resolved === "eggs" ? `Assumes fresh raw eggs in their shells. ${fridge}` : ["rice", "pasta", "beans", "chickpeas", "lentils"].includes(resolved) ? "Assumes cooked and refrigerated. For dry pantry food, include ‘dry’ in the name." : resolved === "tomato" ? "Assumes ripe tomatoes kept refrigerated. Adjust for ripeness and when purchased." : ["onion", "garlic", "potato", "sweet potato", "bread", "bagels"].includes(resolved) ? "Assumes fresh, uncut food stored in a cool, dry place." : fridge, kind: "estimate" };
  return { days: 1, basis: "Food type or preparation is unclear. Review tomorrow and enter a suitable date; this is not a shelf-life estimate.", kind: "review" };
}
type Expirable = { name: string; detectedExpiry?: string | null; inferredExpiry?: string | null; expirySource?: string; addedAt?: string; expiryBasis?: string; expiryKind?: "estimate" | "review" };
export function withDefaultExpiry<T extends Expirable>(item: T, referenceDate = new Date()): T & Expirable {
  if (item.detectedExpiry || item.inferredExpiry) return item;
  const addedAt = item.addedAt ?? localDate(referenceDate);
  const rule = shelfLifeFor(item.name);
  const date = new Date(`${addedAt}T12:00:00`);
  date.setDate(date.getDate() + rule.days);
  return { ...item, addedAt, inferredExpiry: localDate(date), expirySource: "rule", expiryBasis: rule.basis, expiryKind: rule.kind };
}

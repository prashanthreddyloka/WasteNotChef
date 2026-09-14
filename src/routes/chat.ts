import { Router } from "express";
import { z } from "zod";

const messageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string().min(1).max(6000)
});

const bodySchema = z.object({
  messages: z.array(messageSchema).min(1).max(30),
  pantryItems: z.array(z.string().max(80)).max(500).optional(),
  currentPage: z.string().optional()
});

function buildSystemPrompt(pantryItems: string[] = [], currentPage?: string) {
  const pantryContext = pantryItems.length > 0 ? `Current fridge items: ${pantryItems.join(", ")}.` : "No fridge items available yet.";
  const pageContext = currentPage ? `User is currently viewing: ${currentPage}.` : "Current page unknown.";

  return [
    "You are WasteNotChef AI, a thoughtful cooking and food-waste assistant inside a fridge planning app.",
    "Give practical, concise, high-quality answers.",
    "Prefer pantry-first cooking suggestions and low-waste advice.",
    "If the user asks about recipes, suggest realistic cooking options using their current items when possible.",
    pantryContext,
    pageContext
  ].join(" ");
}

function toTranscript(messages: Array<{ role: "user" | "assistant"; content: string }>) {
  return messages.map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${message.content}`).join("\n");
}

async function callGemini(systemPrompt: string, transcript: string) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("Gemini API key is not configured.");
  }

  const model = process.env.GEMINI_MODEL || process.env.AI_MODEL || "gemini-2.5-flash";
  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`, {
    method: "POST",
    signal: AbortSignal.timeout(30000),
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [
        {
          role: "user",
          parts: [{ text: `${systemPrompt}\n\n${transcript}` }]
        }
      ]
    })
  });

  if (!response.ok) {
    throw new Error(`Chat provider returned HTTP ${response.status}.`);
  }

  const data = (await response.json()) as {
    candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
  };

  return data.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join("").trim() || "I could not generate a response right now.";
}

async function callOpenAI(systemPrompt: string, transcript: string) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OpenAI API key is not configured.");
  }

  const model = process.env.OPENAI_MODEL || process.env.AI_MODEL || "gpt-4.1-mini";
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      input: `${systemPrompt}\n\n${transcript}`
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI chat failed: ${errorText}`);
  }

  const data = (await response.json()) as {
    output_text?: string;
  };

  return data.output_text?.trim() || "I could not generate a response right now.";
}

export const chatRouter = Router();

chatRouter.post("/", async (req, res, next) => {
  try {
    const { messages, pantryItems, currentPage } = bodySchema.parse(req.body);
    const systemPrompt = buildSystemPrompt(pantryItems, currentPage);
    const transcript = toTranscript(messages);
    const provider = (process.env.AI_PROVIDER || "gemini").toLowerCase();

    const reply =
      provider === "openai"
        ? await callOpenAI(systemPrompt, transcript)
        : await callGemini(systemPrompt, transcript);

    return res.json({ reply, provider });
  } catch (error) {
    return next(error);
  }
});

import { PaperAirplaneIcon, SparklesIcon, XMarkIcon } from "@heroicons/react/24/outline";
import { useState } from "react";
import type { PantryItem } from "../types";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type ChatWidgetProps = {
  currentPage: string;
  pantryItems: PantryItem[];
};

async function sendChat(messages: ChatMessage[], pantryItems: PantryItem[], currentPage: string) {
  const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "http://localhost:4000/api";
  const response = await fetch(`${apiBase}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.slice(-30).map(message => ({ ...message, content: message.content.slice(0, 6000) })),
      pantryItems: pantryItems.map((item) => item.name),
      currentPage
    })
  });

  if (!response.ok) {
    throw new Error("Chat failed");
  }

  return response.json() as Promise<{ reply: string; provider: string }>;
}

export function ChatWidget({ currentPage, pantryItems }: ChatWidgetProps) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Ask me about recipes, substitutions, meal ideas, food storage, or how to use what is in your fridge."
    }
  ]);

  async function handleSend() {
    const trimmed = input.trim();
    if (!trimmed || busy) {
      return;
    }

    const nextMessages = [...messages, { role: "user" as const, content: trimmed }];
    setMessages(nextMessages);
    setInput("");
    setBusy(true);

    try {
      const result = await sendChat(nextMessages, pantryItems, currentPage);
      setMessages((current) => [...current, { role: "assistant", content: result.reply }]);
    } catch (error) {
      console.warn("Chat unavailable.", error);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: "AI chat is unavailable right now. Add GEMINI_API_KEY or OPENAI_API_KEY on the backend to enable live answers."
        }
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      {open ? (
        <div className="fixed bottom-6 right-6 z-50 w-[min(420px,calc(100vw-2rem))] rounded-[1.8rem] border border-white/70 bg-white/95 shadow-float backdrop-blur">
          <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
            <div className="flex items-center gap-2">
              <SparklesIcon className="h-5 w-5 text-teal-700" />
              <div>
                <div className="text-sm font-semibold text-ink">WasteNotChef AI</div>
                <div className="text-xs text-slate-500">Gemini or GPT-powered cooking help</div>
              </div>
            </div>
            <button type="button" onClick={() => setOpen(false)} className="rounded-full p-1 text-slate-500 hover:bg-slate-100">
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>

          <div className="max-h-96 space-y-3 overflow-y-auto px-4 py-4">
            {messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={`rounded-2xl px-4 py-3 text-sm leading-6 ${
                  message.role === "assistant" ? "bg-mist text-slate-700" : "bg-ink text-white"
                }`}
              >
                {message.content}
              </div>
            ))}
          </div>

          <div className="border-t border-slate-200 p-4">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Ask about recipes, storage, substitutions, or meal ideas..."
                className="min-h-[52px] flex-1 rounded-2xl border border-slate-200 px-4 py-3 outline-none focus:border-teal-400"
              />
              <button
                type="button"
                onClick={() => void handleSend()}
                disabled={busy}
                className="rounded-2xl bg-ink px-4 text-white disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                <PaperAirplaneIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <button
        type="button"
        onClick={() => setOpen(true)}
        className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-2 rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white shadow-float"
      >
        <SparklesIcon className="h-5 w-5" />
        AI Chat
      </button>
    </>
  );
}

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import type { Server } from "node:http";
import { once } from "node:events";
import sharp from "sharp";
const mocks = vi.hoisted(() => ({ save: vi.fn() }));
vi.mock("../src/db/prismaClient", () => ({ prisma: { ingredientRule: { findMany: async () => [] }, pantryItem: { upsert: mocks.save } } }));
import { createServer } from "../src/server";
let server: Server;
let base: string;
async function send(bytes: Uint8Array, type = "image/png") {
  const data = new FormData(); data.append("image", new Blob([Uint8Array.from(bytes).buffer], { type }), "fridge");
  return fetch(`${base}/api/upload-photo`, { method: "POST", body: data });
}
describe("fridge photo uploads", () => {
  beforeAll(async () => {
    vi.stubEnv("GEMINI_API_KEY", ""); vi.stubEnv("MOCK_GEMINI_INGREDIENTS", ""); vi.stubEnv("MOCK_OCR_TEXT", "tomato eggs");
    server = createServer().listen(0, "127.0.0.1"); await once(server, "listening"); base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  });
  afterAll(async () => { vi.unstubAllEnvs(); await new Promise<void>(resolve => server.close(() => resolve())); });
  it("returns labeled ingredients with defaults and stable scan IDs, without shared writes", async () => {
    const png = await sharp({ create: { width: 100, height: 100, channels: 3, background: "white" } }).png().toBuffer();
    const first = await send(png); expect(first.status).toBe(200); const result = await first.json();
    expect(result.recognition).toBe("labels");
    expect(result.items.map((item: { name: string }) => item.name)).toEqual(["tomato", "eggs"]);
    expect(result.items.every((item: { inferredExpiry: string }) => /^\d{4}-\d{2}-\d{2}$/.test(item.inferredExpiry))).toBe(true);
    expect((await (await send(png)).json()).items).toEqual(result.items);
    expect(mocks.save).not.toHaveBeenCalled();
  });
  it("rejects corrupt bytes, unsupported formats, and oversized images", async () => {
    expect((await send(new Uint8Array([1,2,3]))).status).toBe(400);
    expect((await send(new Uint8Array([1]), "text/plain")).status).toBe(400);
    expect((await send(new Uint8Array(10 * 1024 * 1024 + 1))).status).toBe(400);
  });
});

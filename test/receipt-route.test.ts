import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { Server } from "node:http";
import { existsSync } from "node:fs";
import { once } from "node:events";

const mocks = vi.hoisted(() => ({
  analyze: vi.fn(), rules: vi.fn(), upsert: vi.fn(), transaction: vi.fn(), find: vi.fn()
}));
vi.mock("../src/services/analyzeReceipt", () => ({ analyzeReceipt: mocks.analyze }));
vi.mock("../src/db/prismaClient", () => ({ prisma: {
  ingredientRule: { findMany: mocks.rules },
  pantryItem: { upsert: mocks.upsert, findMany: mocks.find }, $transaction: mocks.transaction
} }));
import { createServer } from "../src/server";

let server: Server;
let base: string;
async function send(type = "image/png") {
  const data = new FormData();
  data.append("image", new Blob(["fixture"], { type }), "receipt.png");
  return fetch(`${base}/api/upload-receipt`, { method: "POST", body: data });
}

describe("receipt upload API", () => {
  beforeAll(async () => {
    server = createServer().listen(0, "127.0.0.1");
    await once(server, "listening");
    base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  });
  afterAll(() => new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve())));
  beforeEach(() => {
    vi.resetAllMocks();
    mocks.rules.mockResolvedValue([]);
    mocks.transaction.mockResolvedValue([]);
    mocks.find.mockResolvedValue([]);
  });
  it("saves recognized food in a transaction and returns existing rows on retries", async () => {
    const item = { id: "receipt-abc-0", name: "milk", quantity: "2", confidence: 0.85, inferredExpiry: null, detectedExpiry: null };
    mocks.analyze.mockResolvedValue({ items: [item], skippedLines: 3 });
    mocks.find.mockResolvedValue([item]);
    const response = await send();
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ items: [{ ...item, detectionSource: "ocr", expirySource: "none" }], skippedLines: 3 });
    expect(mocks.upsert).toHaveBeenCalledWith(expect.objectContaining({ where: { id: item.id }, update: {}, create: expect.objectContaining({ source: "receipt" }) }));
    expect(mocks.transaction).toHaveBeenCalledOnce();
  });
  it("rejects missing or unsupported uploads before OCR", async () => {
    expect((await send("text/plain")).status).toBe(400);
    expect((await fetch(`${base}/api/upload-receipt`, { method: "POST" })).status).toBe(400);
    expect(mocks.analyze).not.toHaveBeenCalled();
  });
  it("returns no items without fabricating inventory", async () => {
    mocks.analyze.mockResolvedValue({ items: [], skippedLines: 5 });
    const response = await send();
    expect((await response.json()).items).toEqual([]);
    expect(mocks.upsert).not.toHaveBeenCalled();
  });
  it("does not persist OCR failures and removes the temporary image", async () => {
    mocks.analyze.mockRejectedValue(new Error("Unreadable image"));
    const response = await send();
    expect(response.status).toBe(500);
    expect(mocks.transaction).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(existsSync(mocks.analyze.mock.calls[0][0])).toBe(false));
  });
});

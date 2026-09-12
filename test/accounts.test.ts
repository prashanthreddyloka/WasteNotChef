import { beforeAll, afterAll, describe, expect, it, vi } from "vitest";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { closeSync, mkdirSync, openSync, rmSync } from "node:fs";
import path from "node:path";
import { once } from "node:events";
import type { Server } from "node:http";

vi.hoisted(() => { process.env.DATABASE_URL = `file:${process.cwd().replaceAll("\\", "/")}/tmp/accounts-test-${process.pid}.db`; });
import { createServer } from "../src/server";
import { prisma } from "../src/db/prismaClient";
import { digest } from "../src/services/auth";

let server: Server;
let base: string;
let tokenA: string;
let tokenB: string;
let recovery: string;
let userA: string;
const password = "a-test-password-12345";
const item = { id: "same-client-id", name: "milk", quantity: "1 carton", confidence: 1, detectedExpiry: "2026-10-01", reviewed: true };
async function call(route: string, method = "GET", data?: unknown, token?: string) {
  return fetch(`${base}/api${route}`, { method, headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) }, ...(data ? { body: JSON.stringify(data) } : {}) });
}
describe("accounts and inventory isolation", () => {
  beforeAll(async () => {
    mkdirSync("tmp", { recursive: true });
    closeSync(openSync(path.resolve(`tmp/accounts-test-${process.pid}.db`), "a"));
    const require = createRequire(path.resolve("package.json"));
    execFileSync(process.execPath, [require.resolve("prisma/build/index.js"), "db", "push", "--skip-generate"], { env: process.env, stdio: "pipe" });
    server = createServer().listen(0, "127.0.0.1"); await once(server, "listening");
    base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }, 30000);
  afterAll(async () => {
    if (server) await new Promise<void>(resolve => server.close(() => resolve()));
    await prisma.$disconnect();
    rmSync(path.resolve(`tmp/accounts-test-${process.pid}.db`), { force: true });
    rmSync(path.resolve(`tmp/accounts-test-${process.pid}.db-journal`), { force: true });
  });
  it("creates accounts with hashed passwords and sessions, and rejects unauthenticated inventory", async () => {
    expect((await call("/inventory")).status).toBe(401);
    expect((await call("/auth/register", "POST", { name: "A", email: "a@example.test", password: "short" })).status).toBe(400);
    const a = await call("/auth/register", "POST", { name: "A", email: "a@example.test", password });
    expect(a.status).toBe(201); const dataA = await a.json(); tokenA = dataA.token; recovery = dataA.recoveryCode; userA = dataA.user.id;
    const b = await call("/auth/register", "POST", { name: "B", email: "b@example.test", password }); tokenB = (await b.json()).token;
    const row = await prisma.user.findUniqueOrThrow({ where: { id: userA } });
    expect(row.passwordHash).not.toContain(password); expect(row.recoveryHash).not.toBe(recovery);
    expect(await prisma.authSession.findUnique({ where: { tokenHash: digest(tokenA) } })).not.toBeNull();
    expect((await call("/auth/login", "POST", { email: "a@example.test", password: "incorrect" })).status).toBe(401);
  });
  it("saves, restores, edits and deletes only the current user's inventory", async () => {
    const saved = await call("/inventory", "PUT", { items: [item], version: 0, userId: "another-user" }, tokenA);
    expect(saved.status).toBe(200);
    expect((await (await call("/inventory", "GET", undefined, tokenA)).json()).items).toEqual([item]);
    expect((await (await call("/inventory", "GET", undefined, tokenB)).json()).items).toEqual([]);
    expect((await call("/inventory", "PUT", { items: [{ ...item, name: "tofu" }], version: 0 }, tokenB)).status).toBe(200);
    expect((await call("/inventory", "PUT", { items: [{ ...item, quantity: "2 cartons" }], version: 1 }, tokenA)).status).toBe(200);
    expect((await call("/inventory", "PUT", { items: [], version: 1 }, tokenB)).status).toBe(200);
    expect((await (await call("/inventory", "GET", undefined, tokenA)).json()).items[0].quantity).toBe("2 cartons");
    expect((await call("/inventory", "PUT", { items: [], version: 2 }, tokenA)).status).toBe(200);
  });
  it("rejects stale device saves and invalid inventory without losing existing data", async () => {
    expect((await call("/inventory", "PUT", { items: [item], version: 3 }, tokenA)).status).toBe(200);
    expect((await call("/inventory", "PUT", { items: [], version: 3 }, tokenA)).status).toBe(409);
    expect((await call("/inventory", "PUT", { items: [{ ...item, detectedExpiry: "2026-02-30" }], version: 4 }, tokenA)).status).toBe(400);
    expect((await call("/inventory", "PUT", { items: [item, item], version: 4 }, tokenA)).status).toBe(400);
    expect((await (await call("/inventory", "GET", undefined, tokenA)).json()).items).toEqual([item]);
  });
  it("loads the same inventory through a second login and revokes a logged-out session", async () => {
    const response = await call("/auth/login", "POST", { email: "A@EXAMPLE.TEST", password });
    expect(response.status).toBe(200); const second = (await response.json()).token;
    expect((await (await call("/inventory", "GET", undefined, second)).json()).items).toEqual([item]);
    expect((await call("/auth/logout", "POST", undefined, second)).status).toBe(204);
    expect((await call("/inventory", "GET", undefined, second)).status).toBe(401);
    expect((await call("/inventory", "GET", undefined, tokenA)).status).toBe(200);
  });
  it("recovers an account once and invalidates all old sessions", async () => {
    const reset = await call("/auth/recover", "POST", { email: "a@example.test", recoveryCode: recovery, password: "a-new-test-password-12345" });
    expect(reset.status).toBe(200); const result = await reset.json();
    expect(result.recoveryCode).not.toBe(recovery);
    expect((await call("/inventory", "GET", undefined, tokenA)).status).toBe(401);
    tokenA = result.token;
    expect((await call("/auth/recover", "POST", { email: "a@example.test", recoveryCode: recovery, password })).status).toBe(401);
    expect((await call("/auth/login", "POST", { email: "a@example.test", password })).status).toBe(401);
    expect((await (await call("/inventory", "GET", undefined, result.token)).json()).items).toEqual([item]);
  });
  it("does not expose legacy shared waste records to guests or another account", async () => {
    const day = { scheduledDate: new Date("2026-09-01"), recipeId: "private-recipe", recipeTitle: "Private meal", reasoning: "private", priority: 1, itemsConsumedJson: "[]", leftoversJson: "[]", wasteScore: 90 };
    await prisma.weekPlan.create({ data: { userId: userA, startDate: new Date("2026-09-01"), endDate: new Date("2026-09-02"), wasteScore: 90, reasoning: "private", metadataJson: "{}", dayPlans: { create: day } } });
    await prisma.weekPlan.create({ data: { startDate: new Date("2026-09-01"), endDate: new Date("2026-09-02"), wasteScore: 90, reasoning: "legacy", metadataJson: "{}", dayPlans: { create: { ...day, recipeTitle: "Legacy meal" } } } });
    expect((await (await call("/waste-score?from=2026-09-01&to=2026-09-30", "GET", undefined, tokenA)).json()).timeseries).toEqual([{ date: "2026-09-01", wasteScore: 90, recipeTitle: "Private meal" }]);
    expect((await (await call("/waste-score?from=2026-09-01&to=2026-09-30")).json()).timeseries).toEqual([]);
    expect((await (await call("/waste-score?from=2026-09-01&to=2026-09-30", "GET", undefined, tokenB)).json()).timeseries).toEqual([]);
  });
});


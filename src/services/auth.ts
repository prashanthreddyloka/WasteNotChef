import { createHash, randomBytes, scrypt, timingSafeEqual } from "node:crypto";
import type { RequestHandler } from "express";
import { prisma } from "../db/prismaClient";

export const digest = (value: string) => createHash("sha256").update(value).digest("hex");
function derive(password: string, salt: string): Promise<Buffer> {
  return new Promise((resolve, reject) => scrypt(password, salt, 64, { N: 32768, r: 8, p: 3, maxmem: 64 * 1024 * 1024 }, (error, key) => error ? reject(error) : resolve(key)));
}
export async function hashPassword(password: string) {
  const salt = randomBytes(16).toString("hex");
  return `scrypt-v1:${salt}:${(await derive(password, salt)).toString("hex")}`;
}
export async function verifyPassword(password: string, stored: string) {
  const [, salt, expected] = stored.split(":");
  const actual = await derive(password, salt);
  const hash = Buffer.from(expected, "hex");
  return hash.length === actual.length && timingSafeEqual(hash, actual);
}
export async function createSession(userId: string) {
  const token = randomBytes(32).toString("base64url");
  await prisma.authSession.deleteMany({ where: { expiresAt: { lt: new Date() } } });
  await prisma.authSession.create({ data: { tokenHash: digest(token), userId, expiresAt: new Date(Date.now() + 30 * 86400000) } });
  return token;
}

// Invalid supplied credentials never silently downgrade an account request to guest mode.
export const identifySession: RequestHandler = async (req, res, next) => {
  const header = req.headers.authorization;
  if (!header) return next();
  const token = /^Bearer ([A-Za-z0-9_-]{43})$/.exec(header)?.[1];
  if (!token) { res.status(401).json({ error: "Please sign in again." }); return; }
  try {
    const session = await prisma.authSession.findUnique({ where: { tokenHash: digest(token) }, include: { user: true } });
    if (!session || session.expiresAt.getTime() <= Date.now()) { res.status(401).json({ error: "Your session has expired. Please sign in again." }); return; }
    res.locals.user = session.user;
    res.locals.tokenHash = session.tokenHash;
    next();
  } catch (error) { next(error); }
};
export const requireAccount: RequestHandler = (_req, res, next) => {
  if (!res.locals.user) { res.status(401).json({ error: "Sign in to sync your pantry." }); return; }
  next();
};

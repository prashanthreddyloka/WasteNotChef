import { Router } from "express";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { rateLimit } from "express-rate-limit";
import { z } from "zod";
import { prisma } from "../db/prismaClient";
import { createSession, digest, hashPassword, requireAccount, verifyPassword } from "../services/auth";

export const authRouter = Router();
const email = z.string().trim().toLowerCase().email().max(254);
const password = z.string().min(12, "Use at least 12 characters for your password.").max(128);
const credentials = z.object({ email, password: z.string().min(1).max(128) });
const publicUser = (user: { id: string; name: string; email: string }) => ({ id: user.id, name: user.name, email: user.email, mode: "account" });
const authLimit = rateLimit({ windowMs: 15 * 60 * 1000, limit: 20, standardHeaders: "draft-7", legacyHeaders: false, message: { error: "Too many attempts. Please try again in 15 minutes." } });
authRouter.use((_req, res, next) => { res.setHeader("Cache-Control", "no-store"); next(); });

authRouter.post("/register", authLimit, async (req, res, next) => {
  try {
    const input = z.object({ email, password, name: z.string().trim().min(1).max(80) }).parse(req.body);
    const recoveryCode = randomBytes(24).toString("base64url");
    const user = await prisma.user.create({ data: { email: input.email, name: input.name, passwordHash: await hashPassword(input.password), recoveryHash: digest(recoveryCode) } });
    res.status(201).json({ user: publicUser(user), token: await createSession(user.id), recoveryCode });
  } catch (error) {
    if ((error as { code?: string }).code === "P2002") { res.status(409).json({ error: "An account already uses this email. Sign in or use your recovery code." }); return; }
    next(error);
  }
});
authRouter.post("/login", authLimit, async (req, res, next) => {
  try {
    const input = credentials.parse(req.body);
    const user = await prisma.user.findUnique({ where: { email: input.email } });
    // Run the same expensive derivation for unknown accounts to reduce timing differences.
    const valid = await verifyPassword(input.password, user?.passwordHash ?? `scrypt-v1:${"0".repeat(32)}:${"0".repeat(128)}`);
    if (!user || !valid) { res.status(401).json({ error: "Email or password is incorrect." }); return; }
    res.json({ user: publicUser(user), token: await createSession(user.id) });
  } catch (error) { next(error); }
});
authRouter.post("/recover", authLimit, async (req, res, next) => {
  try {
    const input = z.object({ email, recoveryCode: z.string().min(1).max(128), password }).parse(req.body);
    const user = await prisma.user.findUnique({ where: { email: input.email } });
    const supplied = digest(input.recoveryCode.trim());
    if (!user || !timingSafeEqual(Buffer.from(supplied), Buffer.from(user.recoveryHash))) { res.status(401).json({ error: "Email or recovery code is incorrect." }); return; }
    const recoveryCode = randomBytes(24).toString("base64url");
    await prisma.$transaction(async tx => {
      const changed = await tx.user.updateMany({ where: { id: user.id, recoveryHash: supplied }, data: { passwordHash: await hashPassword(input.password), recoveryHash: digest(recoveryCode) } });
      if (!changed.count) throw new Error("Recovery code already used.");
      await tx.authSession.deleteMany({ where: { userId: user.id } });
    });
    res.json({ user: publicUser(user), token: await createSession(user.id), recoveryCode });
  } catch (error) { next(error); }
});
authRouter.get("/me", requireAccount, (_req, res) => { res.json({ user: publicUser(res.locals.user) }); });
authRouter.post("/logout", requireAccount, async (_req, res, next) => {
  try { await prisma.authSession.deleteMany({ where: { tokenHash: res.locals.tokenHash } }); res.sendStatus(204); } catch (error) { next(error); }
});

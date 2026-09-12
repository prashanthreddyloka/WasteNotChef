import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import { planRouter } from "./routes/plan";
import { recipesRouter } from "./routes/recipes";
import { chatRouter } from "./routes/chat";
import { uploadRouter } from "./routes/upload";
import { wasteRouter } from "./routes/waste";
import { receiptRouter } from "./routes/receipt";
import { authRouter } from "./routes/auth";
import { inventoryRouter } from "./routes/inventory";
import { identifySession } from "./services/auth";
import { ZodError } from "zod";
import { rateLimit } from "express-rate-limit";

dotenv.config();

function resolveCorsOrigins() {
  const configured = process.env.CLIENT_URLS
    ?.split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);

  if (configured && configured.length > 0) {
    return configured;
  }

  if (process.env.CLIENT_URL) {
    return [process.env.CLIENT_URL];
  }

  return ["http://localhost:5173"];
}

export function createServer() {
  const app = express();
  app.set("trust proxy", 1);
  app.use(
    cors({
      origin(origin, callback) {
        const allowedOrigins = resolveCorsOrigins();
        if (!origin || allowedOrigins.includes(origin)) {
          callback(null, true);
          return;
        }

        callback(new Error(`Origin ${origin} not allowed by CORS`));
      }
    })
  );
  app.use(express.json({ limit: "5mb" }));
  app.use("/api", identifySession);
  app.use("/api/auth", authRouter);
  app.use("/api/inventory", inventoryRouter);
  const scanLimit = rateLimit({ windowMs: 60000, limit: 6, standardHeaders: "draft-7", legacyHeaders: false, message: { error: "Please wait a minute before scanning again." } });
  app.use(["/api/upload-photo", "/api/upload-receipt"], scanLimit);
  app.use("/api/upload-photo", uploadRouter);
  app.use("/api/upload-receipt", receiptRouter);
  app.use("/api/recipes", recipesRouter);
  app.use("/api/chat", chatRouter);
  app.use("/api/plan", planRouter);
  app.use("/api/waste", wasteRouter);
  app.use("/api/plan-week", planRouter);
  app.use("/api/waste-score", wasteRouter);
  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, service: "WasteNotChef API" });
  });
  app.use((error: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    if (error instanceof ZodError) { res.status(400).json({ error: error.issues[0]?.message ?? "Invalid input." }); return; }
    console.error(error);
    res.status(500).json({ error: "We could not complete this request. Please try again." });
  });
  return app;
}

import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import { planRouter } from "./routes/plan";
import { recipesRouter } from "./routes/recipes";
import { uploadRouter } from "./routes/upload";
import { wasteRouter } from "./routes/waste";

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
  app.use("/api/upload-photo", uploadRouter);
  app.use("/api/recipes", recipesRouter);
  app.use("/api/plan", planRouter);
  app.use("/api/waste", wasteRouter);
  app.use("/api/plan-week", planRouter);
  app.use("/api/waste-score", wasteRouter);
  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, service: "WasteNotChef API" });
  });
  app.use((error: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    console.error(error);
    res.status(500).json({ error: error.message || "Unexpected server error." });
  });
  return app;
}

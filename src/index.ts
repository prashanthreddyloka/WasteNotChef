import fs from "node:fs";
import path from "node:path";
import dotenv from "dotenv";
import { createServer } from "./server";
import { startReminderScheduler } from "./services/notifications";

dotenv.config();

const app = createServer();
const port = Number(process.env.PORT ?? 4000);
const tmpDir = path.resolve("tmp");

if (!fs.existsSync(tmpDir)) {
  fs.mkdirSync(tmpDir, { recursive: true });
}

app.listen(port, () => {
  console.log(`WasteNotChef API listening on http://localhost:${port}`);
});

startReminderScheduler();

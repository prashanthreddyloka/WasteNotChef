# WasteNotChef

## What I built
WasteNotChef is a camera-first PWA-style web app for turning a quick fridge photo into a practical, animated weekly cooking quest. The repo includes:

- An Express + TypeScript API with OCR-driven upload parsing, deterministic fridge analysis, recipe ranking, EDF week planning, waste scoring, and reminder stubs.
- A React + Vite + Tailwind + Framer Motion client with a polished landing page, onboarding modal, demo upload hero, editable fridge inventory, recipe suggestions, quest-style weekly planner, waste dashboard, and settings surface.
- Prisma + SQLite for easy local persistence, seed data for ingredient shelf-life rules and recipes, tests for OCR/date parsing, scheduling, waste score logic, and one EDF-flavored integration test.
- Local-dev Docker, example API calls, console analytics stubs, and README instructions for getting the product running quickly.

## Next improvements
- Swap the deterministic OCR/keyword pipeline for a real vision detector with bounding-box fusion.
- Improve recipe ranking and scheduling with learned preference models and confidence-aware inventory reasoning.
- Add full mobile push token lifecycle management, service worker subscription sync, and richer notification channels.

## Quick start
1. Copy `.env.example` to `.env`.
2. Copy `client/.env.example` to `client/.env`.
3. Install dependencies:
   - `npm install`
   - `npm install --prefix client`
4. Run Prisma setup:
   - `npx prisma migrate dev --name init`
   - `npm run prisma:seed`
5. Start the app:
   - `npm run dev`
6. Open the client at `http://localhost:5173`.

## Commands
- `npm run dev` starts API + client together.
- `npm test` runs the server-side tests.
- `npm run build` builds the server and client.
- `npm run prisma:seed` reseeds ingredient rules and recipes.

## Publish for a sharable link
Recommended setup:

- Frontend on Vercel
- Backend on Render

### 1. Push to GitHub
- Create a new GitHub repo.
- Push this project to it.

### 2. Deploy the API to Render
- In Render, create a new Blueprint deploy from your GitHub repo, or create a Web Service manually.
- If using the included [render.yaml](C:\Users\prash\Documents\New%20project\render.yaml), Render will prefill most settings.
- Set `CLIENT_URLS` after the frontend is live to your Vercel URL, for example `https://wastenotchef.vercel.app`.
- Keep the persistent disk mounted at `/opt/render/project/src/prisma` so SQLite survives deploys.
- After deploy, note your API URL, for example `https://wastenotchef-api.onrender.com/api`.
- This setup uses `prisma db push` plus the idempotent seed script so you can publish quickly without generating migrations first.

### 3. Deploy the client to Vercel
- Import the same GitHub repo into Vercel.
- Set the project Root Directory to `client`.
- Add environment variable `VITE_API_BASE_URL` with your Render API URL, for example `https://wastenotchef-api.onrender.com/api`.
- Vercel will use the included [vercel.json](C:\Users\prash\Documents\New%20project\client\vercel.json) when the project Root Directory is set to `client`.

### 4. Update backend CORS
- In Render, set `CLIENT_URLS` to your Vercel production URL.
- If you also want preview deploys to work, add comma-separated origins such as:
  `https://wastenotchef.vercel.app,https://wastenotchef-git-main-yourteam.vercel.app`

### 5. Your sharable link
- Share the Vercel frontend URL.
- The frontend will call the Render API using `VITE_API_BASE_URL`.

## Deployment notes
- The client no longer hardcodes localhost; it reads `VITE_API_BASE_URL`.
- The API accepts one or more allowed frontend origins through `CLIENT_URLS`.
- SQLite is fine for demo/content use, but for heavier public traffic a hosted Postgres setup is the next upgrade.

## Architecture notes
- `analyzeFridgePhoto()` is deterministic by design: OCR text is normalized, fuzzy-matched against a keyword vocabulary, date candidates are parsed with regex plus heuristics, then mapped to nearest token boxes where possible.
- The week planner uses Earliest-Deadline-First as the primary optimization goal, then breaks ties using ingredient utilization and preference tags. The implementation is documented in code and tested.
- Notifications are optional. If SMTP or Web Push credentials are missing, reminders are safely logged to the console.

## Local run flow
- Upload or snap a fridge image from the landing page or fridge page.
- Review detected items and adjust expiry dates.
- Generate recipe suggestions and commit a week plan.
- Explore waste trends and export the plan from the planner and settings views.

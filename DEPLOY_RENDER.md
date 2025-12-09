# Deploying to Render.com

This project has two pieces:
- **API**: `server/index.js` (Node/Express) calls `Smallest_Prototype/feature.py` (Python).
- **UI**: `client/` (React + Vite) – served as a static site.

> ⚠️ **Ollama limitation:** Render does not support running Ollama models. The Python pipeline currently expects an `ollama` client. To use Render, either (a) point `get_model_suggestion_from_llama3` to a remote LLM API you control, or (b) stub/mock that call. Otherwise the API service will fail when it tries to reach Ollama locally.

## 1) Prep the repo
Push your code to GitHub. Keep `requirements.txt` (Python) and `server/package.json` (Node).

## 2) API service (Web Service)
- **Blueprint**
  - Environment: Node
  - Branch: main
  - Root directory: repo root
- **Build Command**
  ```
  npm install --prefix server
  pip install -r requirements.txt
  ```
- **Start Command**
  ```
  cd server && node index.js
  ```
- **Environment Variables**
  - `PYTHON_BIN=python3` (or your chosen Python on Render)
  - `PORT=5001` (Render injects `PORT`; `server/index.js` uses it)
  - Optional overrides for Python paths if your layout differs:
    - `ML_LIBRARY_ROOT` (defaults to `<repo>/ML_BOT/ML_Library`)
    - `TRAINED_MODELS_DIR` (defaults to `<repo>/Trained_Models`)
    - `DEFAULT_DATASET_PATH` (defaults to sample CSV)
- **Persistent storage**
  - If you want to keep uploaded datasets or trained models between deploys, add a Render Disk and point uploads/models to that mount path. By default, uploads land in `server/uploads` and models in `Trained_Models` (ephemeral on free tier).

## 3) UI service (Static Site)
- **Blueprint**
  - Environment: Static Site
  - Branch: main
  - Root directory: `client`
- **Build Command**
  ```
  npm install
  npm run build
  ```
- **Publish Directory**
  ```
  client/dist
  ```
- **Environment Variable**
  - `VITE_API_BASE=https://<your-api-service>.onrender.com`

## 4) Connecting UI → API
- In `client/vite.config.js`, the dev proxy hits `http://localhost:5001`. In production, the built app will call `VITE_API_BASE`. Ensure that env var points at the Render Web Service URL.
- CORS is already enabled in `server/index.js`.

## 5) Smoke test locally (recommended)
```
# API
pip install -r requirements.txt
cd server && npm install && npm run dev

# Frontend
cd client && npm install && npm run dev
```
Visit the UI, upload a dataset, add context/target, and confirm the API returns JSON.

## 6) Deployment flow
1. Create the Render Web Service (API) with the build/start commands above.
2. Create the Render Static Site (UI) with the build/publish settings above.
3. Set `VITE_API_BASE` for the UI to the API’s Render URL.
4. Redeploy both services after updates.

## 7) Operational notes
- Training scripts may be interactive; leave `runTraining` unchecked in the UI or calls to `/api/analyze` when deploying on Render, unless you are certain the script exits without stdin.
- If you add non-ASCII paths or filenames, stick to ASCII to avoid issues in container environments.
- Logs from the Python orchestration are returned in the API response; also check Render logs for build/runtime failures.

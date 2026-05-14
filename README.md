# LLM Next-Token Prediction Visualizer

An interactive web app that demonstrates how Large Language Models use **Softmax**, **Temperature**, and **Sampling** to predict the next token. Type a prompt; a real Hugging Face model (**SmolLM2-135M**) returns its raw next-token logits; you then play with temperature and sampling in the browser.

## Live Demo

- **Frontend (GitHub Pages):** <https://eleanorliu12.github.io/Softmax-Demo/>
- **Backend (Hugging Face Spaces):** *deploy your own — see [backend/README.md](backend/README.md), then update `API_BASE_URL` in `index.html`.*

The frontend always loads. The Predict button only works once the backend Space is reachable from `API_BASE_URL`. The free Spaces tier sleeps after ~48h of inactivity, so the first prediction after sleep may take 20–60 seconds to wake the model.

## Try It

1. Open the live demo link above.
2. Type a prompt (e.g., `The cat sat on the`) and click **Predict**.
3. The top-K real logits from SmolLM2-135M populate the table.
4. Drag the **Temperature** slider to watch the softmax distribution reshape.
5. Click **Sample N tokens** to see empirical frequencies converge to the theoretical probabilities (Law of Large Numbers).

## What It Does

Five interactive steps:

1. **Get Next-Token Logits** — Type a prompt; the backend runs a forward pass through SmolLM2-135M and returns the top-K raw logits at the final position. Each row is editable in case you want to tweak.
2. **Set Temperature** — Slider from 0.1 (deterministic) to 5.0 (nearly uniform). The distribution reshapes in real time.
3. **Compare Logits vs. Probabilities** — Side-by-side bar charts show how softmax transforms raw scores into a valid PMF.
4. **Step-by-Step Math** — Live table: `z_i → z_i/T → exp(z_i/T) → p_i`.
5. **Sampling Simulation** — Sample 100 to 50,000 tokens via inverse-CDF and compare empirical frequencies against theoretical probabilities.

## Probability Concepts Covered

- **Discrete Probability Distributions** — Softmax produces a valid PMF (non-negative, sums to 1).
- **Conditional Probability** — LLMs compute P(next token | prompt).
- **Parameterized Distributions** — Temperature reshapes the distribution without changing the model's underlying scores.
- **Law of Large Numbers** — Empirical sampling frequencies converge to theoretical probabilities as N increases.

## Architecture

```
┌──────────────────────────┐       POST /predict       ┌──────────────────────────────┐
│  Frontend (browser)      │  ──────────────────────▶  │  Backend (FastAPI on HF)     │
│  React + Plotly, no build│   { prompt, top_k }       │  SmolLM2-135M forward pass   │
│  Softmax + sampling code │  ◀──────────────────────  │  Returns top-K raw logits    │
└──────────────────────────┘   [ {token, logit}, ... ] └──────────────────────────────┘
```

The backend's only job is to expose pre-softmax scores. Temperature scaling, the math table, and sampling all happen client-side — that's the educational point of the demo.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Plotly.js + Babel Standalone (all via CDN), inline CSS |
| Backend | Python 3.13, FastAPI, `transformers`, PyTorch (CPU) |
| Model | [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) |
| Frontend hosting | GitHub Pages |
| Backend hosting | Hugging Face Spaces (Docker SDK, free CPU tier) |

The frontend has no build step. The backend is one `app.py` plus a Dockerfile.

## Run Locally

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
First start downloads SmolLM2-135M (~270 MB) into `~/.cache/huggingface`.

**Frontend:** in another terminal, from the project root:
```bash
python -m http.server 5500
```
Then open <http://localhost:5500/index.html>. Don't open `index.html` via `file://` — the browser blocks cross-origin `fetch` from a `null` origin.

To change models, set `MODEL_NAME` before launching uvicorn:
```bash
MODEL_NAME=HuggingFaceTB/SmolLM2-360M uvicorn app:app --reload --port 8000
```

## Deploy

- **Frontend → GitHub Pages.** Already wired (the repo serves `index.html` from the root). After you deploy the backend, update `API_BASE_URL` near the top of the `<script>` block in `index.html` to your Space URL and push.
- **Backend → Hugging Face Spaces.** See [backend/README.md](backend/README.md) for the Docker-Space walkthrough.

## Project Structure

```
.
├── index.html              # Frontend (deployed to GitHub Pages)
├── backend/
│   ├── app.py              # FastAPI service exposing /predict
│   ├── requirements.txt
│   ├── Dockerfile          # For Hugging Face Spaces deploy
│   └── README.md           # Local + Spaces deploy instructions
├── DESIGN.md               # Architecture and design decisions
├── commands.txt            # Frequently-used shell commands (gitignored)
└── README.md               # This file
```

## Course Context

Originally built for **STAT 311 — Probability for Data Science**, topic *AI & Probability (Softmax)*. The backend tier was added afterwards to turn a static visualization into a live ML demo.

## License

[MIT](LICENSE) © 2026 Kejun Liu

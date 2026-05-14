# Design Document: LLM Next-Token Prediction Visualizer

## Overview

An interactive web app that demonstrates how Large Language Models use **Softmax**, **Temperature**, and **Sampling** to predict the next token. The frontend is a single HTML page; the backend is a small Python service that runs a real causal language model (**SmolLM2-135M** by default) and exposes its raw next-token logits over HTTP. The frontend feeds those logits into a softmax/temperature/sampling pipeline that the user can manipulate in real time.

Originally built as a STAT 311 (Probability for Data Science) course project; extended into a two-tier portfolio piece that pairs an in-browser visualization with a deployed ML inference API.

## Topic

**AI & Probability (Softmax).** The visualization covers three core probability concepts:

1. **Discrete Probability Distributions** — Softmax converts a real LM's raw output scores into a valid PMF.
2. **Parameterized Distributions** — Temperature reshapes the distribution without changing the underlying scores.
3. **Law of Large Numbers** — Empirical sampling frequencies converge to theoretical probabilities as sample size grows.

## Architecture

The app is split into two cooperating tiers:

```
┌──────────────────────────────┐         POST /predict          ┌─────────────────────────────────┐
│  Frontend (browser)          │  ───────────────────────────▶  │  Backend (FastAPI)              │
│  • Single HTML / React       │  { prompt, top_k }             │  • Loads SmolLM2-135M at boot   │
│  • Plotly charts             │                                │  • One forward pass per request │
│  • Softmax + sampling code   │  ◀───────────────────────────  │  • Returns top-K raw logits     │
└──────────────────────────────┘   [ { token, token_id, logit } ]  └─────────────────────────────────┘
```

The backend's responsibility is small and well-defined: **run a forward pass on the user's prompt, grab `logits[-1]`, return the top-K entries.** It does *not* generate text. The softmax, temperature scaling, and sampling all happen client-side — which is the point of the demo.

### Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | React 18 (via unpkg CDN) | UI components and state management |
| Frontend | Babel Standalone | In-browser JSX compilation — no build step |
| Frontend | Plotly.js 2.27 | Interactive charts |
| Frontend | Inline CSS | Styling |
| Backend  | Python 3.13 + FastAPI | HTTP API |
| Backend  | Hugging Face `transformers` | Model loading and inference |
| Backend  | PyTorch (CPU) | Forward-pass execution |
| Model    | `HuggingFaceTB/SmolLM2-135M` | Modern (2024) small causal LM, ~270 MB |
| Deploy   | Hugging Face Spaces (Docker) | Backend hosting (free CPU tier) |
| Deploy   | GitHub Pages | Frontend hosting |

The frontend has **no build step** — open `index.html` in a browser (or serve it from any static host). The backend is one `app.py` file plus a Dockerfile.

### Why this split?

- **Real logits require running the model.** Most HF Inference API endpoints return generated text or already-softmaxed top-k, not the raw pre-softmax scores the demo needs. Hosting the model ourselves is the only way to expose the tensor directly.
- **The frontend stays viewable as static content.** Softmax math, temperature scaling, charts, and the sampling simulation are all client-side. The only network call is fetching the initial logits.
- **It deploys to a free tier.** A 135M-parameter model fits in HF Spaces' 2 vCPU / 16 GB free CPU tier with ~1s inference latency.

## Backend

### API Contract

`POST /predict`

Request:
```json
{ "prompt": "The cat sat on the", "top_k": 8 }
```

Response:
```json
{
  "model": "HuggingFaceTB/SmolLM2-135M",
  "prompt": "The cat sat on the",
  "predictions": [
    { "token": " bed",    "token_id":  4463, "logit": 18.07 },
    { "token": " edge",   "token_id":  5595, "logit": 17.59 },
    { "token": " window", "token_id":  5700, "logit": 17.28 }
  ]
}
```

`GET /health` — returns `{ "status": "ok", "model": "...", "loaded": true }` once the model finishes loading.

### Inference pipeline

```python
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
last_logits = outputs.logits[0, -1, :]          # shape: (vocab_size,)
top = torch.topk(last_logits, k=top_k)
predictions = [
    { "token": tokenizer.decode([tid]), "token_id": tid, "logit": float(val) }
    for val, tid in zip(top.values, top.indices)
]
```

The model loads once at FastAPI startup via the `lifespan` context manager. Each request is a single `no_grad` forward pass; nothing is cached per-request.

### Why SmolLM2-135M?

- **Modern (Nov 2024).** Trained on ~2 trillion tokens of curated data (FineWeb-Edu, Cosmopedia, Python-Edu, FineMath). The 2024 small-model frontier is dramatically ahead of 2019-era GPT-2.
- **Modern architecture.** Grouped-Query Attention, RoPE positional embeddings, SwiGLU activations, RMSNorm, 8K context window.
- **Fits free tier.** ~270 MB on disk, ~500 MB–1 GB in RAM, sub-second inference on CPU.
- **Outperforms larger old models.** Benchmarks competitive with GPT-2-medium (355M) at ~⅓ the size.

The `MODEL_NAME` env var swaps the model out without code changes (e.g., `MODEL_NAME=HuggingFaceTB/SmolLM2-360M`).

## Frontend

### Component Structure

The app is a single `App` function component using React hooks:

```
App
├── State: tokens[], temperature, numSamples, sampleCounts[]
├── State: prompt, topK, predictStatus, predictError, modelName
├── Effect: fetchPrediction() on mount   ← populates tokens from real model
├── Computed: probs[] (derived via computeSoftmax on every render)
│
├── Section 1: Get Next-Token Logits
│   ├── Prompt textbox + Top-K selector + Predict button
│   ├── Model name + loading/error indicators
│   └── Editable token table (word + logit slider per row, populated from API)
│
├── Section 2: Temperature Slider
│   ├── Range input (0.1 – 5.0)
│   └── Dynamic description text
│
├── Section 3: Dual Bar Charts (Plotly)
│   ├── Left: Raw logits (from the model, edit-able via sliders)
│   └── Right: Softmax probabilities (reacts to T)
│
├── Section 4: Math Breakdown Table
│   └── Token → z_i → z_i/T → exp(z_i/T) → p_i
│
├── Section 5: Sampling Simulation
│   ├── Sample size selector (100 – 50,000)
│   ├── "Sample" button → runs weighted random sampling
│   └── Plotly chart: empirical bars vs. theoretical diamonds
│
└── Explanation (3 paragraphs)
```

### Why React?

Even though the deliverable is a single HTML file, React provides:
- Clean state management — tokens, temperature, sample counts, and the prompt all flow from `useState`.
- Declarative re-rendering — changing temperature instantly recomputes and redraws everything.
- Component-like structure within a single file using hooks.

## Key Algorithms

### Softmax with Numerical Stability

```
computeSoftmax(tokens, T):
    if tokens is empty: return zeros
    maxLogit  = max(tokens.logit)          // for numerical stability
    scaled_i  = (logit_i - maxLogit) / T   // subtract max to avoid overflow
    exp_i     = exp(scaled_i)
    sumExp    = Σ exp_i
    prob_i    = exp_i / sumExp
```

Subtracting the maximum logit before exponentiation prevents `exp()` overflow — a standard trick that does not change the result because the constant cancels in the ratio.

### Weighted Sampling (Inverse CDF)

```
sampleFromDist(probs, n):
    for each of n samples:
        r = uniform random in [0, 1)
        walk the CDF: find first j where cumulative sum >= r
        increment counts[j]
```

This is the inverse transform sampling method applied to a discrete distribution.

## Interactivity Features

| Feature | Controls | What It Demonstrates |
|---|---|---|
| Live model prediction | Prompt textbox + Predict button + Top-K selector | Real LLM softmax inputs — what the model actually sees before sampling |
| Manual logit tweaking | Text input + slider per token | What softmax would do for any hypothetical logit vector |
| Add/remove tokens | Buttons | Vocabulary size affects the distribution |
| Temperature | Continuous slider (0.1–5.0) | T→0 = deterministic, T→∞ = uniform |
| Sample count | Dropdown (100–50,000) | LLN: more samples → convergence |
| Run sampling | Button | Each click is a fresh random experiment |

## Rubric Alignment

| Rubric Category | Points | How We Address It |
|---|---|---|
| Functionality & Depth | /40 | Live LLM logit fetching, multiple interactive controls, sampling simulation with LLN demonstration — "High Marks" level |
| Mathematical Accuracy | /20 | Correct Softmax with numerical stability, proper weighted sampling, step-by-step math table |
| UX & Polish | /15 | Light theme, labeled controls, titled/axis-labeled charts, responsive layout, loading/error states |
| HTML Explanation | /10 | Three detailed paragraphs covering Softmax, Temperature, and Sampling/LLN |
| Reflection Report | /15 | Separate PDF (not part of this repo) |

## Deployment

| Component | Host | Notes |
|---|---|---|
| Frontend | GitHub Pages | Static HTML — already configured (`index.html` in repo root) |
| Backend  | Hugging Face Spaces (Docker SDK, free CPU) | Built from `backend/Dockerfile`; listens on port 7860 |

The free CPU tier sleeps after ~48h of inactivity; the first request after sleep takes 20–60s to wake (model reload). The frontend shows a "Predicting…" indicator while waiting.

Frontend points at the backend via the `API_BASE_URL` constant near the top of the `<script>` block in `index.html`.

## Browser Compatibility

Tested target: any modern browser (Chrome, Firefox, Safari, Edge) with JavaScript enabled and internet access (for CDN scripts and the backend API).

**Note:** the page must be loaded over HTTP (e.g., `python -m http.server`, GitHub Pages, or a local dev server) — opening `index.html` directly via `file://` causes the browser to assign a `null` origin, which some browsers block from making cross-origin `fetch` calls even with permissive CORS on the server.

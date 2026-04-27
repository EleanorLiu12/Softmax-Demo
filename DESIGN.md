# Design Document: LLM Next-Token Prediction Visualizer

## Overview

An interactive, single-page HTML visualization that demonstrates how Large Language Models use **Softmax**, **Temperature**, and **Sampling** to predict the next token. Built for the STAT 311 (Probability for Data Science) course project.

## Topic

**AI & Probability (Softmax)** — one of the approved project topics. The visualization covers three core probability concepts:

1. **Discrete Probability Distributions** — Softmax converts raw scores into a valid PMF
2. **Parameterized Distributions** — Temperature reshapes the distribution without changing the underlying scores
3. **Law of Large Numbers** — Empirical sampling frequencies converge to theoretical probabilities

## Architecture

### Tech Stack

| Technology | Role | Delivery |
|---|---|---|
| React 18 | UI components and state management | CDN (unpkg) |
| Babel Standalone | In-browser JSX compilation | CDN (unpkg) |
| Plotly.js 2.27 | Interactive charts | CDN (plot.ly) |
| CSS (inline) | Styling | Embedded in `<style>` |

Everything is in a single `Liu_Kejun_Project.html` file — no build step, no dependencies to install. Open it in any modern browser.

### Why React?

Even though the deliverable is a single HTML file, React provides:
- Clean state management (tokens, temperature, sample counts all flow from `useState`)
- Declarative re-rendering — changing temperature instantly recomputes and redraws everything
- Component-like structure within a single file using hooks

### Component Structure

The app is a single `App` function component using React hooks:

```
App
├── State: tokens[], temperature, numSamples, sampleCounts[]
├── Computed: probs[] (derived via computeSoftmax on every render)
│
├── Section 1: Token & Logit Input
│   ├── Preset dropdown (4 scenarios)
│   ├── Editable token table (word + logit slider per row)
│   └── Add/Remove token buttons
│
├── Section 2: Temperature Slider
│   ├── Range input (0.1 – 5.0)
│   └── Dynamic description text
│
├── Section 3: Dual Bar Charts (Plotly)
│   ├── Left: Raw logits (static reference)
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

## Key Algorithms

### Softmax with Numerical Stability

```
computeSoftmax(tokens, T):
    maxLogit = max(tokens.logit)          // for numerical stability
    scaled_i = (logit_i - maxLogit) / T   // subtract max to avoid overflow
    exp_i = exp(scaled_i)
    sumExp = Σ exp_i
    prob_i = exp_i / sumExp
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
| Preset scenarios | Dropdown | Different contexts produce different logit patterns |
| Custom tokens | Text input + slider per token | Logits are arbitrary real numbers |
| Add/remove tokens | Buttons | Vocabulary size affects the distribution |
| Temperature | Continuous slider (0.1–5.0) | T→0 = deterministic, T→∞ = uniform |
| Sample count | Dropdown (100–50,000) | LLN: more samples → convergence |
| Run sampling | Button | Each click is a fresh random experiment |

## Rubric Alignment

| Rubric Category | Points | How We Address It |
|---|---|---|
| Functionality & Depth | /40 | Multiple interactive controls, preset + custom inputs, sampling simulation with LLN demonstration — "High Marks" level |
| Mathematical Accuracy | /20 | Correct Softmax with numerical stability, proper weighted sampling, step-by-step math table |
| UX & Polish | /15 | Dark theme, labeled controls, titled/axis-labeled charts, responsive layout |
| HTML Explanation | /10 | Three detailed paragraphs covering Softmax, Temperature, and Sampling/LLN |
| Reflection Report | /15 | Separate PDF (not part of this file) |

## Browser Compatibility

Tested target: any modern browser (Chrome, Firefox, Safari, Edge) with JavaScript enabled and internet access (for CDN scripts). No server required.

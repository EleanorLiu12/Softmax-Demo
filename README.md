# LLM Next-Token Prediction Visualizer

An interactive web visualization demonstrating how Large Language Models use **Softmax**, **Temperature**, and **Sampling** to predict the next token — built for the STAT 311 Probability for Data Science course project.

## Quick Start

1. Open `Liu_Kejun_Project.html` in any modern browser (Chrome, Firefox, Safari, Edge).
2. That's it. No installation, no build step, no server needed.

The file loads React, Plotly.js, and Babel from CDNs, so an internet connection is required on first load.

## What It Does

This tool lets you explore the probability engine behind LLMs through five interactive steps:

1. **Define Tokens & Logits** — Pick a preset scenario or create your own vocabulary with custom logit scores
2. **Set Temperature** — Drag a slider from 0.1 (deterministic) to 5.0 (nearly random) and watch the distribution reshape in real time
3. **Compare Logits vs. Probabilities** — Side-by-side bar charts show how Softmax transforms raw scores into a valid probability distribution
4. **Step-by-Step Math** — A live computation table breaks down the Softmax formula for every token
5. **Sampling Simulation** — Sample 100 to 50,000 tokens and see the empirical frequencies converge to the theoretical probabilities (Law of Large Numbers)

## Probability Concepts Covered

- **Discrete Probability Distributions** — Softmax produces a valid PMF (non-negative, sums to 1)
- **Conditional Probability** — LLMs compute P(next word | context)
- **Parameterized Distributions** — Temperature reshapes the distribution without changing the model's knowledge
- **Law of Large Numbers** — Empirical sampling frequencies converge to theoretical probabilities as N increases

## Tech Stack

| Technology | Purpose |
|---|---|
| React 18 | UI state management and rendering |
| Plotly.js | Interactive charts |
| Babel Standalone | In-browser JSX compilation |
| CSS | Dark-themed responsive layout |

All dependencies are loaded via CDN. The entire project is a single self-contained HTML file.

## Project Structure

```
.
├── Liu_Kejun_Project.html   # The visualization (deliverable)
├── DESIGN.md                # Architecture and design decisions
└── README.md                # This file
```

## Course Context

- **Course:** STAT 311 — Probability for Data Science
- **Topic:** AI & Probability (Softmax) — approved project topic
- **Reference:** "Probability in the Machine" course document

## License

Academic project. Not licensed for redistribution.

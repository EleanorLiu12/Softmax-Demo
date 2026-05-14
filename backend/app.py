"""
Next-token logit server for the Softmax visualizer.

Runs a small causal LM (SmolLM2-135M by default), executes one forward pass
on a user-supplied prompt, and returns the top-K *raw logits* at the final
position. Softmax + temperature + sampling all happen client-side — this
server's only job is to expose the pre-softmax scores.
"""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M")
MAX_PROMPT_TOKENS = 512
MAX_TOP_K = 50

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    state["tokenizer"] = tokenizer
    state["model"] = model
    yield
    state.clear()


app = FastAPI(title="Softmax Demo: Next-Token Logits", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(8, ge=2, le=MAX_TOP_K)


class Prediction(BaseModel):
    token: str
    token_id: int
    logit: float


class PredictResponse(BaseModel):
    model: str
    prompt: str
    predictions: list[Prediction]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": "model" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    tokenizer = state.get("tokenizer")
    model = state.get("model")
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    inputs = tokenizer(
        req.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # logits shape: (batch=1, seq_len, vocab_size) → take final position
    last_logits = outputs.logits[0, -1, :]
    top = torch.topk(last_logits, k=req.top_k)

    predictions = [
        Prediction(
            token=tokenizer.decode([int(tid)]),
            token_id=int(tid),
            logit=float(val),
        )
        for val, tid in zip(top.values.tolist(), top.indices.tolist())
    ]

    return PredictResponse(
        model=MODEL_NAME,
        prompt=req.prompt,
        predictions=predictions,
    )

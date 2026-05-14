# Softmax Demo — Backend

FastAPI service that loads a small causal LM (default: `HuggingFaceTB/SmolLM2-135M`) and exposes its raw next-token logits over HTTP. The frontend (`../index.html`) calls this service and runs softmax/temperature/sampling client-side.

## API

`POST /predict`

```json
{ "prompt": "The cat sat on the", "top_k": 8 }
```

Response:

```json
{
  "model": "HuggingFaceTB/SmolLM2-135M",
  "prompt": "The cat sat on the",
  "predictions": [
    { "token": " floor", "token_id": 4314, "logit": 4.12 },
    { "token": " ground", "token_id": 1568, "logit": 3.88 },
    ...
  ]
}
```

`GET /health` — returns `{"status":"ok","model":"...","loaded":true}` once the model has finished loading.

## Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

First start downloads the model (~270 MB) into `~/.cache/huggingface` and takes 10–30 seconds. Subsequent starts are cached.

Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The cat sat on the","top_k":8}'
```

## Override the model

Set `MODEL_NAME` env var:

```bash
MODEL_NAME=HuggingFaceTB/SmolLM2-360M uvicorn app:app --port 8000
```

Any causal LM available on the Hugging Face Hub works.

## Deploy to Hugging Face Spaces

1. Create a new Space at <https://huggingface.co/new-space>. Pick **Docker** as the SDK and the free CPU hardware.
2. Clone the Space repo locally and copy `app.py`, `requirements.txt`, and `Dockerfile` into it.
3. Add a `README.md` in the Space root with the Spaces frontmatter:

   ```yaml
   ---
   title: Softmax Demo Backend
   emoji: 🔢
   colorFrom: blue
   colorTo: purple
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

4. `git push` to the Space. The first build takes a few minutes (installing torch).
5. Your endpoint is `https://<username>-<space-name>.hf.space`. Plug that into `API_BASE_URL` at the top of `../index.html`.

The free CPU tier sleeps after ~48h of inactivity; the first request after sleep takes 20–60s to wake.

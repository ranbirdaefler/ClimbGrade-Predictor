---
title: V2 In My Gym
emoji: 🧗
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# V2 In My Gym — Is your gym soft?

Upload a photo of a climbing wall, tap the holds on your route, set the wall
angle, and get an AI-predicted V-grade with a confidence range.

**Live:** https://v2inmygym.net

## How it works

1. Every tapped hold is segmented (LAB colour distance around the tap) and
   cropped to 224x224.
2. A frozen **DINOv2 ViT-B/14** turns each crop into a 768-d embedding.
3. The route becomes a fully connected graph: holds are nodes, edges carry
   dx / dy / distance / angle, wall angle is a global feature.
4. A 3-layer **GATv2** graph network trained on 35,000+ Kilter Board routes
   regresses a difficulty score that maps to V0-V13.

## Layout

- `server.py` — FastAPI JSON API (`/api/session`, `/api/hold`, `/api/predict`, `/api/feedback`) and static hosting
- `static/` — hand-written single-page frontend (canvas hold picker, no framework, no build step)
- `src/inference/` — crop → embed → graph → GNN pipeline
- `src/feedback.py` — saves every prediction (photo + holds) and syncs it to a HF dataset

## Run locally

```bash
pip install -r requirements.txt
uvicorn server:app --port 7860 --reload
```

# Is My Gym Soft? — AI Climbing Grade Predictor

Upload a photo of a climbing route, tap the holds, and get an AI-predicted V-grade. Built to give climbers an unbiased second opinion on route difficulty, independent of gym setting.

**Try it live:** [v2inmygym.net](v2inmygym.net)

## How It Works

The system uses a two-stage pipeline: a vision model to understand individual holds, and a graph neural network to reason about the route as a whole.

### 1. Hold Embedding (DINOv2)

Each hold is cropped from the wall photo using color-based segmentation in LAB space around the user's tap point. The crop is resized to 224×224 and passed through a frozen **DINOv2 ViT-B/14** backbone, producing a 768-dimensional embedding that captures the hold's visual features — shape, size, texture, and color.

### 2. Route Reasoning (GATv2 Graph Neural Network)

The route is modeled as a fully connected graph where each node is a hold and edges encode spatial relationships between holds.

**Node features:**
- DINOv2 embedding (768-d → projected to 64-d)
- Hold role embedding (start / middle / finish / foot — 8-d learned embedding)
- Normalized x, y position on the wall
- Wall angle (0–70°, normalized)

**Edge features (5-d):**
- Relative dx, dy between holds
- Euclidean distance
- sin and cos of the angle between holds

The GNN uses **3 GATv2 layers** with **4 attention heads**, residual connections, and layer normalization. Graph-level readout combines mean pooling with learned attentional aggregation. A final MLP (129 → 64 → 32 → 1) regresses a continuous difficulty score that maps to V-grades (V0–V13+).

**Architecture summary:**
| Component | Details |
|---|---|
| Vision backbone | DINOv2 ViT-B/14 (frozen, 768-d CLS token) |
| Projection | 768 → 256 → 64 (ReLU, dropout 0.1) |
| GNN | 3× GATv2Conv, 4 heads, hidden dim 64 |
| Edge features | 5-d geometric (dx, dy, dist, sin, cos) |
| Pooling | Mean + GlobalAttention (concatenated) |
| Output | MLP → scalar difficulty (10–30 scale) |
| Parameters | ~255K trainable (GNN only, ViT is frozen) |

## Training

The model was trained on **35,000+ routes** from the Kilter Board, a standardized climbing training board with digitally rendered holds and community-graded routes.

**Training setup:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Loss: MSE on predicted vs actual difficulty
- Batch size: 256
- Early stopping: patience 50 on validation MAE
- Data augmentation: 80% foot-hold dropout (randomly maps foot-only holds to generic middle holds during training)

**Results (held-out test set):**
| Metric | Value |
|---|---|
| MAE | 1.27 difficulty units |
| RMSE | 1.63 difficulty units |
| Within ±1 grade | ~49% |

The deployment model is retrained on 95% of the data with 5% validation, achieving a best validation MAE of **1.21**.

## Data Pipeline

```
Kilter Board hold images
    → Crop individual holds (224×224)
    → DINOv2 embedding (768-d per hold)
    → Build fully-connected graph per route
    → GATv2 GNN → difficulty prediction
```

For inference on gym photos, the pipeline adapts:

```
User uploads wall photo
    → Tap each hold → color segmentation crop (LAB space)
    → DINOv2 embedding
    → Build graph with spatial + role features
    → GNN prediction → V-grade
```

## The App

The web app is built with **Streamlit** and deployed on **HuggingFace Spaces** with a T4 GPU.

**User flow:**
1. Upload or snap a photo of a climbing wall
2. Click each hold on the route (auto-detected via color segmentation)
3. Assign roles (start, hand, finish, foot, volume)
4. Set the wall angle
5. Get an AI-predicted V-grade with confidence range

**Data collection:** Every prediction automatically saves the wall photo, hold positions, wall angle, and predicted grade to a HuggingFace dataset. Users can optionally submit the actual gym grade as feedback. This data will be used to fine-tune the model on real gym routes.

## Project Structure

```
climb-pred/
├── app.py                      # Streamlit web app
├── configs/
│   └── inference.yaml          # Model and crop config
├── src/
│   ├── inference/
│   │   ├── crop.py             # Color-based hold cropping
│   │   ├── embed.py            # DINOv2 hold embedder
│   │   ├── graph.py            # PyG graph construction
│   │   ├── model.py            # GATv2 GNN architecture
│   │   └── predict.py          # Full inference pipeline
│   ├── feedback.py             # Data collection + HF sync
│   └── logo_b64.py             # Embedded logo
├── models/
│   └── deployment_model.pt     # Trained model checkpoint
├── Dockerfile                  # HF Spaces deployment
└── requirements-deploy.txt     # Production dependencies
```

## Tech Stack

- **Vision:** DINOv2 ViT-B/14 (Meta)
- **GNN:** GATv2 via PyTorch Geometric
- **Frontend:** Streamlit(Will change later maybe)
- **Deployment:** HuggingFace Spaces (Docker, T4 GPU)
- **Data storage:** HuggingFace Datasets

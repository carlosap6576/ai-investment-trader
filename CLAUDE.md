# CLAUDE.md - Technical Reference

This file provides technical guidance for Claude Code and developers working with this repository.

```
CORE COMMANDS PIPELINE (DO NOT DELETE):
  Option A: Maximum Intraday Data (1 month, 5-minute)

  python download.py -s AAPL -p 1mo -i 5m -n 2000
  python train.py -s AAPL -b 0.1 --sell-threshold -0.1 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300
  python test.py -s AAPL -b 0.1 --sell-threshold -0.1 --samples 10 --summary

  Option B: More History (3 months, hourly)

  python download.py -s AAPL -p 3mo -i 1h -n 2000
  python train.py -s AAPL -b 0.3 --sell-threshold -0.3 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300
  python test.py -s AAPL -b 0.3 --sell-threshold -0.3 --samples 10 --summary

  Option C: Maximum History (1 year, daily)

  python download.py -s AAPL -p 1y -i 1d -n 2000
  python train.py -s AAPL -b 1.0 --sell-threshold -1.0 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300
  python test.py -s AAPL -b 1.0 --sell-threshold -1.0 --samples 10 --summary

```

---

## Table of Contents

- [Project Structure](#project-structure)
- [File Reference](#file-reference)
- [CLI Commands Reference](#cli-commands-reference)
- [Configuration Parameters](#configuration-parameters)
- [Advanced Threshold Tuning](#advanced-threshold-tuning)
- [Model Architecture](#model-architecture)
- [Data Flow](#data-flow)
- [Smart Training Guard](#smart-training-guard)
- [Parameter Consistency Rules](#parameter-consistency-rules)
- [Device Support](#device-support)
- [Known Issues](#known-issues)
- [Future Optimizations](#future-optimizations)

---

## Project Structure

```
ai-investment-trader/
├── download.py                 # Data collection script
├── train.py                    # Model training script
├── test.py                     # Model evaluation script
├── requirements.txt            # Python dependencies
├── README.md                   # Learning guide (start here)
├── CLAUDE.md                   # Technical reference (this file)
│
├── models/
│   └── gemma_transformer_classifier.py  # Neural network model
│
└── datasets/
    └── {SYMBOL}/               # Per-symbol data folder
        ├── {SYMBOL}.pth        # Trained model weights
        ├── .training_meta.json # Training history & metadata
        ├── historical_data.json # Price data (regenerable)
        ├── news.json           # Raw news articles (regenerable)
        └── news_with_price.json # Merged training data (regenerable)
```

### File Categories

| Category | Files | On Delete |
|----------|-------|-----------|
| **Protected** | `*.pth` | Hours of training to recreate |
| **Regenerable** | `*.json` | Re-download in seconds |
| **Metadata** | `.training_meta.json` | Auto-regenerated |

---

## File Reference

### Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `download.py` | ~822 | Fetches price data (yfinance) and news (Yahoo Finance), merges by timestamp |
| `train.py` | ~1039 | Loads data, trains model, saves weights, implements Smart Guard |
| `test.py` | ~515 | Loads model, evaluates on test data, reports accuracy/F1 |

### Model

| File | Purpose |
|------|---------|
| `models/gemma_transformer_classifier.py` | Defines `SimpleGemmaTransformerClassifier` with Gemma embeddings + Transformer encoder |

### Data Files (per symbol)

| File | Created By | Used By | Content |
|------|------------|---------|---------|
| `historical_data.json` | download.py | download.py | Price data (timestamp → price) |
| `news.json` | download.py | download.py | Raw news articles |
| `news_with_price.json` | download.py | train.py, test.py | Merged training data with labels |
| `{SYMBOL}.pth` | train.py | test.py | Trained model weights |
| `.training_meta.json` | train.py | train.py | Training history, data hash |

---

## CLI Commands Reference

### download.py

```bash
python download.py -s SYMBOL [OPTIONS]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--symbol` | `-s` | *required* | Trading symbol (BTC-USD, AAPL, etc.) |
| `--period` | `-p` | `1mo` | How far back (1d, 5d, 1mo, 3mo, 6mo, 1y, max) |
| `--interval` | `-i` | `5m` | Candle interval (1m, 2m, 5m, 15m, 30m, 1h, 1d) |
| `--news-count` | `-n` | `100` | Number of news articles to fetch |

**Intraday Data Limits (yfinance):**

| Interval | Maximum Period |
|----------|----------------|
| 1m | 7 days |
| 2m, 5m, 15m, 30m | 60 days |
| 60m, 90m, 1h | 730 days |
| 1d+ | Unlimited |

### train.py

```bash
python train.py -s SYMBOL [OPTIONS]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--symbol` | `-s` | *required* | Trading symbol |
| `--buy-threshold` | `-b` | `1.0` | Price increase % for BUY label |
| `--sell-threshold` | | `-1.0` | Price decrease % for SELL label |
| `--epochs` | `-e` | `20` | Training epochs |
| `--learning-rate` | `-l` | `0.005` | Optimizer learning rate |
| `--optimizer` | `-o` | `SGD` | Optimizer (SGD or AdamW) |
| `--batch-size` | | `1` | Samples per gradient update |
| `--split` | | `0.8` | Train/test split ratio |
| `--hidden-dim` | | `256` | Model internal dimension |
| `--num-layers` | | `2` | Transformer encoder layers |
| `--fresh` | | `False` | Start from scratch (ignore existing .pth) |
| `--force` | | `False` | Bypass Smart Training Guard |
| `--min-new-samples` | | `5` | Minimum new samples to train |
| `--cooldown` | | `5` | Minutes between training sessions |

### test.py

```bash
python test.py -s SYMBOL [OPTIONS]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--symbol` | `-s` | *required* | Trading symbol (must match train.py) |
| `--buy-threshold` | `-b` | `1.0` | Must match train.py |
| `--sell-threshold` | | `-1.0` | Must match train.py |
| `--split` | | `0.8` | Must match train.py |
| `--samples` | | `3` | Sample predictions to display |
| `--hidden-dim` | | `256` | Must match train.py |
| `--num-layers` | | `2` | Must match train.py |

---

## Configuration Parameters

### Model Architecture Parameters

| Parameter | Values | Model Size | Use Case |
|-----------|--------|------------|----------|
| `--hidden-dim 256` | Default | ~2.6 MB | Fast, general use |
| `--hidden-dim 512` | Larger | ~10 MB | Higher capacity |
| `--num-layers 2` | Default | Faster | Quick training |
| `--num-layers 4` | More | Slower | Better patterns |

### Training Profiles

| Profile | Settings | Training Time | Use Case |
|---------|----------|---------------|----------|
| **Speed** | `--batch-size 32 -e 20 -l 0.01` | ~10 min | Prototyping |
| **Balanced** | `--batch-size 8 -e 50 -l 0.005` | ~30 min | General use |
| **Precision** | `--batch-size 1 -e 200 -l 0.001 -o AdamW` | ~4 hours | Production |

### Optimizer Comparison

| Optimizer | Behavior | Best For |
|-----------|----------|----------|
| `SGD` | Fixed step size | Speed, prototyping |
| `AdamW` | Adaptive step size | Precision, production |

---

## Advanced Threshold Tuning

### By Time Interval

| Interval | Threshold Range |
|----------|-----------------|
| 1 minute | ±0.2% to ±0.5% |
| 5 minutes | ±0.5% to ±1.5% |
| 15 minutes | ±1% to ±2% |
| 1 hour | ±2% to ±4% |
| 1 day | ±3% to ±7% |

### By Market Conditions

| Condition | Adjustment | Example (BTC) |
|-----------|------------|---------------|
| Major news event | +50-100% | ±1.5% to ±2.0% |
| Normal trading | Baseline | ±1.0% |
| Weekends | -30-50% | ±0.5% to ±0.7% |
| Holidays | -50-70% | ±0.3% to ±0.5% |

### By Trader Profile

| Style | Threshold | Signals/Day |
|-------|-----------|-------------|
| Conservative | Wider (+30%) | Fewer |
| Balanced | Baseline | Moderate |
| Aggressive | Tighter (-30%) | More |

### Quick Calibration

Check label distribution after training:

| Distribution | Problem | Action |
|--------------|---------|--------|
| All HOLD | Thresholds too wide | Lower thresholds |
| No HOLD | Thresholds too tight | Raise thresholds |
| 20-40% each | Good | Keep thresholds |

---

## Model Architecture

### Architecture Flow

```
Input: "Price: 95000, Headline: Bitcoin crashes..."
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Gemma Embedding      │  ← Frozen (300M params)
                 │   (1024 dimensions)    │
                 └───────────┬────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Linear Projection    │  ← Trainable
                 │   1024 → 256 dims      │
                 └───────────┬────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  TransformerEncoder    │  ← Trainable
                 │  (2 layers, 4 heads)   │
                 └───────────┬────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Classifier Head      │  ← Trainable
                 │   256 → 3 classes      │
                 └───────────┬────────────┘
                              │
                              ▼
                 Output: [SELL, HOLD, BUY]
```

### Trainable Parameters

| Component | Parameters | Notes |
|-----------|------------|-------|
| Gemma Embedding | 300M | Frozen |
| Linear Projection | ~262K | Trainable |
| Transformer | ~400K | Trainable |
| Classifier | ~770 | Trainable |
| **Total Trainable** | ~660K | |

### Key Model Methods

| Method | Purpose |
|--------|---------|
| `model(texts)` | Forward pass, returns logits |
| `model.predict(text)` | Convenience method, returns dict |
| `model.train()` | Enable training mode |
| `model.eval()` | Enable evaluation mode |

---

## Data Flow

```
┌─────────────┐
│   yfinance  │
│  (prices)   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         download.py                     │
│  • Fetches 5-min price candles           │
│  • Fetches news headlines                │
│  • Merges by timestamp                   │
│  • Calculates 5-min future price change  │
│  • Assigns SELL/HOLD/BUY labels          │
└──────┬──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ news_with_price.json         │
│ [                            │
│   {                          │
│     "title": "...",          │
│     "price": 259.95,         │
│     "future_price": 260.0,   │
│     "percentage": 0.019      │
│   }                          │
│ ]                            │
└──────┬───────────────────────┘
       │
       ├─────────────┬──────────────┐
       ▼             ▼              ▼
   [train]      [test]         [train.py]
     80%          20%             │
       │             │            ▼
       ▼             ▼      ┌────────────┐
    Training     Testing    │ {SYMBOL}   │
    Loop         Loop       │   .pth     │
                            └────────────┘
```

### Label Assignment

```
Price Change        Condition              Label
─────────────       ─────────              ─────
< -1.0%             pct < SELL_THRESHOLD   → SELL
-1.0% to +1.0%      (everything else)      → HOLD
> +1.0%             pct > BUY_THRESHOLD    → BUY
```

---

## Smart Training Guard

Prevents overfitting by checking if training is needed.

### Guard Checks

| Check | Priority | Default | Purpose |
|-------|----------|---------|---------|
| Data Hash | PRIMARY | - | Skip if data unchanged |
| Cooldown | Secondary | 5 min | Prevent rapid-fire training |
| Min Samples | Tertiary | 5 | Skip if too few new samples |

### Tuning Guards

| Data Interval | Cooldown | Min Samples |
|---------------|----------|-------------|
| 1 minute | 1 | 3 |
| 5 minutes | 5 | 5 |
| 15 minutes | 15 | 5 |
| 1 hour | 60 | 10 |

### Bypass Options

```bash
# Force training (bypass all guards)
python train.py -s BTC-USD --force

# Custom guard settings
python train.py -s BTC-USD --min-new-samples 20 --cooldown 120
```

### Metadata File

Smart Guard stores history in `.training_meta.json`:

```json
{
  "last_training": {
    "timestamp": "2025-01-04T15:30:00+00:00",
    "data_hash": "sha256:a1b2c3d4...",
    "sample_count": 156,
    "epochs": 20,
    "accuracy": 0.72,
    "f1_score": 0.68
  }
}
```

---

## Parameter Consistency Rules

**CRITICAL**: Parameters must match between train.py and test.py.

| Parameter | Mismatch Effect | Severity |
|-----------|-----------------|----------|
| `--buy-threshold` | Wrong labels | CRITICAL |
| `--sell-threshold` | Wrong labels | CRITICAL |
| `--split` | Data leakage | CRITICAL |
| `--hidden-dim` | Model incompatible | CRITICAL |
| `--num-layers` | Model incompatible | CRITICAL |
| `--samples` | Display only | Safe |

### Correct Usage

```bash
# Training
python train.py -s BTC-USD -b 0.1 --sell-threshold -0.1 --hidden-dim 512 --num-layers 4

# Testing (MUST MATCH)
python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1 --hidden-dim 512 --num-layers 4
```

---

## Device Support

Auto-detection priority:
1. **CUDA** - NVIDIA GPUs (Linux/Windows)
2. **MPS** - Apple Silicon (M1/M2/M3)
3. **CPU** - Fallback

### Enabling CUDA

If PyTorch shows CPU instead of GPU:

```bash
# Check GPU
nvidia-smi

# Check PyTorch
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Reinstall with CUDA
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

| CUDA Version | Install Command |
|--------------|-----------------|
| 11.8+ | `--index-url https://download.pytorch.org/whl/cu118` |
| 12.1+ | `--index-url https://download.pytorch.org/whl/cu121` |
| 12.4+ | `--index-url https://download.pytorch.org/whl/cu124` |

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "Using CPU" with NVIDIA GPU | Reinstall PyTorch with CUDA |
| "CUDA out of memory" | Reduce `--batch-size` |
| `nvidia-smi` not found | Install NVIDIA drivers |

---

## Known Issues

All major issues have been resolved:

| Issue | Status |
|-------|--------|
| Hardcoded MPS device | Fixed - uses auto-detection |
| Inverted label calculation | Fixed - uses `future_price - price` |
| Wrong threshold values | Fixed - uses ±1.0% |
| Unnecessary model save in test.py | Fixed - removed |

---

## Future Optimizations

### High Priority

#### 1. Embedding Caching to Disk

**Problem**: Gemma embedding is slow (~10-20 min for 1000 samples), recomputed each training run.

**Solution**: Persist embeddings to `datasets/{SYMBOL}/.embedding_cache.pkl`

**Expected Benefit**: 10-100x faster subsequent runs.

---

#### 2. Learning Rate Scheduling

**Problem**: Constant learning rate throughout training.

**Options**:
- `StepLR`: Decay every N epochs
- `CosineAnnealingLR`: Smooth decay
- `ReduceLROnPlateau`: Decay when loss plateaus

```bash
python train.py -s BTC-USD --lr-schedule cosine
```

---

### Medium Priority

#### 3. Time of Day Features

Add temporal context from timestamps:
- Hour of day
- Day of week
- Market open/close status

---

#### 4. Market Volatility Indicator

Add rolling statistics:
- Median % change over past N intervals
- Standard deviation (volatility measure)
- Direction bias

---

#### 5. Sequence Data

Instead of single news → prediction, use sequence of recent news:
- Concatenate last N articles
- Capture sentiment momentum

---

### Lower Priority

#### 6. Dropout Rate Tuning

Expose dropout as CLI parameter (currently hardcoded at 0.1).

#### 7. Multi-stage Batch Training

Vary batch size during training:
- Start large, decrease to small
- Or start small, increase to large

#### 8. Reinforcement Learning

Replace supervised classification with RL to optimize for profit directly.

---

## Priority Matrix

| Item | Complexity | Benefit |
|------|------------|---------|
| Embedding caching | Low-Medium | HIGH |
| Learning rate scheduling | Low-Medium | Medium |
| Dropout tuning | Low | Low-Medium |
| Time features | Medium | Medium |
| Market volatility | Medium | Medium |
| Sequence data | Medium-High | Medium-High |
| Multi-stage batch | Low-Medium | Low |
| Reinforcement learning | HIGH | HIGH |

---

**Document Version**: 2.0 | **Last Updated**: January 2025

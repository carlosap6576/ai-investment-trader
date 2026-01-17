# CLAUDE.md - Technical Reference

This file provides technical guidance for Claude Code and developers working with this repository.

```
CORE COMMANDS PIPELINE (DO NOT DELETE):
  Option A: Maximum Intraday Data (1 month, 5-minute)

  python -m cli.finbert_sentiment.download -s AAPL -p 1mo -i 5m -n 2000
  python -m cli.finbert_sentiment.train -s AAPL -b 0.1 --sell-threshold -0.1 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300
  python -m cli.finbert_sentiment.test -s AAPL -b 0.1 --sell-threshold -0.1 --samples 10 --summary

  Option B: More History (3 months, hourly)

  python -m cli.finbert_sentiment.download -s AAPL -p 3mo -i 1h -n 2000
  python -m cli.finbert_sentiment.train -s AAPL -b 0.3 --sell-threshold -0.3 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300
  python -m cli.finbert_sentiment.test -s AAPL -b 0.3 --sell-threshold -0.3 --samples 10 --summary

  Option C: Maximum History (1 year, daily)

  python -m cli.finbert_sentiment.download -s AAPL -p 1y -i 1d -n 2000
  python -m cli.finbert_sentiment.train -s AAPL -b 1.0 --sell-threshold -1.0 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300
  python -m cli.finbert_sentiment.test -s AAPL -b 1.0 --sell-threshold -1.0 --samples 10 --summary

```

---

## Table of Contents

- [Project Structure](#project-structure)
- [File Reference](#file-reference)
- [CLI Commands Reference](#cli-commands-reference)
- [Configuration Parameters](#configuration-parameters)
- [Advanced Threshold Tuning](#advanced-threshold-tuning)
- [Model Architecture](#model-architecture)
- [Hierarchical Sentiment Features](#hierarchical-sentiment-features)
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
├── cli/                               # Command-line interface
│   └── finbert_sentiment/             # FinBERT sentiment strategy
│       ├── __init__.py
│       ├── download.py                # Data collection with sentiment analysis
│       ├── train.py                   # Model training with Smart Guard
│       └── test.py                    # Evaluation with AI summary
│
├── src/                               # Core application logic
│   ├── data/
│   │   ├── news_classifier.py         # MARKET/SECTOR/TICKER classification
│   │   ├── schemas.py                  # Data structures (40 features)
│   │   └── sector_mapping.py           # GICS sector codes
│   │
│   ├── features/
│   │   ├── sentiment_aggregator.py     # Multi-level sentiment aggregation
│   │   └── feature_builder.py          # Temporal sequence construction
│   │
│   ├── models/
│   │   ├── transformer.py              # HierarchicalSentimentTransformer
│   │   ├── finbert_sentiment.py        # FinBERT analyzer with caching
│   │   ├── report_summarizer.py        # Flan-T5 AI summary generator
│   │   └── gemma_transformer_classifier.py  # Legacy Gemma-based model
│   │
│   └── evaluation/
│       ├── metrics.py                  # Trading-specific metrics
│       └── report_generator.py         # Formatted evaluation reports
│
├── ai/                                # Reusable AI components
│   ├── sentiment/
│   │   └── finbert.py                  # FinBERT sentiment analyzer
│   ├── embeddings/
│   │   └── gemma.py                    # Gemma text embeddings
│   └── summarizers/
│       └── flan_t5.py                  # Flan-T5 report summarizer
│
├── core/                              # Shared infrastructure
│   ├── config/
│   │   └── constants.py                # Centralized defaults
│   ├── utils/
│   │   └── device.py                   # GPU/CPU detection
│   └── cli/
│       └── parser.py                   # FriendlyArgumentParser
│
├── models/                            # Legacy model definitions
│   └── gemma_transformer_classifier.py
│
├── datasets/                          # Per-symbol data and models
│   └── {SYMBOL}/
│       ├── {SYMBOL}.pth               # Trained model weights
│       ├── {SYMBOL}_metadata.json     # Model architecture config
│       ├── .training_meta.json        # Smart Guard metadata
│       ├── historical_data.json       # Price data (regenerable)
│       ├── news.json                  # Raw news articles (regenerable)
│       └── news_with_price.json       # Training data with sentiment
│
├── requirements.txt                   # Python dependencies
├── README.md                          # Learning guide
├── CLAUDE.md                          # Technical reference (this file)
└── hierarchical_sentiment_implementation_plan.md  # Architecture documentation
```

### File Categories

| Category | Files | On Delete |
|----------|-------|-----------|
| **Protected** | `*.pth`, `*_metadata.json` | Hours of training to recreate |
| **Regenerable** | `*.json` (except metadata) | Re-download in seconds |
| **Metadata** | `.training_meta.json` | Auto-regenerated |

---

## File Reference

### CLI Commands

| File | Lines | Purpose |
|------|-------|---------|
| `cli/finbert_sentiment/download.py` | ~1800 | Fetches price data, news, runs FinBERT sentiment |
| `cli/finbert_sentiment/train.py` | ~1150 | Trains hierarchical model, implements Smart Guard |
| `cli/finbert_sentiment/test.py` | ~1050 | Evaluates model, generates AI summary |

### Core Models

| File | Purpose |
|------|---------|
| `src/models/transformer.py` | HierarchicalSentimentTransformer with cross-level attention |
| `src/models/finbert_sentiment.py` | FinBERT analyzer with MD5 caching |
| `src/models/report_summarizer.py` | Flan-T5 AI summary generator |

### Data Processing

| File | Purpose |
|------|---------|
| `src/data/news_classifier.py` | Classifies news into MARKET/SECTOR/TICKER levels |
| `src/data/schemas.py` | Data structures for 40 hierarchical features |
| `src/features/sentiment_aggregator.py` | Aggregates sentiment by level and time window |

### Data Files (per symbol)

| File | Created By | Used By | Content |
|------|------------|---------|---------|
| `historical_data.json` | download | download | Price data (timestamp → price) |
| `news.json` | download | download | Raw news articles |
| `news_with_price.json` | download | train, test | Training data with sentiment |
| `{SYMBOL}.pth` | train | test | Trained model weights |
| `{SYMBOL}_metadata.json` | train | test | Model architecture config |
| `.training_meta.json` | train | train | Smart Guard metadata |

---

## CLI Commands Reference

### Download Command

```bash
python -m cli.finbert_sentiment.download -s SYMBOL [OPTIONS]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--symbol` | `-s` | *required* | Trading symbol (BTC-USD, AAPL, etc.) |
| `--period` | `-p` | `1mo` | How far back (1d, 5d, 1mo, 3mo, 6mo, 1y, max) |
| `--interval` | `-i` | `5m` | Candle interval (1m, 2m, 5m, 15m, 30m, 1h, 1d) |
| `--news-count` | `-n` | `100` | Number of news articles to fetch |
| `--no-filter` | | `False` | Disable news relevance filtering |
| `--no-sentiment` | | `False` | Skip FinBERT sentiment analysis |

**Intraday Data Limits (yfinance):**

| Interval | Maximum Period |
|----------|----------------|
| 1m | 7 days |
| 2m, 5m, 15m, 30m | 60 days |
| 60m, 90m, 1h | 730 days |
| 1d+ | Unlimited |

### Train Command

```bash
python -m cli.finbert_sentiment.train -s SYMBOL [OPTIONS]
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
| `--seq-length` | | `20` | Temporal sequence length |
| `--fresh` | | `False` | Start from scratch (ignore existing .pth) |
| `--force` | | `False` | Bypass Smart Training Guard |
| `--min-new-samples` | | `5` | Minimum new samples to train |
| `--cooldown` | | `5` | Minutes between training sessions |

### Test Command

```bash
python -m cli.finbert_sentiment.test -s SYMBOL [OPTIONS]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--symbol` | `-s` | *required* | Trading symbol (must match train) |
| `--buy-threshold` | `-b` | `1.0` | Must match train |
| `--sell-threshold` | | `-1.0` | Must match train |
| `--split` | | `0.8` | Must match train |
| `--samples` | | `3` | Sample predictions to display |
| `--hidden-dim` | | `256` | Must match train |
| `--num-layers` | | `2` | Must match train |
| `--summary` | | `False` | Generate AI-powered summary |
| `--summary-model` | | `flan-t5-xl` | Model for AI summary |

**AI Summary Models:**
- `google/flan-t5-small` (80M) - Fastest
- `google/flan-t5-base` (250M) - Balanced
- `google/flan-t5-large` (780M) - Good quality
- `google/flan-t5-xl` (3B) - Best quality (default)

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

| Profile | Settings | Use Case |
|---------|----------|----------|
| **Speed** | `--batch-size 32 -e 20 -l 0.01` | Prototyping |
| **Balanced** | `--batch-size 8 -e 50 -l 0.005` | General use |
| **Precision** | `--batch-size 1 -e 200 -l 0.001 -o AdamW` | Production |

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

### Quick Calibration

Check label distribution after training:

| Distribution | Problem | Action |
|--------------|---------|--------|
| All HOLD | Thresholds too wide | Lower thresholds |
| No HOLD | Thresholds too tight | Raise thresholds |
| 20-40% each | Good | Keep thresholds |

---

## Model Architecture

### HierarchicalSentimentTransformer

```
Input: [batch, seq_len, 40 features]
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Level-Specific Encoders                                │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────┐ │
│  │ Ticker    │  │ Sector    │  │ Market    │  │Cross │ │
│  │ (12→128)  │  │ (10→128)  │  │ (10→128)  │  │(8→128)│ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └──┬───┘ │
│        └──────────────┼──────────────┼───────────┘     │
│                       ▼                                │
│  ┌────────────────────────────────────────────┐        │
│  │     Cross-Level Attention Module           │        │
│  │  • Ticker → Sector attention               │        │
│  │  • Ticker → Market attention               │        │
│  │  • Residual connections                    │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Fusion Layer (512 → 64)                                │
│  LayerNorm → GELU → Dropout                             │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Positional Encoding (Sinusoidal)                       │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Temporal Transformer Encoder                           │
│  • Multi-head self-attention (4 heads)                  │
│  • 2 encoder layers                                     │
│  • Feed-forward with GELU                               │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Classification Head                                    │
│  Linear(64 → 3) → [SELL, HOLD, BUY]                    │
└─────────────────────────────────────────────────────────┘
```

### Key Model Methods

| Method | Purpose |
|--------|---------|
| `model(x)` | Forward pass, returns logits [batch, 3] |
| `model.predict(x)` | Returns class, confidence, probabilities |
| `model.get_level_importance()` | Returns attention weights per level |

---

## Hierarchical Sentiment Features

### Feature Summary (40 Total)

| Level | Count | Examples |
|-------|-------|----------|
| **Ticker** | 12 | sentiment_mean, momentum_1d, news_volume |
| **Sector** | 10 | breadth, dispersion, leader_sentiment |
| **Market** | 10 | fed_sentiment, fear_index, regime |
| **Cross-Level** | 8 | ticker_vs_sector_deviation, divergence_score |

### News Classification

| Level | Triggers |
|-------|----------|
| **MARKET** | Fed, FOMC, GDP, inflation, unemployment, tariffs |
| **SECTOR** | Industry-wide, regulatory, supply chain |
| **TICKER** | Company name, earnings, CEO, product launch |

### FinBERT Sentiment

- **Model:** ProsusAI/finbert
- **Score Range:** -1 (negative) to +1 (positive)
- **Caching:** MD5-based to avoid recomputation

---

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    DOWNLOAD PHASE                        │
├─────────────────────────────────────────────────────────┤
│  yfinance → Price data                                  │
│  Yahoo/Alpha Vantage → News articles                    │
│  NewsLevelClassifier → MARKET/SECTOR/TICKER labels      │
│  FinBERT → Sentiment scores (-1 to +1)                  │
│  → news_with_price.json                                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     TRAIN PHASE                          │
├─────────────────────────────────────────────────────────┤
│  news_with_price.json                                   │
│  → SentimentAggregator (40 features per timestep)       │
│  → TemporalFeatureBuilder (sequences of 20)             │
│  → HierarchicalSentimentTransformer                     │
│  → {SYMBOL}.pth + metadata                              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      TEST PHASE                          │
├─────────────────────────────────────────────────────────┤
│  Load model from {SYMBOL}.pth                           │
│  Evaluate on test split (20%)                           │
│  Compute metrics (accuracy, F1, trading PnL)            │
│  Generate AI summary with Flan-T5                       │
└─────────────────────────────────────────────────────────┘
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

### Bypass Options

```bash
# Force training (bypass all guards)
python -m cli.finbert_sentiment.train -s BTC-USD --force

# Custom guard settings
python -m cli.finbert_sentiment.train -s BTC-USD --min-new-samples 20 --cooldown 120
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

**CRITICAL**: Parameters must match between train and test.

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
python -m cli.finbert_sentiment.train -s BTC-USD -b 0.1 --sell-threshold -0.1 --hidden-dim 512 --num-layers 4

# Testing (MUST MATCH)
python -m cli.finbert_sentiment.test -s BTC-USD -b 0.1 --sell-threshold -0.1 --hidden-dim 512 --num-layers 4
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
| Root-level scripts cluttering project | Fixed - moved to cli/ |

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
python -m cli.finbert_sentiment.train -s BTC-USD --lr-schedule cosine
```

---

### Medium Priority

#### 3. Time of Day Features

Add temporal context from timestamps:
- Hour of day
- Day of week
- Market open/close status

---

#### 4. Additional Strategies

Add new strategy folders:
- `cli/rsi/` - RSI-based technical strategy
- `cli/macd/` - MACD crossover strategy
- `cli/combined/` - Multi-signal ensemble

---

### Lower Priority

#### 5. Dropout Rate Tuning

Expose dropout as CLI parameter (currently hardcoded at 0.1).

#### 6. Multi-stage Batch Training

Vary batch size during training.

#### 7. Reinforcement Learning

Replace supervised classification with RL to optimize for profit directly.

---

## Priority Matrix

| Item | Complexity | Benefit |
|------|------------|---------|
| Embedding caching | Low-Medium | HIGH |
| Learning rate scheduling | Low-Medium | Medium |
| Dropout tuning | Low | Low-Medium |
| Time features | Medium | Medium |
| Additional strategies | Medium | HIGH |
| Reinforcement learning | HIGH | HIGH |

---

**Document Version**: 3.0 | **Last Updated**: January 2025

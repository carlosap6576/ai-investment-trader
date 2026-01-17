# Hierarchical Multi-Level Sentiment Analysis - Implementation Reference

## Overview

This document describes the **implemented** hierarchical multi-level news sentiment architecture for the trading signal classifier. The system processes news at Market, Sector, and Ticker levels to predict **Buy/Sell/Hold** signals.

**Status: FULLY IMPLEMENTED**

---

## Executive Summary

### Architecture
- Three-tier hierarchical sentiment processing (Market → Sector → Ticker)
- FinBERT integration for financial-domain sentiment analysis
- Cross-level feature engineering with attention-based fusion
- HierarchicalSentimentTransformer with temporal sequence modeling
- 40 total features across 4 feature groups

### CLI Commands
```bash
# Download data with hierarchical sentiment analysis
python -m cli.finbert_sentiment.download -s AAPL -p 1mo -i 5m

# Train the hierarchical model
python -m cli.finbert_sentiment.train -s AAPL -b 0.1 --sell-threshold -0.1 -e 300

# Evaluate model performance
python -m cli.finbert_sentiment.test -s AAPL -b 0.1 --sell-threshold -0.1 --summary
```

---

## Project Structure

```
ai-investment-trader/
├── cli/
│   └── finbert_sentiment/
│       ├── __init__.py
│       ├── download.py      # Data collection with hierarchical sentiment
│       ├── train.py         # Model training with Smart Guard
│       └── test.py          # Evaluation with AI-powered summary
│
├── src/
│   ├── data/
│   │   ├── news_classifier.py    # NewsLevelClassifier (MARKET/SECTOR/TICKER)
│   │   ├── schemas.py            # Data structures (40 features)
│   │   └── sector_mapping.py     # GICS sector codes and ticker mappings
│   │
│   ├── features/
│   │   ├── sentiment_aggregator.py  # Multi-level sentiment aggregation
│   │   └── feature_builder.py       # Temporal sequence construction
│   │
│   ├── models/
│   │   ├── transformer.py           # HierarchicalSentimentTransformer
│   │   ├── finbert_sentiment.py     # FinBERT analyzer with caching
│   │   └── report_summarizer.py     # Flan-T5 AI summary generator
│   │
│   └── evaluation/
│       ├── metrics.py               # Trading-specific metrics
│       └── report_generator.py      # Formatted evaluation reports
│
├── ai/                              # Reusable AI components
│   ├── sentiment/
│   │   └── finbert.py               # FinBERT sentiment analyzer
│   ├── embeddings/
│   │   └── gemma.py                 # Gemma text embeddings
│   └── summarizers/
│       └── flan_t5.py               # Flan-T5 report summarizer
│
├── core/                            # Shared infrastructure
│   ├── config/
│   │   └── constants.py             # Centralized defaults
│   ├── utils/
│   │   └── device.py                # GPU/CPU detection
│   └── cli/
│       └── parser.py                # FriendlyArgumentParser
│
└── datasets/
    └── {SYMBOL}/                    # Per-symbol data and models
        ├── {SYMBOL}.pth             # Trained model weights
        ├── {SYMBOL}_metadata.json   # Model architecture config
        ├── .training_meta.json      # Smart Guard metadata
        └── news_with_price.json     # Training data with sentiment
```

---

## Part 1: Data Layer

### 1.1 News Classification Module

**File:** `src/data/news_classifier.py` (384 lines)

**Implementation:**

```python
class NewsLevelClassifier:
    """
    Classifies news articles into hierarchical levels:
    - MARKET: Fed announcements, GDP, unemployment, geopolitical events
    - SECTOR: Industry-wide news, regulatory changes affecting sectors
    - TICKER: Company-specific earnings, management, products, lawsuits
    """

    def __init__(self, target_ticker: str, strict_filtering: bool = True):
        self.target_ticker = target_ticker.upper()
        self.strict_filtering = strict_filtering

    def classify(self, news_item: dict) -> Optional[str]:
        """Returns 'MARKET', 'SECTOR', 'TICKER', or None if irrelevant."""

    def classify_with_details(self, news_item: dict) -> dict:
        """Returns classification with reasoning and confidence."""
```

**Classification Keywords:**

| Level | Example Keywords (66+ market, 33+ sector) |
|-------|-------------------------------------------|
| MARKET | Fed, FOMC, interest rate, GDP, inflation, unemployment, recession, tariff, trade war, treasury, central bank, monetary policy |
| SECTOR | regulatory, supply chain, industry-wide, labor shortage, chip shortage, oil prices, banking sector |
| TICKER | Company name, ticker symbol, CEO, earnings, quarterly results, product launch |

**Relevance Filtering:**
- Filters out news not relevant to target ticker
- Configurable via `--no-filter` CLI flag
- Reports filtering statistics

---

### 1.2 News Data Schema

**File:** `src/data/schemas.py` (269 lines)

**NewsItem Structure:**

```python
@dataclass
class NewsItem:
    # Core fields
    headline: str
    summary: Optional[str]
    timestamp: datetime
    source: str
    url: Optional[str]

    # Hierarchical classification
    level: Literal["MARKET", "SECTOR", "TICKER"]

    # Entity associations
    primary_ticker: Optional[str]
    sector_gics: Optional[str]

    # FinBERT sentiment (populated during download)
    sentiment_score: float           # -1 to +1
    sentiment_confidence: float      # 0 to 1
    sentiment_label: Literal["positive", "negative", "neutral"]

    # Price data (for labeling)
    price: Optional[float]
    future_price: Optional[float]
    price_change_pct: Optional[float]
```

---

### 1.3 Sentiment Aggregation

**File:** `src/features/sentiment_aggregator.py` (429 lines)

**Aggregation Windows:**
- Trading-hour aligned: 9:30 AM to 9:30 AM next day
- Configurable lookback periods

**Implemented Aggregations:**

```python
class SentimentAggregator:
    def aggregate_ticker_sentiment(self, ticker: str, window: TimeWindow) -> TickerSentimentFeatures:
        """12 ticker-level features"""

    def aggregate_sector_sentiment(self, sector: str, window: TimeWindow) -> SectorSentimentFeatures:
        """10 sector-level features"""

    def aggregate_market_sentiment(self, window: TimeWindow) -> MarketSentimentFeatures:
        """10 market-level features"""

    def compute_cross_level_features(self, ticker, sector, market) -> CrossLevelFeatures:
        """8 cross-level features"""
```

---

## Part 2: Feature Engineering

### 2.1 Hierarchical Feature Vector

**File:** `src/features/feature_builder.py` (456 lines)

**Total: 40 Features**

#### Ticker-Level Features (12)

| Feature | Description |
|---------|-------------|
| `ticker_mean_sentiment` | Average sentiment score |
| `ticker_sentiment_std` | Sentiment volatility |
| `ticker_sentiment_skew` | Asymmetry in sentiment distribution |
| `ticker_news_volume` | News count (log-transformed) |
| `ticker_sentiment_momentum_1d` | 1-day sentiment change |
| `ticker_sentiment_momentum_5d` | 5-day sentiment change |
| `ticker_sentiment_max` | Most positive news |
| `ticker_sentiment_min` | Most negative news |
| `ticker_positive_ratio` | % of positive news |
| `ticker_negative_ratio` | % of negative news |
| `ticker_news_recency` | Time since last news (hours) |
| `ticker_source_diversity` | Entropy of news sources |

#### Sector-Level Features (10)

| Feature | Description |
|---------|-------------|
| `sector_sentiment_mean` | Sector average sentiment |
| `sector_sentiment_breadth` | % of sector tickers positive |
| `sector_news_volume` | Total sector news volume |
| `sector_sentiment_dispersion` | Cross-ticker disagreement |
| `sector_leader_sentiment` | Sentiment of sector leader |
| `sector_sentiment_momentum` | Sector sentiment trend |
| `sector_relative_strength` | Sector vs. market sentiment |
| `sector_news_concentration` | Herfindahl index of news |
| `sector_peer_sentiment` | Average of top 5 peers |
| `sector_laggard_sentiment` | Worst-performing peer sentiment |

#### Market-Level Features (10)

| Feature | Description |
|---------|-------------|
| `market_sentiment_index` | Aggregate market sentiment |
| `market_news_volume` | Total market news volume |
| `fed_sentiment` | Federal Reserve news sentiment |
| `economic_sentiment` | GDP, employment news sentiment |
| `geopolitical_sentiment` | Trade, conflict news sentiment |
| `market_fear_index` | Negative news concentration |
| `market_sentiment_momentum` | Market sentiment trend |
| `market_breadth_sentiment` | % of all tickers positive |
| `vix_level` | VIX value (if available) |
| `market_regime` | Bull/Bear/Neutral (0/1/2) |

#### Cross-Level Features (8)

| Feature | Description |
|---------|-------------|
| `ticker_vs_sector_deviation` | Ticker - Sector sentiment |
| `ticker_vs_market_deviation` | Ticker - Market sentiment |
| `sector_vs_market_deviation` | Sector - Market sentiment |
| `ticker_sector_correlation` | Rolling correlation |
| `ticker_market_beta` | Sensitivity to market sentiment |
| `sentiment_divergence_score` | Composite divergence measure |
| `relative_news_attention` | Ticker news share vs. normal |
| `sentiment_surprise` | Actual vs. expected sentiment |

---

### 2.2 Feature Normalization

**Normalization Strategies:**

| Feature Type | Method |
|--------------|--------|
| Sentiment scores (-1 to +1) | Z-score with rolling window |
| Volume features | Log-transform then Z-score |
| Ratio features (0 to 1) | Logit transform |
| Deviation features | Z-score with MAD estimator |

**Implementation:**

```python
class FeatureNormalizer:
    def __init__(self, lookback_days: int = 60):
        self.lookback = lookback_days
        self.normalizers = {}  # Separate per feature group

    def fit(self, features: HierarchicalFeatureVector):
        """Learn rolling statistics"""

    def transform(self, features: HierarchicalFeatureVector) -> np.ndarray:
        """Apply normalization"""
```

---

## Part 3: Sentiment Analysis Engine

### 3.1 FinBERT Integration

**Files:**
- `src/models/finbert_sentiment.py` (279 lines)
- `ai/sentiment/finbert.py` (250 lines)

**Model:** `ProsusAI/finbert`

**Implementation:**

```python
class FinBERTSentimentAnalyzer:
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: Optional[str] = None, use_cache: bool = True):
        self.device = device or self._detect_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self._cache = {}  # MD5-based caching

    def analyze(self, text: str) -> SentimentResult:
        """
        Returns:
            score: P(positive) - P(negative), range -1 to +1
            confidence: Max probability
            label: "positive", "negative", or "neutral"
        """

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[SentimentResult]:
        """Batch processing with smart cache usage"""
```

**Advantages over Generic Models:**
- 15-20% improvement on financial text
- Handles financial jargon (liability, depreciation, exposure)
- Trained on Financial PhraseBank + Reuters TRC2

**Caching:**
- MD5-based text hashing
- Avoids recomputation for duplicate headlines
- Cache statistics via `get_stats()`

---

## Part 4: Model Architecture

### 4.1 HierarchicalSentimentTransformer

**File:** `src/models/transformer.py` (583 lines)

**Architecture:**

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

**Implementation:**

```python
class HierarchicalSentimentTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        temporal_dim: int = 64,
        num_temporal_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        sequence_length: int = 20,
    ):
        # Level encoders (ticker: 12, sector: 10, market: 10, cross: 8)
        self.ticker_encoder = LevelEncoder(12, hidden_dim)
        self.sector_encoder = LevelEncoder(10, hidden_dim)
        self.market_encoder = LevelEncoder(10, hidden_dim)
        self.cross_encoder = LevelEncoder(8, hidden_dim)

        # Cross-level attention
        self.ticker_sector_attn = CrossLevelAttention(hidden_dim, num_heads)
        self.ticker_market_attn = CrossLevelAttention(hidden_dim, num_heads)

        # Fusion and temporal processing
        self.fusion = nn.Linear(hidden_dim * 4, temporal_dim)
        self.pos_encoding = PositionalEncoding(temporal_dim, sequence_length)
        self.temporal_transformer = nn.TransformerEncoder(...)

        # Classification
        self.classifier = nn.Linear(temporal_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, 40] feature tensor
        Returns:
            logits: [batch, 3] class logits
        """

    def get_level_importance(self) -> Dict[str, float]:
        """Returns attention weights for interpretability"""
```

**Configurable Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128/256 | Level encoder output dimension |
| `temporal_dim` | 64 | Temporal transformer dimension |
| `num_temporal_layers` | 2 | Transformer encoder layers |
| `num_heads` | 4 | Attention heads |
| `dropout` | 0.1 | Dropout probability |
| `sequence_length` | 20 | Temporal sequence length |

---

## Part 5: CLI Commands

### 5.1 Download Command

```bash
python -m cli.finbert_sentiment.download -s SYMBOL [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-s, --symbol` | Required | Trading symbol (AAPL, BTC-USD) |
| `-p, --period` | `1mo` | Lookback period (1d, 5d, 1mo, 3mo, 1y) |
| `-i, --interval` | `5m` | Candle interval (1m, 5m, 15m, 1h, 1d) |
| `-n, --news-count` | `100` | News articles to fetch |
| `--no-filter` | `False` | Disable relevance filtering |
| `--no-sentiment` | `False` | Skip FinBERT analysis |

**Output Files:**
- `historical_data.json` - Price data
- `news.json` - Raw news articles
- `news_with_price.json` - Training data with sentiment

---

### 5.2 Train Command

```bash
python -m cli.finbert_sentiment.train -s SYMBOL [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-s, --symbol` | Required | Trading symbol |
| `-b, --buy-threshold` | `1.0` | % increase for BUY label |
| `--sell-threshold` | `-1.0` | % decrease for SELL label |
| `-e, --epochs` | `20` | Training epochs |
| `-l, --learning-rate` | `0.005` | Optimizer learning rate |
| `-o, --optimizer` | `SGD` | Optimizer (SGD, AdamW) |
| `--batch-size` | `1` | Samples per gradient update |
| `--split` | `0.8` | Train/test split ratio |
| `--hidden-dim` | `256` | Model hidden dimension |
| `--num-layers` | `2` | Transformer layers |
| `--seq-length` | `20` | Temporal sequence length |
| `--fresh` | `False` | Ignore existing model |
| `--force` | `False` | Bypass Smart Training Guard |

**Smart Training Guard:**
- Checks data hash to detect changes
- Enforces cooldown between runs (default: 5 min)
- Requires minimum new samples (default: 5)
- Stores metadata in `.training_meta.json`

---

### 5.3 Test Command

```bash
python -m cli.finbert_sentiment.test -s SYMBOL [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-s, --symbol` | Required | Trading symbol |
| `-b, --buy-threshold` | `1.0` | Must match training |
| `--sell-threshold` | `-1.0` | Must match training |
| `--split` | `0.8` | Must match training |
| `--samples` | `3` | Predictions to display |
| `--hidden-dim` | `256` | Must match training |
| `--num-layers` | `2` | Must match training |
| `--summary` | `False` | Generate AI summary |
| `--summary-model` | `flan-t5-xl` | Summary model |

**AI Summary Models:**
- `google/flan-t5-small` (80M) - Fastest
- `google/flan-t5-base` (250M) - Balanced
- `google/flan-t5-large` (780M) - Good quality
- `google/flan-t5-xl` (3B) - Best quality

---

## Part 6: Evaluation Metrics

### 6.1 Classification Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| F1 Score (Macro) | Unweighted class average |
| F1 Score (Weighted) | Support-weighted average |
| Precision/Recall | Per-class metrics |
| Confusion Matrix | Prediction breakdown |

### 6.2 Trading Metrics

| Metric | Description |
|--------|-------------|
| Directional Accuracy | Correct BUY/SELL direction |
| Simulated PnL | Cumulative return % |
| Sharpe Ratio | Risk-adjusted return |
| Max Drawdown | Largest peak-to-trough |
| Win Rate | % profitable trades |

### 6.3 Calibration Metrics

| Metric | Description |
|--------|-------------|
| Brier Score | Probability calibration |
| Expected Calibration Error | Confidence vs accuracy |
| Confidence (Correct) | Avg confidence on correct |
| Confidence (Incorrect) | Avg confidence on wrong |

---

## Part 7: Sample Evaluation Report

```
================================================================================
HIERARCHICAL SENTIMENT TRADING SIGNAL CLASSIFIER - EVALUATION REPORT
================================================================================

## 1. OVERALL PERFORMANCE
----------------------------------------
  Accuracy:              0.6834 (68.34%)
  F1 Score (Macro):      0.6521
  F1 Score (Weighted):   0.6798
  Directional Accuracy:  0.7123

## 2. PER-CLASS PERFORMANCE
----------------------------------------
  Class     | Precision | Recall  | F1-Score | Support
  ----------|-----------|---------|----------|--------
  SELL      | 0.6912    | 0.6543  | 0.6722   | 45
  HOLD      | 0.6389    | 0.7012  | 0.6686   | 89
  BUY       | 0.7234    | 0.6891  | 0.7058   | 52

## 3. TRADING PERFORMANCE (Simulated)
----------------------------------------
  Simulated PnL:         +12.34%
  Sharpe Ratio:          1.456
  Max Drawdown:          -8.23%
  Win Rate:              58.34%

## 4. HIERARCHICAL LEVEL IMPORTANCE
----------------------------------------
  How much each level contributed to predictions:

  TICKER-LEVEL:     ████████████████████ 35.2%
  SECTOR-LEVEL:     ████████████         22.1%
  MARKET-LEVEL:     ██████████████       25.8%
  CROSS-LEVEL:      ██████               16.9%

## 5. TOP 10 MOST IMPORTANT FEATURES
----------------------------------------
   1. ticker_sentiment_momentum_1d         █████████████████████████ 0.0891
   2. market_sentiment_index               ████████████████████████  0.0823
   3. ticker_vs_sector_deviation           ███████████████████       0.0712
   4. sector_sentiment_breadth             ██████████████████        0.0698
   5. ticker_sentiment_mean                █████████████████         0.0654

## 6. SAMPLE PREDICTIONS
----------------------------------------
  Sample 1: CORRECT
    Ticker:      AAPL
    Date:        2024-01-15 10:30
    Headline:    Apple reports record Q4 earnings...
    Prediction:  BUY (confidence: 0.78)
    Actual:      BUY
    Probs:       SELL=0.08, HOLD=0.14, BUY=0.78

## 7. CALIBRATION ANALYSIS
----------------------------------------
  Brier Score:                    0.2134
  Expected Calibration Error:     0.0456
  Avg Confidence (Correct):       0.7234
  Avg Confidence (Incorrect):     0.5123

================================================================================
END OF REPORT
================================================================================
```

---

## Implementation Checklist

### Phase 1: Data Layer
- [x] Create `NewsLevelClassifier` class
- [x] Update data schemas with hierarchical fields
- [x] Create `SentimentAggregator` class
- [x] Implement GICS sector mapping

### Phase 2: Sentiment Engine
- [x] Integrate FinBERT model
- [x] Implement sentiment caching
- [x] Create batch processing pipeline

### Phase 3: Feature Engineering
- [x] Implement 40-feature `HierarchicalFeatureVector`
- [x] Create `TemporalFeatureBuilder`
- [x] Implement feature normalization

### Phase 4: Model Architecture
- [x] Implement `HierarchicalSentimentTransformer`
- [x] Create level-specific encoders
- [x] Implement cross-level attention
- [x] Add attention weight extraction

### Phase 5: Pipeline Integration
- [x] Create CLI download command
- [x] Create CLI train command
- [x] Create CLI test command
- [x] Implement Smart Training Guard

### Phase 6: Evaluation
- [x] Implement trading-specific metrics
- [x] Create `ReportGenerator`
- [x] Integrate Flan-T5 AI summaries

### Phase 7: Production Features
- [x] Device auto-detection (CUDA/MPS/CPU)
- [x] Comprehensive caching
- [x] Error handling with fallbacks
- [x] Module-based CLI structure

---

## Code Statistics

| Module | Lines | Status |
|--------|-------|--------|
| `src/data/schemas.py` | 269 | Complete |
| `src/data/news_classifier.py` | 384 | Complete |
| `src/data/sector_mapping.py` | 300+ | Complete |
| `src/features/sentiment_aggregator.py` | 429 | Complete |
| `src/features/feature_builder.py` | 456 | Complete |
| `src/models/finbert_sentiment.py` | 279 | Complete |
| `src/models/transformer.py` | 583 | Complete |
| `src/models/report_summarizer.py` | 496 | Complete |
| `ai/sentiment/finbert.py` | 250 | Complete |
| `ai/embeddings/gemma.py` | 153 | Complete |
| `ai/summarizers/flan_t5.py` | 429 | Complete |
| `cli/finbert_sentiment/*.py` | ~1800 | Complete |
| **Total** | **~5,800** | **Production Ready** |

---

## Quick Start

```bash
# 1. Download data for AAPL with 1 month of 5-minute data
python -m cli.finbert_sentiment.download -s AAPL -p 1mo -i 5m -n 500

# 2. Train the model (lower thresholds for intraday)
python -m cli.finbert_sentiment.train -s AAPL -b 0.1 --sell-threshold -0.1 \
    --hidden-dim 512 --num-layers 4 -e 300 -o AdamW

# 3. Evaluate with AI-powered summary
python -m cli.finbert_sentiment.test -s AAPL -b 0.1 --sell-threshold -0.1 \
    --hidden-dim 512 --num-layers 4 --samples 10 --summary
```

---

**Document Version**: 2.0
**Last Updated**: January 2025
**Status**: Implementation Complete

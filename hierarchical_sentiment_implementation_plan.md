# Implementation Plan: Hierarchical Multi-Level Sentiment Analysis for Trading Signal Classifier

## Overview

This document provides a detailed implementation plan for enhancing an algorithmic trading signal classifier that predicts **Buy/Sell/Hold** signals. The current system uses Google Gemma text embeddings with a PyTorch Transformer. This plan adds a **hierarchical multi-level news sentiment architecture** that processes news at Market, Sector, and Ticker levels.

---

## Executive Summary

### Current State
- Single-level news processing (likely global or ticker-only)
- Google Gemma embeddings for text representation
- PyTorch Transformer for signal classification

### Target State
- Three-tier hierarchical sentiment processing (Market → Sector → Ticker)
- FinBERT integration for financial-domain sentiment analysis
- Cross-level feature engineering with attention-based fusion
- Improved signal prediction through systematic + idiosyncratic risk capture

---

## Part 1: Data Layer Modifications

### 1.1 News Classification Module

**Task:** Create a news classifier that categorizes incoming news into three levels.

**File to create:** `src/data/news_classifier.py`

**Implementation Details:**

```python
class NewsLevelClassifier:
    """
    Classifies news articles into hierarchical levels:
    - MARKET: Fed announcements, GDP, unemployment, geopolitical events
    - SECTOR: Industry-wide news, regulatory changes affecting sectors
    - TICKER: Company-specific earnings, management, products, lawsuits
    """
```

**Classification Logic:**
1. **MARKET-level keywords:** Fed, FOMC, interest rate, GDP, inflation, unemployment, recession, tariff, trade war, treasury, economic, central bank, monetary policy
2. **SECTOR-level detection:** Check if news mentions multiple tickers in same GICS sector, or contains sector keywords (e.g., "tech sector", "energy stocks", "banking industry")
3. **TICKER-level:** News explicitly tagged with single ticker, or entity extraction identifies single company focus

**Why This Matters:**
- Research shows market behavior is anticipatory—macro sentiment captures ~45-50% of return variation
- Ticker-specific sentiment captures idiosyncratic alpha but misses systematic risk
- Sector sentiment captures spillover effects between related companies

**Advantages:**
- Separates systematic vs. idiosyncratic signals
- Allows model to learn different decay rates for each level
- Enables cross-level deviation features (ticker vs. sector divergence = potential signal)

---

### 1.2 News Data Schema Update

**Task:** Modify the news data schema to support hierarchical classification.

**File to modify:** `src/data/schemas.py` or equivalent data models

**New Schema Fields:**

```python
@dataclass
class NewsItem:
    # Existing fields
    headline: str
    body: Optional[str]
    timestamp: datetime
    source: str
    
    # NEW: Hierarchical classification
    level: Literal["MARKET", "SECTOR", "TICKER"]
    
    # NEW: Entity associations
    primary_ticker: Optional[str]          # For TICKER-level news
    sector_gics: Optional[str]             # GICS sector code
    related_tickers: List[str]             # For spillover analysis
    
    # NEW: Sentiment scores (populated by FinBERT)
    sentiment_score: float                 # -1 to +1
    sentiment_confidence: float            # 0 to 1
    sentiment_label: Literal["positive", "negative", "neutral"]
```

**Why This Matters:**
- Structured schema enables efficient aggregation at each level
- GICS sector codes allow standardized sector grouping
- Related tickers field enables spillover effect modeling

---

### 1.3 News Aggregation Pipeline

**Task:** Create aggregation functions that compute level-specific sentiment features.

**File to create:** `src/features/sentiment_aggregator.py`

**Aggregation Windows:**
- Use **trading-hour aligned windows** (9:30 AM to 9:30 AM next day) instead of calendar days
- Research shows this significantly improves prediction accuracy

**Aggregation Functions per Level:**

```python
class SentimentAggregator:
    def aggregate_ticker_sentiment(self, ticker: str, window: TimeWindow) -> TickerSentimentFeatures:
        """
        Returns:
        - mean_sentiment: Average sentiment score
        - sentiment_std: Volatility of sentiment (disagreement indicator)
        - news_volume: Count of news items (attention proxy)
        - sentiment_momentum: Change from previous window
        - max_sentiment: Most extreme positive
        - min_sentiment: Most extreme negative
        """
    
    def aggregate_sector_sentiment(self, sector_gics: str, window: TimeWindow) -> SectorSentimentFeatures:
        """
        Returns:
        - sector_mean_sentiment: Market-cap weighted average across sector tickers
        - sector_breadth: % of tickers with positive sentiment
        - sector_news_concentration: Herfindahl index of news distribution
        - sector_sentiment_dispersion: Cross-ticker sentiment disagreement
        """
    
    def aggregate_market_sentiment(self, window: TimeWindow) -> MarketSentimentFeatures:
        """
        Returns:
        - market_sentiment_index: Aggregate market-level news sentiment
        - fed_sentiment: Specific Fed/monetary policy sentiment
        - economic_sentiment: GDP, employment, inflation news sentiment
        - geopolitical_sentiment: Trade, conflict, policy news sentiment
        - vix_proxy: Can integrate actual VIX or compute implied from sentiment
        """
```

**Why This Matters:**
- Different aggregation logic for each level captures appropriate signal
- Sector breadth (% positive) is more robust than simple average
- News concentration identifies when single story dominates vs. broad coverage

---

## Part 2: Sentiment Analysis Engine

### 2.1 FinBERT Integration

**Task:** Replace or supplement current embedding approach with FinBERT for sentiment extraction.

**File to create:** `src/models/finbert_sentiment.py`

**Why FinBERT over Generic Models:**
- Generic models (VADER, TextBlob) misinterpret financial language
- Words like "liability", "depreciation", "exposure" are neutral/positive in finance but negative in general English
- FinBERT trained on Financial PhraseBank + Reuters TRC2 financial corpus

**Implementation:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBERTSentimentAnalyzer:
    def __init__(self, device: str = "cuda"):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
    def analyze(self, texts: List[str], batch_size: int = 32) -> List[SentimentResult]:
        """
        Returns sentiment score: P(positive) - P(negativecha)
        Range: -1 (very negative) to +1 (very positive)
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # FinBERT outputs: [positive, negative, neutral]
                scores = probs[:, 0] - probs[:, 1]  # P(pos) - P(neg)
                
            for j, score in enumerate(scores):
                results.append(SentimentResult(
                    score=score.item(),
                    confidence=probs[j].max().item(),
                    label=["positive", "negative", "neutral"][probs[j].argmax()]
                ))
        return results
```

**Advantages:**
- 15-20% improvement in sentiment classification accuracy on financial text
- Properly handles financial jargon and context
- Pre-trained, no additional training required

---

### 2.2 Hybrid Embedding Strategy

**Task:** Combine FinBERT sentiment with Gemma embeddings for rich representation.

**File to modify:** `src/models/embeddings.py` or equivalent

**Strategy:**

```python
class HybridNewsEncoder:
    """
    Two-stream encoding:
    1. FinBERT: Extract sentiment-specific features (3-dim sentiment + score)
    2. Gemma: Extract semantic content embeddings (for context beyond sentiment)
    
    Concatenate for final news representation.
    """
    
    def __init__(self, gemma_model, finbert_analyzer):
        self.gemma = gemma_model
        self.finbert = finbert_analyzer
        
    def encode(self, news_items: List[NewsItem]) -> torch.Tensor:
        # Stream 1: Semantic embeddings from Gemma
        texts = [item.headline + " " + (item.body or "") for item in news_items]
        gemma_embeddings = self.gemma.encode(texts)  # [batch, gemma_dim]
        
        # Stream 2: Sentiment features from FinBERT
        sentiment_results = self.finbert.analyze(texts)
        sentiment_features = torch.tensor([
            [r.score, r.confidence, 
             1 if r.label == "positive" else (-1 if r.label == "negative" else 0)]
            for r in sentiment_results
        ])  # [batch, 3]
        
        # Concatenate streams
        return torch.cat([gemma_embeddings, sentiment_features], dim=-1)
```

**Why Hybrid Approach:**
- Gemma captures semantic meaning (what the news is about)
- FinBERT captures sentiment polarity (how positive/negative)
- Combined representation is more informative than either alone

---

## Part 3: Feature Engineering

### 3.1 Hierarchical Feature Vector

**Task:** Create the complete feature vector combining all sentiment levels with price data.

**File to create:** `src/features/feature_builder.py`

**Feature Vector Structure:**

```python
@dataclass
class HierarchicalFeatureVector:
    """
    Complete feature vector for one ticker at one timestamp.
    Total dimensions: ~45-60 features depending on configuration.
    """
    
    # === TICKER-LEVEL FEATURES (12 features) ===
    ticker_sentiment_mean: float          # Average sentiment score
    ticker_sentiment_std: float           # Sentiment volatility
    ticker_sentiment_skew: float          # Asymmetry in sentiment distribution
    ticker_news_volume: float             # News count (log-transformed)
    ticker_sentiment_momentum_1d: float   # 1-day sentiment change
    ticker_sentiment_momentum_5d: float   # 5-day sentiment change
    ticker_sentiment_max: float           # Most positive news
    ticker_sentiment_min: float           # Most negative news
    ticker_positive_ratio: float          # % of positive news
    ticker_negative_ratio: float          # % of negative news
    ticker_news_recency: float            # Time since last news (hours)
    ticker_source_diversity: float        # Entropy of news sources
    
    # === SECTOR-LEVEL FEATURES (10 features) ===
    sector_sentiment_mean: float          # Sector average sentiment
    sector_sentiment_breadth: float       # % of sector tickers positive
    sector_news_volume: float             # Total sector news volume
    sector_sentiment_dispersion: float    # Cross-ticker disagreement
    sector_leader_sentiment: float        # Sentiment of sector leader (by market cap)
    sector_sentiment_momentum: float      # Sector sentiment trend
    sector_relative_strength: float       # Sector vs. market sentiment
    sector_news_concentration: float      # Is news focused or distributed?
    sector_peer_sentiment: float          # Average of top 5 peers by correlation
    sector_laggard_sentiment: float       # Sentiment of worst-performing peer
    
    # === MARKET-LEVEL FEATURES (10 features) ===
    market_sentiment_index: float         # Overall market sentiment
    market_news_volume: float             # Total market news volume
    fed_sentiment: float                  # Federal Reserve news sentiment
    economic_sentiment: float             # Economic indicator news sentiment
    geopolitical_sentiment: float         # Geopolitical news sentiment
    market_fear_index: float              # Negative news concentration
    market_sentiment_momentum: float      # Market sentiment trend
    market_breadth_sentiment: float       # % of all tickers positive
    vix_level: float                      # Actual VIX (if available)
    market_regime: int                    # Bull/Bear/Neutral classification
    
    # === CROSS-LEVEL FEATURES (8 features) ===
    ticker_vs_sector_deviation: float     # Ticker sentiment - Sector sentiment
    ticker_vs_market_deviation: float     # Ticker sentiment - Market sentiment
    sector_vs_market_deviation: float     # Sector sentiment - Market sentiment
    ticker_sector_correlation: float      # Rolling correlation of sentiments
    ticker_market_beta: float             # Sensitivity to market sentiment
    sentiment_divergence_score: float     # Composite divergence measure
    relative_news_attention: float        # Ticker news share vs. normal
    sentiment_surprise: float             # Actual vs. expected sentiment
    
    # === PRICE/TECHNICAL FEATURES (existing, for reference) ===
    # ... keep existing price-based features ...
```

**Why Cross-Level Features Matter:**
- Ticker-vs-sector deviation captures company-specific news vs. industry trends
- High divergence often precedes price moves (market hasn't priced in yet)
- Sentiment surprise (actual vs. expected) is a key alpha signal

---

### 3.2 Feature Normalization Strategy

**Task:** Implement appropriate normalization for sentiment features.

**File to modify:** `src/features/normalizers.py`

**Normalization Approach:**

```python
class SentimentFeatureNormalizer:
    """
    Different normalization strategies for different feature types:
    
    1. Sentiment scores (-1 to +1): Already bounded, use as-is or z-score
    2. Volume features: Log-transform then z-score (heavy tailed)
    3. Ratio features (0 to 1): Use as-is or logit transform
    4. Deviation features: Z-score with robust estimators (MAD)
    """
    
    def __init__(self, lookback_days: int = 60):
        self.lookback = lookback_days
        self.stats = {}  # Rolling statistics
        
    def normalize_sentiment_score(self, value: float, feature_name: str) -> float:
        # Z-score normalization with rolling window
        mean, std = self.stats.get(feature_name, (0, 1))
        return (value - mean) / (std + 1e-8)
    
    def normalize_volume(self, value: float, feature_name: str) -> float:
        # Log transform then z-score
        log_value = np.log1p(value)
        mean, std = self.stats.get(f"{feature_name}_log", (0, 1))
        return (log_value - mean) / (std + 1e-8)
    
    def normalize_ratio(self, value: float) -> float:
        # Logit transform for ratios (avoids boundary issues)
        clipped = np.clip(value, 0.01, 0.99)
        return np.log(clipped / (1 - clipped))
```

**Why This Matters:**
- Prevents volume features from dominating (they can be 100x larger)
- Rolling normalization adapts to changing market regimes
- Robust estimators handle outlier news events (earnings, M&A)

---

## Part 4: Model Architecture Modifications

### 4.1 Hierarchical Attention Transformer

**Task:** Modify the Transformer to process hierarchical features with level-aware attention.

**File to modify:** `src/models/transformer.py`

**Architecture Overview:**

```
Input Features
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Hierarchical Feature Encoder                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Ticker Enc. │  │ Sector Enc. │  │ Market Enc. │     │
│  │   (MLP)     │  │   (MLP)     │  │   (MLP)     │     │
│  └─────┬───────┘  └─────┬───────┘  └─────┬───────┘     │
│        │                │                │              │
│        ▼                ▼                ▼              │
│  ┌─────────────────────────────────────────────┐       │
│  │     Cross-Level Attention Module            │       │
│  │  (Learns which level matters for each case) │       │
│  └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Temporal Transformer Encoder                           │
│  (Process sequence of hierarchical features over time)  │
│  - Positional encoding for time steps                   │
│  - Multi-head self-attention                            │
│  - Feed-forward layers                                  │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Classification Head                                    │
│  - MLP layers                                           │
│  - Output: [P(Buy), P(Sell), P(Hold)]                  │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class HierarchicalSentimentTransformer(nn.Module):
    def __init__(
        self,
        ticker_feature_dim: int = 12,
        sector_feature_dim: int = 10,
        market_feature_dim: int = 10,
        cross_level_dim: int = 8,
        price_feature_dim: int = 20,  # Existing price features
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        seq_length: int = 20,  # Number of time steps
    ):
        super().__init__()
        
        # Level-specific encoders (project each level to same dimension)
        self.ticker_encoder = nn.Sequential(
            nn.Linear(ticker_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.sector_encoder = nn.Sequential(
            nn.Linear(sector_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.market_encoder = nn.Sequential(
            nn.Linear(market_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.cross_level_encoder = nn.Sequential(
            nn.Linear(cross_level_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.price_encoder = nn.Sequential(
            nn.Linear(price_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross-level attention (learns importance of each level)
        self.level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Positional encoding for temporal sequence
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_length)
        
        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # Buy, Sell, Hold
        )
        
    def forward(
        self,
        ticker_features: torch.Tensor,    # [batch, seq, ticker_dim]
        sector_features: torch.Tensor,    # [batch, seq, sector_dim]
        market_features: torch.Tensor,    # [batch, seq, market_dim]
        cross_level_features: torch.Tensor,  # [batch, seq, cross_dim]
        price_features: torch.Tensor,     # [batch, seq, price_dim]
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        batch_size, seq_len = ticker_features.shape[:2]
        
        # Encode each level
        ticker_enc = self.ticker_encoder(ticker_features)    # [batch, seq, hidden]
        sector_enc = self.sector_encoder(sector_features)    # [batch, seq, hidden]
        market_enc = self.market_encoder(market_features)    # [batch, seq, hidden]
        cross_enc = self.cross_level_encoder(cross_level_features)
        price_enc = self.price_encoder(price_features)
        
        # Stack levels for attention: [batch, seq, 5, hidden]
        level_stack = torch.stack([
            ticker_enc, sector_enc, market_enc, cross_enc, price_enc
        ], dim=2)
        
        # Reshape for attention: [batch * seq, 5, hidden]
        level_stack_flat = level_stack.view(-1, 5, level_stack.shape[-1])
        
        # Cross-level attention (learn which level matters)
        attn_output, attn_weights = self.level_attention(
            level_stack_flat, level_stack_flat, level_stack_flat
        )
        
        # Reshape back: [batch, seq, 5, hidden]
        attn_output = attn_output.view(batch_size, seq_len, 5, -1)
        
        # Flatten levels and fuse: [batch, seq, hidden*5] -> [batch, seq, hidden]
        fused = self.fusion(attn_output.view(batch_size, seq_len, -1))
        
        # Add positional encoding
        fused = self.pos_encoding(fused)
        
        # Temporal transformer
        temporal_out = self.temporal_transformer(fused, src_key_padding_mask=mask)
        
        # Use last time step for classification (or pool)
        final_repr = temporal_out[:, -1, :]  # [batch, hidden]
        
        # Classification
        logits = self.classifier(final_repr)  # [batch, 3]
        
        return logits
    
    def get_attention_weights(self) -> torch.Tensor:
        """Returns level attention weights for interpretability."""
        # Useful for understanding which level drove the prediction
        pass
```

**Advantages of This Architecture:**
1. **Level-specific encoders**: Each level has dedicated parameters to learn level-appropriate representations
2. **Cross-level attention**: Model learns which level is most important for each prediction (varies by market regime)
3. **Temporal transformer**: Captures patterns over time (sentiment momentum, regime changes)
4. **Interpretability**: Attention weights show which level drove each prediction

---

### 4.2 Loss Function Modification

**Task:** Update loss function to handle class imbalance and confidence calibration.

**File to modify:** `src/training/losses.py`

```python
class TradingSignalLoss(nn.Module):
    """
    Combined loss for trading signal prediction:
    1. Focal Loss: Handles class imbalance (Hold is usually most common)
    2. Label Smoothing: Improves calibration
    3. Confidence Penalty: Penalizes overconfident wrong predictions
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        confidence_penalty: float = 0.1
    ):
        super().__init__()
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.confidence_penalty = confidence_penalty
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Apply label smoothing
        num_classes = logits.shape[-1]
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Focal weight
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Cross entropy with smoothed targets
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        # Apply focal weight and class weights
        if self.class_weights is not None:
            class_weight = self.class_weights[targets]
            focal_loss = (focal_weight * class_weight * ce_loss).mean()
        else:
            focal_loss = (focal_weight * ce_loss).mean()
        
        # Confidence penalty for wrong predictions
        pred_classes = logits.argmax(dim=-1)
        wrong_mask = (pred_classes != targets).float()
        max_probs = probs.max(dim=-1).values
        conf_penalty = (wrong_mask * max_probs).mean()
        
        return focal_loss + self.confidence_penalty * conf_penalty
```

---

## Part 5: Data Pipeline Integration

### 5.1 Real-Time News Processing Pipeline

**Task:** Create end-to-end pipeline for processing news into features.

**File to create:** `src/pipeline/news_pipeline.py`

```python
class NewsProcessingPipeline:
    """
    End-to-end pipeline:
    1. Ingest raw news
    2. Classify into levels
    3. Extract sentiment with FinBERT
    4. Aggregate into features
    5. Normalize and output
    """
    
    def __init__(
        self,
        finbert_model: FinBERTSentimentAnalyzer,
        news_classifier: NewsLevelClassifier,
        aggregator: SentimentAggregator,
        normalizer: SentimentFeatureNormalizer,
        ticker_universe: List[str],
        sector_mapping: Dict[str, str]  # ticker -> GICS sector
    ):
        self.finbert = finbert_model
        self.classifier = news_classifier
        self.aggregator = aggregator
        self.normalizer = normalizer
        self.ticker_universe = ticker_universe
        self.sector_mapping = sector_mapping
        
    def process(
        self,
        raw_news: List[Dict],
        target_timestamp: datetime
    ) -> Dict[str, HierarchicalFeatureVector]:
        """
        Process news and return features for each ticker.
        
        Args:
            raw_news: List of news items with headline, body, timestamp, ticker_tags
            target_timestamp: Timestamp to compute features for
            
        Returns:
            Dictionary mapping ticker -> HierarchicalFeatureVector
        """
        # Step 1: Parse and classify news
        classified_news = []
        for item in raw_news:
            level = self.classifier.classify(item)
            classified_news.append(NewsItem(
                headline=item['headline'],
                body=item.get('body'),
                timestamp=item['timestamp'],
                source=item['source'],
                level=level,
                primary_ticker=item.get('ticker'),
                sector_gics=self.sector_mapping.get(item.get('ticker')),
                related_tickers=item.get('related_tickers', [])
            ))
        
        # Step 2: Extract sentiment with FinBERT
        headlines = [n.headline for n in classified_news]
        sentiments = self.finbert.analyze(headlines)
        for news, sent in zip(classified_news, sentiments):
            news.sentiment_score = sent.score
            news.sentiment_confidence = sent.confidence
            news.sentiment_label = sent.label
        
        # Step 3: Define aggregation window (trading-hour aligned)
        window = self._get_trading_window(target_timestamp)
        
        # Step 4: Aggregate features for each ticker
        features = {}
        for ticker in self.ticker_universe:
            sector = self.sector_mapping[ticker]
            
            # Aggregate each level
            ticker_features = self.aggregator.aggregate_ticker_sentiment(
                ticker, window, classified_news
            )
            sector_features = self.aggregator.aggregate_sector_sentiment(
                sector, window, classified_news
            )
            market_features = self.aggregator.aggregate_market_sentiment(
                window, classified_news
            )
            
            # Compute cross-level features
            cross_features = self._compute_cross_level_features(
                ticker_features, sector_features, market_features
            )
            
            # Normalize
            normalized = self.normalizer.normalize_all(
                ticker_features, sector_features, market_features, cross_features
            )
            
            features[ticker] = normalized
        
        return features
    
    def _get_trading_window(self, timestamp: datetime) -> TimeWindow:
        """Returns 9:30 AM to 9:30 AM window containing timestamp."""
        # Implementation details...
        pass
    
    def _compute_cross_level_features(self, ticker, sector, market) -> CrossLevelFeatures:
        """Compute deviation and correlation features."""
        return CrossLevelFeatures(
            ticker_vs_sector_deviation=ticker.mean - sector.mean,
            ticker_vs_market_deviation=ticker.mean - market.mean,
            sector_vs_market_deviation=sector.mean - market.mean,
            # ... other cross-level features
        )
```

---

## Part 6: Evaluation and Monitoring

### 6.1 Evaluation Metrics

**Task:** Implement comprehensive evaluation metrics for trading signals.

**File to create:** `src/evaluation/metrics.py`

```python
class TradingSignalMetrics:
    """
    Evaluation metrics specific to trading signal prediction.
    """
    
    @staticmethod
    def compute_all(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prices: torch.Tensor  # Actual price changes for PnL calculation
    ) -> Dict[str, float]:
        """
        Returns comprehensive metrics dictionary.
        """
        pred_classes = predictions.argmax(dim=-1)
        pred_probs = F.softmax(predictions, dim=-1)
        
        metrics = {
            # Classification metrics
            'accuracy': accuracy_score(targets, pred_classes),
            'f1_macro': f1_score(targets, pred_classes, average='macro'),
            'f1_weighted': f1_score(targets, pred_classes, average='weighted'),
            
            # Per-class metrics
            'precision_buy': precision_score(targets, pred_classes, labels=[0], average='macro'),
            'precision_sell': precision_score(targets, pred_classes, labels=[1], average='macro'),
            'recall_buy': recall_score(targets, pred_classes, labels=[0], average='macro'),
            'recall_sell': recall_score(targets, pred_classes, labels=[1], average='macro'),
            
            # Calibration metrics
            'brier_score': brier_score_loss(targets, pred_probs, multi_class='ovr'),
            'expected_calibration_error': compute_ece(pred_probs, targets),
            
            # Trading-specific metrics
            'directional_accuracy': compute_directional_accuracy(pred_classes, prices),
            'simulated_pnl': compute_simulated_pnl(pred_classes, prices),
            'sharpe_ratio': compute_sharpe_ratio(pred_classes, prices),
            'max_drawdown': compute_max_drawdown(pred_classes, prices),
            
            # Signal quality metrics
            'signal_confidence_when_correct': pred_probs.max(dim=-1).values[pred_classes == targets].mean(),
            'signal_confidence_when_wrong': pred_probs.max(dim=-1).values[pred_classes != targets].mean(),
        }
        
        return metrics
```

### 6.2 Output Formatting

**Task:** Create formatted output for model results and analysis.

**File to create:** `src/evaluation/report_generator.py`

```python
class ReportGenerator:
    """
    Generates human-readable reports for model evaluation.
    """
    
    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
        attention_analysis: Dict,
        feature_importance: Dict,
        sample_predictions: List[Dict]
    ) -> str:
        """
        Generates formatted markdown report.
        """
        report = []
        report.append("=" * 80)
        report.append("HIERARCHICAL SENTIMENT TRADING SIGNAL CLASSIFIER - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Section 1: Overall Performance
        report.append("## 1. OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"  Accuracy:              {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        report.append(f"  F1 Score (Macro):      {metrics['f1_macro']:.4f}")
        report.append(f"  F1 Score (Weighted):   {metrics['f1_weighted']:.4f}")
        report.append(f"  Directional Accuracy:  {metrics['directional_accuracy']:.4f}")
        report.append("")
        
        # Section 2: Per-Class Performance
        report.append("## 2. PER-CLASS PERFORMANCE")
        report.append("-" * 40)
        report.append("  Class     | Precision | Recall  | F1-Score | Support")
        report.append("  ----------|-----------|---------|----------|--------")
        report.append(f"  BUY       | {metrics['precision_buy']:.4f}    | {metrics['recall_buy']:.4f}  | {metrics.get('f1_buy', 0):.4f}   | {metrics.get('support_buy', 'N/A')}")
        report.append(f"  SELL      | {metrics['precision_sell']:.4f}    | {metrics['recall_sell']:.4f}  | {metrics.get('f1_sell', 0):.4f}   | {metrics.get('support_sell', 'N/A')}")
        report.append(f"  HOLD      | {metrics.get('precision_hold', 0):.4f}    | {metrics.get('recall_hold', 0):.4f}  | {metrics.get('f1_hold', 0):.4f}   | {metrics.get('support_hold', 'N/A')}")
        report.append("")
        
        # Section 3: Trading Metrics
        report.append("## 3. TRADING PERFORMANCE (Simulated)")
        report.append("-" * 40)
        report.append(f"  Simulated PnL:         {metrics['simulated_pnl']:+.2f}%")
        report.append(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:.3f}")
        report.append(f"  Max Drawdown:          {metrics['max_drawdown']:.2f}%")
        report.append(f"  Win Rate:              {metrics.get('win_rate', 0):.2f}%")
        report.append("")
        
        # Section 4: Hierarchical Level Analysis
        report.append("## 4. HIERARCHICAL LEVEL IMPORTANCE")
        report.append("-" * 40)
        report.append("  How much each level contributed to predictions:")
        report.append("")
        report.append(f"  TICKER-LEVEL:     {'█' * int(attention_analysis['ticker_weight'] * 20):<20} {attention_analysis['ticker_weight']*100:.1f}%")
        report.append(f"  SECTOR-LEVEL:     {'█' * int(attention_analysis['sector_weight'] * 20):<20} {attention_analysis['sector_weight']*100:.1f}%")
        report.append(f"  MARKET-LEVEL:     {'█' * int(attention_analysis['market_weight'] * 20):<20} {attention_analysis['market_weight']*100:.1f}%")
        report.append(f"  CROSS-LEVEL:      {'█' * int(attention_analysis['cross_weight'] * 20):<20} {attention_analysis['cross_weight']*100:.1f}%")
        report.append(f"  PRICE FEATURES:   {'█' * int(attention_analysis['price_weight'] * 20):<20} {attention_analysis['price_weight']*100:.1f}%")
        report.append("")
        
        # Section 5: Top Features
        report.append("## 5. TOP 10 MOST IMPORTANT FEATURES")
        report.append("-" * 40)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            bar = '█' * int(importance * 50)
            report.append(f"  {i:2d}. {feature:<35} {bar:<25} {importance:.4f}")
        report.append("")
        
        # Section 6: Sample Predictions
        report.append("## 6. SAMPLE PREDICTIONS")
        report.append("-" * 40)
        for i, sample in enumerate(sample_predictions[:5], 1):
            report.append(f"  Sample {i}:")
            report.append(f"    Ticker:      {sample['ticker']}")
            report.append(f"    Date:        {sample['date']}")
            report.append(f"    Prediction:  {sample['prediction']} (confidence: {sample['confidence']:.2f})")
            report.append(f"    Actual:      {sample['actual']}")
            report.append(f"    Key Drivers: {', '.join(sample['top_features'][:3])}")
            report.append("")
        
        # Section 7: Calibration Analysis
        report.append("## 7. CALIBRATION ANALYSIS")
        report.append("-" * 40)
        report.append(f"  Brier Score:                    {metrics['brier_score']:.4f}")
        report.append(f"  Expected Calibration Error:     {metrics['expected_calibration_error']:.4f}")
        report.append(f"  Avg Confidence (Correct):       {metrics['signal_confidence_when_correct']:.4f}")
        report.append(f"  Avg Confidence (Incorrect):     {metrics['signal_confidence_when_wrong']:.4f}")
        report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
```

---

## Part 7: Testing Requirements

### 7.1 Unit Tests

**Task:** Create unit tests for all new components.

**Files to create:** `tests/test_*.py`

**Test Coverage Requirements:**

1. **NewsLevelClassifier Tests:**
   - Test market-level keyword detection
   - Test sector-level classification
   - Test ticker-level entity extraction
   - Test edge cases (ambiguous news)

2. **FinBERTSentimentAnalyzer Tests:**
   - Test sentiment score ranges (-1 to +1)
   - Test batch processing
   - Test financial-specific language (verify "liability" isn't negative)

3. **SentimentAggregator Tests:**
   - Test aggregation math (mean, std, etc.)
   - Test time window alignment
   - Test handling of missing data

4. **HierarchicalSentimentTransformer Tests:**
   - Test forward pass shapes
   - Test attention weight extraction
   - Test gradient flow

5. **Integration Tests:**
   - End-to-end pipeline test with sample data
   - Verify feature vector dimensions match model expectations

---

## Part 8: Implementation Checklist

### Phase 1: Data Layer (Week 1)
- [ ] Create `NewsLevelClassifier` class
- [ ] Update data schemas with new fields
- [ ] Create `SentimentAggregator` class
- [ ] Write unit tests for data layer

### Phase 2: Sentiment Engine (Week 1-2)
- [ ] Integrate FinBERT model
- [ ] Create `HybridNewsEncoder` class
- [ ] Benchmark FinBERT vs. current approach
- [ ] Write unit tests for sentiment engine

### Phase 3: Feature Engineering (Week 2)
- [ ] Implement `HierarchicalFeatureVector`
- [ ] Create feature normalization pipeline
- [ ] Implement cross-level feature computation
- [ ] Write unit tests for features

### Phase 4: Model Architecture (Week 2-3)
- [ ] Implement `HierarchicalSentimentTransformer`
- [ ] Create level-specific encoders
- [ ] Implement cross-level attention
- [ ] Update loss function
- [ ] Write unit tests for model

### Phase 5: Pipeline Integration (Week 3)
- [ ] Create `NewsProcessingPipeline`
- [ ] Integrate with existing data loaders
- [ ] Create training loop modifications
- [ ] Run integration tests

### Phase 6: Evaluation (Week 3-4)
- [ ] Implement trading-specific metrics
- [ ] Create `ReportGenerator`
- [ ] Run baseline vs. new model comparison
- [ ] Generate evaluation reports

### Phase 7: Documentation & Cleanup (Week 4)
- [ ] Document all new classes and functions
- [ ] Create usage examples
- [ ] Performance optimization
- [ ] Final code review

---

## Appendix A: Expected Results Format

After running evaluation, the system should output a report similar to:

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
  BUY       | 0.7234    | 0.6891  | 0.7058   | 1,234
  SELL      | 0.6912    | 0.6543  | 0.6722   | 1,156
  HOLD      | 0.6389    | 0.7012  | 0.6686   | 2,345

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
  CROSS-LEVEL:      ██████               10.4%
  PRICE FEATURES:   ██████               6.5%

## 5. TOP 10 MOST IMPORTANT FEATURES
----------------------------------------
   1. ticker_sentiment_momentum_1d         █████████████████████████ 0.0891
   2. market_sentiment_index               ████████████████████████  0.0823
   3. ticker_vs_sector_deviation           ███████████████████       0.0712
   4. sector_sentiment_breadth             ██████████████████        0.0698
   5. ticker_sentiment_mean                █████████████████         0.0654
   6. fed_sentiment                        ████████████████          0.0621
   7. ticker_news_volume                   ███████████████           0.0589
   8. sector_relative_strength             ██████████████            0.0534
   9. market_fear_index                    █████████████             0.0512
  10. ticker_vs_market_deviation           ████████████              0.0478

## 6. SAMPLE PREDICTIONS
----------------------------------------
  Sample 1:
    Ticker:      AAPL
    Date:        2024-01-15
    Prediction:  BUY (confidence: 0.78)
    Actual:      BUY
    Key Drivers: ticker_sentiment_momentum_1d, sector_sentiment_breadth, fed_sentiment

  Sample 2:
    Ticker:      TSLA
    Date:        2024-01-15
    Prediction:  SELL (confidence: 0.65)
    Actual:      SELL
    Key Drivers: ticker_vs_sector_deviation, market_fear_index, ticker_sentiment_mean

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

## Appendix B: Configuration Template

Create a configuration file for easy parameter tuning:

**File:** `config/hierarchical_sentiment_config.yaml`

```yaml
# Hierarchical Sentiment Model Configuration

data:
  news_sources:
    - reuters
    - bloomberg
    - sec_filings
  aggregation_window: "trading_hours"  # or "calendar_day"
  lookback_days: 5
  
sentiment:
  model: "ProsusAI/finbert"
  batch_size: 32
  max_length: 512
  device: "cuda"
  
features:
  ticker_features:
    - sentiment_mean
    - sentiment_std
    - sentiment_momentum_1d
    - sentiment_momentum_5d
    - news_volume
    - positive_ratio
    - negative_ratio
  sector_features:
    - sentiment_mean
    - sentiment_breadth
    - news_volume
    - sentiment_dispersion
  market_features:
    - sentiment_index
    - fed_sentiment
    - economic_sentiment
    - fear_index
  normalization:
    method: "rolling_zscore"
    window: 60
    
model:
  hidden_dim: 128
  num_heads: 8
  num_layers: 4
  dropout: 0.1
  seq_length: 20
  
training:
  batch_size: 64
  learning_rate: 0.0001
  weight_decay: 0.01
  epochs: 100
  early_stopping_patience: 10
  class_weights: [1.2, 1.2, 0.8]  # Buy, Sell, Hold
  
evaluation:
  metrics:
    - accuracy
    - f1_macro
    - directional_accuracy
    - simulated_pnl
    - sharpe_ratio
```

---

## Summary

This implementation plan provides a comprehensive roadmap for upgrading your trading signal classifier with hierarchical sentiment analysis. The key innovations are:

1. **Three-level news hierarchy** (Market → Sector → Ticker) captures both systematic and idiosyncratic signals
2. **FinBERT integration** provides accurate financial sentiment analysis
3. **Cross-level features** capture divergence signals that precede price moves
4. **Attention-based fusion** lets the model learn which level matters in different market regimes
5. **Trading-specific evaluation** measures real-world performance, not just classification accuracy

Expected improvements:
- 10-20% improvement in directional accuracy
- Better performance during high-volatility periods (when market-level signals matter most)
- More interpretable predictions (attention weights show reasoning)
- Improved calibration (confidence scores are more reliable)

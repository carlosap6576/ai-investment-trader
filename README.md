# Tuning and Optimization (TODO)

---

## Pending - Detailed Plans

### 1. Capture median price % deltas as market indicator

**Objective**: Add market volatility context to help the model distinguish between volatile and calm market conditions.

**Current State**: We only use the price at news time and future price. The model has no context about overall market conditions.

**Data Availability (yfinance)**: ✅ Available - we already fetch price intervals, can calculate deltas from existing data.

**Implementation Plan**:
- [ ] **Step 1.1**: In `download.py`, calculate rolling statistics from historical prices
  - Median % change over past N intervals (e.g., last 12 intervals = 1 hour of 5-min data)
  - Standard deviation of % changes (volatility measure)
  - Direction bias (% of intervals that were positive)
- [ ] **Step 1.2**: Add new fields to `news_with_price.json`:
  ```json
  {
    "title": "...",
    "price": 95000,
    "future_price": 96000,
    "percentage": 1.05,
    "market_volatility": 0.45,
    "market_median_delta": 0.12,
    "market_direction_bias": 0.58
  }
  ```
- [ ] **Step 1.3**: Update `train.py` and `test.py` to include market context in input text:
  ```
  Price: 95000
  Market Volatility: HIGH (0.45%)
  Market Trend: SLIGHTLY BULLISH (58% positive)
  Headline: Bitcoin surges...
  ```
- [ ] **Step 1.4**: Alternatively, add as separate numerical features (requires model architecture change)

**Expected Benefit**: Model can learn that same news has different impact during volatile vs calm periods.

**Complexity**: Medium

---

### 2. Reinforcement learning

**Objective**: Replace supervised classification with RL to directly optimize for trading profit rather than prediction accuracy.

**Current State**: Supervised learning predicts BUY/SELL/HOLD labels based on historical price movement. This may not align perfectly with actual trading profitability.

**Why RL might be better**:
- Classification treats all correct predictions equally
- RL can weight decisions by profit magnitude
- RL can learn when NOT to trade (transaction costs, slippage)
- RL can optimize for risk-adjusted returns (Sharpe ratio)

**Implementation Plan**:
- [ ] **Step 2.1**: Design trading environment
  - State: Current news embedding + price + position + portfolio value
  - Actions: BUY, SELL, HOLD (or continuous position sizing)
  - Reward: Profit/loss from action, minus transaction costs
- [ ] **Step 2.2**: Choose RL algorithm
  - **PPO** (Proximal Policy Optimization): Stable, good for continuous action spaces
  - **DQN** (Deep Q-Network): Simpler, good for discrete actions
  - **A2C/A3C**: Actor-Critic methods for faster training
- [ ] **Step 2.3**: Create backtesting simulator
  - Replay historical data chronologically
  - Track portfolio value, positions, transaction costs
  - Calculate rewards based on actual profit/loss
- [ ] **Step 2.4**: Modify model architecture
  - Replace classifier head with policy network (action probabilities)
  - Add value network for advantage estimation
- [ ] **Step 2.5**: Training loop changes
  - Collect trajectories (state, action, reward sequences)
  - Update policy to maximize expected cumulative reward
- [ ] **Step 2.6**: Evaluation metrics
  - Total return, Sharpe ratio, max drawdown
  - Compare against buy-and-hold baseline

**Expected Benefit**: Model directly optimizes what we care about (profit) rather than proxy metric (classification accuracy).

**Complexity**: HIGH - Fundamental architecture change, requires trading simulator, new training paradigm.

**Dependencies**: Should complete other optimizations first (embedding caching, time features) before this major rewrite.

---

### 3. Multi-stage batch training

**Objective**: Vary batch size during training for better convergence - start small for fine-grained learning, increase for stability.

**Current State**: Constant batch size throughout training.

**Research Background**:
- Small batches: More noise in gradients, can escape local minima, but unstable
- Large batches: Smoother gradients, stable convergence, but may miss fine details
- Multi-stage: Get benefits of both

**Implementation Plan**:
- [ ] **Step 3.1**: Design batch schedule strategies
  - **Strategy A - Warmup**: Start large (32), decrease to small (1)
  - **Strategy B - Annealing**: Start small (1), increase to large (32)
  - **Strategy C - Cyclic**: Alternate between small and large
- [ ] **Step 3.2**: Add `--batch-schedule` parameter to train.py
  ```bash
  python train.py -s BTC-USD --batch-schedule warmup
  python train.py -s BTC-USD --batch-schedule annealing
  ```
- [ ] **Step 3.3**: Implement batch size scheduler
  ```python
  def get_batch_size(epoch, total_epochs, schedule='annealing'):
      if schedule == 'annealing':
          return 1 + int(31 * epoch / total_epochs)
      elif schedule == 'warmup':
          return 32 - int(31 * epoch / total_epochs)
  ```
- [ ] **Step 3.4**: Modify training loop to use dynamic batch size
- [ ] **Step 3.5**: Benchmark against constant batch size

**Expected Benefit**: Better convergence, potentially higher final accuracy.

**Complexity**: Low-Medium

**Alternative**: Learning rate scheduling achieves similar effect and is more common.

---

### 4. Dropout rate tuning

**Objective**: Expose dropout rate as configurable parameter for regularization tuning.

**Current State**: Dropout is hardcoded at 0.1 (10%) in `models/gemma_transformer_classifier.py`.

**Background**:
- Dropout randomly disables neurons during training to prevent overfitting
- Too low (0.0): Model may overfit to training data
- Too high (0.5+): Model may underfit, not learn enough
- Typical range: 0.1 - 0.3

**Implementation Plan**:
- [ ] **Step 4.1**: Add `--dropout` parameter to train.py CLI
- [ ] **Step 4.2**: Pass dropout rate to model constructor
  ```python
  model = SimpleGemmaTransformerClassifier(
      hidden_dim=config.hidden_dim,
      num_layers=config.num_layers,
      dropout=config.dropout
  )
  ```
- [ ] **Step 4.3**: Update test.py to also accept `--dropout` parameter (must match training)
- [ ] **Step 4.4**: Document recommended ranges in CLAUDE.md:
  - Small dataset (<500 samples): 0.3-0.5 (more regularization)
  - Medium dataset (500-5000): 0.1-0.3 (default)
  - Large dataset (>5000): 0.0-0.1 (less regularization needed)
- [ ] **Step 4.5**: Add to training metadata for architecture tracking

**Expected Benefit**: Better control over overfitting, especially for small datasets.

**Complexity**: Low

---

### 5. Learning rate scheduling

**Objective**: Adjust learning rate during training for better convergence.

**Current State**: Constant learning rate throughout training (default: 0.005).

**Background**:
- **Warmup**: Start low, increase gradually (helps with large batch sizes)
- **Step Decay**: Reduce LR by factor every N epochs
- **Cosine Annealing**: Smoothly decrease LR following cosine curve
- **ReduceOnPlateau**: Reduce LR when loss stops improving

**Implementation Plan**:
- [ ] **Step 5.1**: Add `--lr-schedule` parameter to train.py
  ```bash
  python train.py -s BTC-USD --lr-schedule cosine
  python train.py -s BTC-USD --lr-schedule step --lr-step-size 10 --lr-gamma 0.5
  python train.py -s BTC-USD --lr-schedule plateau
  ```
- [ ] **Step 5.2**: Implement scheduler options using PyTorch's `torch.optim.lr_scheduler`:
  - `StepLR`: Decay by gamma every step_size epochs
  - `CosineAnnealingLR`: Cosine decay to min LR
  - `ReduceLROnPlateau`: Reduce when loss plateaus
  - `OneCycleLR`: Warmup then decay (popular for transformers)
- [ ] **Step 5.3**: Add to training loop:
  ```python
  scheduler = create_scheduler(optimizer, config)
  for epoch in range(epochs):
      train_one_epoch()
      scheduler.step()
  ```
- [ ] **Step 5.4**: Log current learning rate during training
- [ ] **Step 5.5**: Document best practices in CLAUDE.md

**Expected Benefit**: Faster convergence, better final accuracy, avoid overshooting optimal weights.

**Complexity**: Low-Medium

**Recommended Default**: `OneCycleLR` or `CosineAnnealingLR` work well for transformers.

---

### 6. Time of day features (vectorized day of week)

**Objective**: Add temporal context to help model learn time-based patterns (e.g., Monday morning volatility, Friday afternoon calm).

**Current State**: We have timestamps but don't use them as features. Model only sees price and text.

**Data Availability (yfinance)**: ✅ Available - `pubDate` field already contains timestamp.

**Implementation Plan**:
- [ ] **Step 6.1**: Extract temporal features from `pubDate` in download.py:
  ```python
  from datetime import datetime
  dt = datetime.fromisoformat(pub_date)

  features = {
      'hour': dt.hour,
      'day_of_week': dt.weekday(),
      'is_market_open': is_market_hours(dt),
      'minutes_since_open': calc_minutes(dt),
  }
  ```
- [ ] **Step 6.2**: Choose encoding method:
  - **Option A - Text**: Add to input string
    ```
    Price: 95000
    Time: Monday 09:35 (Market Open: 5 min)
    Headline: ...
    ```
  - **Option B - Cyclical**: Encode as sin/cos for hour and day
    ```python
    hour_sin = sin(2 * pi * hour / 24)
    hour_cos = cos(2 * pi * hour / 24)
    ```
  - **Option C - One-hot**: Separate features for each hour/day
- [ ] **Step 6.3**: Update JSON schema:
  ```json
  {
    "title": "...",
    "price": 95000,
    "hour": 9,
    "day_of_week": 0,
    "is_market_open": true
  }
  ```
- [ ] **Step 6.4**: Update train.py/test.py to include temporal features in input
- [ ] **Step 6.5**: For Option B/C, modify model to accept additional numerical inputs

**Expected Benefit**: Model learns patterns like "negative news on Monday morning has bigger impact than Friday afternoon".

**Complexity**: Medium

**Recommendation**: Start with Option A (text-based) for simplicity, as Gemma can understand temporal text.

---

### 7. Sequence data (merge sequences together)

**Objective**: Instead of single news → prediction, use sequence of recent news to capture sentiment trends.

**Current State**: Each sample is independent - one news article predicts one label. Model has no memory of recent news.

**Rationale**:
- Single negative article might be noise
- Three negative articles in a row = stronger sell signal
- Sentiment momentum matters

**Implementation Plan**:
- [ ] **Step 7.1**: Define sequence strategy:
  - **Option A - Concatenation**: Merge last N articles into one input
    ```
    [News 1] Bitcoin drops...
    [News 2] Regulatory concerns...
    [News 3] Market selloff...
    → Stronger SELL signal
    ```
  - **Option B - Embedding Average**: Average embeddings of last N articles
  - **Option C - Sequence Model**: Use LSTM/GRU over sequence of embeddings
- [ ] **Step 7.2**: Add `--sequence-length` parameter (default: 1 = current behavior)
- [ ] **Step 7.3**: Modify data loading to create sequences:
  ```python
  sequences = []
  for i in range(len(data)):
      recent_news = data[max(0, i-seq_len):i+1]
      sequences.append({
          'texts': [n['title'] for n in recent_news],
          'label': data[i]['label']
      })
  ```
- [ ] **Step 7.4**: For Option A, concatenate texts with separators
- [ ] **Step 7.5**: For Option C, add LSTM layer after embedding:
  ```
  Texts → Gemma Embeddings → LSTM → Classifier
  ```
- [ ] **Step 7.6**: Handle edge cases (first N samples have shorter sequences)

**Expected Benefit**: Model captures sentiment momentum, not just point-in-time sentiment.

**Complexity**: Medium-High

**Recommendation**: Start with Option A (concatenation) - simplest and Gemma handles long text well.

---

### 8. Embedding caching to disk (save by hash)

**Objective**: Persist computed embeddings to disk to avoid recomputation on repeated training runs.

**Current State**:
- In-memory cache exists in model (`self.embedding_cache`)
- Cache is lost when model is reloaded
- Same text gets re-embedded every training run

**Problem**:
- Gemma embedding is SLOW (the bottleneck)
- With 1000 samples, embedding takes ~10-20 minutes
- Re-running train.py re-computes ALL embeddings

**Implementation Plan**:
- [ ] **Step 8.1**: Design cache file structure:
  ```
  datasets/{SYMBOL}/.embedding_cache.pkl
  ```
- [ ] **Step 8.2**: Add cache loading on model initialization:
  ```python
  def load_embedding_cache(self, cache_path):
      if os.path.exists(cache_path):
          self.embedding_cache = pickle.load(open(cache_path, 'rb'))
  ```
- [ ] **Step 8.3**: Add cache saving after training:
  ```python
  def save_embedding_cache(self, cache_path):
      pickle.dump(self.embedding_cache, open(cache_path, 'wb'))
  ```
- [ ] **Step 8.4**: Add `--cache-embeddings` flag to train.py (default: True)
- [ ] **Step 8.5**: Add `--clear-cache` flag to force re-computation
- [ ] **Step 8.6**: Handle cache invalidation:
  - Different embedding model → clear cache
  - Store metadata with cache (embedding model name, version)
- [ ] **Step 8.7**: Estimate cache size:
  - 768 floats × 4 bytes × 1000 samples = ~3 MB per 1000 samples

**Expected Benefit**:
- First run: Normal speed (compute + save)
- Subsequent runs: 10-100x faster (load from cache)
- Enables rapid iteration on training parameters

**Complexity**: Low-Medium

**Priority**: HIGH - This is a quick win with major speedup. Should implement early.

---

## Implementation Priority

| Priority | Item | Complexity | Benefit |
|----------|------|------------|---------|
| 1 | Embedding caching | Low-Medium | HIGH |
| 2 | Learning rate scheduling | Low-Medium | Medium |
| 3 | Dropout rate tuning | Low | Low-Medium |
| 4 | Time of day features | Medium | Medium |
| 5 | Market volatility indicator | Medium | Medium |
| 6 | Sequence data | Medium-High | Medium-High |
| 7 | Multi-stage batch training | Low-Medium | Low |
| 8 | Reinforcement learning | HIGH | HIGH |

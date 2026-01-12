# AI Investment Trader

An algorithmic trading signal classifier that predicts **Buy/Sell/Hold** signals using news headlines and price data. The model uses a Hierarchical Sentiment Transformer architecture with:
- Multi-level news classification (Market / Sector / Ticker)
- FinBERT-based financial sentiment analysis
- Cross-level attention between sentiment levels
- Temporal sequences for pattern detection

> **IMPORTANT**: This is a **SHORT-TERM, HIGH-FREQUENCY trading tool**, NOT a long-term investment tool. The model predicts what will happen in the **next 5 minutes** after news is published.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Choose Your Trading Style](#choose-your-trading-style)
- [How It Works](#how-it-works)
- [Beginner's Guide](#beginners-guide)
  - [The Cooking Analogy](#the-cooking-analogy)
  - [The Student's Notebook](#the-students-notebook-continuous-learning)
  - [The Teacher Checks Materials](#the-teacher-checks-materials-smart-training-guard)
  - [The Report Card](#the-report-card-understanding-evaluation)
- [Understanding the Trading Time Scale](#understanding-the-trading-time-scale)
- [Threshold Quick Reference](#threshold-quick-reference)
- [Complete Pipeline Commands](#complete-pipeline-commands)
- [Understanding Your Results](#understanding-your-results)
- [Glossary](#glossary)
- [Advanced Configuration](#advanced-configuration)

---

## Quick Start

Get the model running in 3 commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with HuggingFace (required for Gemma model)
huggingface-cli login

# 3. Run the full pipeline
python download.py -s BTC-USD    # Download data
python train.py -s BTC-USD       # Train model
python test.py -s BTC-USD        # Evaluate
```

**That's it!** Your trained model is saved to `datasets/BTC-USD/BTC-USD.pth`.

For other assets, just change the symbol:
```bash
python download.py -s AAPL -p 1mo -i 5m -n 500
python train.py -s AAPL -b 0.1 --sell-threshold -0.1
python test.py -s AAPL -b 0.1 --sell-threshold -0.1
```

---

## Choose Your Trading Style

Pick the command set that matches your trading approach. Each option is optimized for different time horizons and risk profiles.

### Option A: Scalper / Day Trader

**Best for:** Active traders who monitor markets in real-time and make multiple trades per day.

| Characteristic | Description |
|----------------|-------------|
| **Time Horizon** | Minutes to hours |
| **Data Interval** | 5-minute candles |
| **History** | 1 month (max for 5m data) |
| **Volatility Capture** | Micro price movements |
| **Required Attention** | High - must act quickly on signals |

```bash
# 1. DOWNLOAD - Maximum intraday data
python download.py -s AAPL -p 1mo -i 5m -n 2000

# 2. TRAIN - Precision model for micro-movements
python train.py -s AAPL -b 0.1 --sell-threshold -0.1 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300

# 3. TEST - Evaluate with AI-powered summary
python test.py -s AAPL -b 0.1 --sell-threshold -0.1 --samples 10 --summary
```

**Threshold Guide for Scalping:**
| Asset Type | Thresholds | Why |
|------------|------------|-----|
| Large Cap (AAPL) | ±0.1% | Low volatility, small moves matter |
| Growth (TSLA) | ±0.3% | Higher volatility |
| Crypto (BTC) | ±0.5% | Very high volatility |

---

### Option B: Swing Trader

**Best for:** Traders who hold positions for days to weeks, checking markets a few times per day.

| Characteristic | Description |
|----------------|-------------|
| **Time Horizon** | Days to weeks |
| **Data Interval** | 1-hour candles |
| **History** | 3 months |
| **Volatility Capture** | Intraday trends |
| **Required Attention** | Medium - check a few times daily |

```bash
# 1. DOWNLOAD - 3 months of hourly data
python download.py -s AAPL -p 3mo -i 1h -n 2000

# 2. TRAIN - Model for intraday trends
python train.py -s AAPL -b 0.3 --sell-threshold -0.3 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300

# 3. TEST - Evaluate with AI-powered summary
python test.py -s AAPL -b 0.3 --sell-threshold -0.3 --samples 10 --summary
```

**Threshold Guide for Swing Trading:**
| Asset Type | Thresholds | Why |
|------------|------------|-----|
| Large Cap (AAPL) | ±0.3% | Capture meaningful hourly moves |
| Growth (TSLA) | ±0.5% | Higher volatility stocks |
| Crypto (BTC) | ±1.0% | Significant hourly swings |

---

### Option C: Position Trader

**Best for:** Traders who hold positions for weeks to months, making fewer but larger trades.

| Characteristic | Description |
|----------------|-------------|
| **Time Horizon** | Weeks to months |
| **Data Interval** | Daily candles |
| **History** | 1 year (unlimited) |
| **Volatility Capture** | Major trend shifts |
| **Required Attention** | Low - check daily or weekly |

```bash
# 1. DOWNLOAD - 1 year of daily data
python download.py -s AAPL -p 1y -i 1d -n 2000

# 2. TRAIN - Model for major trend identification
python train.py -s AAPL -b 1.0 --sell-threshold -1.0 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 300

# 3. TEST - Evaluate with AI-powered summary
python test.py -s AAPL -b 1.0 --sell-threshold -1.0 --samples 10 --summary
```

**Threshold Guide for Position Trading:**
| Asset Type | Thresholds | Why |
|------------|------------|-----|
| Large Cap (AAPL) | ±1.0% | Filter daily noise |
| Growth (TSLA) | ±2.0% | Capture significant daily moves |
| Crypto (BTC) | ±3.0% | Major daily trend shifts only |

---

### Quick Comparison

| Style | Data Interval | Max History | Thresholds | Check Frequency |
|-------|---------------|-------------|------------|-----------------|
| **A: Scalper** | 5 minutes | 1 month | ±0.1% to ±0.5% | Constant |
| **B: Swing** | 1 hour | 3 months | ±0.3% to ±1.0% | Few times/day |
| **C: Position** | 1 day | 1+ years | ±1.0% to ±3.0% | Daily/weekly |

> **Note:** The `--summary` flag uses Google's Flan-T5-XL (3B parameters) to generate a human-readable analysis of your model's performance and trading readiness.

---

## How It Works

### Trading Thesis

News headlines during volatile periods correlate with 5-minute price direction. The model learns patterns like:
- "Regulatory concerns" → often precedes price drops → **SELL** signal
- "Institutional adoption" → often precedes price pumps → **BUY** signal
- Neutral news → sideways movement → **HOLD** signal

### Early Signal Detection

The model gives you an **early heads-up** based on news sentiment:

```
NEWS PUBLISHED                          5 MINUTES LATER
      │                                       │
      ▼                                       ▼
"Bitcoin ETF approved!"    ───────────►   Price goes UP
      │
      └── Model sees positive sentiment
          → Predicts BUY
          → You buy BEFORE the rise

"SEC investigating crypto" ───────────►   Price goes DOWN
      │
      └── Model sees negative sentiment
          → Predicts SELL
          → You sell BEFORE the drop
```

### Your Trading Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR TRADING WORKFLOW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. NEWS BREAKS: "Major bank announces Bitcoin partnership"             │
│                                    ↓                                    │
│  2. MODEL ANALYZES: Sees positive sentiment, similar to past news       │
│                     that preceded price increases                       │
│                                    ↓                                    │
│  3. MODEL PREDICTS: BUY (82% confident)                                 │
│                                    ↓                                    │
│  4. YOU ACT: Buy now, BEFORE the expected price rise                    │
│                                    ↓                                    │
│  5. RESULT: You bought early at a lower price ✓                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Important Caveat**: This is a **predictive model**, not a guarantee. The market doesn't always react the same way to similar news. Use it as **one signal among many**, not as financial advice.

---

## Beginner's Guide

This section explains how the project works using simple analogies.

### The Cooking Analogy

Imagine you want to teach a robot to cook **pizza**.

**Step 1: Gather Recipes (download.py)**
```
You collect:
- 100 pizza recipes
- What ingredients were used
- How they turned out (good/bad/okay)

This becomes your "recipe book" → datasets/BTC-USD/news_with_price.json
```

**Step 2: Train the Robot (train.py)**
```
The robot reads ALL 100 recipes and learns patterns:
- "When dough is thin + high heat → crispy crust"
- "Too much cheese → soggy middle"
- "Fresh tomatoes → better taste"

The robot's BRAIN after learning → BTC-USD.pth
```

**Step 3: Test the Robot (test.py)**
```
Give the robot NEW recipes it hasn't seen.
See if it can predict: "Will this pizza be good?"
```

### The Pipeline Simplified

```
download.py -s BTC-USD  →  "Collect study materials for Bitcoin"
                                ↓
                    datasets/BTC-USD/news_with_price.json (the textbook)
                                ↓
train.py -s BTC-USD     →  "Model reads textbook, learns patterns"
                                ↓
                    datasets/BTC-USD/BTC-USD.pth (model's brain)
                                ↓
test.py -s BTC-USD      →  "Quiz the model on new questions"
```

### What's in Each File?

| File | Simple Explanation |
|------|-------------------|
| `download.py` | "Go to the library and get study materials" |
| `train.py` | "Study the materials and learn patterns" |
| `test.py` | "Take a quiz to see how much was learned" |
| `*.json` files | "The textbooks with raw information" |
| `*.pth` file | "The brain after studying (learned patterns)" |

### The "Only Knows Bitcoin" Problem

The model is like a student who only attended one class:

```
┌──────────────────────────────────────────────────────────────┐
│  MODEL: "Bitcoin Brain"                                      │
│                                                              │
│  Training Data Seen:                                         │
│    ✅ Bitcoin news articles                                  │
│    ✅ Bitcoin price movements                                │
│    ❌ Apple Stock (never studied)                            │
│    ❌ Tesla Stock (never studied)                            │
│                                                              │
│  If you ask about AAPL:                                      │
│    🤷 "I don't know... I only learned Bitcoin patterns"      │
└──────────────────────────────────────────────────────────────┘
```

**To analyze a different symbol**, you must:
1. Run `download.py -s AAPL` to get new data
2. Run `train.py -s AAPL` to train a NEW model
3. The old `.pth` file only knows Bitcoin!

---

### The Student's Notebook (Continuous Learning)

By default, `train.py` **continues from existing knowledge** rather than starting from scratch:

```
┌─────────────────────────────────────────────────────────────────────────┐
│               CONTINUOUS LEARNING (Default Behavior)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  THE SMART STUDENT (default):                                           │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Week 1: Student learns A, B, C → saves to notebook               │ │
│  │  Week 2: READS OLD NOTES first → then learns D, E, F              │ │
│  │  Week 3: READS OLD NOTES → then learns G, H, I                    │ │
│  │                                                                   │ │
│  │  ✓ Result: Student accumulates knowledge over time!              │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  THE FORGETFUL STUDENT (--fresh flag):                                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Week 1: Student learns A, B, C                                   │ │
│  │  Week 2: THROWS AWAY old notebook! Learns D, E, F from scratch    │ │
│  │  Week 3: THROWS AWAY notebook again! Learns G, H, I only          │ │
│  │                                                                   │ │
│  │  ❌ Result: Student only knows the LAST thing learned            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**How to use:**
```bash
# DEFAULT: Continue from existing knowledge (RECOMMENDED)
python train.py -s BTC-USD

# FRESH START: Throw away old notes, start from scratch
python train.py -s BTC-USD --fresh
```

---

### The Teacher Checks Materials (Smart Training Guard)

The Smart Training Guard prevents overfitting when running automated training:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SMART TRAINING GUARD                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Think of it like a teacher who checks before starting class:           │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  CHECK 1: "Is there new material since last class?"               │ │
│  │           (Data hash comparison)                                  │ │
│  │                                                                   │ │
│  │  CHECK 2: "Are there enough new topics to teach?"                 │ │
│  │           (Minimum new samples threshold)                         │ │
│  │                                                                   │ │
│  │  CHECK 3: "Has enough time passed since last class?"              │ │
│  │           (Cooldown period)                                       │ │
│  │                                                                   │ │
│  │  If ALL checks pass → "Let's learn!"                              │ │
│  │  If ANY check fails → "Class dismissed, come back later."         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this matters:** Without Smart Guard, running `train.py` repeatedly on the same data causes the model to "memorize" instead of "learn patterns."

---

### The Report Card (Understanding Evaluation)

After testing, you'll see:

```
Results:
  Accuracy:       10/12 = 83.33%
  F1 Score:       0.7576
```

#### What is ACCURACY?

**"How many did you get right?"** - Simple counting.

```
Teacher gives you a 12-question test.
You answer all 12 questions.
Teacher grades: 10 correct, 2 wrong.

Your grade: 10/12 = 83.33%
```

#### What is F1 SCORE?

**"How confident AND thorough are you?"** - Quality measurement.

F1 balances two things:
- **Precision**: "When you raised your hand, were you right?"
- **Recall**: "Did you catch all the ones you should have?"

#### The Lazy Student Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE LAZY STUDENT PROBLEM                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Test with 100 questions: 80 HOLD, 10 BUY, 10 SELL                      │
│                                                                         │
│  LAZY STUDENT (always answers "HOLD"):                                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Accuracy: 80/100 = 80%  ← Looks good!                            │ │
│  │  F1 Score: 0.30          ← Reveals the truth!                     │ │
│  │                                                                   │ │
│  │  The student learned NOTHING - just guessed the most common!      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  SMART STUDENT (actually learned patterns):                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Accuracy: 75%           ← Slightly lower...                      │ │
│  │  F1 Score: 0.72          ← But much better quality!               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  LESSON: High accuracy + Low F1 = Model is cheating!                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Score Benchmarks

| Accuracy | F1 Score | Grade | Meaning |
|----------|----------|-------|---------|
| < 40% | < 0.35 | F | Model learned nothing |
| 40-55% | 0.35-0.50 | D | Barely learning |
| 55-65% | 0.50-0.60 | C | Some patterns found |
| 65-75% | 0.60-0.70 | B | Good! Learning patterns |
| 75-85% | 0.70-0.80 | A | Very good! Solid predictions |
| > 95% | > 0.90 | ??? | Suspicious - check for data leakage! |

---

## Understanding the Trading Time Scale

This is a **5-MINUTE** trading tool, not a long-term investment tool.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOW THIS TOOL IS MEANT TO BE USED                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ❌ WRONG WAY:                                                          │
│     1. Look at monthly chart                                            │
│     2. Run model once                                                   │
│     3. Hold for weeks/months                                            │
│                                                                         │
│  ✓ CORRECT WAY:                                                         │
│     1. News breaks: "SEC announces new crypto rules"                    │
│     2. IMMEDIATELY run model on this news                               │
│     3. Model predicts: "In the NEXT 5 MINUTES, price will drop"         │
│     4. You act NOW (sell within minutes)                                │
│     5. Repeat for each news event                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Understanding HOLD

**HOLD is NOT "price stays exactly the same"** - it's a **range**:

```
     SELL              HOLD                BUY
◄──────────┼────────────────────────┼──────────►
         -1%           0%          +1%

Examples (5-minute price changes):
  -2.5%  → SELL  (significant drop)
  -0.8%  → HOLD  (minor movement)
  +0.3%  → HOLD  (minor movement)
  +1.5%  → BUY   (significant rise)
```

---

## Threshold Quick Reference

Different assets have different volatility. Use appropriate thresholds:

| Asset Type | Example | Threshold | Command |
|------------|---------|-----------|---------|
| **Crypto** | BTC-USD | ±1.0% | `-b 1.0 --sell-threshold -1.0` |
| **Volatile Stock** | TSLA | ±0.5% | `-b 0.5 --sell-threshold -0.5` |
| **Large Cap** | AAPL, MSFT | ±0.3% | `-b 0.3 --sell-threshold -0.3` |
| **Index ETF** | SPY, QQQ | ±0.2% | `-b 0.2 --sell-threshold -0.2` |

### How to Check if Your Thresholds Are Good

After running `train.py`, check the label distribution:

```
✅ GOOD: Label distribution: SELL=45, HOLD=120, BUY=52
   (All three classes have samples - model can learn)

❌ BAD:  Label distribution: SELL=0, HOLD=199, BUY=0
   (All HOLD - thresholds too wide, lower them!)
```

**Quick Calibration:**
1. Run `train.py` and check the label distribution
2. If all HOLD → Lower thresholds
3. If almost no HOLD → Raise thresholds
4. Aim for roughly 20-40% in each class

---

## Complete Pipeline Commands

Copy-paste these commands for different asset types:

### Cryptocurrency (BTC-USD, ETH-USD)

```bash
python download.py -s BTC-USD -p 1mo -i 5m -n 1000
python train.py -s BTC-USD -b 1.0 --sell-threshold -1.0 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200
python test.py -s BTC-USD -b 1.0 --sell-threshold -1.0 --samples 10 --hidden-dim 512 --num-layers 4
```

### Volatile Stock (TSLA, GME)

```bash
python download.py -s TSLA -p 1mo -i 5m -n 500
python train.py -s TSLA -b 0.5 --sell-threshold -0.5 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200
python test.py -s TSLA -b 0.5 --sell-threshold -0.5 --samples 10 --hidden-dim 512 --num-layers 4
```

### Large Cap Stock (AAPL, MSFT, GOOGL)

```bash
python download.py -s AAPL -p 1mo -i 5m -n 500
python train.py -s AAPL -b 0.1 --sell-threshold -0.1 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200
python test.py -s AAPL -b 0.1 --sell-threshold -0.1 --samples 10 --hidden-dim 512 --num-layers 4
```

### Index ETF (SPY, QQQ)

```bash
python download.py -s SPY -p 1mo -i 5m -n 300
python train.py -s SPY -b 0.05 --sell-threshold -0.05 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200
python test.py -s SPY -b 0.05 --sell-threshold -0.05 --samples 10 --hidden-dim 512 --num-layers 4
```

---

## Understanding Your Results

### Quick Reference

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| **Accuracy** | > 70% | < 50% |
| **F1 Score** | > 0.65 | < 0.45 |
| **Both similar** | Acc ≈ F1 | Acc >> F1 (cheating!) |

### Red Flags

| Symptom | Problem | Solution |
|---------|---------|----------|
| Accuracy 80%, F1 0.30 | Guessing most common class | Fix threshold balance |
| Accuracy > 95% | Data leakage | Check train/test split matches |
| F1 varies wildly | Too few test samples | Get more data |
| Both scores < 40% | Model learned nothing | More epochs, better thresholds |

### How to Improve

1. **More data**: `python download.py -s AAPL -n 1000`
2. **More epochs**: `python train.py -s AAPL -e 300`
3. **Better thresholds**: Adjust until label distribution is balanced
4. **Bigger model**: `--hidden-dim 512 --num-layers 4`

---

## Glossary

| Term | Definition |
|------|------------|
| **Batch** | Number of samples processed before updating weights |
| **Embedding** | Converting text to numbers the model can understand |
| **Epoch** | One complete pass through all training data |
| **F1 Score** | Balance of precision and recall (0-1, higher is better) |
| **Learning Rate** | How big of steps the model takes when learning |
| **Logits** | Raw model outputs before converting to probabilities |
| **Optimizer** | Algorithm that updates model weights (SGD, AdamW) |
| **Threshold** | The % price change that triggers BUY or SELL |
| **Transformer** | The neural network architecture used for learning |

---

## Advanced Configuration

For detailed technical documentation including:
- CLI parameter reference
- Model architecture deep dive
- Advanced threshold tuning by market conditions
- Device support & GPU setup
- Project file structure

See **[CLAUDE.md](CLAUDE.md)** - the technical reference guide.

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KEY TAKEAWAYS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. This is a 5-MINUTE trading tool, not long-term investing            │
│                                                                         │
│  2. Models learn PATTERNS, not predictions                              │
│     "Negative news often precedes drops" ≠ "This news WILL cause drop"  │
│                                                                         │
│  3. Different assets need different thresholds                          │
│     Crypto moves 1%+ in 5 min, but AAPL rarely moves 0.3%               │
│                                                                         │
│  4. Past patterns don't guarantee future results                        │
│     Use as ONE signal among many, not as financial advice               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Happy Trading! May your F1 scores be high and your losses be low.*

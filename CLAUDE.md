# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```
CORE COMMANDS PILELINE (DO NOT DELETE):
python download.py -s AAPL -p 1mo -i 5m -n 1000
python train.py -s AAPL -b 0.1 --sell-threshold -0.1 --batch-size 1 -l 0.005 -o AdamW --hidden-dim 512 --num-layers 4 -e 500 --continue
python test.py -s AAPL -b 0.1 --sell-threshold -0.1 --samples 10 --hidden-dim 512 --num-layers 4
```
---

## Table of Contents

### Getting Started
- [Project Overview](#project-overview) - What this tool does
- [Quick Start](#quick-start) - Get running in 3 commands
- [How to Read This Guide](#how-to-read-this-guide) - Navigation tips

### Understanding the Tool
- [Understanding the Trading Time Scale](#understanding-the-trading-time-scale) - Why this is a 5-minute tool
- [Threshold Best Practices](#threshold-best-practices-by-asset-class) - Tuning for different assets
- [Complete Pipeline Commands](#complete-pipeline-commands-by-asset-type) - Ready-to-use commands
- [Live Trading Implementation](#live-trading-implementation-future) - Real-time trading setup

### The Beginner's Guide (Learning with Stories)
- [The Cooking Analogy](#the-cooking-analogy) - Understanding ML basics
- [What's Actually in Each File?](#whats-actually-in-each-file) - Textbooks vs trained brains
- [The Pipeline Simplified](#the-pipeline-simplified) - How data flows
- [Understanding the .pth File](#understanding-the-pth-file) - What gets saved
- [The "Only Knows Bitcoin" Problem](#the-only-knows-bitcoin-problem) - Why models are asset-specific

### Training Strategies
- [Continuous Learning](#continuous-learning-the-students-notebook) - Smart Student vs Forgetful Student
- [Smart Training Guard](#smart-training-guard-daemon-safe-automation) - Preventing overfitting
- [Production Retraining Guide](#production-retraining-guide) - Keeping models fresh

### Commands & Configuration
- [Commands](#commands) - All CLI commands
- [Configuration Reference](#configuration-reference) - All parameters explained
- [Understanding Evaluation Results](#understanding-evaluation-results-the-report-card) - Reading your scores

### Technical Deep Dives
- [Model Architecture Deep Dive](#model-architecture-deep-dive) - Inside the neural network
- [Understanding Model Queries](#understanding-model-queries-classification-vs-chat-models) - How to query your model
- [PyTorch Functions Reference](#pytorch-functions-reference) - Key functions explained

### Reference
- [Libraries and Dependencies](#libraries-and-dependencies) - What each library does
- [File Reference](#file-reference) - What each file does
- [Glossary of Terms](#glossary-of-terms-and-acronyms) - Technical definitions
- [Device Support](#device-support) - GPU acceleration guide
- [Known Issues & TODO](#known-issues) - Current status

---

## How to Read This Guide

This guide is designed for different reading styles:

| If you want to... | Start here |
|-------------------|------------|
| **Get running quickly** | [Quick Start](#quick-start) |
| **Understand how it works** | [Beginner's Guide](#beginners-guide-how-this-project-works) |
| **Learn through stories** | [The Cooking Analogy](#the-cooking-analogy) |
| **Set up automated training** | [Smart Training Guard](#smart-training-guard-daemon-safe-automation) |
| **Tune for your asset** | [Threshold Best Practices](#threshold-best-practices-by-asset-class) |
| **Understand the code** | [Model Architecture Deep Dive](#model-architecture-deep-dive) |
| **Look up a term** | [Glossary](#glossary-of-terms-and-acronyms) |

### Educational Stories Used in This Guide

Throughout this documentation, we use relatable analogies to explain complex ML concepts:

| Story | Concept | Section |
|-------|---------|---------|
| **The Cooking Analogy** | How ML training works | [Link](#the-cooking-analogy) |
| **The Student's Notebook** | Continuous learning & model persistence | [Link](#continuous-learning-the-students-notebook) |
| **The Teacher Checks for New Material** | Smart Training Guard | [Link](#smart-training-guard-daemon-safe-automation) |
| **The Report Card** | Understanding accuracy & F1 scores | [Link](#understanding-evaluation-results-the-report-card) |
| **The Lazy Student Problem** | Why F1 matters more than accuracy | [Link](#why-f1-matters-more-than-accuracy) |
| **The Answer Key Problem** | Parameter consistency | [Link](#parameter-consistency-why-its-critical) |

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
python download.py -s AAPL -b 0.3 --sell-threshold -0.3
python train.py -s AAPL -b 0.3 --sell-threshold -0.3
python test.py -s AAPL -b 0.3 --sell-threshold -0.3
```

---

## Project Overview

Algorithmic trading signal classifier that predicts **Buy/Sell/Hold** signals for BTC-USD using news headlines and price data. The model combines Google Gemma text embeddings (300M parameters) with a PyTorch Transformer classifier to learn correlations between news sentiment and short-term price movements.

> **IMPORTANT**: This is a **SHORT-TERM, HIGH-FREQUENCY trading tool**, NOT a long-term investment tool. The model predicts what will happen in the **next 5 minutes** after news is published. It requires near-real-time usage and active monitoring. See [Understanding the Trading Time Scale](#understanding-the-trading-time-scale) for details.

### Target Market Conditions

The model is designed for volatile markets with:
- Rapid price swings (see `target-profit.png` for examples)
- High news event frequency
- High trading volume

### Trading Thesis

News headlines during volatile periods correlate with 5-minute price direction. The model learns patterns like:
- "Regulatory concerns" → often precedes price drops → SELL signal
- "Institutional adoption" → often precedes price pumps → BUY signal
- Neutral news → sideways movement → HOLD signal

### How the Model Helps You Trade (Early Signal Detection)

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

**The Key Insight**: The model learned patterns from historical data where it saw:
- What the news said (headline + summary)
- What the price did 5 minutes later

So when new news comes in, it gives you a heads-up based on what **typically happened** after similar news in the past.

**Trading Flow:**

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
│                     "The model predicts the price will RISE.            │
│                      Consider buying."                                  │
│                                    ↓                                    │
│  4. YOU ACT: Buy now, BEFORE the expected price rise                    │
│                                    ↓                                    │
│  5. MARKET REACTS: Price rises as traders respond to news               │
│                                    ↓                                    │
│  6. RESULT: You bought early at a lower price ✓                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**The Opposite Case:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. NEWS BREAKS: "Government announces crypto crackdown"                │
│                                    ↓                                    │
│  2. MODEL ANALYZES: Sees negative sentiment, similar to past news       │
│                     that preceded price drops                           │
│                                    ↓                                    │
│  3. MODEL PREDICTS: SELL (75% confident)                                │
│                     "The model predicts the price will DROP.            │
│                      Consider selling."                                 │
│                                    ↓                                    │
│  4. YOU ACT: Sell now, BEFORE the expected price drop                   │
│                                    ↓                                    │
│  5. MARKET REACTS: Price drops as traders panic sell                    │
│                                    ↓                                    │
│  6. RESULT: You sold early at a higher price ✓                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Important Caveat**: This is a **predictive model**, not a guarantee. The market doesn't always react the same way to similar news. Use it as **one signal among many**, not as financial advice. Past patterns don't guarantee future results.

---

## Understanding the Trading Time Scale

This section clarifies an important concept that can cause confusion: the relationship between the long-term charts shown in `target-profit.png` and the short-term (5-minute) nature of the actual trading tool.

### The Time-Scale Disconnect (Common Confusion)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE DISCONNECT YOU MIGHT NOTICE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  target-profit.png shows:                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 6 MONTHS of data (May → October)                             │   │
│  │  • Price range: $85,000 → $120,000 (35%+ swings)                │   │
│  │  • Pink circles = "look at these volatile waves"                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  The CODE actually works on:                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 5 MINUTE intervals                                           │   │
│  │  • ±1% thresholds                                               │   │
│  │  • Predicts next 5 minutes, not next month                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  COMMON QUESTION:                                                       │
│  "If I'm looking at monthly data, prices move 30%... HOLD at ±1%        │
│   would NEVER trigger because prices are always way different!"         │
│                                                                         │
│  ANSWER:                                                                │
│  The ±1% threshold is for 5-MINUTE changes, not monthly changes.        │
│  The image shows WHEN to use the tool, not the time scale it works on.  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What target-profit.png Actually Shows

The image shows **WHEN to use this tool** (during volatile market periods), **NOT how it works**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CORRECT INTERPRETATION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  target-profit.png (6-month view):                                      │
│                                                                         │
│       Price                                                             │
│     $120K ┤            ╭─╮    ╭─╮                      ╭╮               │
│           │      ╭────╯  ╰╮ ╭╯  ╰─╮    ╭──╮          ╭╯╰╮              │
│     $110K ┤   ╭─╯        ╰─╯      ╰────╯  ╰╮   ╭─────╯   │              │
│           │  ╭╯                             ╰──╯         ╰─             │
│     $100K ┤ ╭╯                                                          │
│           │╭╯       ↑         ↑        ↑       ↑        ↑               │
│      $90K ┤╯        │         │        │       │        │               │
│           └─────────┴─────────┴────────┴───────┴────────┴──────         │
│              May    Jun      Jul      Aug     Sep      Oct              │
│                                                                         │
│              These pink circles highlight VOLATILE PERIODS              │
│              = "This is when you want to actively use this tool"        │
│                                                                         │
│  ZOOMED INTO ONE VOLATILE PERIOD (what the tool actually sees):         │
│                                                                         │
│       Price     ← One pink circle zoomed in = many 5-minute candles     │
│    $115.5K ┤        ╭╮                                                  │
│            │      ╭─╯╰╮    ╭╮                                           │
│    $115.0K ┤   ╭──╯   ╰──╮╭╯╰╮                                          │
│            │ ╭─╯         ╰╯  ╰╮                                         │
│    $114.5K ┤╯                 ╰──                                       │
│            └─────┬─────┬─────┬─────┬─────┬─────┬─────                   │
│               9:00  9:05  9:10  9:15  9:20  9:25  9:30                  │
│                                                                         │
│               ↑ THIS is where the model works (5-minute intervals)      │
│               ↑ The ±1% threshold makes sense at THIS scale             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### This is a SHORT-TERM Trading Tool

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOW THIS TOOL IS MEANT TO BE USED                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ❌ WRONG WAY (what the long-term image might suggest):                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. Look at monthly chart                                       │   │
│  │  2. Run model once                                              │   │
│  │  3. Hold for weeks/months                                       │   │
│  │  ❌ At this scale, ±1% threshold is meaningless                 │   │
│  │  ❌ HOLD would never trigger (prices always differ by >1%)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ✓ CORRECT WAY (actual design intent):                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. News breaks: "SEC announces new crypto rules"               │   │
│  │  2. IMMEDIATELY run model on this news                          │   │
│  │  3. Model predicts: "In the NEXT 5 MINUTES, price will drop"    │   │
│  │  4. You act NOW (sell within minutes)                           │   │
│  │  5. Repeat for each news event                                  │   │
│  │  ✓ At 5-minute scale, ±1% threshold makes sense                 │   │
│  │  ✓ HOLD triggers when price moves <1% (common in 5 min)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Understanding the HOLD Threshold

**HOLD is NOT "price stays exactly the same"** - it's a **range**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LABEL ASSIGNMENT LOGIC                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Price Change        Condition              Label                      │
│   ─────────────       ─────────              ─────                      │
│   < -1.0%             pct < SELL_THRESHOLD   → SELL                     │
│   -1.0% to +1.0%      (everything else)      → HOLD                     │
│   > +1.0%             pct > BUY_THRESHOLD    → BUY                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

VISUAL NUMBER LINE:

     SELL              HOLD                BUY
◄──────────┼────────────────────────┼──────────►
         -1%           0%          +1%

Examples (5-minute price changes):
  -2.5%  → SELL  (below -1%, significant drop)
  -0.8%  → HOLD  (between -1% and +1%, minor movement)
  +0.3%  → HOLD  (between -1% and +1%, minor movement)
  +0.99% → HOLD  (still under +1%, not significant enough)
  +1.5%  → BUY   (above +1%, significant rise)
```

**HOLD means**: "The price moved, but not enough to be significant" (±1% is considered noise/sideways movement at the 5-minute scale).

### Why ±1% Makes Sense at 5-Minute Scale

```
BITCOIN 5-MINUTE PRICE MOVEMENTS (typical):

  Most 5-min periods:  -0.5% to +0.5%  → HOLD (no significant move)
  Volatile periods:    -2% to +2%      → BUY or SELL signals
  Extreme events:      -5% to +5%      → Strong BUY or SELL

The ±1% threshold captures "significant" 5-minute moves while
filtering out normal market noise.
```

### Threshold Best Practices by Time Interval

If you modify the `PRICE_INTERVAL` in `download.py`, you should also adjust the thresholds in `train.py` to match:

| Time Interval | Suggested Threshold | Rationale |
|---------------|---------------------|-----------|
| 1 minute | ±0.2% to ±0.5% | Very small moves expected in 1 min |
| **5 minutes** | **±0.5% to ±1.5%** | **Current setting (±1.0%) - reasonable** |
| 15 minutes | ±1% to ±2% | Larger moves expected |
| 30 minutes | ±1.5% to ±3% | More time for price movement |
| 1 hour | ±2% to ±4% | Significant moves possible |
| 4 hours | ±3% to ±6% | Larger swings common |
| 1 day | ±3% to ±7% | Daily volatility range |
| 1 week | ±5% to ±15% | Weekly swings |

**Rule of Thumb**: As time interval increases, thresholds should increase proportionally because prices have more time to move.

### Threshold Best Practices by Asset Class

Different assets have different volatility levels. Using the same thresholds for all assets will cause problems:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE VOLATILITY PROBLEM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Using ±1% threshold for ALL assets:                                    │
│                                                                         │
│  BITCOIN (BTC-USD) - Highly Volatile                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Typical 5-min moves: -2% to +2%                                │   │
│  │  With ±1% threshold:                                            │   │
│  │    SELL: 25%  |  HOLD: 40%  |  BUY: 35%                         │   │
│  │  ✓ Good distribution - model can learn patterns                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  APPLE (AAPL) - Low Volatility                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Typical 5-min moves: -0.3% to +0.3%                            │   │
│  │  With ±1% threshold:                                            │   │
│  │    SELL: 0%   |  HOLD: 100% |  BUY: 0%                          │   │
│  │  ❌ BAD! Model has nothing to learn (all HOLD)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Recommended Thresholds by Asset Class:**

| Asset Class | Example Symbols | Suggested Threshold | Rationale |
|-------------|-----------------|---------------------|-----------|
| **Cryptocurrency** | BTC-USD, ETH-USD | ±1.0% to ±2.0% | Highly volatile, large swings common |
| **Meme Stocks** | GME, AMC | ±1.0% to ±1.5% | High volatility, news-driven |
| **Growth Tech** | TSLA, NVDA | ±0.5% to ±1.0% | Moderate-high volatility |
| **Large Cap Tech** | AAPL, MSFT, GOOGL | ±0.3% to ±0.5% | Lower volatility, stable |
| **Blue Chip / Index** | SPY, DIA, JPM | ±0.2% to ±0.4% | Very stable, small moves |
| **Commodities** | GC=F (Gold), CL=F (Oil) | ±0.5% to ±1.0% | Moderate volatility |

**How to Adjust for Your Asset:**

```python
# In train.py - Choose thresholds based on your asset

# For Cryptocurrency (BTC-USD, ETH-USD)
BUY_THRESHOLD = 1.0
SELL_THRESHOLD = -1.0

# For Growth/Volatile Stocks (TSLA, NVDA)
BUY_THRESHOLD = 0.7
SELL_THRESHOLD = -0.7

# For Large Cap Stocks (AAPL, MSFT, GOOGL)
BUY_THRESHOLD = 0.3
SELL_THRESHOLD = -0.3

# For Blue Chip / Index (SPY, DIA)
BUY_THRESHOLD = 0.2
SELL_THRESHOLD = -0.2
```

**Important**: After changing thresholds in `train.py`, you must also update `test.py` to match!

**How to Check if Your Thresholds Are Good:**

After running `train.py`, look at the label distribution:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CHECKING LABEL DISTRIBUTION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GOOD DISTRIBUTION (model can learn):                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Label distribution: SELL=45, HOLD=120, BUY=52                  │   │
│  │                                                                 │   │
│  │  ✓ All three classes have samples                               │   │
│  │  ✓ Reasonable balance (not too skewed)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  BAD DISTRIBUTION (model cannot learn):                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Label distribution: SELL=0, HOLD=59, BUY=0                     │   │
│  │                                                                 │   │
│  │  ❌ Only HOLD samples - thresholds too wide!                    │   │
│  │  ❌ Lower your thresholds (e.g., ±1.0% → ±0.3%)                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ALSO BAD (opposite problem):                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Label distribution: SELL=80, HOLD=2, BUY=85                    │   │
│  │                                                                 │   │
│  │  ❌ Almost no HOLD samples - thresholds too tight!              │   │
│  │  ❌ Raise your thresholds (e.g., ±0.1% → ±0.5%)                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Quick Calibration Method:**

1. Run `train.py` and check the label distribution
2. If all HOLD → Lower thresholds
3. If almost no HOLD → Raise thresholds
4. Aim for roughly 20-40% in each class (doesn't need to be perfect)

### Complete Pipeline Commands by Asset Type

Copy-paste these commands for different asset types. All commands use precision settings optimized for accuracy.

#### Volatility Spectrum

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VOLATILITY SPECTRUM (5-minute moves)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LOW                                                          HIGH      │
│  ◄─────────────────────────────────────────────────────────────────►   │
│                                                                         │
│  SPY     AAPL    MSFT    GOOGL    NVDA    TSLA    ETH-USD   BTC-USD    │
│  (ETF)   (Large Cap)     (Tech)   (Growth) (Meme)  (Crypto)  (Crypto)  │
│                                                                         │
│  ±0.05%  ±0.1%   ±0.1%   ±0.15%   ±0.3%   ±0.5%   ±0.8%     ±1.0%     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Quick Reference Table

| Asset Type | Example | Buy | Sell | News | Typical 5min Move |
|------------|---------|-----|------|------|-------------------|
| **Index ETF** | SPY, QQQ | `0.05` | `-0.05` | 300 | ±0.02% - 0.1% |
| **Large Cap** | AAPL, MSFT | `0.1` | `-0.1` | 500 | ±0.05% - 0.2% |
| **Growth Tech** | NVDA, AMD | `0.3` | `-0.3` | 500 | ±0.1% - 0.5% |
| **Volatile Stock** | TSLA, GME | `0.5` | `-0.5` | 500 | ±0.2% - 1.0% |
| **Crypto** | BTC-USD | `1.0` | `-1.0` | 1000 | ±0.3% - 2.0% |

#### Index ETF (SPY, QQQ, DIA) - Very Low Volatility

```bash
# Download
python download.py -s SPY -p 1mo -i 5m -n 300

# Train
python train.py -s SPY -b 0.05 --sell-threshold -0.05 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200

# Test
python test.py -s SPY -b 0.05 --sell-threshold -0.05 --samples 10
```

#### Large Cap Stock (AAPL, MSFT, GOOGL) - Low Volatility

```bash
# Download
python download.py -s AAPL -p 1mo -i 5m -n 500

# Train
python train.py -s AAPL -b 0.1 --sell-threshold -0.1 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200

# Test
python test.py -s AAPL -b 0.1 --sell-threshold -0.1 --samples 10
```

#### Growth Stock (NVDA, AMD) - Medium Volatility

```bash
# Download
python download.py -s NVDA -p 1mo -i 5m -n 500

# Train
python train.py -s NVDA -b 0.3 --sell-threshold -0.3 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200

# Test
python test.py -s NVDA -b 0.3 --sell-threshold -0.3 --samples 10
```

#### Volatile Stock (TSLA, GME, AMC) - High Volatility

```bash
# Download
python download.py -s TSLA -p 1mo -i 5m -n 500

# Train
python train.py -s TSLA -b 0.5 --sell-threshold -0.5 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200

# Test
python test.py -s TSLA -b 0.5 --sell-threshold -0.5 --samples 10
```

#### Cryptocurrency (BTC-USD, ETH-USD) - Very High Volatility

```bash
# Download (crypto has 24/7 data)
python download.py -s BTC-USD -p 1mo -i 5m -n 1000

# Train
python train.py -s BTC-USD -b 1.0 --sell-threshold -1.0 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -e 200

# Test
python test.py -s BTC-USD -b 1.0 --sell-threshold -1.0 --samples 10
```

#### How to Calibrate Your Thresholds

After running `train.py`, check the label distribution in the output:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WHAT YOU SEE                      WHAT TO DO                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SELL=0%  HOLD=100% BUY=0%         Thresholds TOO WIDE                  │
│                                    → Make SMALLER (±1% → ±0.3%)         │
│                                                                         │
│  SELL=45% HOLD=10%  BUY=45%        Thresholds TOO TIGHT                 │
│                                    → Make BIGGER (±0.1% → ±0.3%)        │
│                                                                         │
│  SELL=25% HOLD=50%  BUY=25%        ✓ PERFECT!                           │
│                                    → Keep these thresholds              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Threshold Best Practices by Market Conditions

Market conditions significantly affect price volatility. The same asset can behave very differently during different times:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MARKET CONDITIONS & VOLATILITY                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  HIGH VOLATILITY (use wider thresholds):                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Market open (first 30-60 minutes)                            │   │
│  │  • Major news events (Fed announcements, earnings)              │   │
│  │  • Market close (last 30 minutes)                               │   │
│  │  • Crypto: 24/7 but peaks during US/EU trading hours            │   │
│  │  • Geopolitical events (wars, elections)                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LOW VOLATILITY (use tighter thresholds):                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Holidays (Christmas, New Year, Thanksgiving)                 │   │
│  │  • Weekends (stocks closed, crypto low volume)                  │   │
│  │  • Mid-day lull (11am - 2pm ET for US stocks)                   │   │
│  │  • Summer months (July-August typically slower)                 │   │
│  │  • Between major news events                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Threshold Adjustments by Market Condition:**

| Market Condition | Volatility | Threshold Adjustment | Example (BTC) |
|------------------|------------|----------------------|---------------|
| Major news event | Very High | Wider (+50-100%) | ±1.5% to ±2.0% |
| Normal trading | Normal | Baseline | ±1.0% |
| Weekends | Low | Tighter (-30-50%) | ±0.5% to ±0.7% |
| Holidays | Very Low | Much tighter (-50-70%) | ±0.3% to ±0.5% |
| After-hours (stocks) | Low | Tighter (-30-50%) | ±0.3% to ±0.5% |

### Trading Hours Reference

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRADING HOURS (US Eastern Time)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STOCKS (NYSE/NASDAQ):                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Pre-market:    4:00 AM - 9:30 AM  (low volume)                 │   │
│  │  Market open:   9:30 AM - 10:30 AM (HIGH volatility)            │   │
│  │  Mid-day:       10:30 AM - 3:00 PM (moderate)                   │   │
│  │  Power hour:    3:00 PM - 4:00 PM  (HIGH volatility)            │   │
│  │  After-hours:   4:00 PM - 8:00 PM  (low volume)                 │   │
│  │  Closed:        Weekends & holidays                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  CRYPTOCURRENCY (24/7):                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  US hours:      9:00 AM - 5:00 PM ET   (high volume)            │   │
│  │  EU hours:      3:00 AM - 11:00 AM ET  (high volume)            │   │
│  │  Asia hours:    8:00 PM - 4:00 AM ET   (moderate volume)        │   │
│  │  Weekends:      Lower volume but still trading                  │   │
│  │  Best times:    US/EU overlap (9 AM - 11 AM ET)                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Threshold Best Practices by Trader Profile

Different trading styles require different threshold settings:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRADER PROFILES                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CONSERVATIVE TRADER:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Goal: Fewer trades, higher confidence signals only             │   │
│  │  Risk: Low                                                      │   │
│  │  Threshold: WIDER (more HOLD, fewer BUY/SELL)                   │   │
│  │                                                                 │   │
│  │  BTC-USD:  ±1.5% to ±2.0%                                       │   │
│  │  AAPL:     ±0.5% to ±0.7%                                       │   │
│  │  SPY:      ±0.4% to ±0.5%                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  MID-AGGRESSIVE TRADER (RECOMMENDED):                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Goal: Balanced approach, capture good opportunities            │   │
│  │  Risk: Medium                                                   │   │
│  │  Threshold: MODERATE (balanced distribution)                    │   │
│  │                                                                 │   │
│  │  BTC-USD:  ±0.8% to ±1.2%                                       │   │
│  │  AAPL:     ±0.25% to ±0.4%                                      │   │
│  │  SPY:      ±0.2% to ±0.3%                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  AGGRESSIVE TRADER:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Goal: Many trades, capture small movements                     │   │
│  │  Risk: High                                                     │   │
│  │  Threshold: TIGHTER (more BUY/SELL, fewer HOLD)                 │   │
│  │                                                                 │   │
│  │  BTC-USD:  ±0.4% to ±0.7%                                       │   │
│  │  AAPL:     ±0.1% to ±0.2%                                       │   │
│  │  SPY:      ±0.1% to ±0.15%                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Recommended Settings for Mid-Aggressive Trader:**

| Asset | BUY_THRESHOLD | SELL_THRESHOLD | Expected Signals/Day |
|-------|---------------|----------------|----------------------|
| **BTC-USD** | +1.0% | -1.0% | 5-15 (volatile days) |
| **ETH-USD** | +1.2% | -1.2% | 5-12 |
| **TSLA** | +0.5% | -0.5% | 3-8 |
| **AAPL** | +0.3% | -0.3% | 2-5 |
| **MSFT** | +0.25% | -0.25% | 2-4 |
| **SPY** | +0.2% | -0.2% | 1-3 |

### Complete Threshold Selection Guide

Use this decision tree to select your thresholds:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THRESHOLD SELECTION DECISION TREE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: What asset are you trading?                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Cryptocurrency (BTC, ETH)     → Start with ±1.0%               │   │
│  │  Meme/Growth stocks (TSLA)     → Start with ±0.5%               │   │
│  │  Large cap tech (AAPL, MSFT)   → Start with ±0.3%               │   │
│  │  Index funds (SPY, QQQ)        → Start with ±0.2%               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  STEP 2: What's the market condition?                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Holiday/Weekend/Low volume    → Reduce threshold by 50%        │   │
│  │  Normal trading day            → Keep baseline                  │   │
│  │  Major news event              → Increase threshold by 50%      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  STEP 3: What's your risk tolerance?                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Conservative                  → Increase threshold by 30%      │   │
│  │  Mid-aggressive (recommended)  → Keep as-is                     │   │
│  │  Aggressive                    → Reduce threshold by 30%        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  STEP 4: Calibrate with your data                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Run train.py and check label distribution:                     │   │
│  │  • All HOLD? → Lower thresholds                                 │   │
│  │  • No HOLD?  → Raise thresholds                                 │   │
│  │  • Good mix? → You're set!                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example: Calculating Thresholds for AAPL on a Holiday

```
STEP 1: AAPL (Large cap tech)         → Base: ±0.3%
STEP 2: Holiday (low volume)          → ×0.5 = ±0.15%
STEP 3: Mid-aggressive trader         → ×1.0 = ±0.15%
STEP 4: Run train.py, check distribution

RESULT: Use BUY_THRESHOLD = 0.15, SELL_THRESHOLD = -0.15
```

### Example: Calculating Thresholds for BTC-USD During Major News

```
STEP 1: BTC-USD (Cryptocurrency)      → Base: ±1.0%
STEP 2: Major news event              → ×1.5 = ±1.5%
STEP 3: Mid-aggressive trader         → ×1.0 = ±1.5%
STEP 4: Run train.py, check distribution

RESULT: Use BUY_THRESHOLD = 1.5, SELL_THRESHOLD = -1.5
```

### Common Mistakes to Avoid

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMMON THRESHOLD MISTAKES                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ❌ MISTAKE 1: Using same thresholds for all assets                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Problem: BTC moves 1-2% easily, AAPL rarely moves 0.5%         │   │
│  │  Result: All HOLD for stocks, too many signals for crypto       │   │
│  │  Fix: Adjust thresholds per asset class                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ❌ MISTAKE 2: Not adjusting for market conditions                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Problem: Holiday data has tiny movements (your current issue!) │   │
│  │  Result: All samples become HOLD                                │   │
│  │  Fix: Lower thresholds during low-volatility periods            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ❌ MISTAKE 3: Making thresholds too tight                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Problem: Using ±0.01% captures noise, not real signals         │   │
│  │  Result: Many false signals, poor model accuracy                │   │
│  │  Fix: Threshold should be above typical market noise            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ❌ MISTAKE 4: Not matching train.py and test.py thresholds             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Problem: Training with ±1% but testing with ±0.5%              │   │
│  │  Result: Mismatched labels, meaningless accuracy                │   │
│  │  Fix: ALWAYS keep both files in sync                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quick Reference Card for Mid-Aggressive Trader

```
┌─────────────────────────────────────────────────────────────────────────┐
│            QUICK REFERENCE: MID-AGGRESSIVE TRADER SETTINGS              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CRYPTOCURRENCY (BTC-USD, ETH-USD):                                     │
│    Normal:     BUY = +1.0%    SELL = -1.0%                              │
│    Holiday:    BUY = +0.5%    SELL = -0.5%                              │
│    News event: BUY = +1.5%    SELL = -1.5%                              │
│                                                                         │
│  VOLATILE STOCKS (TSLA, NVDA, AMD):                                     │
│    Normal:     BUY = +0.5%    SELL = -0.5%                              │
│    Holiday:    BUY = +0.25%   SELL = -0.25%                             │
│    News event: BUY = +0.8%    SELL = -0.8%                              │
│                                                                         │
│  LARGE CAP (AAPL, MSFT, GOOGL):                                         │
│    Normal:     BUY = +0.3%    SELL = -0.3%                              │
│    Holiday:    BUY = +0.15%   SELL = -0.15%                             │
│    News event: BUY = +0.5%    SELL = -0.5%                              │
│                                                                         │
│  INDEX FUNDS (SPY, QQQ, DIA):                                           │
│    Normal:     BUY = +0.2%    SELL = -0.2%                              │
│    Holiday:    BUY = +0.1%    SELL = -0.1%                              │
│    News event: BUY = +0.3%    SELL = -0.3%                              │
│                                                                         │
│  Remember: Always run train.py and check label distribution!            │
│  Target: 20-40% in each class (SELL, HOLD, BUY)                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Expected Usage Pattern (Real-Time Trading)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPECTED USAGE PATTERN                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PREPARATION PHASE (do this regularly):                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  DAILY:                                                         │   │
│  │  • Run download.py to get fresh news + prices                   │   │
│  │  • This accumulates more training data                          │   │
│  │                                                                 │   │
│  │  WEEKLY (or when performance drops):                            │   │
│  │  • Run train.py to retrain model on recent data                 │   │
│  │  • Run test.py to verify accuracy                               │   │
│  │  • This keeps the "brain" updated with recent patterns          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ACTIVE TRADING PHASE (real-time):                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. Monitor news feeds (Twitter, news sites, RSS)               │   │
│  │  2. When significant news breaks → immediately run:             │   │
│  │                                                                 │   │
│  │     result = model.predict("Price: $95000                       │   │
│  │                             Headline: SEC approves Bitcoin ETF  │   │
│  │                             Summary: ...")                      │   │
│  │                                                                 │   │
│  │  3. Get prediction for NEXT 5 MINUTES:                          │   │
│  │     → "BUY (82% confident) - Price will RISE. Consider buying." │   │
│  │                                                                 │   │
│  │  4. Act immediately if signal is strong                         │   │
│  │  5. Repeat for each news event                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ⚠️  This is NOT "set and forget" investing!                           │
│  ⚠️  This requires active monitoring and quick decision-making         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Comparison: Short-Term vs Long-Term Trading

| Aspect | This Tool (Short-Term) | Long-Term Investing |
|--------|------------------------|---------------------|
| **Time horizon** | 5 minutes | Months to years |
| **Decision frequency** | Multiple times per day | Rarely |
| **Threshold** | ±1% (small moves matter) | ±10-20% (only big moves matter) |
| **News reaction** | Immediate (within minutes) | Slow (days/weeks) |
| **Monitoring** | Active, real-time | Passive, periodic |
| **Trading style** | High-frequency, news-reactive | Buy-and-hold |
| **Risk profile** | Higher (many small trades) | Lower (few large positions) |

### Live Trading Implementation (Future)

To use this model for actual trading, you would need to connect it to a real-time data stream:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LIVE TRADING ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │  yfinance   │     │   Trained   │     │   Trading   │              │
│   │  WebSocket  │ ──► │    Model    │ ──► │   Decision  │              │
│   │  (prices)   │     │  .predict() │     │  BUY/SELL   │              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│         │                    ▲                                          │
│         │                    │                                          │
│         ▼                    │                                          │
│   ┌─────────────┐     ┌─────────────┐                                  │
│   │    News     │     │   Format    │                                  │
│   │    Feed     │ ──► │   Input     │                                  │
│   │  (Yahoo)    │     │   Text      │                                  │
│   └─────────────┘     └─────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

1. **Set up real-time price stream** using yfinance WebSocket:
   - Documentation: https://ranaroussi.github.io/yfinance/reference/yfinance.websocket.html

2. **Monitor for price signals** (significant price movements)

3. **When triggered, fetch latest news** (similar to `download.py`)

4. **Format input and run prediction**:
   ```python
   input_text = f"Price: {current_price}\nHeadline: {headline}\nSummary: {summary}"
   result = model.predict(input_text)
   if result['prediction'] == 'BUY' and result['confidence'] > 0.7:
       # Execute buy order
   ```

5. **Execute trade** via your broker's API

> ⚠️ **WARNING**: This is NOT financial advice. The model is NOT guaranteed to be profitable. Always test thoroughly with paper trading before using real money. Past performance does not guarantee future results.

### Summary: Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KEY TAKEAWAYS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. This is a SHORT-TERM (5-minute) trading tool                        │
│     • NOT for long-term investing                                       │
│     • Requires active, real-time monitoring                             │
│                                                                         │
│  2. The ±1% HOLD threshold is for 5-minute price changes                │
│     • At 5-min scale, most moves are <1% (hence HOLD is valid)          │
│     • Significant news causes >1% moves (BUY or SELL signals)           │
│                                                                         │
│  3. target-profit.png shows WHEN to use the tool                        │
│     • Pink circles = volatile periods = ideal conditions                │
│     • The time scale in the image is NOT the trading time scale         │
│                                                                         │
│  4. Adjust thresholds if you change time intervals                      │
│     • Longer intervals → larger thresholds                              │
│     • See "Threshold Best Practices" table above                        │
│                                                                         │
│  5. Expected workflow:                                                  │
│     • Prepare: download.py (daily), train.py (weekly)                   │
│     • Trade: model.predict() when news breaks, act within minutes       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Beginner's Guide: How This Project Works

This section explains the project in simple terms for those new to machine learning.

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

The robot's BRAIN after learning → gemma_transformer_classifier.pth
```

**Step 3: Test the Robot (test.py)**
```
Give the robot NEW recipes it hasn't seen.
See if it can predict: "Will this pizza be good?"
```

### What's Actually in Each File?

```
datasets/BTC-USD/
├── news_with_price.json              BTC-USD.pth
│   ┌───────────────────────────┐     ┌───────────────────────────────┐
│   │ News: "Bitcoin drops!"    │     │                               │
│   │ Price: $100,000           │     │   NOT the news articles       │
│   │ Future: $98,000           │ →   │   NOT the prices              │
│   │ Result: SELL              │     │                               │
│   │                           │     │   Just PATTERNS:              │
│   │ News: "Elon buys BTC!"    │     │   "scary words → sell"        │
│   │ Price: $95,000            │     │   "hype words → buy"          │
│   │ Future: $99,000           │     │   (stored as math numbers)    │
│   │ Result: BUY               │     │                               │
│   │                           │     │   ~500,000 learned parameters │
│   │ ... 100s more examples    │     │                               │
│   └───────────────────────────┘     └───────────────────────────────┘
│          THE TEXTBOOK                      THE TRAINED BRAIN
│         (raw examples)                    (learned patterns)
└── Both files live together in the same symbol folder!
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

Each symbol has its own folder with data AND model together:
    datasets/
    ├── BTC-USD/
    │   ├── news_with_price.json
    │   └── BTC-USD.pth
    ├── AAPL/
    │   ├── news_with_price.json
    │   └── AAPL.pth
    └── TSLA/
        ├── news_with_price.json
        └── TSLA.pth
```

### Understanding the .pth File

The `.pth` file is a PyTorch model checkpoint - a binary file containing all the learned weights and biases from training. Think of it as the "brain" of your trained model saved to disk.

**Key points:**
- It does NOT contain the original news articles or prices
- It contains ~500,000 numbers (weights) that represent learned patterns
- These numbers are adjusted during training to minimize prediction errors
- When you load the model, these numbers are restored so it "remembers" what it learned

**Analogy:**
- **Training** = studying for an exam, then writing notes
- **`.pth` file** = the notes
- **Testing** = reading your notes to answer questions

### The "Only Knows Bitcoin" Problem

The model is like a student who only attended one class:

```
┌──────────────────────────────────────────────────────────────┐
│  MODEL: "Bitcoin Brain"                                      │
│                                                              │
│  Training Data Seen:                                         │
│    ✅ Bitcoin news articles                                  │
│    ✅ Bitcoin price movements                                │
│    ✅ Bitcoin market patterns                                │
│    ❌ Apple Stock (never studied)                            │
│    ❌ Tesla Stock (never studied)                            │
│    ❌ Gold Futures (never studied)                           │
│                                                              │
│  If you ask about AAPL:                                      │
│    🤷 "I don't know... I only learned Bitcoin patterns"      │
└──────────────────────────────────────────────────────────────┘
```

The model learned Bitcoin-specific patterns like:
- "When news says 'regulation' → Bitcoin usually drops"
- "When news says 'adoption' → Bitcoin usually rises"

But these patterns might NOT work for Apple stock because:
- Apple reacts to different news (iPhone sales, not crypto regulation)
- Apple stock moves differently than Bitcoin
- The vocabulary and context are completely different

**To analyze a different symbol**, you must:
1. Change `SYMBOL` in all scripts
2. Run `download.py` to get new data
3. Run `train.py` to train a NEW model
4. The old `.pth` file only knows Bitcoin!

### Quick Reference: File Purposes

| File | Simple Explanation |
|------|-------------------|
| `download.py` | "Go to the library and get study materials" |
| `train.py` | "Study the materials and learn patterns" |
| `test.py` | "Take a quiz to see how much was learned" |
| `*.json` files | "The textbooks with raw information" |
| `*.pth` file | "The brain after studying (learned patterns)" |

### How the .pth File Is Created (The Brain Pipeline)

The `.pth` file is the trained model - the "brain" that contains all learned patterns. Here's exactly how it's created:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CREATING THE .pth FILE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: download.py                                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Creates: datasets/BTC-USD/news_with_price.json                   │ │
│  │  (The training data - news + prices + labels)                     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│  STEP 2: train.py                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  1. Loads the JSON data                                           │ │
│  │  2. Creates a new empty model (random weights)                    │ │
│  │  3. Trains the model (adjusts weights)                            │ │
│  │  4. SAVES the trained weights ← THIS CREATES THE .pth FILE       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│  OUTPUT: gemma_transformer_classifier.pth                               │
│  (The trained "brain" - ~2.6 MB of learned patterns)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**The exact code that creates it** (in `train.py`, line 200-201):

```python
def save_model(model):
    """Save model weights to disk."""
    torch.save(model.state_dict(), MODEL_FILE)  # ← THIS LINE CREATES THE .pth FILE
    print(f"\nModel saved to {MODEL_FILE}")
```

### Visual Timeline of .pth Creation

```
TIME ──────────────────────────────────────────────────────────────────────►

download.py                    train.py
    │                              │
    │  Fetch data                  │  Load JSON
    │  from Yahoo                  │      │
    │      │                       │      ▼
    │      ▼                       │  Create empty model
    │  Merge news                  │  (random weights)
    │  + prices                    │      │
    │      │                       │      ▼
    │      ▼                       │  Train 20 epochs
    │  Calculate                   │  (adjust weights)
    │  labels                      │      │
    │      │                       │      ▼
    │      ▼                       │  ┌─────────────────────────┐
    │  ┌────────────────────┐      │  │ torch.save(             │
    │  │ Save JSON file     │      │  │   model.state_dict(),   │
    │  │                    │      │  │   'gemma_transformer_   │
    │  │ BTC-USD_news_      │ ───► │  │    classifier.pth'      │
    │  │ with_price.json    │      │  │ )                       │
    │  └────────────────────┘      │  └─────────────────────────┘
    │                              │              │
    ▼                              ▼              ▼

                               gemma_transformer_classifier.pth
                               (THE TRAINED BRAIN IS BORN!)
```

### File Creation Reference

| Script | Creates | Used By |
|--------|---------|---------|
| `download.py` | `datasets/{SYMBOL}/news_with_price.json` | `train.py` |
| `train.py` | `gemma_transformer_classifier.pth` | `test.py` |
| `test.py` | Nothing (just reads and evaluates) | - |

### If You Delete the .pth File

If you delete `gemma_transformer_classifier.pth`, you can recreate it:

```bash
# Option 1: Full pipeline (fresh data + new model)
python download.py    # Get latest news/prices
python train.py       # Train model → creates NEW .pth file

# Option 2: Just retrain (if JSON data already exists)
python train.py       # Train model → creates NEW .pth file
```

**Note**: The new `.pth` file will have different weights since training starts from random initialization. Results may vary slightly.

### Continuous Learning (The Student's Notebook)

By default, `train.py` now **continues from existing knowledge** rather than starting from scratch. Think of it like a student who keeps their notes from previous classes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│               CONTINUOUS LEARNING (Default Behavior)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  THE SMART STUDENT (--fresh NOT used, default):                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │  Week 1 (Monday):                                                 │ │
│  │    Student attends class → learns patterns A, B, C               │ │
│  │    Teacher says "Save your notes!" → saves to notebook           │ │
│  │    📓 Notebook contains: A, B, C                                  │ │
│  │                                                                   │ │
│  │  Week 1 (Wednesday):                                              │ │
│  │    Student READS OLD NOTES first → remembers A, B, C             │ │
│  │    Then attends new class → learns D, E, F                       │ │
│  │    Updates notebook → A, B, C, D, E, F                           │ │
│  │    📓 Notebook now contains: A, B, C, D, E, F                     │ │
│  │                                                                   │ │
│  │  Week 2 (Monday):                                                 │ │
│  │    Student READS OLD NOTES → remembers A through F               │ │
│  │    Attends new class → learns G, H, I                            │ │
│  │    Updates notebook → A through I                                │ │
│  │    📓 Notebook now contains: A, B, C, D, E, F, G, H, I            │ │
│  │                                                                   │ │
│  │  ✓ Result: Student accumulates knowledge over time!              │ │
│  │  ✓ Each training session BUILDS on previous learning             │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  THE FORGETFUL STUDENT (--fresh flag used):                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │  Week 1 (Monday):                                                 │ │
│  │    Student learns A, B, C → saves to notebook                    │ │
│  │    📓 Notebook contains: A, B, C                                  │ │
│  │                                                                   │ │
│  │  Week 1 (Wednesday):                                              │ │
│  │    Student THROWS AWAY old notebook! Starts with blank pages     │ │
│  │    Learns D, E, F from scratch                                   │ │
│  │    📓 Notebook contains: D, E, F (lost A, B, C!)                  │ │
│  │                                                                   │ │
│  │  Week 2 (Monday):                                                 │ │
│  │    Student THROWS AWAY notebook again!                           │ │
│  │    Learns G, H, I from scratch                                   │ │
│  │    📓 Notebook contains: G, H, I (lost everything else!)         │ │
│  │                                                                   │ │
│  │  ❌ Result: Student only knows the LAST thing learned            │ │
│  │  ❌ All previous learning is LOST                                │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### How to Use Continuous Learning

```bash
# DEFAULT: Continue from existing knowledge (RECOMMENDED)
# The student reads their old notes, then learns more
python train.py -s BTC-USD

# FRESH START: Throw away old notes, start from scratch
# Only use when you REALLY want to forget everything
python train.py -s BTC-USD --fresh
```

#### When to Use Each Mode

| Scenario | Command | Why |
|----------|---------|-----|
| **Daily/Weekly retraining** | `python train.py -s BTC-USD` | Keep building on learned patterns |
| **New data available** | `python train.py -s BTC-USD` | Add new knowledge to existing |
| **Performance dropping** | `python train.py -s BTC-USD` | Reinforce with fresh examples |
| **Completely new strategy** | `python train.py -s BTC-USD --fresh` | Start over with different approach |
| **Model seems corrupted** | `python train.py -s BTC-USD --fresh` | Reset to clean state |
| **Changing architecture** | `python train.py -s BTC-USD --fresh` | New hidden_dim/layers needs fresh weights |

#### Console Output: What to Expect

When you run `train.py`, you'll see one of three messages in the "MODEL INITIALIZED" section:

```
CONTINUING from existing model:
┌─────────────────────────────────────────────────────────────────────────┐
│  ✓ CONTINUING from existing model!                                      │
│    Loaded weights from: datasets/BTC-USD/BTC-USD.pth                    │
│    (The student is reading their old notes before learning more)        │
└─────────────────────────────────────────────────────────────────────────┘

FRESH START (--fresh flag used):
┌─────────────────────────────────────────────────────────────────────────┐
│  ✗ FRESH START (--fresh flag)                                           │
│    Training from scratch with random weights                            │
│    (The student is starting with a blank notebook)                      │
└─────────────────────────────────────────────────────────────────────────┘

NEW MODEL (no existing .pth file found):
┌─────────────────────────────────────────────────────────────────────────┐
│  ○ NEW MODEL (no existing model found)                                  │
│    Will save to: datasets/BTC-USD/BTC-USD.pth                           │
│    (First day of class - student has no previous notes)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

#### The Safe Default Philosophy

We made continuous learning the **default** because:

1. **Protects your investment**: Training takes time and compute resources
2. **Accumulates knowledge**: Each session builds on previous learning
3. **Prevents accidents**: You won't accidentally overwrite a well-trained model
4. **Mimics real learning**: Just like humans, AI should build on existing knowledge

If you truly want to start fresh:
- Use the `--fresh` flag explicitly
- Or delete the `.pth` file manually

This design ensures you never accidentally lose a trained model!

### Smart Training Guard (Daemon-Safe Automation)

The Smart Training Guard prevents overfitting when running automated training processes. It detects if training is actually needed before proceeding.

#### The Problem: Over-Studying the Same Material

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE OVER-STUDYING PROBLEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Your daemon runs every hour:                                           │
│                                                                         │
│  Hour 1:  download → train → ✓ New data, productive training            │
│  Hour 2:  download → train → ⚠️ Same data, wasted compute               │
│  Hour 3:  download → train → ⚠️ Same data, starting to overfit          │
│  Hour 4:  download → train → ❌ Same data, overfitting!                 │
│  ...                                                                    │
│  Hour 24: download → train → ❌ Model ruined, memorized everything      │
│                                                                         │
│  Without Smart Guard, the model keeps "re-reading the same textbook"    │
│  until it memorizes answers instead of understanding patterns.          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### The Solution: Teacher Checks for New Material

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SMART TRAINING GUARD                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Think of it like a teacher who checks before starting class:           │
│                                                                         │
│  Teacher arrives at classroom:                                          │
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

#### Guard Checks Explained

| Check | What It Does | Default | Priority | School Analogy |
|-------|--------------|---------|----------|----------------|
| **Data Hash** | SHA256 fingerprint of training data | - | **PRIMARY** | "Is this a new textbook or the same one?" |
| **Min New Samples** | Require X new data points | 5 | Secondary | "Are there enough new chapters to teach?" |
| **Cooldown** | Wait N minutes between sessions | 5 min | Tertiary | "Did students have time to rest and absorb?" |

**Important**: Defaults are tuned for 5-minute trading intervals (`PRICE_INTERVAL="5m"` in download.py).

#### Guard Priority: Why Data Hash is King

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GUARD PRIORITY HIERARCHY                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PRIORITY 1: Data Hash (PRIMARY - Most Important)                       │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Question: "Is this the EXACT SAME data as last training?"        │ │
│  │                                                                   │ │
│  │  • SHA256 hash of all training data content                       │ │
│  │  • If SAME hash → SKIP (no new information to learn)              │ │
│  │  • If DIFFERENT hash → proceed to next check                      │ │
│  │                                                                   │ │
│  │  This catches 90%+ of redundant training attempts!                │ │
│  │  Even if cooldown passed, same data = no training.                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│  PRIORITY 2: Cooldown (Secondary)                                       │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Question: "Has enough time passed since last training?"          │ │
│  │                                                                   │ │
│  │  • Prevents rapid-fire training even with new data                │ │
│  │  • Gives model time to "settle" between sessions                  │ │
│  │  • Should match your data interval (5min data → 5min cooldown)    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│  PRIORITY 3: Min New Samples (Tertiary)                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Question: "Is there ENOUGH new data to be meaningful?"           │ │
│  │                                                                   │ │
│  │  • Even if data changed, is it worth training?                    │ │
│  │  • 1-2 new samples = probably noise, skip                         │ │
│  │  • 5+ new samples = meaningful update, train                      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│  ALL CHECKS PASSED → TRAINING APPROVED ✓                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Tuning Guards for Different Data Intervals

The default guards are optimized for 5-minute trading. If you change `PRICE_INTERVAL` in `download.py`, adjust your guards accordingly:

| Data Interval | Cooldown | Min New Samples | Use Case |
|---------------|----------|-----------------|----------|
| 1 minute | `--cooldown 1` | `--min-new-samples 3` | Ultra high-frequency |
| **5 minutes** | **`--cooldown 5`** | **`--min-new-samples 5`** | **Default (recommended)** |
| 15 minutes | `--cooldown 15` | `--min-new-samples 5` | Medium frequency |
| 30 minutes | `--cooldown 30` | `--min-new-samples 5` | Lower frequency |
| 1 hour | `--cooldown 60` | `--min-new-samples 10` | Hourly updates |
| 1 day | `--cooldown 1440` | `--min-new-samples 20` | Daily retraining |

**Rule of Thumb**: Cooldown should match your data interval. If data comes every 5 minutes, cooldown should be 5 minutes.

#### Tuning Guards for Different Trading Styles

| Trading Style | Settings | Rationale |
|---------------|----------|-----------|
| **Aggressive (High-Frequency)** | `--cooldown 5 --min-new-samples 3` | Train often, capture every pattern |
| **Balanced (Recommended)** | `--cooldown 5 --min-new-samples 5` | Good balance of freshness and stability |
| **Conservative** | `--cooldown 15 --min-new-samples 10` | Train less often, more stable model |
| **Very Conservative** | `--cooldown 60 --min-new-samples 20` | Hourly training, very stable |

#### Example: 5-Minute Trading Daemon

Here's what happens when you run a daemon every 5 minutes with the default settings:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              5-MINUTE DAEMON WITH DEFAULT GUARDS                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TIME    NEWS    DATA HASH    COOLDOWN    MIN SAMPLES    RESULT        │
│  ─────   ─────   ──────────   ────────    ───────────    ──────        │
│  9:00    15      Changed      Passed      15 ≥ 5         ✓ TRAIN       │
│  9:05    8       Changed      Passed      8 ≥ 5          ✓ TRAIN       │
│  9:10    0       SAME         -           -              ⏭️ SKIP (hash) │
│  9:15    12      Changed      Passed      12 ≥ 5         ✓ TRAIN       │
│  9:20    3       Changed      Passed      3 < 5          ⏭️ SKIP (min)  │
│  9:25    20      Changed      Passed      20 ≥ 5         ✓ TRAIN       │
│  9:30    0       SAME         -           -              ⏭️ SKIP (hash) │
│  9:35    7       Changed      Passed      7 ≥ 5          ✓ TRAIN       │
│                                                                         │
│  Result: 5 training sessions in 35 minutes (when meaningful)            │
│  Skipped: 3 times (2x same data, 1x not enough samples)                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Weekend/Low Activity Handling

During weekends or low-activity periods, the Data Hash guard automatically handles this:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              WEEKEND SCENARIO (Bitcoin - Low Activity)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Saturday:                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  12:00 - download → 2 new articles  → train (first of day)       │ │
│  │  12:05 - download → 0 new articles  → SKIP (hash unchanged)       │ │
│  │  12:10 - download → 0 new articles  → SKIP (hash unchanged)       │ │
│  │  ...                                                              │ │
│  │  15:00 - download → 0 new articles  → SKIP (hash unchanged)       │ │
│  │  15:05 - download → 3 new articles  → SKIP (< 5 min samples)      │ │
│  │  15:10 - download → 5 new articles  → train (enough new data)     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  The daemon runs every 5 minutes but only trains TWICE all day!         │
│  No manual intervention needed - Smart Guard handles it automatically.  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### How to Use Smart Guard

```bash
# Normal run (Smart Guard enabled by default)
python train.py -s BTC-USD
# → Checks if training is needed, skips if not

# Force training (bypass all guards)
python train.py -s BTC-USD --force
# → Always trains, ignores guards

# Custom thresholds
python train.py -s BTC-USD --min-new-samples 20 --cooldown 120
# → Require 20 new samples, 2 hour cooldown
```

#### Console Output Examples

```
TRAINING APPROVED:
┌─────────────────────────────────────────────────────────────────────────┐
│  SMART TRAINING GUARD                                                   │
│  ------------------------------------------------------------           │
│  ✓ TRAINING APPROVED                                                    │
│    Reason: New data detected - training approved                        │
│    new_samples: 24                                                      │
│    data_hash: changed                                                   │
│    cooldown: passed                                                     │
└─────────────────────────────────────────────────────────────────────────┘

TRAINING SKIPPED (data unchanged):
┌─────────────────────────────────────────────────────────────────────────┐
│  SMART TRAINING GUARD                                                   │
│  ------------------------------------------------------------           │
│  ⏭️  TRAINING SKIPPED                                                   │
│    Reason: Data unchanged since last training                           │
│    last_trained: 2025-01-04T15:30:00+00:00                              │
│    sample_count: 156                                                    │
│    data_hash: unchanged                                                 │
│                                                                         │
│  To force training anyway: python train.py -s SYMBOL --force            │
└─────────────────────────────────────────────────────────────────────────┘

TRAINING SKIPPED (cooldown):
┌─────────────────────────────────────────────────────────────────────────┐
│  SMART TRAINING GUARD                                                   │
│  ------------------------------------------------------------           │
│  ⏭️  TRAINING SKIPPED                                                   │
│    Reason: Cooldown not met (45min < 60min)                             │
│    last_trained: 2025-01-04T15:30:00+00:00                              │
│    minutes_since: 45.2                                                  │
│    cooldown_required: 60                                                │
│                                                                         │
│  To force training anyway: python train.py -s SYMBOL --force            │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Training Metadata File

Smart Guard stores training history in `datasets/{SYMBOL}/.training_meta.json`:

```json
{
  "last_training": {
    "timestamp": "2025-01-04T15:30:00+00:00",
    "data_hash": "sha256:a1b2c3d4...",
    "sample_count": 156,
    "epochs": 20,
    "accuracy": 0.72,
    "f1_score": 0.68
  },
  "training_history": [
    {"timestamp": "2025-01-04T15:30:00", "samples": 156, "accuracy": 0.72},
    {"timestamp": "2025-01-03T10:00:00", "samples": 142, "accuracy": 0.70}
  ]
}
```

#### Daemon-Safe Automation

Smart Guard makes `train.py` safe for cron jobs and background daemons:

```bash
#!/bin/bash
# daemon_train.sh - Safe to run every hour via cron

# Download fresh data (always safe to run)
python download.py -s BTC-USD

# Train ONLY if needed (Smart Guard handles the logic)
python train.py -s BTC-USD

# The script:
# - Trains if new data is available AND cooldown passed
# - Skips gracefully if no changes detected
# - Exits with code 0 either way (daemon-friendly)
```

**Crontab Example:**

```bash
# Run every hour - Smart Guard prevents over-training
0 * * * * cd /path/to/project && python download.py -s BTC-USD && python train.py -s BTC-USD >> /var/log/btc_training.log 2>&1
```

#### When to Use --force

| Scenario | Use --force? | Why |
|----------|--------------|-----|
| Regular automated training | No | Let Smart Guard protect you |
| Testing new hyperparameters | Yes | Need to train regardless of data |
| After fixing a bug in code | Yes | Want fresh training with fix |
| Model seems wrong | Yes | Need to retrain to verify |
| Debugging training issues | Yes | Need to see full training output |

### Production Retraining Guide

In production, **retraining is expected and necessary**. Models get stale as markets change.

**Why Models Get Stale:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WHY MODELS GET STALE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  JANUARY 2025:                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Model trained on: Oct-Dec 2024 data                              │ │
│  │  Learned pattern: "ETF news → price goes UP"                      │ │
│  │  Result: Works great! ✅                                          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  JUNE 2025:                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Same model, but market has changed:                              │ │
│  │  - ETF hype is old news now                                       │ │
│  │  - New patterns: "Regulation news" matters more                   │ │
│  │  - Old patterns no longer work                                    │ │
│  │  Result: Model predictions are wrong! ❌                          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  SOLUTION: Retrain with fresh data!                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Production Retraining Workflow (with Continuous Learning):**

```
┌─────────────────────────────────────────────────────────────────────────┐
│           TYPICAL PRODUCTION WORKFLOW (Continuous Learning)             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   DAILY/WEEKLY:                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  python download.py -s BTC-USD   # Get fresh news + prices      │  │
│   │                                   # Cleans old data, fetches new │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│   WEEKLY/MONTHLY (or when performance drops):                           │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  python train.py -s BTC-USD   # Continue training!              │  │
│   │                                # LOADS existing .pth first      │  │
│   │                                # Learns from new data           │  │
│   │                                # Saves updated knowledge        │  │
│   │                                                                 │  │
│   │  ✓ Model gets SMARTER over time                                 │  │
│   │  ✓ Old patterns are PRESERVED                                   │  │
│   │  ✓ New patterns are ADDED                                       │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│   AFTER RETRAINING:                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  python test.py -s BTC-USD   # Verify model still works         │  │
│   │                               # Check accuracy didn't drop      │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**When to Retrain:**

| Trigger | Why | How Often |
|---------|-----|-----------|
| **Scheduled** | Keep model fresh with recent patterns | Weekly or Monthly |
| **Performance drop** | Model accuracy falling below threshold | When detected |
| **Market regime change** | Bull → Bear market, major events | As needed |
| **New data accumulated** | Enough new training examples | When data doubles |

**Production Best Practices:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BEST PRACTICES                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. BACKUP BEFORE RETRAINING                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  # Before retraining, save the old model                          │ │
│  │  cp gemma_transformer_classifier.pth backups/model_2025_01_03.pth │ │
│  │                                                                   │ │
│  │  # Then retrain                                                   │ │
│  │  python train.py                                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  2. VERSION YOUR MODELS                                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  backups/                                                         │ │
│  │  ├── model_2025_01_01.pth  (accuracy: 72%)                        │ │
│  │  ├── model_2025_01_15.pth  (accuracy: 75%)                        │ │
│  │  └── model_2025_02_01.pth  (accuracy: 71%) ← rollback candidate   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  3. VALIDATE BEFORE DEPLOYING                                           │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  # After retraining, always test                                  │ │
│  │  python test.py                                                   │ │
│  │                                                                   │ │
│  │  # If accuracy drops significantly, rollback:                     │ │
│  │  cp backups/model_2025_01_15.pth gemma_transformer_classifier.pth │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  4. MONITOR PERFORMANCE                                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Track these metrics over time:                                   │ │
│  │  - Accuracy (% correct predictions)                               │ │
│  │  - F1 Score (balance of precision/recall)                         │ │
│  │  - Profit/Loss if using for real trading                          │ │
│  │                                                                   │ │
│  │  If metrics decline → time to retrain!                            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Simple Production Script Example:**

```bash
#!/bin/bash
# retrain.sh - Weekly retraining script for a specific symbol

SYMBOL=${1:-BTC-USD}  # Default to BTC-USD if no argument
DATE=$(date +%Y_%m_%d)
MODEL_FILE="datasets/${SYMBOL}/${SYMBOL}.pth"

echo "=== Starting weekly retrain: $SYMBOL @ $DATE ==="

# 1. Backup current model
echo "Backing up current model..."
mkdir -p datasets/${SYMBOL}/backups
cp $MODEL_FILE datasets/${SYMBOL}/backups/${SYMBOL}_$DATE.pth

# 2. Download fresh data
echo "Downloading fresh data..."
python download.py -s $SYMBOL

# 3. Retrain
echo "Retraining model..."
python train.py -s $SYMBOL

# 4. Test new model
echo "Testing new model..."
python test.py

echo "=== Retrain complete: $MODEL_FILE ==="

# Usage:
#   ./retrain.sh          # Retrains BTC-USD
#   ./retrain.sh AAPL     # Retrains AAPL
#   ./retrain.sh ETH-USD  # Retrains ETH-USD
```

**The Model Lifecycle:**

```
     ┌──────────────────────────────────────────────────────────────┐
     │                                                              │
     ▼                                                              │
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐           │
│ COLLECT │ → │  TRAIN  │ → │  TEST   │ → │ DEPLOY  │           │
│  DATA   │    │  MODEL  │    │  MODEL  │    │  MODEL  │           │
└─────────┘    └─────────┘    └─────────┘    └────┬────┘           │
                                                  │                │
                                                  ▼                │
                                           ┌─────────────┐         │
                                           │  MONITOR    │         │
                                           │ PERFORMANCE │         │
                                           └──────┬──────┘         │
                                                  │                │
                                    ┌─────────────┴─────────────┐  │
                                    ▼                           ▼  │
                              Performance OK?            Performance Bad?
                                    │                           │  │
                                    ▼                           │  │
                               Keep using                       │  │
                                                                └──┘
                                                          (Retrain!)
```

### Understanding Model Queries (Classification vs Chat Models)

This is a **classification model**, NOT a chat model like Claude or GPT. They work differently:

```
CHAT MODEL (Claude, GPT)              CLASSIFICATION MODEL (This Project)
┌─────────────────────────┐           ┌─────────────────────────┐
│ System: "You are a chef"│           │                         │
│ User: "How do I cook    │           │ Input: "Bitcoin drops   │
│        eggs?"           │  →        │         on Fed news"    │
│                         │           │                         │
│ Assistant: "First,      │           │ Output: [0.8, 0.1, 0.1] │
│ crack the eggs into..." │           │         SELL  HOLD BUY  │
│ (generates text)        │           │         (just numbers)  │
└─────────────────────────┘           └─────────────────────────┘
     CONVERSATION                          PREDICTION
```

**This model doesn't "answer questions" - it predicts one of three labels: SELL, HOLD, or BUY.**

### Where is the "Query"?

The "query" is hidden in this one line:

```python
logits = model(input_text)
```

Here's what happens step by step:

```
YOUR "QUERY":
input_text = "Price: 95000\nHeadline: Bitcoin crashes\nSummary: ..."

         ↓ model(input_text)

INSIDE THE MODEL:
1. Gemma converts text → 1024 numbers (embedding)
2. Numbers go through Transformer layers
3. Classifier outputs 3 numbers

         ↓

MODEL'S "ANSWER":
logits = tensor([2.1, 0.3, -1.2])  ← raw scores for [SELL, HOLD, BUY]
probs  = tensor([0.75, 0.20, 0.05]) ← after softmax (percentages)

INTERPRETATION:
"75% confident this is a SELL signal"
```

### The evaluate() Function Explained Simply

```python
def evaluate(model, test_features, test_labels):
```
**"Teacher is going to quiz the student (model) and count correct answers"**

```python
    model.eval()
```
**"Tell the student: This is a TEST, not practice. Be serious!"**
(Disables training behaviors like dropout)

```python
    with torch.no_grad():
```
**"Don't write notes during the test - just answer questions"**
(Saves memory by not tracking calculations for learning)

```python
        for i in range(len(test_features)):
            input_text = [test_features[i]]
```
**"For each question on the test..."**

```python
            logits = model(input_text)  # ← THIS IS THE "QUERY"!
```
**"Ask the student the question and get their answer"**

```python
            probs = logits.softmax(dim=-1)
```
**"Convert raw scores to percentages (must add up to 100%)"**

```python
            predicted = torch.argmax(probs, dim=-1)
```
**"Which answer has the highest percentage? That's the final answer"**
- If probs = [0.75, 0.20, 0.05] → argmax = 0 (SELL)
- If probs = [0.10, 0.30, 0.60] → argmax = 2 (BUY)

```python
            if predicted.item() == actual.item():
                correct += 1
```
**"Did the student get it right? Add to their score!"**

```python
    accuracy = correct / total
```
**"Final grade: 80 correct out of 100 = 80%"**

### PyTorch Functions Reference

Key PyTorch functions used in this project and what they do:

| Function | What It Does | Simple Analogy |
|----------|--------------|----------------|
| `model(input)` | Pass input through neural network, get output | "Ask the student a question" |
| `model.eval()` | Switch to evaluation mode (no learning) | "This is a test, not practice" |
| `model.train()` | Switch to training mode (learning enabled) | "This is practice, learn from mistakes" |
| `torch.no_grad()` | Don't track calculations (saves memory) | "Don't take notes, just answer" |
| `torch.tensor([...])` | Create a tensor (multi-dimensional array) | "Put numbers in a special box" |
| `.softmax(dim=-1)` | Convert scores to probabilities (sum to 1.0) | "Turn scores into percentages" |
| `torch.argmax(...)` | Find index of largest value | "Which answer is highest?" |
| `.item()` | Extract single Python number from tensor | "Take the number out of the box" |
| `.to(device)` | Move tensor to CPU/GPU | "Move data to the calculator" |
| `.cpu()` | Move tensor to CPU | "Bring data back from GPU" |
| `.float()` | Convert to floating-point numbers | "Use decimal numbers" |
| `torch.load(...)` | Load saved model from file | "Load the brain from disk" |
| `torch.save(...)` | Save model to file | "Save the brain to disk" |
| `.state_dict()` | Get all model weights as dictionary | "Export all learned patterns" |
| `.load_state_dict()` | Load weights into model | "Import learned patterns" |
| `.backward()` | Calculate gradients (for learning) | "Figure out what to improve" |
| `.zero_grad()` | Reset gradients to zero | "Forget previous improvements" |
| `optimizer.step()` | Update weights based on gradients | "Apply the improvements" |

### How to Query Your Model (Practical Example)

Here's how you would "ask" your trained model about new data:

```python
import torch
from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier

# Load the trained model (the "brain")
model = SimpleGemmaTransformerClassifier()
model.load_state_dict(torch.load('gemma_transformer_classifier.pth', weights_only=True))
model.eval()  # Set to test mode

# Your "query" - the news you want to analyze
my_question = """Price: 95000
Headline: Bitcoin crashes as SEC announces new regulations
Summary: The SEC has announced strict new regulations on cryptocurrency."""

# Ask the model (THIS IS THE QUERY!)
with torch.no_grad():
    logits = model([my_question])
    probs = logits.softmax(dim=-1).cpu()

# Interpret the answer
labels = ["SELL", "HOLD", "BUY"]
predicted_index = torch.argmax(probs).item()
confidence = probs[0][predicted_index].item() * 100

print(f"Prediction: {labels[predicted_index]}")
print(f"Confidence: {confidence:.1f}%")
print(f"All probabilities: SELL={probs[0][0]:.2f}, HOLD={probs[0][1]:.2f}, BUY={probs[0][2]:.2f}")
```

**Example Output:**
```
Prediction: SELL
Confidence: 78.5%
All probabilities: SELL=0.78, HOLD=0.15, BUY=0.07
```

### The Cookbook Analogy for Queries

If your model was trained on a cookbook:

```
CHAT MODEL (Claude):
Q: "How do I cook eggs?"
A: "First, crack the eggs into a bowl. Heat a pan with butter..."

YOUR CLASSIFICATION MODEL:
Q: "Eggs, butter, pan, medium heat, 3 minutes"
A: [0.05, 0.15, 0.80]  →  "This will likely taste GOOD"
   [BAD]  [OK]  [GOOD]

It doesn't EXPLAIN anything - it just PREDICTS a category!
```

---

## Commands

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# HuggingFace authentication (required for Gemma model)
huggingface-cli login

# Full workflow for BTC-USD
python download.py -s BTC-USD
python train.py -s BTC-USD
python test.py
```

### Command-Line Interface (CLI)

All scripts now support CLI arguments for production-ready usage. Run any script with `--help` for full documentation.

#### download.py - Data Collection

```bash
# Basic usage (required: symbol)
python download.py -s BTC-USD

# Custom period and interval
python download.py -s BTC-USD -p 3mo -i 15m

# More news articles
python download.py -s AAPL -p 1mo -i 5m -n 500

# Show full help
python download.py --help
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--symbol` | `-s` | *required* | Trading symbol (e.g., BTC-USD, AAPL, TSLA) |
| `--period` | `-p` | `1mo` | How far back to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, max) |
| `--interval` | `-i` | `5m` | Price candle interval (1m, 2m, 5m, 15m, 30m, 1h, 1d) |
| `--news-count` | `-n` | `100` | Number of news articles to fetch |

**Intraday Data Limits:**

| Interval | Maximum Period | Example |
|----------|----------------|---------|
| 1m | 7 days | `-i 1m -p 5d` |
| 2m, 5m, 15m, 30m | 60 days | `-i 5m -p 1mo` |
| 60m, 90m, 1h | 730 days | `-i 1h -p 1y` |
| 1d and above | Unlimited | `-i 1d -p max` |

#### train.py - Model Training

```bash
# Basic usage (required: symbol)
python train.py -s BTC-USD

# Custom thresholds for less volatile stocks
python train.py -s AAPL -b 0.3 --sell-threshold -0.3

# More epochs and different optimizer
python train.py -s TSLA -e 50 -o AdamW

# Full customization
python train.py -s ETH-USD -b 1.5 --sell-threshold -1.5 -e 100 -l 0.001 -o AdamW

# Show full help
python train.py --help
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--symbol` | `-s` | *required* | Trading symbol to train on |
| `--buy-threshold` | `-b` | `1.0` | Price increase % to trigger BUY |
| `--sell-threshold` | | `-1.0` | Price decrease % to trigger SELL |
| `--epochs` | `-e` | `20` | Number of training epochs |
| `--learning-rate` | `-l` | `0.005` | Optimizer learning rate |
| `--optimizer` | `-o` | `SGD` | Optimizer type (SGD or AdamW) |
| `--batch-size` | | `1` | Samples per gradient update |
| `--split` | | `0.8` | Train/test split ratio |
| `--model-file` | | `gemma_transformer_classifier.pth` | Output model file |
| `--hidden-dim` | | `256` | Model internal dimension (256=~2.6MB, 512=~10MB) |
| `--num-layers` | | `2` | Transformer layers (2=fast, 4=thorough) |

#### test.py - Model Evaluation

```bash
# Test with saved model (uses hardcoded config - must match train.py)
python test.py
```

**Note**: `test.py` still uses hardcoded configuration. Ensure its thresholds match `train.py`.

---

## Configuration Reference

All scripts use **CLI arguments** for configuration. Run any script with `--help` for full documentation. Default values are used when arguments are not provided.

### download.py - Data Collection

Parameters are passed via CLI arguments (see [Commands > download.py](#downloadpy---data-collection) for usage).

**Output Files:**

| File | Description |
|------|-------------|
| `datasets/{SYMBOL}/historical_data.json` | Price data (timestamp → price) |
| `datasets/{SYMBOL}/news.json` | Raw news articles |
| `datasets/{SYMBOL}/news_with_price.json` | Merged training data with labels |

#### Selective Deletion Pattern (Safe Re-downloads)

When you run `download.py`, it uses **selective deletion** to ensure fresh data while protecting your trained models:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SELECTIVE DELETION PATTERN                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  datasets/BTC-USD/                                                      │
│  ├── historical_data.json    ← DELETED (regenerable)                    │
│  ├── news.json               ← DELETED (regenerable)                    │
│  ├── news_with_price.json    ← DELETED (regenerable)                    │
│  └── BTC-USD.pth             ★ PRESERVED (expensive to recreate)        │
│                                                                         │
│  WHY THIS MATTERS:                                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  • Data files can be re-downloaded in seconds                     │ │
│  │  • Model files (.pth) take HOURS of training to create            │ │
│  │  • You can run download.py multiple times per day safely          │ │
│  │  • Your trained models are NEVER deleted                          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Example Output:**

```
[2/4] PREPARING FRESH DIRECTORY
------------------------------------------------------------
  Cleaning data files (preserving trained models)...
    ✓ Removed: historical_data.json
    ✓ Removed: news.json
    ✓ Removed: news_with_price.json
    ★ Preserved: BTC-USD.pth (trained model)
  ✓ Cleaned 3 data file(s), preserved 1 model(s)
```

**File Categories:**

| Category | Files | Action | Reason |
|----------|-------|--------|--------|
| **Data files** | `*.json` | DELETE | Can re-download in seconds |
| **Model files** | `*.pth` | PRESERVE | Hours of training to recreate |

**Safe Workflow for Repeated Downloads:**

```bash
# Morning: Download fresh data
python download.py -s BTC-USD

# Train your model (takes time)
python train.py -s BTC-USD -e 200 ...

# Afternoon: Want fresh data again?
python download.py -s BTC-USD    # ← SAFE! Model is preserved

# Your BTC-USD.pth is still there!
python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1
```

This pattern ensures you can refresh your market data as often as needed without ever losing your trained models.

#### How Download Parameters Work Together (The Story)

Imagine you're a detective trying to figure out: **"Does news affect Bitcoin's price?"**

**Step 1: Gather Price Evidence**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PRICE_PERIOD = "1mo"                                                   │
│  "Go back 1 month and collect ALL price data"                           │
│                                                                         │
│  Options:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "1d"  = Last 1 day      (very recent, little data)             │   │
│  │  "5d"  = Last 5 days     (recent week)                          │   │
│  │  "1mo" = Last 1 month    ← DEFAULT (good balance)               │   │
│  │  "3mo" = Last 3 months   (more patterns, but older)             │   │
│  │  "1y"  = Last 1 year     (lots of data, but old patterns)       │   │
│  │  "max" = All available   (maximum data)                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PRICE_INTERVAL = "5m"                                                  │
│  "Record the price every 5 minutes"                                     │
│                                                                         │
│  Think of it like taking photos:                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "1m"  = Photo every 1 minute   (very detailed, lots of data)   │   │
│  │  "5m"  = Photo every 5 minutes  ← DEFAULT (good for news)       │   │
│  │  "15m" = Photo every 15 minutes (less detail)                   │   │
│  │  "1h"  = Photo every 1 hour     (big picture only)              │   │
│  │  "1d"  = Photo every 1 day      (daily trends only)             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Result: A timeline of prices like a heartbeat monitor                  │
│                                                                         │
│  TIME:   9:00   9:05   9:10   9:15   9:20   9:25   9:30                │
│  PRICE:  $100K  $101K  $99K   $98K   $102K  $103K  $101K               │
│          ──●──────●──────●──────●──────●──────●──────●──                │
└─────────────────────────────────────────────────────────────────────────┘
```

**Step 2: Gather News Evidence**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  NEWS_COUNT = 1000                                                      │
│  "Collect up to 1000 news articles about Bitcoin"                       │
│                                                                         │
│  Each article has a timestamp:                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Article 1: "Bitcoin surges on ETF news"     @ 9:07:23 AM       │   │
│  │  Article 2: "Regulation concerns grow"       @ 9:12:45 AM       │   │
│  │  Article 3: "Whale moves 10,000 BTC"         @ 9:18:02 AM       │   │
│  │  ... up to 1000 articles                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Step 3: Match News to Prices (The Magic)**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PRICE_INTERVAL_SECONDS = 300                                           │
│  "Each time slot is 300 seconds (5 minutes)"                            │
│                                                                         │
│  This is how we match news to prices:                                   │
│                                                                         │
│  NEWS ARTICLE: "Bitcoin surges!" published at 9:07:23 AM                │
│                                                                         │
│  STEP A: Round down to nearest 5-minute slot                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  9:07:23 → rounds down to → 9:05:00                             │   │
│  │                                                                 │   │
│  │  (Because 9:07 is between 9:05 and 9:10,                        │   │
│  │   we use the 9:05 price slot)                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  STEP B: Get the price at that slot                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Price at 9:05:00 = $100,500 (price when news came out)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  STEP C: Get the price 5 minutes LATER                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Price at 9:10:00 = $102,000 (price after news had time to      │   │
│  │                               affect the market)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  STEP D: Calculate what happened                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Price went UP: $102,000 - $100,500 = +$1,500 (+1.5%)           │   │
│  │                                                                 │   │
│  │  LABEL: BUY (because price went up after this news)             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Visual Timeline of Matching:**

```
TIME        9:00    9:05    9:10    9:15    9:20    9:25    9:30
            │       │       │       │       │       │       │
PRICES:     $100K   $100.5K $102K   $99K    $98K    $99.5K  $101K
            ●───────●───────●───────●───────●───────●───────●
                    ▲       ▲
                    │       │
                    │       └── FUTURE PRICE (5 min later)
                    │
            ┌───────┴───────┐
            │ NEWS ARTICLE  │
            │ "BTC surges!" │
            │ @ 9:07:23     │
            │               │
            │ Rounded to    │
            │ 9:05 slot     │
            └───────────────┘

CALCULATION:
  Current price (9:05):  $100,500
  Future price (9:10):   $102,000
  Change: +$1,500 (+1.5%)
  Label: BUY ✓
```

**Why 5 Minutes?**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  The 5-minute window is a HYPOTHESIS:                                   │
│                                                                         │
│  "News affects Bitcoin's price within 5 minutes"                        │
│                                                                         │
│  Too Short (1 min):                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ❌ Not enough time for market to react                         │   │
│  │  ❌ Price changes might be random noise                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Too Long (1 hour):                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ❌ Other news might come out                                   │   │
│  │  ❌ Can't tell which news caused the change                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  5 Minutes (our choice):                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ✓ Enough time for traders to read and react                    │   │
│  │  ✓ Short enough to link news → price change                     │   │
│  │  ✓ Good balance for crypto markets (24/7, fast-moving)          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  You can EXPERIMENT with different intervals!                           │
│  Change PRICE_INTERVAL to "15m" and PRICE_INTERVAL_SECONDS to 900      │
│  to test if 15-minute predictions work better.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Complete Example:**

```
INPUT (what download.py collects):

  News: "SEC approves Bitcoin ETF!"
  Published: 2025-01-15 at 14:07:33

  Price Timeline:
  14:00 = $95,000
  14:05 = $95,200  ← News rounded to this slot
  14:10 = $97,500  ← Price 5 min after news
  14:15 = $98,000

OUTPUT (what goes into training data):

  {
    "title": "SEC approves Bitcoin ETF!",
    "price": 95200,           ← Price when news came out
    "future_price": 97500,    ← Price 5 min later
    "difference": 2300,       ← How much it changed
    "percentage": 2.42,       ← Percentage change (+2.42%)
    "pubDate": "2025-01-15T14:07:33Z"
  }

  Since +2.42% > +1% threshold → Label = BUY
```

### train.py - Model Training

Parameters are passed via CLI arguments (see [Commands > train.py](#trainpy---model-training) for usage).

#### Understanding Epochs

An **epoch** is one complete pass through ALL your training data:

```
YOUR DATA: 159 training samples

EPOCH 1:  Model sees all 159 samples (shuffled) → learns patterns, adjusts weights
EPOCH 2:  Model sees all 159 samples (reshuffled) → refines what it learned
...
EPOCH 20: Model sees all 159 samples again → final refinements
```

**How Many Epochs Do You Need?**

| Data Size | Recommended Epochs | Why |
|-----------|-------------------|-----|
| Small (<500 samples) | 50-200 | Few examples = need more repetitions to learn |
| Medium (500-5000) | 20-50 | Default (20) is reasonable starting point |
| Large (>10000) | 5-20 | Plenty of examples = fewer passes needed |

**Signs You Need More Epochs:**
- Loss still decreasing at the final epoch
- Accuracy improving steadily (not plateaued)

**Signs You Have Too Many:**
- Loss stopped improving (plateaued)
- Test accuracy drops while train accuracy rises (**overfitting**)

**Example:**
```bash
# Small dataset - use more epochs
python train.py -s BTC-USD -e 100

# Large dataset - fewer epochs needed
python train.py -s BTC-USD -e 10
```

#### Understanding Batch Size

**Batch size** = How many samples the model looks at before updating its weights.

```
YOUR DATA: 159 training samples
BATCH SIZE: 1 (default)

EPOCH 1:
  Sample 1 → Predict → Calculate error → Update weights
  Sample 2 → Predict → Calculate error → Update weights
  Sample 3 → Predict → Calculate error → Update weights
  ...
  Sample 159 → Update weights

  Total weight updates per epoch: 159
```

**With Batch Size = 8:**

```
EPOCH 1:
  Samples 1-8  → Predict all 8 → Average errors → Update weights ONCE
  Samples 9-16 → Predict all 8 → Average errors → Update weights ONCE
  ...

  Total weight updates per epoch: 159 ÷ 8 = ~20
```

**Batch Size Comparison:**

| Batch Size | Updates/Epoch | Speed | Learning Behavior |
|------------|---------------|-------|-------------------|
| **1** (default) | 159 | Slowest | Noisy but can escape local minima |
| **8** | ~20 | Faster | Good balance of speed and stability |
| **32** | ~5 | Much faster | Very stable, may miss fine details |
| **159** (full) | 1 | Fastest | Too smooth, can get stuck |

**Analogy - Learning to Cook:**

```
BATCH SIZE 1:
┌────────────────────────────────────────────────────────────────┐
│  Cook 1 dish → Taste → Adjust recipe immediately              │
│  Cook 1 dish → Taste → Adjust recipe immediately              │
│                                                                │
│  Very reactive - but one bad dish = one bad adjustment         │
└────────────────────────────────────────────────────────────────┘

BATCH SIZE 8:
┌────────────────────────────────────────────────────────────────┐
│  Cook 8 dishes → Taste ALL → Adjust based on average feedback │
│                                                                │
│  More stable - one bad dish doesn't ruin everything            │
└────────────────────────────────────────────────────────────────┘
```

**Recommendations by Dataset Size:**

| Dataset Size | Recommended Batch Size | Why |
|--------------|------------------------|-----|
| Small (<500) | 1-8 | Few samples, need frequent updates |
| Medium (500-5000) | 8-32 | Balance speed and learning |
| Large (>10000) | 32-128 | Faster training, stable gradients |

**Example:**
```bash
# Default (batch size 1) - slow but thorough
python train.py -s BTC-USD -e 50

# Faster training with batch size 8
python train.py -s BTC-USD -e 50 --batch-size 8

# Even faster (good for larger datasets)
python train.py -s BTC-USD -e 50 --batch-size 32
```

**Note:** Larger batch sizes may require adjusting the learning rate. A common rule: when you double batch size, increase learning rate by ~1.4x.

#### Training Profiles: Speed vs Precision

Choose your training profile based on how critical accuracy is:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       TRAINING PROFILES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SPEED-OPTIMIZED (prototyping, non-critical):                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  --batch-size 32 --epochs 20 -l 0.01                            │   │
│  │  Fast iteration, good enough, occasional errors acceptable      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  BALANCED (general trading, most use cases):                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  --batch-size 8 --epochs 50 -l 0.005                            │   │
│  │  Good balance of speed and accuracy                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PRECISION-OPTIMIZED (financial, medical, legal, safety-critical):     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  --batch-size 1 --epochs 200 -l 0.001 -o AdamW                  │   │
│  │  Slow but thorough, errors are NOT acceptable                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Profile Comparison:**

| Profile | Batch | Epochs | Learning Rate | Optimizer | Training Time | Use Case |
|---------|-------|--------|---------------|-----------|---------------|----------|
| Speed | 32 | 20 | 0.01 | SGD | ~5-15 min | Prototyping, testing ideas |
| Balanced | 8 | 50 | 0.005 | SGD | ~30-60 min | General trading signals |
| Precision | 1 | 200 | 0.001 | AdamW | ~2-8 hours | Financial, medical, critical |

**Why Precision Settings Work:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--batch-size 1` | 1 | Every sample gets individual attention - rare patterns won't be averaged away |
| `--epochs 200` | 200 | Model sees each pattern 200 times - thorough learning |
| `--learning-rate 0.001` | 0.001 | Small precise weight adjustments - won't overshoot optimal values |
| `--optimizer AdamW` | AdamW | Adapts learning rate per-parameter, includes regularization |

**Example Commands:**

```bash
# SPEED: Quick prototyping
python train.py -s BTC-USD --batch-size 32 -e 20 -l 0.01

# BALANCED: General use (recommended for most trading)
python train.py -s BTC-USD --batch-size 8 -e 50 -l 0.005

# PRECISION: Critical financial decisions
python train.py -s BTC-USD --batch-size 1 -e 200 -l 0.001 -o AdamW

# PRECISION: Medical/safety-critical data
python train.py -s MEDICAL-DATA --batch-size 1 -e 200 -l 0.001 -o AdamW
```

**Multiple Training Runs (for critical applications):**

For maximum confidence in critical applications, run multiple training sessions and compare:

```bash
# Run 1: Standard precision
python train.py -s BTC-USD --batch-size 1 -e 200 -l 0.001 -o AdamW \
    --model-file models/btc_precision_run1.pth

# Run 2: Slightly different settings
python train.py -s BTC-USD --batch-size 2 -e 150 -l 0.0005 -o AdamW \
    --model-file models/btc_precision_run2.pth

# Run 3: More epochs, tiny learning rate
python train.py -s BTC-USD --batch-size 1 -e 300 -l 0.0001 -o AdamW \
    --model-file models/btc_precision_run3.pth

# Compare all three with test.py - use the one with best F1 score
```

**Training Time Estimates (10,000 samples):**

| Profile | Estimated Time |
|---------|----------------|
| Speed | ~5-15 minutes |
| Balanced | ~30-60 minutes |
| Precision | ~2-8 hours |

For financial and medical applications, the extra training time is worth the improved accuracy.

#### Understanding Model Architecture (Beginner-Friendly)

This section explains the four key training parameters using simple analogies.

**The School Learning Analogy:**

Imagine you're learning to identify animals (CATS, DOGS, BIRDS) from a picture book. Each parameter controls a different aspect of how you learn:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EPOCHS = How many times you read the picture book                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  epochs=20 (read 20 times):                                             │
│    📖 → 📖 → 📖 → ... (20 times)                                        │
│    "I think that's a cat? Maybe a small dog?"                           │
│    You learned basics but might confuse similar animals                 │
│                                                                         │
│  epochs=200 (read 200 times):                                           │
│    📖 → 📖 → 📖 → ... (200 times!)                                      │
│    "That's definitely a tabby cat - I can tell by the                   │
│     M-shaped marking on its forehead!"                                  │
│    You know every detail by heart                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  BATCH SIZE = How many pictures before teacher checks your answer       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  batch_size=1 (teacher checks EVERY picture):                           │
│    🖼️ "Cat!" → Teacher: "Correct! ✓"                                    │
│    🖼️ "Dog!" → Teacher: "No, that's a wolf. Here's why..."             │
│    SLOW but you learn from EVERY mistake immediately                    │
│                                                                         │
│  batch_size=32 (teacher checks after 32 pictures):                      │
│    🖼️🖼️🖼️🖼️🖼️🖼️🖼️🖼️ ... (32 pictures)                              │
│    Teacher: "Overall you got 25/32 right. Try to improve."              │
│    FAST but individual mistakes might get lost in the average           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  HIDDEN_DIM = How big is your notebook for taking notes                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  hidden_dim=256 (small notebook):                                       │
│    📓 Your notes:                                                       │
│    ┌─────────────────────────────────────┐                              │
│    │ CAT: 4 legs, fur, says meow         │                              │
│    │ DOG: 4 legs, fur, says woof         │                              │
│    └─────────────────────────────────────┘                              │
│    Basic info only - might confuse cat and dog!                         │
│    Model size: ~2.6 MB                                                  │
│                                                                         │
│  hidden_dim=512 (big notebook):                                         │
│    📒 Your notes:                                                       │
│    ┌─────────────────────────────────────────────────────────┐          │
│    │ CAT: 4 legs, fur, meow, whiskers, pointy ears,          │          │
│    │      retractable claws, slit pupils, purrs...           │          │
│    │ DOG: 4 legs, fur, woof, floppy ears, round pupils,      │          │
│    │      non-retractable claws, wags tail...                │          │
│    └─────────────────────────────────────────────────────────┘          │
│    Lots of detail - can tell the difference easily!                     │
│    Model size: ~10 MB                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  NUM_LAYERS = How many smart friends double-check your answer           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  num_layers=2 (2 friends check):                                        │
│    You: "I think it's a cat"                                            │
│      ↓                                                                  │
│    Friend 1: "Looks right to me"                                        │
│      ↓                                                                  │
│    Friend 2: "Yeah, probably a cat"                                     │
│      ↓                                                                  │
│    Final: "Cat" ✓                                                       │
│    Quick check - good for easy cases                                    │
│                                                                         │
│  num_layers=4 (4 friends check):                                        │
│    You: "I think it's a cat"                                            │
│      ↓                                                                  │
│    Friend 1 (fur expert): "Fur pattern looks feline"                    │
│      ↓                                                                  │
│    Friend 2 (ear expert): "Pointy ears - confirms cat"                  │
│      ↓                                                                  │
│    Friend 3 (eye expert): "Slit pupils - definitely cat"                │
│      ↓                                                                  │
│    Friend 4 (behavior expert): "Body language says cat"                 │
│      ↓                                                                  │
│    Final: "Cat" ✓✓✓✓ (very confident!)                                  │
│    Thorough check - catches subtle details                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### What Affects Model Size?

**Important:** Not all parameters affect the model file size!

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WHAT AFFECTS MODEL SIZE?                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DOES NOT CHANGE MODEL SIZE:                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • epochs (20 vs 200)        → Same .pth file size              │   │
│  │  • batch_size (1 vs 32)      → Same .pth file size              │   │
│  │  • learning_rate             → Same .pth file size              │   │
│  │  • training data amount      → Same .pth file size              │   │
│  │                                                                 │   │
│  │  These only change the VALUES inside the model, not the size    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  DOES CHANGE MODEL SIZE (architecture):                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • --hidden-dim (256 → 512)  → ~4x larger model file            │   │
│  │  • --num-layers (2 → 4)      → ~2x larger model file            │   │
│  │                                                                 │   │
│  │  These add more "boxes" for the model to store patterns         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Model Size Reference:**

| hidden_dim | num_layers | Approximate Size | Use Case |
|------------|------------|------------------|----------|
| 256 | 2 | ~2.6 MB | Fast, general use (default) |
| 256 | 4 | ~4 MB | More thorough checking |
| 512 | 2 | ~8 MB | Detailed pattern recognition |
| 512 | 4 | ~10 MB | Maximum precision (medical/financial) |

#### FAST vs PRECISE: Complete Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FAST MODE (Speed Priority)                           │
│                    Like: Fast Food Restaurant 🍔                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  epochs=20        → Read the recipe book 20 times                       │
│  batch_size=32    → Cook 32 burgers, then check quality                 │
│  hidden_dim=256   → Small checklist: "Has bun? Has patty? Done!"        │
│  num_layers=2     → 2 people check: cook + manager                      │
│                                                                         │
│  Result: Fast, cheap, good enough                                       │
│  Mistakes: Sometimes wrong order, cold fries - acceptable               │
│                                                                         │
│  📁 Model size: ~2.6 MB                                                 │
│  ⏱️ Training time: ~10 minutes                                          │
│                                                                         │
│  Command:                                                               │
│  python train.py -s BTC-USD -e 20 --batch-size 32                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    PRECISE MODE (Quality Priority)                      │
│                    Like: Hospital Surgery Room 🏥                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  epochs=200       → Study the medical book 200 times                    │
│  batch_size=1     → Check EVERY patient individually                    │
│  hidden_dim=512   → Detailed checklist: vitals, history, allergies...   │
│  num_layers=4     → 4 doctors review: surgeon + anesthesiologist        │
│                     + nurse + specialist                                │
│                                                                         │
│  Result: Slow, expensive, but ZERO mistakes                             │
│  Mistakes: NOT ALLOWED - lives depend on it!                            │
│                                                                         │
│  📁 Model size: ~10 MB                                                  │
│  ⏱️ Training time: ~4-8 hours                                           │
│                                                                         │
│  Command:                                                               │
│  python train.py -s BTC-USD -e 200 --batch-size 1 -l 0.001 -o AdamW \   │
│      --hidden-dim 512 --num-layers 4                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Quick Reference: Parameter Summary

| Parameter | What It Controls | Small Value | Large Value | Affects Size? |
|-----------|------------------|-------------|-------------|---------------|
| `--epochs` | Times reading data | Quick skim | Know by heart | No |
| `--batch-size` | Samples before feedback | Every one checked | Batch feedback | No |
| `--hidden-dim` | Notebook size | Short notes | Detailed notes | **Yes** |
| `--num-layers` | Expert reviewers | Quick check | Thorough review | **Yes** |
| `--split` | Data division | More test data | More training data | No |
| `--optimizer` | Learning strategy | SGD (steady) | AdamW (adaptive) | No |

#### Understanding Train/Test Split

The `--split` parameter divides your data into two parts. This is **NOT** an accuracy score!

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAIN/TEST SPLIT EXPLAINED                           │
│                    (The Hidden Test Analogy)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Imagine you have 100 flashcards to learn animals:                      │
│                                                                         │
│  🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏  (100 cards total)           │
│                                                                         │
│  --split 0.8 means:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TRAINING SET (80 cards):                                       │   │
│  │  🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏                       │   │
│  │  🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏                       │   │
│  │  🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏                       │   │
│  │  🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏                       │   │
│  │                                                                 │   │
│  │  You STUDY these. You can look at them over and over.           │   │
│  │  The model learns patterns from these 80 cards.                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TEST SET (20 cards):                                           │   │
│  │  🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏🃏                       │   │
│  │                                                                 │   │
│  │  Teacher HIDES these. You never see them during study.          │   │
│  │  On test day, teacher uses ONLY these 20 to quiz you.           │   │
│  │  This shows if you REALLY learned, or just memorized.           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why This Matters:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE HONEST TEST PROBLEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WITHOUT split (testing on training data):                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Teacher: "What animal is this?" (shows card you studied)       │   │
│  │  You: "Cat!" (you memorized this exact card)                    │   │
│  │  Teacher: "100% correct! You're a genius!"                      │   │
│  │                                                                 │   │
│  │  ❌ FAKE SUCCESS! You just memorized, didn't truly learn.       │   │
│  │  ❌ Give you a NEW animal picture → you'd fail!                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  WITH split (testing on hidden data):                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Teacher: "What animal is this?" (shows card you NEVER saw)     │   │
│  │  You: "Hmm... it has whiskers, pointy ears... Cat!"             │   │
│  │  Teacher: "Correct! You truly understand cats!"                 │   │
│  │                                                                 │   │
│  │  ✓ HONEST TEST! You learned the patterns, not the cards.        │   │
│  │  ✓ You can identify NEW cats you've never seen before!          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Common Mistake - Confusing Split with Accuracy:**

```
This output:    "Train/Test split: 80% / 20%"

WRONG interpretation: "My model is 80% accurate"
RIGHT interpretation: "80% of data for learning, 20% for testing"

The ACCURACY comes later in the results:
  "Accuracy: 45/59 = 76.27%"  ← THIS is your score!
```

**Split Recommendations:**

| Split | Train % | Test % | Use Case |
|-------|---------|--------|----------|
| `--split 0.9` | 90% | 10% | Large datasets (10,000+ samples) |
| `--split 0.8` | 80% | 20% | **Default** - good balance |
| `--split 0.7` | 70% | 30% | Small datasets, need more validation |
| `--split 0.6` | 60% | 40% | Very small datasets, extra validation |

**Example:**

```bash
# Default: 80% train, 20% test
python train.py -s BTC-USD

# More training data (when you have lots of samples)
python train.py -s BTC-USD --split 0.9

# More test data (when you want extra validation)
python train.py -s BTC-USD --split 0.7
```

#### Understanding Optimizers (SGD vs AdamW)

The optimizer controls HOW the model learns from its mistakes. Think of it as the "learning strategy."

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE BASKETBALL THROWING ANALOGY                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Imagine you're learning to throw a basketball into a hoop.             │
│  You start FAR from the target (high error) and want to get closer      │
│  (lower error).                                                         │
│                                                                         │
│     START                                              GOAL             │
│       ●                                                 🏀              │
│     (high error)                                    (zero error)        │
│       │                                                                 │
│       │  ← How do you get there?                                        │
│       ▼                                                                 │
│                                                                         │
│  Two strategies:                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  SGD = The Steady Walker 🚶                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Takes the SAME size step every time, no matter what.                   │
│                                                                         │
│     START                                              GOAL             │
│       ●────────●────────●────────●────────●────────●────🏀             │
│             step     step     step     step     step                    │
│            (same)   (same)   (same)   (same)   (same)                   │
│                                                                         │
│  ✓ Simple and predictable                                               │
│  ✓ Works well when the path is straightforward                          │
│  ✓ Less memory needed                                                   │
│                                                                         │
│  ✗ Might step OVER the target if steps are too big                      │
│  ✗ Might take forever if steps are too small                            │
│  ✗ Same speed everywhere, even when it should slow down                 │
│                                                                         │
│  Best for: Simple problems, quick prototyping                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  AdamW = The Smart Runner 🏃                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ADAPTS step size based on the situation:                               │
│  - Far from goal? Take BIG steps!                                       │
│  - Close to goal? Take TINY steps!                                      │
│                                                                         │
│     START                                              GOAL             │
│       ●━━━━━━━━━━━━━━━━●━━━━━━━━●━━━━●━━●━●─●🏀                        │
│              BIG          MED    small tiny *                           │
│            (far away)                    (close!)                       │
│                                                                         │
│  ✓ Adapts to the terrain - smart about when to speed up/slow down       │
│  ✓ Less likely to overshoot the target                                  │
│  ✓ Handles tricky paths better                                          │
│  ✓ Includes "weight decay" - prevents overconfidence                    │
│                                                                         │
│  ✗ More complex, uses more memory                                       │
│  ✗ Has more settings that could go wrong                                │
│                                                                         │
│  Best for: Complex problems, precision-critical tasks                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Quick Comparison:**

| Aspect | SGD 🚶 | AdamW 🏃 |
|--------|--------|----------|
| Step size | Fixed (same every time) | Adaptive (changes as needed) |
| Speed | Constant | Fast when far, slow when close |
| Memory usage | Lower | Higher |
| Complexity | Simple | More sophisticated |
| Risk of overshooting | Higher | Lower |
| Best for | General use, prototyping | Precision, complex patterns |
| Our recommendation | Speed mode | Precision mode |

**When to Use Each:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  USE SGD (-o SGD) WHEN:                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  • Quick prototyping and testing ideas                                  │
│  • Simple datasets with clear patterns                                  │
│  • You want faster training with less memory                            │
│  • The task is not mission-critical                                     │
│                                                                         │
│  Example:                                                               │
│  python train.py -s BTC-USD -o SGD                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  USE AdamW (-o AdamW) WHEN:                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  • Precision is critical (financial, medical, legal)                    │
│  • Complex datasets with subtle patterns                                │
│  • You want the model to fine-tune near the optimum                     │
│  • Training time is not a concern                                       │
│                                                                         │
│  Example:                                                               │
│  python train.py -s BTC-USD -o AdamW -l 0.001                           │
│                                                                         │
│  Note: AdamW often works better with smaller learning rates (0.001)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Recommended Combinations:**

| Mode | Optimizer | Learning Rate | Use Case |
|------|-----------|---------------|----------|
| **Speed** | SGD | 0.005 - 0.01 | Prototyping, testing |
| **Balanced** | SGD | 0.005 | General trading |
| **Precision** | AdamW | 0.001 | Financial, medical, critical |

**Example Commands:**

```bash
# Speed mode (default): SGD with standard learning rate
python train.py -s BTC-USD -o SGD -l 0.005

# Precision mode: AdamW with smaller learning rate
python train.py -s BTC-USD -o AdamW -l 0.001 -e 200 --batch-size 1
```

#### Model Creation Analysis: Finding the Optimal Command

This section demonstrates how to analyze and compare training commands to find the optimal configuration for maximum precision and data integrity.

**The Golden Rule:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KEY INSIGHT                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   DATA QUALITY > TRAINING TIME                                          │
│                                                                         │
│   A "simple" model trained on GOOD data                                 │
│   beats a "powerful" model trained on EMPTY data                        │
│                                                                         │
│   The BEST model combines:                                              │
│   • GOOD data (correct thresholds for your volatility)                  │
│   • DEEP learning (maximum epochs)                                      │
│   • HIGH capacity (large hidden dim, many layers)                       │
│   • SMART optimizer (AdamW with small learning rate)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Three-Way Command Comparison Example:**

| Label | Command |
|-------|---------|
| **A** | `python train.py -s BTC-USD -e 50 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -b 0.1 --sell-threshold -0.1` |
| **B** | `python train.py -s BTC-USD -e 200 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4` |
| **C** | `python train.py -s BTC-USD -e 200 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -b 0.1 --sell-threshold -0.1` |

**Parameter Comparison Matrix:**

| Parameter | Command A | Command B | Command C | Optimal |
|-----------|-----------|-----------|-----------|---------|
| **Epochs** | 50 | 200 | 200 | 200 ✓ |
| **Batch Size** | 1 | 1 | 1 | 1 ✓ |
| **Learning Rate** | 0.001 | 0.001 | 0.001 | 0.001 ✓ |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW ✓ |
| **Hidden Dim** | 512 | 512 | 512 | 512 ✓ |
| **Num Layers** | 4 | 4 | 4 | 4 ✓ |
| **Buy Threshold** | 0.1% | 1.0% (default) | 0.1% | 0.1% ✓ |
| **Sell Threshold** | -0.1% | -1.0% (default) | -0.1% | -0.1% ✓ |

**Expected Label Distribution (Low-Volatility Data):**

| Command | SELL | HOLD | BUY | Data Quality |
|---------|------|------|-----|--------------|
| **A** (±0.1%) | 21 (10.6%) | 160 (80.4%) | 18 (9.0%) | ✓ Balanced |
| **B** (±1.0%) | 0 (0.0%) | 199 (100%) | 0 (0.0%) | ❌ Useless |
| **C** (±0.1%) | 21 (10.6%) | 160 (80.4%) | 18 (9.0%) | ✓ Balanced |

**Why Thresholds Matter More Than Epochs:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE THRESHOLD PROBLEM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WRONG thresholds (Command B with ±1.0%):                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  All 199 samples become HOLD (100%)                             │   │
│  │  Model learns: "Always predict HOLD"                            │   │
│  │  200 epochs of learning NOTHING useful                          │   │
│  │  → Powerful brain, completely EMPTY                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  CORRECT thresholds (Commands A & C with ±0.1%):                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SELL: 21 samples, HOLD: 160 samples, BUY: 18 samples           │   │
│  │  Model learns: Real patterns for each signal                    │   │
│  │  → Brain has REAL knowledge                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Visual Comparison:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  COMMAND A: Good Data, Less Study (50 epochs, ±0.1%)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  📚 TEXTBOOK: Filled with real examples                                 │
│  📖 STUDY: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  50/200        │
│  🧠 RESULT: Good understanding, could go deeper                         │
│  Score: ⭐⭐⭐⭐☆ (4.25/5)                                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  COMMAND B: Empty Data, Maximum Study (200 epochs, ±1.0%)               │
├─────────────────────────────────────────────────────────────────────────┤
│  📚 TEXTBOOK: Blank! (all HOLD, nothing to learn)                       │
│  📖 STUDY: ████████████████████████████████████████████████  200/200    │
│  🧠 RESULT: Memorized "always say HOLD" - USELESS!                      │
│  Score: ⭐☆☆☆☆ (3.10/5)                                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  COMMAND C: Rich Data, Maximum Study (200 epochs, ±0.1%)                │
├─────────────────────────────────────────────────────────────────────────┤
│  📚 TEXTBOOK: Filled with real examples                                 │
│  📖 STUDY: ████████████████████████████████████████████████  200/200    │
│  🧠 RESULT: EXPERT-LEVEL mastery of all patterns                        │
│  Score: ⭐⭐⭐⭐⭐ (5.00/5)                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Quality Scoring Breakdown:**

| Quality Factor | Weight | Command A | Command B | Command C |
|----------------|--------|-----------|-----------|-----------|
| **Data Diversity** | 35% | ⭐⭐⭐⭐⭐ (5) | ⭐ (1) | ⭐⭐⭐⭐⭐ (5) |
| **Learning Depth** | 25% | ⭐⭐⭐ (3) | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐⭐ (5) |
| **Model Capacity** | 20% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐⭐ (5) |
| **Optimizer Quality** | 10% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐⭐ (5) |
| **Precision Settings** | 10% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐⭐ (5) |
| **TOTAL** | 100% | **4.25/5** | **3.10/5** | **5.00/5** |

**Final Ranking:**

| Rank | Command | Score | Verdict |
|------|---------|-------|---------|
| 🥇 1st | **C** | 5.00/5 | OPTIMAL - Rich data + Maximum study |
| 🥈 2nd | **A** | 4.25/5 | GOOD - Rich data, but less study time |
| 🥉 3rd | **B** | 3.10/5 | FAILED - Powerful brain learning nothing |

**The Optimal Precision Command:**

```bash
python train.py -s BTC-USD -e 200 --batch-size 1 -l 0.001 -o AdamW --hidden-dim 512 --num-layers 4 -b 0.1 --sell-threshold -0.1
```

| Parameter | Value | Why |
|-----------|-------|-----|
| `-e 200` | 200 epochs | Maximum learning depth |
| `--batch-size 1` | 1 | Every sample gets individual attention |
| `-l 0.001` | 0.001 | Tiny precise adjustments (Smart Runner) |
| `-o AdamW` | AdamW | Adaptive optimizer for precision |
| `--hidden-dim 512` | 512 | Large pattern storage capacity |
| `--num-layers 4` | 4 | Maximum expert review layers |
| `-b 0.1` | 0.1% | Matches low-volatility data |
| `--sell-threshold -0.1` | -0.1% | Matches low-volatility data |

**Expected Model Behavior:**

| News Scenario | Command A | Command B | Command C |
|---------------|-----------|-----------|-----------|
| "SEC cracks down" | SELL ✓ | HOLD ❌ | SELL ✓✓ |
| "ETF approved" | BUY ✓ | HOLD ❌ | BUY ✓✓ |
| "Markets quiet" | HOLD ✓ | HOLD ✓ | HOLD ✓✓ |
| **Confidence** | Medium | None | High |
| **Pattern Recognition** | Good | None | Excellent |

**Key Lessons:**

1. **Thresholds must match your data's volatility** - Check label distribution first!
2. **200 epochs on empty data = 0 learning** - Quality over quantity
3. **The optimal command combines ALL factors** - Data + Epochs + Architecture + Optimizer
4. **Always verify label distribution** - Aim for 10-30% in SELL and BUY classes

#### Threshold Guidelines by Asset Type

Different assets have different volatility. Use appropriate thresholds:

| Asset Type | Buy Threshold | Sell Threshold | Example Command |
|------------|---------------|----------------|-----------------|
| Crypto (BTC, ETH) | 1.0% | -1.0% | `python train.py -s BTC-USD` |
| Volatile (TSLA, GME) | 0.5% | -0.5% | `python train.py -s TSLA -b 0.5 --sell-threshold -0.5` |
| Large Cap (AAPL, MSFT) | 0.3% | -0.3% | `python train.py -s AAPL -b 0.3 --sell-threshold -0.3` |
| Index (SPY, QQQ) | 0.2% | -0.2% | `python train.py -s SPY -b 0.2 --sell-threshold -0.2` |

**Check Your Label Distribution:**

After running `train.py`, check the label distribution output:

```
✅ GOOD: Label distribution: SELL=45, HOLD=120, BUY=52
   (All three classes have samples - model can learn)

❌ BAD:  Label distribution: SELL=0, HOLD=199, BUY=0
   (All HOLD - thresholds too wide, lower them!)
```

If >90% of samples are HOLD, `train.py` will warn you and suggest adjusted thresholds.

### test.py - Model Evaluation

The test script evaluates your trained model on held-out test data (default: 20% of the dataset).

#### CLI Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--symbol` | `-s` | Required | Trading symbol (must match train.py) |
| `--buy-threshold` | `-b` | `1.0` | Must match train.py threshold |
| `--sell-threshold` | | `-1.0` | Must match train.py threshold |
| `--split` | | `0.8` | Must match train.py split |
| `--model-file` | | `datasets/{SYMBOL}/{SYMBOL}.pth` | Path to trained model |
| `--samples` | | `3` | Number of sample predictions to show |

#### The `--samples` Parameter (The Grading Analogy)

Think of testing your model like a **teacher grading a student's test**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE GRADING ANALOGY                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WHAT THE TEACHER DOES:                                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  1. Grade ALL 40 test questions (this is the ACCURACY score)     │ │
│  │  2. Calculate the final grade: "32/40 = 80%"                     │ │
│  │  3. Show a FEW example answers for parent-teacher conference     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  The --samples parameter controls step 3: HOW MANY examples to show    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**What `--samples` DOES vs DOESN'T DO:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ✓ DOES: Show more/fewer example predictions for human review           │
│  ✓ DOES: Help you understand HOW the model thinks                       │
│  ✓ DOES: Let you spot patterns in model behavior                        │
│                                                                         │
│  ✗ DOESN'T: Change the accuracy score (ALL questions are graded)        │
│  ✗ DOESN'T: Change the F1 score                                         │
│  ✗ DOESN'T: Make the model "better" or "worse"                          │
│  ✗ DOESN'T: Affect the actual test in any way                           │
│                                                                         │
│  It's like asking the teacher to show more answers at parent-teacher    │
│  conference - the grade is already final!                               │
└─────────────────────────────────────────────────────────────────────────┘
```

**When to use different sample counts:**

| Samples | Use Case |
|---------|----------|
| `--samples 0` | Just want the scores, no examples |
| `--samples 3` | Quick sanity check (default) |
| `--samples 10` | Deeper review of model behavior |
| `--samples 20+` | Detailed analysis/debugging |

#### Matching Train and Test Commands

**CRITICAL**: Test thresholds MUST match training thresholds!

```bash
# If you trained with:
python train.py -s BTC-USD -b 0.1 --sell-threshold -0.1 -e 200

# Then test with THE SAME thresholds:
python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1

# Different thresholds = WRONG labels = meaningless accuracy!
```

#### Example Test Commands

```bash
# Basic test (uses default thresholds ±1.0%)
python test.py -s BTC-USD

# Test with custom thresholds (match your training!)
python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1

# Test with more sample predictions
python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1 --samples 10

# Test with custom model file
python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1 --model-file backups/model_v2.pth

# Just scores, no sample predictions
python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1 --samples 0
```

#### Sample Output

```
============================================================
        TRADING SIGNAL CLASSIFIER - TESTING
============================================================

CONFIGURATION:
  [Data]
  Symbol:           BTC-USD
  Data file:        datasets/BTC-USD/news_with_price.json
  Model file:       datasets/BTC-USD/BTC-USD.pth

  [Signal Thresholds]
  Buy threshold:    > 0.1% price increase
  Sell threshold:   < -0.1% price decrease

  [Test Parameters]
  Train/Test split: 80% / 20%
  Sample predictions: 3

============================================================

LOADING MODEL...
------------------------------------------------------------
  Model file:       datasets/BTC-USD/BTC-USD.pth
  Device:           cuda

LOADING TEST DATA...
------------------------------------------------------------
  Data file:        datasets/BTC-USD/news_with_price.json
  Total samples:    199
  Test samples:     40 (20%)
  Label distribution:
    SELL:    21  ( 10.6%)
    HOLD:   160  ( 80.4%)
    BUY:     18  (  9.0%)

------------------------------------------------------------
EVALUATION
------------------------------------------------------------
  Evaluating 40 samples...

  Results:
    Accuracy:       32/40 = 80.00%
    F1 Score:       0.7842

  Classification Report:
              precision    recall  f1-score   support
        SELL       0.75      0.86      0.80         7
        HOLD       0.83      0.79      0.81        29
         BUY       0.67      0.75      0.71         4
    accuracy                           0.80        40

------------------------------------------------------------
SAMPLE PREDICTIONS
------------------------------------------------------------
  Showing 3 sample predictions:

    Sample 1:
      Headline: Bitcoin surges on ETF approval news...
      → BUY (82% confident) - The model predicts the price will RISE.

    Sample 2:
      Headline: SEC announces new cryptocurrency regulations...
      → SELL (75% confident) - The model predicts the price will DROP.

    Sample 3:
      Headline: Market remains steady amid uncertainty...
      → HOLD (68% confident) - The model predicts NO significant change.

------------------------------------------------------------
Done!
------------------------------------------------------------
```

**Important**: Test accuracy depends on BOTH training quality AND threshold matching. A well-trained model with mismatched thresholds will show poor accuracy because the labels are wrong, not because the model is bad!

#### Parameter Consistency: Why It's Critical

All parameters (except `--samples` and `--model-file`) MUST match between training and testing. Here's why:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER MATCHING CRITICALITY                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Parameter          │ Criticality │ What Happens If Mismatched         │
│  ───────────────────┼─────────────┼────────────────────────────────────│
│  --buy-threshold    │ 🔴 CRITICAL │ WRONG LABELS → meaningless test    │
│  --sell-threshold   │ 🔴 CRITICAL │ WRONG LABELS → meaningless test    │
│  --split            │ 🔴 CRITICAL │ DATA LEAKAGE → fake high accuracy  │
│  --symbol           │ 🔴 CRITICAL │ Wrong data → meaningless test      │
│  --samples          │ 🟢 SAFE     │ Display only, no effect            │
│  --model-file       │ 🟡 OPTIONAL │ Valid for comparing models         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Problem 1: Mismatched Thresholds (WRONG ANSWERS)**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE ANSWER KEY PROBLEM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SCHOOL ANALOGY:                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Training:                                                        │ │
│  │    Teacher: "A grade of 90+ is an A, 80-89 is B, below 80 is C"   │ │
│  │    Student studies with THIS grading scale                        │ │
│  │                                                                   │ │
│  │  Testing (MISMATCHED):                                            │ │
│  │    Teacher: "Actually, 95+ is A, 85-94 is B, below 85 is C"       │ │
│  │    Student's 92% answer → Was "A" in training, now "B" in test!   │ │
│  │                                                                   │ │
│  │  Result: Student looks WRONG even though they learned correctly!  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

REAL EXAMPLE:

  Price change: +0.5% (price went up slightly)

  TRAINING (±0.1% thresholds):
    +0.5% > +0.1%  →  Label = BUY
    Model learns: "This news pattern = BUY"

  TESTING (±1.0% thresholds - WRONG!):
    +0.5% is between -1.0% and +1.0%  →  Label = HOLD
    Model predicts BUY (correctly!)
    Test says: "WRONG! Answer was HOLD"

    ❌ Model is penalized for being RIGHT!
```

**Visual: Same Data, Different Labels**

```
PRICE CHANGE:  -2%   -0.5%   +0.3%   +0.8%   +1.5%
               │      │       │       │       │
               ▼      ▼       ▼       ▼       ▼

TRAINING (±0.1%):
  SELL ◄───────┼──────┼───────┼───────┼───────┼──────► BUY
              -0.1%        HOLD        +0.1%
  Labels:    SELL   SELL    BUY     BUY     BUY


TESTING (±1.0% - MISMATCHED!):
  SELL ◄───────────────┼───────────────┼───────────────► BUY
                      -1%    HOLD     +1%
  Labels:    SELL   HOLD    HOLD    HOLD    BUY
                    ^^^^    ^^^^    ^^^^
                    DIFFERENT LABELS!

Result: 3 out of 5 samples have WRONG "correct answers"
        Model accuracy will be ~40% even if model is PERFECT!
```

**Problem 2: Mismatched Split (DATA LEAKAGE)**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE CHEATING PROBLEM                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SCHOOL ANALOGY:                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Training (80% split):                                            │ │
│  │    Student studies questions 1-80                                 │ │
│  │    Questions 81-100 are HIDDEN for the final exam                 │ │
│  │                                                                   │ │
│  │  Testing (70% split - MISMATCHED!):                               │ │
│  │    Exam uses questions 71-100                                     │ │
│  │    But wait... student already SAW questions 71-80 during study!  │ │
│  │                                                                   │ │
│  │  Result: Student gets those 10 questions "free" - CHEATING!       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

REAL EXAMPLE:

  Total samples: 100

  TRAINING (split=0.8):
    Samples 0-79:  Used for TRAINING (model memorizes these)
    Samples 80-99: Reserved for TESTING (model never sees these)

  TESTING (split=0.7 - MISMATCHED!):
    Test uses samples 70-99

    Samples 70-79: Model ALREADY SAW these during training!
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   DATA LEAKAGE - model memorized the answers!

    Result: Artificially HIGH accuracy (maybe 95% instead of 75%)
            Model looks great but will FAIL on real new data!
```

**Visual: Data Leakage**

```
SAMPLES:  [0]─────────────────────────────────────────────────────[99]

TRAINING (split=0.8):
          ├──────── TRAIN (0-79) ────────┤ TEST (80-99) │
          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░░░░░│

TESTING (split=0.7 - WRONG!):
          ├────── TRAIN (0-69) ──────┤─── TEST (70-99) ───│
                                     │▓▓▓▓▓▓▓▓▓▓│
                                     │  LEAKED! │
                                     │ Model saw│
                                     │  these!  │

The overlapping region (70-79) is DATA LEAKAGE:
- Model memorized these during training
- Now being "tested" on memorized answers
- Fake high score!
```

**The Three Golden Rules**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE THREE GOLDEN RULES                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RULE 1: THRESHOLDS MUST MATCH                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  train.py -b 0.1 --sell-threshold -0.1                            │ │
│  │  test.py  -b 0.1 --sell-threshold -0.1  ← SAME!                   │ │
│  │                                                                   │ │
│  │  Why: Labels are calculated from thresholds                       │ │
│  │  Risk: Wrong labels = meaningless accuracy (could be 0%!)         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  RULE 2: SPLIT MUST MATCH                                               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  train.py --split 0.8                                             │ │
│  │  test.py  --split 0.8  ← SAME!                                    │ │
│  │                                                                   │ │
│  │  Why: Determines which samples are train vs test                  │ │
│  │  Risk: Data leakage = fake high accuracy (misleading!)            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  RULE 3: SYMBOL MUST MATCH                                              │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  train.py -s BTC-USD                                              │ │
│  │  test.py  -s BTC-USD  ← SAME!                                     │ │
│  │                                                                   │ │
│  │  Why: Model learned Bitcoin patterns, not Apple patterns          │ │
│  │  Risk: Testing wrong data = completely meaningless                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Quick Reference: Correct vs Wrong Commands**

```bash
# ✅ CORRECT: All parameters match
python train.py -s BTC-USD -b 0.1 --sell-threshold -0.1 --split 0.8 -e 200
python test.py  -s BTC-USD -b 0.1 --sell-threshold -0.1 --split 0.8

# ❌ WRONG: Threshold mismatch (labels will be wrong)
python train.py -s BTC-USD -b 0.1 --sell-threshold -0.1
python test.py  -s BTC-USD -b 1.0 --sell-threshold -1.0  # DIFFERENT!

# ❌ WRONG: Split mismatch (data leakage)
python train.py -s BTC-USD --split 0.8
python test.py  -s BTC-USD --split 0.7  # DIFFERENT!

# ❌ WRONG: Symbol mismatch (testing on different asset)
python train.py -s BTC-USD
python test.py  -s AAPL  # DIFFERENT!
```

**Parameter Criticality Summary**

| Parameter | If Mismatched | Severity | Result |
|-----------|---------------|----------|--------|
| `--buy-threshold` | Labels change | 🔴 **CRITICAL** | 0-40% accuracy even with perfect model |
| `--sell-threshold` | Labels change | 🔴 **CRITICAL** | 0-40% accuracy even with perfect model |
| `--split` | Data leakage | 🔴 **CRITICAL** | Fake 90%+ accuracy, fails on real data |
| `--symbol` | Wrong data | 🔴 **CRITICAL** | Completely meaningless test |
| `--samples` | Display only | 🟢 Safe | No effect on accuracy |
| `--model-file` | Different model | 🟡 Intentional | Valid for comparing models |

---

## Understanding Evaluation Results (The Report Card)

After training and testing, you'll see results like this:

```
------------------------------------------------------------
EVALUATION
------------------------------------------------------------
  Test samples:     12
  Results:
    Accuracy:       10/12 = 83.33%
    F1 Score:       0.7576
```

This section explains what these metrics mean and how to interpret them.

### What is ACCURACY?

Accuracy is the simplest metric - **"How many did you get right?"**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ACCURACY = "How many did you get right?"             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SCHOOL ANALOGY:                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Teacher gives you a 12-question test.                            │ │
│  │  You answer all 12 questions.                                     │ │
│  │  Teacher grades: 10 correct, 2 wrong.                             │ │
│  │                                                                   │ │
│  │  Your grade: 10/12 = 83.33%                                       │ │
│  │                                                                   │ │
│  │  Simple! Just count the right answers.                            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  YOUR MODEL:                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  12 news articles to classify (BUY/SELL/HOLD)                     │ │
│  │  Model predicted 10 correctly                                     │ │
│  │  Model got 2 wrong                                                │ │
│  │                                                                   │ │
│  │  Accuracy: 10/12 = 83.33%                                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What is F1 SCORE?

F1 is more complex - it measures the **quality** of your answers, not just quantity. It balances two things:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    F1 SCORE = "How confident AND thorough are you?"     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SCHOOL ANALOGY - The Science Fair Judge:                               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │  PRECISION = "When you raised your hand, were you right?"         │ │
│  │  ─────────────────────────────────────────────────────────────    │ │
│  │    You raised your hand 5 times to answer "BUY"                   │ │
│  │    4 times you were correct                                       │ │
│  │    Precision = 4/5 = 80%                                          │ │
│  │                                                                   │ │
│  │    High precision = "I only speak when I'm sure"                  │ │
│  │                                                                   │ │
│  │  RECALL = "Did you catch all the ones you should have?"           │ │
│  │  ─────────────────────────────────────────────────────────────    │ │
│  │    There were 6 questions where "BUY" was correct                 │ │
│  │    You only raised your hand for 4 of them                        │ │
│  │    Recall = 4/6 = 67%                                             │ │
│  │                                                                   │ │
│  │    High recall = "I caught most opportunities"                    │ │
│  │                                                                   │ │
│  │  F1 = Balance of BOTH (harmonic mean)                             │ │
│  │  ─────────────────────────────────────────────────────────────    │ │
│  │    F1 = 2 × (Precision × Recall) / (Precision + Recall)           │ │
│  │    F1 = 2 × (0.80 × 0.67) / (0.80 + 0.67) = 0.73                  │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why F1 Matters More Than Accuracy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE LAZY STUDENT PROBLEM                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Imagine a test with 100 questions:                                     │
│    - 80 questions = HOLD                                                │
│    - 10 questions = BUY                                                 │
│    - 10 questions = SELL                                                │
│                                                                         │
│  LAZY STUDENT (always answers "HOLD"):                                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Answers: HOLD, HOLD, HOLD, HOLD... (100 times)                   │ │
│  │                                                                   │ │
│  │  Accuracy: 80/100 = 80%  ← Looks good!                            │ │
│  │  F1 Score: 0.30          ← Reveals the truth!                     │ │
│  │                                                                   │ │
│  │  The student learned NOTHING - just guessed the most common!      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  SMART STUDENT (actually learned patterns):                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Answers: Varies based on the question                            │ │
│  │                                                                   │ │
│  │  Accuracy: 75%           ← Slightly lower...                      │ │
│  │  F1 Score: 0.72          ← But much better quality!               │ │
│  │                                                                   │ │
│  │  This student actually learned to identify BUY and SELL signals!  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  LESSON: High accuracy + Low F1 = Model is cheating (always guessing   │
│          the most common answer)                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Score Benchmarks: What's a Good Score?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SCORE BENCHMARKS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  BASELINE (Random Guessing):                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  With 3 classes (BUY/SELL/HOLD), random guessing = 33%            │ │
│  │  Anything ABOVE 33% means the model learned SOMETHING             │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  SCORE INTERPRETATION:                                                  │
│                                                                         │
│  Accuracy    F1 Score    Grade    Meaning                               │
│  ─────────   ─────────   ─────    ───────────────────────────────────   │
│  < 40%       < 0.35      ❌ F     Model learned nothing (near random)   │
│  40-55%      0.35-0.50   ⚠️ D     Barely learning, needs improvement    │
│  55-65%      0.50-0.60   📊 C     Some patterns found, room to grow     │
│  65-75%      0.60-0.70   📈 B     Good! Model is learning patterns      │
│  75-85%      0.70-0.80   ✅ A     Very good! Solid predictions          │
│  85-95%      0.80-0.90   🌟 A+    Excellent! Production-ready           │
│  > 95%       > 0.90      🤔 ???   Suspicious - check for data leakage!  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quick Reference: Reading Your Report Card

| Metric | What It Measures | Good Sign | Bad Sign |
|--------|------------------|-----------|----------|
| **Accuracy** | % of correct predictions | > 70% | < 50% |
| **F1 Score** | Quality & balance | > 0.65 | < 0.45 |
| **Both similar** | Honest learning | Acc ≈ F1 | Acc >> F1 |

### Red Flags to Watch For

| Symptom | Problem | Solution |
|---------|---------|----------|
| Accuracy 80%, F1 0.30 | Guessing most common class | Fix threshold balance |
| Accuracy > 95% | Data leakage (cheating) | Check train/test split matches |
| F1 varies wildly | Too few test samples | Get more data |
| Both scores < 40% | Model learned nothing | More epochs, better thresholds |

### How to Improve Your Scores

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOW TO GET BETTER GRADES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. MORE DATA (More study material)                                     │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Problem: Too few test samples (e.g., only 12)                    │ │
│  │  Solution: Download more news articles                            │ │
│  │                                                                   │ │
│  │  python download.py -s AAPL -p 1mo -i 5m -n 1000                  │ │
│  │                                       ^^^^                        │ │
│  │                                       More articles!              │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  2. MORE EPOCHS (Study longer)                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Problem: Model didn't study enough                               │ │
│  │  Solution: Increase epochs                                        │ │
│  │                                                                   │ │
│  │  python train.py -s AAPL ... -e 300                               │ │
│  │                              ^^^^^                                │ │
│  │                              More studying!                       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  3. BETTER THRESHOLDS (Right grading scale)                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Problem: Label distribution is unbalanced (all HOLD)             │ │
│  │  Solution: Adjust thresholds until ~25% SELL, ~50% HOLD, ~25% BUY │ │
│  │                                                                   │ │
│  │  Check training output for label distribution!                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  4. BIGGER MODEL (Bigger brain)                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Problem: Model too simple to capture complex patterns            │ │
│  │  Solution: Increase model size                                    │ │
│  │                                                                   │ │
│  │  --hidden-dim 512 --num-layers 4                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  5. LOWER LEARNING RATE (Careful studying)                              │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Problem: Model learning too fast, missing details                │ │
│  │  Solution: Lower learning rate                                    │ │
│  │                                                                   │ │
│  │  python train.py -s AAPL ... -l 0.0005                            │ │
│  │                              ^^^^^^^^                             │ │
│  │                              Slower, more careful learning        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example: Interpreting Real Results

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EXAMPLE RESULTS:                                                       │
│    Accuracy: 83.33%                                                     │
│    F1 Score: 0.7576                                                     │
│                                                                         │
│  INTERPRETATION:                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  ✓ Accuracy (83%) is GOOD - model gets most predictions right     │ │
│  │  ✓ F1 (0.76) is GOOD - model actually learned patterns            │ │
│  │    (not just guessing the most common class)                      │ │
│  │  ✓ Both metrics are close = balanced, honest learning             │ │
│  │                                                                   │ │
│  │  Grade: ✅ A (Very good!)                                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ⚠️  Note: Small test samples (12) may cause variance.                 │
│      Consider getting more data for more reliable results.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture Deep Dive

This section explains the neural network model in `models/gemma_transformer_classifier.py`.

### What the Model Does (Simple Explanation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         THE MODEL'S JOB                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   INPUT:  "Price: 95000, Headline: Bitcoin crashes on Fed news"         │
│                              ↓                                          │
│                         [MAGIC BOX]                                     │
│                              ↓                                          │
│   OUTPUT: [0.75, 0.15, 0.10]  →  "75% sure this is a SELL signal"       │
│           SELL  HOLD  BUY                                               │
│                                                                         │
│   The model does NOT explain WHY - it just predicts a category!         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Understanding Dimensions (What "256 dim" Means)

"Dim" = **dimensions** = **how many numbers describe something**.

```
1 DIMENSION (1D):
┌─────────────────────────────────────────────────────────────────────────┐
│  Your age: 25                                                           │
│                                                                         │
│  Just ONE number describes it.                                          │
│  [25]                                                                   │
└─────────────────────────────────────────────────────────────────────────┘

2 DIMENSIONS (2D):
┌─────────────────────────────────────────────────────────────────────────┐
│  A point on a map: (latitude, longitude)                                │
│                                                                         │
│  TWO numbers describe it.                                               │
│  [40.7, -74.0]  ← New York City                                         │
└─────────────────────────────────────────────────────────────────────────┘

3 DIMENSIONS (3D):
┌─────────────────────────────────────────────────────────────────────────┐
│  A point in space: (x, y, z)                                            │
│                                                                         │
│  THREE numbers describe it.                                             │
│  [10, 20, 5]  ← Position of a drone                                     │
└─────────────────────────────────────────────────────────────────────────┘

1024 DIMENSIONS:
┌─────────────────────────────────────────────────────────────────────────┐
│  The "meaning" of a sentence (Gemma embedding)                          │
│                                                                         │
│  1024 numbers describe it!                                              │
│  [0.23, -0.45, 0.78, 0.12, ..., 0.56]  ← 1024 numbers total             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why 1024 numbers for text?** Because meaning is complex:

```
"Bitcoin crashes" → [0.82, -0.34, 0.91, ..., 0.12]  (1024 numbers)
                     │      │      │
                     │      │      └── Maybe captures "financial topic"
                     │      └── Maybe captures "negative sentiment"
                     └── Maybe captures "cryptocurrency related"

Each number captures some tiny aspect of meaning.
More dimensions = more nuance can be captured.
```

**Why compress 1024 → 256?**

| 1024 dimensions | 256 dimensions |
|-----------------|----------------|
| More detailed | Less detailed |
| Slower to process | Faster to process |
| More memory | Less memory |
| May contain noise | Focuses on important patterns |

**Real-world analogy:**

```
ORIGINAL (1024 dim):
"The Securities and Exchange Commission announced new regulatory
frameworks targeting cryptocurrency exchanges, causing Bitcoin
to decline sharply in early morning trading."

COMPRESSED (256 dim):
"SEC regulation → Bitcoin drops"

Same core meaning, fewer "numbers" to describe it!
```

### What's Inside the "Magic Box"?

The model has 4 main parts, like a factory assembly line:

```
STEP 1: GEMMA EMBEDDING (The Translator)
┌─────────────────────────────────────────────────────────────────────────┐
│  "Bitcoin crashes on news"  →  [0.23, -0.45, 0.78, ..., 0.12]           │
│                                 (1024 numbers)                          │
│                                                                         │
│  Gemma is a pre-trained model from Google. It converts ANY text into    │
│  a list of 1024 numbers that capture the "meaning" of the text.         │
│  This part is FROZEN - we don't change it during training.              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 2: LINEAR PROJECTION (The Compressor)
┌─────────────────────────────────────────────────────────────────────────┐
│  [1024 numbers]  →  [256 numbers]                                       │
│                                                                         │
│  Reduces the size to make processing faster.                            │
│  This part IS trained - it learns what information to keep.             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 3: TRANSFORMER ENCODER (The Pattern Finder)
┌─────────────────────────────────────────────────────────────────────────┐
│  [256 numbers]  →  [256 numbers] (but transformed)                      │
│                                                                         │
│  Uses "attention" to find patterns in the numbers.                      │
│  2 layers, 4 attention heads.                                           │
│  This part IS trained - it learns what patterns matter.                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 4: CLASSIFIER HEAD (The Decision Maker)
┌─────────────────────────────────────────────────────────────────────────┐
│  [256 numbers]  →  [3 numbers]  →  SELL / HOLD / BUY                    │
│                                                                         │
│  Final layer that outputs one score for each possible answer.           │
│  Sigmoid activation squashes scores to 0-1 range.                       │
│  This part IS trained - it learns how to make the final decision.       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Model Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `"google/embeddinggemma-300m"` | Pre-trained Gemma model from HuggingFace |
| `NUM_CLASSES` | `3` | Output classes (SELL, HOLD, BUY) |
| `HIDDEN_DIM` | `256` | Internal dimension after projection |
| `NUM_LAYERS` | `2` | Number of Transformer encoder layers |
| `NUM_HEADS` | `4` | Number of attention heads per layer |
| `DROPOUT` | `0.1` | Dropout rate (10%) for regularization |

### Key Model Methods

| Method | What It Does | When to Use |
|--------|--------------|-------------|
| `model(texts)` | Main forward pass - returns raw logits | Used by train.py and test.py |
| `model.predict(text)` | Convenience method - returns dict with prediction | For quick single predictions |
| `model.embedding(text)` | Convert text to 1024-dim vector | Internal use (cached) |
| `model.train()` | Enable training mode | Before training loop |
| `model.eval()` | Enable evaluation mode | Before testing/prediction |

### Using the predict() Method

The model now includes a convenient `predict()` method:

```python
from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier
import torch

# Load trained model
model = SimpleGemmaTransformerClassifier()
model.load_state_dict(torch.load('gemma_transformer_classifier.pth', weights_only=True))

# Get prediction
result = model.predict("Bitcoin surges on ETF approval news")

print(result)
# Output:
# {
#     'prediction': 'BUY',
#     'confidence': 0.82,
#     'probabilities': {'SELL': 0.08, 'HOLD': 0.10, 'BUY': 0.82}
# }
```

### What Gets Trained vs What's Frozen

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINABLE vs FROZEN PARTS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   FROZEN (Pre-trained, NOT updated):                                    │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Gemma Embedding Model (300 million parameters)                 │  │
│   │  Already trained by Google on massive text data                 │  │
│   │  Knows how to convert ANY text to meaningful numbers            │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   TRAINABLE (Updated during training):                                  │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  • Linear Projection: 1024 → 256 (~262,000 parameters)          │  │
│   │  • Transformer Encoder: 2 layers (~400,000 parameters)          │  │
│   │  • Classifier Head: 256 → 3 (~770 parameters)                   │  │
│   │                                                                 │  │
│   │  Total trainable: ~660,000 parameters                           │  │
│   │  These learn Bitcoin-specific patterns from YOUR data           │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Complete Dimension Flow (Visual Summary)

Here's how dimensions change as data flows through the model:

```
INPUT: "Price: 95000\nHeadline: Bitcoin crashes on regulation news"
       (just text - no dimensions yet)
                              │
                              ▼
              ┌───────────────────────────────┐
              │      GEMMA EMBEDDING          │
              │      (The Translator)         │
              │                               │
              │   Converts text to numbers    │
              │   using pre-trained model     │
              └───────────────┬───────────────┘
                              │
                              ▼
              [■■■■■■■■■■■■■■■■■■■■■■■■■■■■]
                     1024 DIMENSIONS
              (1024 numbers describing meaning)
                              │
                              ▼
              ┌───────────────────────────────┐
              │     LINEAR PROJECTION         │
              │     (The Compressor)          │
              │                               │
              │   nn.Linear(1024, 256)        │
              │   Learnable compression       │
              └───────────────┬───────────────┘
                              │
                              ▼
                    [■■■■■■■■]
                  256 DIMENSIONS
              (compressed but meaningful)
                              │
                              ▼
              ┌───────────────────────────────┐
              │   TRANSFORMER ENCODER         │
              │   (The Pattern Finder)        │
              │                               │
              │   2 layers, 4 attention heads │
              │   Input: 256 → Output: 256    │
              └───────────────┬───────────────┘
                              │
                              ▼
                    [■■■■■■■■]
                  256 DIMENSIONS
              (patterns extracted)
                              │
                              ▼
              ┌───────────────────────────────┐
              │     CLASSIFIER HEAD           │
              │     (The Decision Maker)      │
              │                               │
              │   nn.Linear(256, 3)           │
              │   + Sigmoid activation        │
              └───────────────┬───────────────┘
                              │
                              ▼
                      [■ ■ ■]
                   3 DIMENSIONS
              (one score per class)
                              │
                              ▼
OUTPUT: [0.75, 0.15, 0.10] → SELL (75% confident)
         SELL  HOLD  BUY
```

### The Embedding Cache

The model caches embeddings to avoid recomputing them:

```python
# First time seeing this text:
model(["Bitcoin crashes"])  # Computes embedding, stores in cache

# Second time (same text):
model(["Bitcoin crashes"])  # Returns cached embedding - FAST!

# Different text:
model(["Bitcoin rises"])    # Computes new embedding, stores in cache
```

Cache key is SHA256 hash of the text, so identical text = cache hit.

**Note**: Cache is in-memory only - cleared when model is reloaded.

---

## Architecture

### Data Pipeline

```
[download.py]                    [train.py]                 [test.py]
     │                                │                          │
     ▼                                ▼                          ▼
┌─────────────┐              ┌─────────────────┐         ┌──────────────┐
│ Yahoo       │              │ Load merged     │         │ Load saved   │
│ Finance API │              │ JSON data       │         │ .pth weights │
└─────┬───────┘              └────────┬────────┘         └──────┬───────┘
      │                               │                         │
      ▼                               ▼                         ▼
┌─────────────┐              ┌─────────────────┐         ┌──────────────┐
│ Price data  │              │ Create text     │         │ Run on test  │
│ (5-min)     │              │ inputs          │         │ data (20%)   │
└─────┬───────┘              └────────┬────────┘         └──────┬───────┘
      │                               │                         │
      ▼                               ▼                         ▼
┌─────────────┐              ┌─────────────────┐         ┌──────────────┐
│ News        │              │ Feed through    │         │ Report       │
│ headlines   │              │ model           │         │ Accuracy/F1  │
└─────┬───────┘              └────────┬────────┘         └──────────────┘
      │                               │
      ▼                               ▼
┌─────────────┐              ┌─────────────────┐
│ Merge by    │              │ Backprop +      │
│ timestamp   │              │ optimize        │
└─────┬───────┘              └────────┬────────┘
      │                               │
      ▼                               ▼
┌─────────────┐              ┌─────────────────┐
│ Calculate   │              │ Save weights    │
│ % change    │              │ to .pth file    │
└─────────────┘              └─────────────────┘
```

### Model Architecture (`models/gemma_transformer_classifier.py`)

```
Input: "Price: 95000, Headline: Fed raises rates, Summary: ..."
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Gemma Embedding      │
                 │   (1024 dimensions)    │
                 └───────────┬────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Linear Projection    │
                 │   1024 → 256 dims      │
                 └───────────┬────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  TransformerEncoder    │
                 │  (2 layers, 4 heads)   │
                 │  GELU activation       │
                 └───────────┬────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Classifier Head      │
                 │   256 → 3 classes      │
                 │   Sigmoid activation   │
                 └───────────┬────────────┘
                              │
                              ▼
                 Output: [sell, hold, buy]
                 e.g., [0.1, 0.2, 0.7] → BUY
```

### Label Generation

| Price Change (5 min) | Label | One-Hot Encoding |
|----------------------|-------|------------------|
| < -1%                | SELL  | [1, 0, 0]        |
| -1% to +1%           | HOLD  | [0, 1, 0]        |
| > +1%                | BUY   | [0, 0, 1]        |

### Training Parameters (current defaults in `train.py`)

| Parameter      | Value | Description                          |
|----------------|-------|--------------------------------------|
| learning_rate  | 0.005 | Step size for weight updates         |
| batch          | 1     | Samples per gradient update          |
| epochs         | 20    | Full passes through training data    |
| optimizer      | SGD   | Stochastic Gradient Descent          |
| loss function  | CrossEntropyLoss | For multi-class classification |

---

## Libraries and Dependencies

This project uses the following libraries:

| Library | Purpose | Description |
|---------|---------|-------------|
| **yfinance** | Data Source | Downloads historical price data and news from Yahoo Finance. Provides easy access to market data without API keys. |
| **PyTorch** | Deep Learning | Core framework for building and training the neural network. Handles tensors, gradients, and GPU acceleration. |
| **sentence-transformers** | Text Embeddings | Loads and runs the Gemma embedding model. Converts text into numerical vectors. |
| **transformers** | Model Support | HuggingFace library for loading pre-trained models like Gemma. |
| **NumPy** | Numerical Computing | Fast array operations and mathematical functions. Foundation for scientific computing in Python. |
| **Pandas** | Data Manipulation | Data analysis library for handling tabular data (DataFrames). Used internally by yfinance. |
| **scikit-learn** | ML Utilities | Provides evaluation metrics (F1 score, classification report) and preprocessing utilities. |
| **matplotlib** | Visualization | Plotting library for creating charts and graphs (used for analysis). |
| **TensorFlow** | Backend Support | Required as a backend dependency for some model operations. |
| **tf-keras** | Keras Support | TensorFlow-Keras integration for compatibility with certain model components. |

### Core Model Components

| Component | Model/Library | Parameters | Description |
|-----------|---------------|------------|-------------|
| **Gemma Embedding** | `google/embeddinggemma-300m` | 300M | Google's text embedding model. Converts text to 768-dimensional vectors capturing semantic meaning. |
| **Transformer Encoder** | PyTorch `nn.TransformerEncoder` | ~660K (trainable) | Learns patterns in the embedded text to identify trading signals. |
| **Classification Head** | PyTorch `nn.Linear` + `nn.Sigmoid` | ~770 (trainable) | Final layer that outputs probabilities for SELL/HOLD/BUY. |

### Installation

All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note**: The Gemma model requires HuggingFace authentication:
```bash
huggingface-cli login
```

---

## File Reference

| File | Purpose |
|------|---------|
| `download.py` | Fetches price data and news from Yahoo Finance, merges by timestamp |
| `train.py` | Trains model on 80% of data, saves weights to symbol folder |
| `test.py` | Evaluates model on 20% held-out test data |
| `models/gemma_transformer_classifier.py` | Neural network model definition |

**Per-Symbol Files (in `datasets/{SYMBOL}/`):**

| File | Purpose |
|------|---------|
| `datasets/{SYMBOL}/historical_data.json` | Price data (unix timestamp ms → price) |
| `datasets/{SYMBOL}/news.json` | Raw news articles from Yahoo Finance |
| `datasets/{SYMBOL}/news_with_price.json` | Merged training data with labels |
| `datasets/{SYMBOL}/{SYMBOL}.pth` | Trained model weights for this symbol |

**Example structure:**
```
datasets/
├── BTC-USD/
│   ├── historical_data.json
│   ├── news.json
│   ├── news_with_price.json
│   └── BTC-USD.pth          ← Model lives with its data
└── AAPL/
    ├── historical_data.json
    ├── news.json
    ├── news_with_price.json
    └── AAPL.pth
```

| Other Files | Purpose |
|-------------|---------|
| `target-profit.png` | Visual example of target volatile market conditions |

---

## Known Issues

All previously known issues have been resolved:

| Issue | Status | Resolution |
|-------|--------|------------|
| ~~Hardcoded MPS device~~ | ✅ Fixed | All scripts use `model.device` for auto-detection |
| ~~Inverted label calculation~~ | ✅ Fixed | Changed to `future_price - price` (was backwards) |
| ~~Wrong threshold values~~ | ✅ Fixed | Now uses ±1.0 (percentage points) |
| ~~Unnecessary model save in test.py~~ | ✅ Fixed | Removed - only `train.py` saves |
| ~~Hardcoded magic numbers~~ | ✅ Fixed | All parameters in config sections at top of files |

---

## TODO List

### Completed
- [x] Save and load model
- [x] Calculate the cost (loss tracking)
- [x] Separate training and validation data
- [x] More data (pull from download.py)
- [x] SGD optimizer
- [x] Activation functions (GELU)
- [x] Configurable label thresholds (via BUY_THRESHOLD/SELL_THRESHOLD)
- [x] Cross-platform device support (CUDA/MPS/CPU)
- [x] Production-ready config sections in all scripts
- [x] Comprehensive documentation (CLAUDE.md beginner's guide)
- [x] Model architecture documentation with visual diagrams
- [x] PyTorch functions reference table
- [x] predict() convenience method for easy model queries

### Pending
- [ ] Capture median price % deltas as market indicator
- [ ] Reinforcement learning
- [ ] Reduce dimensionality of embeddings
- [ ] Add more symbols (AAPL, TSLA, GC=F)
- [ ] Batch size tuning
- [ ] Multi-stage batch training
- [ ] Dropout rate tuning
- [ ] Learning rate adjustments/scheduling
- [ ] Time of day features (vectorized day of week)
- [ ] Sequence data (merge sequences together)
- [ ] Embedding caching to disk (save by hash)

---

## Glossary of Terms and Acronyms

| Term | Full Name | Definition |
|------|-----------|------------|
| **Adam** | Adaptive Moment Estimation | Optimizer that adapts learning rate per-parameter using momentum |
| **AdamW** | Adam with Weight Decay | Variant of Adam with improved regularization |
| **API** | Application Programming Interface | Method for software components to communicate |
| **Backprop** | Backpropagation | Algorithm to compute gradients for neural network training |
| **Batch** | Batch Size | Number of samples processed before updating weights |
| **BTC-USD** | Bitcoin to US Dollar | Trading pair for Bitcoin priced in US dollars |
| **CPU** | Central Processing Unit | Main processor in a computer |
| **CrossEntropyLoss** | Cross-Entropy Loss | Loss function measuring difference between predicted and actual probability distributions |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's parallel computing platform for GPUs |
| **Dropout** | Dropout Regularization | Technique that randomly disables neurons during training to prevent overfitting |
| **Embedding** | Vector Embedding | Dense numerical representation of text/data in continuous vector space |
| **Epoch** | Training Epoch | One complete pass through the entire training dataset |
| **F1 Score** | F1 Score | Harmonic mean of precision and recall (0-1, higher is better) |
| **GELU** | Gaussian Error Linear Unit | Smooth activation function: x * Φ(x) |
| **Gemma** | Gemma | Google's family of lightweight open language models |
| **GPU** | Graphics Processing Unit | Processor optimized for parallel computations |
| **HF** | HuggingFace | Platform hosting ML models and datasets |
| **JSON** | JavaScript Object Notation | Text format for structured data storage |
| **Logits** | Logits | Raw model outputs before applying softmax/sigmoid |
| **LR** | Learning Rate | Step size for gradient descent weight updates |
| **MPS** | Metal Performance Shaders | Apple's GPU computing framework for Mac/iOS |
| **NLP** | Natural Language Processing | AI field dealing with human language understanding |
| **One-Hot** | One-Hot Encoding | Representing categories as binary vectors (e.g., [0,1,0]) |
| **Optimizer** | Optimizer | Algorithm that updates model weights to minimize loss |
| **PTH** | PyTorch | File extension for PyTorch saved model weights |
| **ReLU** | Rectified Linear Unit | Activation function: max(0, x) |
| **SGD** | Stochastic Gradient Descent | Optimizer updating weights using random sample batches |
| **SHA256** | Secure Hash Algorithm 256-bit | Cryptographic hash function producing 256-bit output |
| **Sigmoid** | Sigmoid Function | Activation that squashes values to range (0, 1) |
| **Softmax** | Softmax Function | Converts logits to probability distribution summing to 1 |
| **Tensor** | Tensor | Multi-dimensional array (the fundamental data structure in PyTorch) |
| **Transformer** | Transformer Architecture | Neural network architecture using self-attention mechanisms |
| **yfinance** | Yahoo Finance | Python library for downloading market data from Yahoo Finance |

---

## Device Support

The model auto-detects the best available device in this priority order:
1. **CUDA** - NVIDIA GPUs (Linux/Windows)
2. **MPS** - Apple Silicon (M1/M2/M3 Macs)
3. **CPU** - Fallback for all systems

All scripts use `model.device` for automatic cross-platform compatibility. No manual configuration needed.

### Enabling GPU Acceleration (NVIDIA)

If you have an NVIDIA GPU but see `Using CPU` when running the scripts, PyTorch was likely installed without CUDA support.

**Step 1: Check Your GPU and CUDA Version**

```bash
nvidia-smi
```

Look for the CUDA Version in the output:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.44                 Driver Version: 591.44         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060      WDDM  |   00000000:01:00.0  On |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

**Step 2: Check if PyTorch Sees Your GPU**

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If you see `CUDA: False`, PyTorch needs to be reinstalled with CUDA support.

**Step 3: Reinstall PyTorch with CUDA**

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

| Your CUDA Version | PyTorch Install Command |
|-------------------|-------------------------|
| 11.8 or higher | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 or higher | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 or higher | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |

**Note**: CUDA is backward compatible. If your driver shows CUDA 13.1, you can use PyTorch built for CUDA 12.4.

**Step 4: Verify Installation**

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Expected output:**
```
CUDA: True | GPU: NVIDIA GeForce RTX 4060
```

Now when you run `train.py`, you should see:
```
Using GPU: NVIDIA GeForce RTX 4060
```

### Troubleshooting GPU Issues

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GPU TROUBLESHOOTING                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem: "Using CPU" even though you have NVIDIA GPU                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Cause: PyTorch installed without CUDA support                  │   │
│  │  Fix: Reinstall PyTorch with CUDA (see Step 3 above)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Problem: "CUDA out of memory" error                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Cause: GPU memory full (model + embeddings too large)          │   │
│  │  Fix: Reduce BATCH_SIZE in train.py (try 1 or 2)                │   │
│  │  Fix: Close other GPU applications (games, browsers)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Problem: nvidia-smi not found                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Cause: NVIDIA drivers not installed                            │   │
│  │  Fix: Download drivers from https://www.nvidia.com/drivers      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Problem: torch.cuda.is_available() returns False after reinstall       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Cause: Wrong CUDA version or conda/pip conflict                │   │
│  │  Fix: Check PyTorch version matches your CUDA:                  │   │
│  │       python -c "import torch; print(torch.__version__)"        │   │
│  │       Should show: 2.x.x+cu124 (not +cpu)                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Performance Comparison

| Device | Training Speed (approx) | Notes |
|--------|-------------------------|-------|
| **RTX 4060** | ~5-10x faster | Recommended for serious training |
| **RTX 3080** | ~8-15x faster | Great performance |
| **Apple M1/M2** | ~3-5x faster | Uses MPS backend |
| **CPU (Intel/AMD)** | Baseline | Slow, but works everywhere |

Using a GPU significantly speeds up training, especially with larger datasets.

---

## Target Symbols

| Symbol | Description |
|--------|-------------|
| BTC-USD | Bitcoin to US Dollar (currently implemented) |
| AAPL | Apple Inc. (planned) |
| TSLA | Tesla Inc. (planned) |
| GC=F | Gold Futures (planned) |

---

## Summary: Everything You've Learned

Congratulations! If you've read through this guide, you now understand how to build, train, and deploy an AI trading signal classifier. Let's recap the key concepts using our educational stories:

### The Big Picture (The Cooking Analogy)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    YOUR ML JOURNEY                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. GATHER RECIPES (download.py)                                        │
│     "Collect 100 pizza recipes and note how they turned out"            │
│     → Creates: news_with_price.json                                     │
│                                                                         │
│  2. TRAIN THE CHEF (train.py)                                           │
│     "Robot reads all recipes and learns patterns"                       │
│     → Creates: {SYMBOL}.pth (the trained "brain")                       │
│                                                                         │
│  3. TEST THE CHEF (test.py)                                             │
│     "Give robot new recipes it hasn't seen"                             │
│     → Reports: Accuracy & F1 Score (the "report card")                  │
│                                                                         │
│  4. USE THE CHEF (model.predict())                                      │
│     "Ask: Will this pizza be good?"                                     │
│     → Returns: BUY / HOLD / SELL with confidence                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Concepts Recap

| Concept | Story | Key Takeaway |
|---------|-------|--------------|
| **Model Training** | Cooking Analogy | Models learn patterns, not memorize answers |
| **Continuous Learning** | Student's Notebook | Keep notes between sessions (default: `--fresh` off) |
| **Smart Guard** | Teacher Checks Materials | Only train when there's new data to learn |
| **Overfitting** | Over-Studying Same Chapter | Training on same data = memorization, not learning |
| **Accuracy vs F1** | Report Card vs Lazy Student | F1 catches cheaters who guess common answers |
| **Parameter Consistency** | Answer Key Problem | Train/test thresholds MUST match |
| **Asset-Specific Models** | Only Knows Bitcoin | Each asset needs its own trained model |

### The Three Commands You Need

```bash
# Daily workflow (daemon-safe)
python download.py -s BTC-USD    # Get fresh data
python train.py -s BTC-USD       # Train (Smart Guard decides if needed)
python test.py -s BTC-USD        # Verify performance
```

### The Three Guards That Protect Your Model

| Guard | Question | Default |
|-------|----------|---------|
| **Data Hash** | "Is this new material?" | Always checked (PRIMARY) |
| **Cooldown** | "Has enough time passed?" | 5 minutes |
| **Min Samples** | "Is there enough to learn?" | 5 samples |

### Quick Reference: Threshold by Asset

| Asset Type | Threshold | Example Command |
|------------|-----------|-----------------|
| **Crypto** | ±1.0% | `python train.py -s BTC-USD` |
| **Volatile Stock** | ±0.5% | `python train.py -s TSLA -b 0.5 --sell-threshold -0.5` |
| **Large Cap** | ±0.3% | `python train.py -s AAPL -b 0.3 --sell-threshold -0.3` |
| **Index ETF** | ±0.2% | `python train.py -s SPY -b 0.2 --sell-threshold -0.2` |

### What Makes a Good Model?

| Metric | Bad | Okay | Good | Excellent |
|--------|-----|------|------|-----------|
| **Accuracy** | < 40% | 40-60% | 60-75% | > 75% |
| **F1 Score** | < 0.3 | 0.3-0.5 | 0.5-0.7 | > 0.7 |

### Final Words

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REMEMBER                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. This is a 5-MINUTE trading tool, not a long-term investment tool    │
│                                                                         │
│  2. Models learn PATTERNS, not predictions                              │
│     "Negative news often precedes drops" ≠ "This news WILL cause drop"  │
│                                                                         │
│  3. Past patterns don't guarantee future results                        │
│     Use as ONE signal among many, not as financial advice               │
│                                                                         │
│  4. Keep your model fresh with continuous learning                      │
│     Markets change → patterns change → model needs retraining           │
│                                                                         │
│  5. Different assets need different thresholds                          │
│     Crypto moves 1%+ in 5 min, but AAPL rarely moves 0.3%               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Happy Trading! May your F1 scores be high and your losses be low.* 🎯

---

**Document Version**: 2.0 | **Last Updated**: January 2025 | **Total Educational Stories**: 6

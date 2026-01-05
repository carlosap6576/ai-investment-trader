"""
Download and prepare market data for training.
Fetches price data and news, then merges them for model training.
"""

import argparse
import json
import datetime
import os
import shutil
import sys
import textwrap

# Delay yfinance import to allow --help to work without dependencies
yf = None

# =============================================================================
# DEFAULTS - Used when CLI arguments are not provided
# =============================================================================

DEFAULT_PERIOD = "1mo"          # How far back to fetch
DEFAULT_INTERVAL = "5m"         # Candle interval
DEFAULT_NEWS_COUNT = 100        # Number of news articles to fetch
DATASETS_DIR = "datasets"

# =============================================================================
# VALID OPTIONS AND CONSTRAINTS (based on market data API)
# =============================================================================

# Valid periods - NOTE: there is NO 2mo, 4mo, 7mo, etc.
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

# Interval to seconds mapping
INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "90m": 5400,
    "1h": 3600,
    "1d": 86400,
    "5d": 432000,
    "1wk": 604800,
    "1mo": 2592000,
    "3mo": 7776000,
}

VALID_INTERVALS = list(INTERVAL_SECONDS.keys())

# Period to approximate days mapping (for constraint validation)
PERIOD_TO_DAYS = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
    "ytd": 365,   # Approximation
    "max": 99999,  # Unlimited
}

# Intraday data limits (based on market data provider constraints)
INTERVAL_MAX_DAYS = {
    "1m": 7,       # 1-minute data: last 7 days only
    "2m": 60,      # 2-minute data: last 60 days
    "5m": 60,      # 5-minute data: last 60 days
    "15m": 60,     # 15-minute data: last 60 days
    "30m": 60,     # 30-minute data: last 60 days
    "60m": 730,    # 60-minute data: last 730 days (~2 years)
    "90m": 730,    # 90-minute data: last 730 days
    "1h": 730,     # 1-hour data: last 730 days
    "1d": 99999,   # Daily+: unlimited
    "5d": 99999,
    "1wk": 99999,
    "1mo": 99999,
    "3mo": 99999,
}

# =============================================================================
# END CONFIGURATION
# =============================================================================


# =============================================================================
# HELP TEXT AND CLI INTERFACE
# =============================================================================

HELP_TEXT = """
================================================================================
                        MARKET DATA DOWNLOADER
                    Trading Signal Training Data Tool
================================================================================

DESCRIPTION:
    Downloads price data and news for a given symbol, then prepares
    training data for the trading signal classifier.

--------------------------------------------------------------------------------
USAGE:
--------------------------------------------------------------------------------

    python download.py -s SYMBOL [OPTIONS]
    python download.py --symbol SYMBOL [OPTIONS]

--------------------------------------------------------------------------------
REQUIRED ARGUMENT:
--------------------------------------------------------------------------------

    -s, --symbol SYMBOL
        Trading symbol to download data for.

        Examples:
          Crypto:    BTC-USD, ETH-USD, DOGE-USD
          Stocks:    AAPL, TSLA, MSFT, GOOGL, NVDA
          ETFs:      SPY, QQQ, DIA
          Futures:   GC=F (Gold), CL=F (Oil)

--------------------------------------------------------------------------------
OPTIONAL ARGUMENTS:
--------------------------------------------------------------------------------

    -p, --period PERIOD
        How far back to fetch historical data.
        Default: 1mo

        Valid values:
          1d      Last 1 day
          5d      Last 5 days
          1mo     Last 1 month      <-- default
          3mo     Last 3 months
          6mo     Last 6 months
          1y      Last 1 year
          2y      Last 2 years
          5y      Last 5 years
          10y     Last 10 years
          ytd     Year to date
          max     All available data

        NOTE: There is NO 2mo, 4mo, 7mo, etc. Only the values listed above!

    -i, --interval INTERVAL
        Candle/bar interval for price data.
        Default: 5m

        Valid values:
          1m      1 minute        (max 7 days of data)
          2m      2 minutes       (max 60 days)
          5m      5 minutes       (max 60 days)   <-- default
          15m     15 minutes      (max 60 days)
          30m     30 minutes      (max 60 days)
          60m     60 minutes      (max 730 days)
          90m     90 minutes      (max 730 days)
          1h      1 hour          (max 730 days)
          1d      1 day           (unlimited)
          5d      5 days          (unlimited)
          1wk     1 week          (unlimited)
          1mo     1 month         (unlimited)
          3mo     3 months        (unlimited)

    -n, --news-count COUNT
        Number of news articles to fetch.
        Default: 100

    -h, --help
        Show this help message and exit.

--------------------------------------------------------------------------------
INTRADAY DATA LIMITS:
--------------------------------------------------------------------------------

    Intraday data has limited historical availability:

    +------------------+-------------+--------------------------------+
    | Interval         | Max Period  | Recommendation                 |
    +------------------+-------------+--------------------------------+
    | 1m               | 7 days      | Use -p 5d or -p 1d             |
    | 2m, 5m, 15m, 30m | 60 days     | Use -p 1mo                     |
    | 60m, 90m, 1h     | 730 days    | Use -p 1y or -p 2y             |
    | 1d, 5d, 1wk+     | Unlimited   | Any period works               |
    +------------------+-------------+--------------------------------+

    If you request too much data for an intraday interval, you'll get an error
    with suggestions for valid combinations.

--------------------------------------------------------------------------------
EXAMPLES:
--------------------------------------------------------------------------------

    # Basic usage (downloads 1 month of 5-minute data)
    python download.py -s BTC-USD

    # Download stock with 15-minute candles
    python download.py -s AAPL -p 1mo -i 15m

    # Download with hourly data for 1 year
    python download.py -s TSLA -p 1y -i 1h

    # Download with daily data for 5 years
    python download.py -s ETH-USD -p 5y -i 1d

    # Download with more news articles
    python download.py -s NVDA -p 1mo -i 5m -n 500

    # Full form (same as short form)
    python download.py --symbol MSFT --period 3mo --interval 1h --news-count 200

--------------------------------------------------------------------------------
OUTPUT:
--------------------------------------------------------------------------------

    Data is saved to: datasets/{SYMBOL}/

    Files created:
      - historical_data.json    Price data (timestamp -> price)
      - news.json               Raw news articles
      - news_with_price.json    Training data (news + price labels)

================================================================================
"""


def print_help():
    """Print the full help text."""
    print(HELP_TEXT)


def validate_period_interval(period, interval):
    """
    Validate that the period is compatible with the interval.
    Intraday data has limits on how far back it is available.

    Returns: (is_valid, error_message)
    """
    period_days = PERIOD_TO_DAYS.get(period, 0)
    max_days = INTERVAL_MAX_DAYS.get(interval, 99999)

    if period_days > max_days:
        # Build helpful suggestion
        suggestions = []
        for p, days in PERIOD_TO_DAYS.items():
            if days <= max_days and p in VALID_PERIODS:
                suggestions.append(p)

        suggestion_str = ", ".join(suggestions[:5])

        return False, (
            f"\n"
            f"ERROR: Invalid combination!\n"
            f"\n"
            f"  You requested:  --period {period} --interval {interval}\n"
            f"\n"
            f"  Problem: {interval} data is only available for the last {max_days} days.\n"
            f"           Your requested period ({period}) is approximately {period_days} days.\n"
            f"\n"
            f"  Valid periods for {interval} interval: {suggestion_str}\n"
            f"\n"
            f"  Solutions:\n"
            f"    1. Use a shorter period:  python download.py -s SYMBOL -p 1mo -i {interval}\n"
            f"    2. Use a longer interval: python download.py -s SYMBOL -p {period} -i 1d\n"
            f"\n"
            f"  Run 'python download.py --help' for more information.\n"
        )

    return True, None


class FriendlyArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that shows friendly help on errors."""

    def error(self, message):
        """Override error to show full help on any error."""
        sys.stderr.write(f"\n{'='*70}\n")
        sys.stderr.write(f"ERROR: {message}\n")
        sys.stderr.write(f"{'='*70}\n\n")

        # Show quick usage hint
        sys.stderr.write("QUICK USAGE:\n")
        sys.stderr.write("  python download.py -s SYMBOL [-p PERIOD] [-i INTERVAL] [-n COUNT]\n\n")

        sys.stderr.write("EXAMPLES:\n")
        sys.stderr.write("  python download.py -s BTC-USD\n")
        sys.stderr.write("  python download.py -s AAPL -p 1mo -i 15m\n")
        sys.stderr.write("  python download.py -s TSLA -p 1y -i 1h -n 500\n\n")

        sys.stderr.write("VALID PERIODS:\n")
        sys.stderr.write(f"  {', '.join(VALID_PERIODS)}\n")
        sys.stderr.write("  (Note: there is NO 2mo, 4mo, 7mo, etc.)\n\n")

        sys.stderr.write("VALID INTERVALS:\n")
        sys.stderr.write(f"  {', '.join(VALID_INTERVALS)}\n\n")

        sys.stderr.write("For full help, run: python download.py --help\n\n")
        sys.exit(2)

    def print_help(self, file=None):
        """Override to show our custom help."""
        print_help()


def parse_args():
    """Parse command-line arguments."""
    # Check for help flag first (before creating parser)
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) == 1:
        print_help()
        sys.exit(0)

    parser = FriendlyArgumentParser(
        add_help=False  # We handle help ourselves
    )

    parser.add_argument("-s", "--symbol", required=True)
    parser.add_argument("-p", "--period", default=DEFAULT_PERIOD, choices=VALID_PERIODS)
    parser.add_argument("-i", "--interval", default=DEFAULT_INTERVAL, choices=VALID_INTERVALS)
    parser.add_argument("-n", "--news-count", type=int, default=DEFAULT_NEWS_COUNT)
    parser.add_argument("-h", "--help", action="store_true")

    args = parser.parse_args()

    # Validate period/interval combination
    is_valid, error_msg = validate_period_interval(args.period, args.interval)
    if not is_valid:
        sys.stderr.write(error_msg)
        sys.exit(1)

    return args


class Config:
    """Configuration container for download parameters."""

    def __init__(self, symbol, period, interval, news_count):
        self.symbol = symbol.upper()
        self.period = period
        self.interval = interval
        self.interval_seconds = INTERVAL_SECONDS[interval]
        self.news_count = news_count

        # Derived paths
        self.symbol_dir = f"{DATASETS_DIR}/{self.symbol}"
        self.historical_data_file = f"{self.symbol_dir}/historical_data.json"
        self.news_file = f"{self.symbol_dir}/news.json"
        self.training_data_file = f"{self.symbol_dir}/news_with_price.json"

    def __str__(self):
        return (
            f"  Symbol: {self.symbol}\n"
            f"  Period: {self.period}\n"
            f"  Interval: {self.interval} ({self.interval_seconds}s)\n"
            f"  News count: {self.news_count}\n"
            f"  Output dir: {self.symbol_dir}/"
        )


# =============================================================================
# TICKER VALIDATION AND CLEANUP
# =============================================================================

def validate_ticker(symbol):
    """
    Validate that yfinance supports the given ticker symbol.

    This MUST be called before any file operations to fail fast on invalid symbols.

    Args:
        symbol: The ticker symbol to validate (e.g., "BTC-USD", "AAPL")

    Returns:
        tuple: (is_valid, error_message)
               is_valid: True if ticker is valid, False otherwise
               error_message: None if valid, helpful error string if invalid
    """
    import io
    import logging

    print(f"  Validating ticker '{symbol}'...")

    try:
        # Suppress library output during validation
        # This prevents noisy error messages from being printed
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)

        # Redirect stdout/stderr to suppress any print statements from the library
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            ticker = yf.Ticker(symbol)
            # Try to get minimal historical data to validate the ticker exists
            # Using 1 day of daily data - minimal API call
            hist = ticker.history(period="1d")
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        if hist.empty:
            # No data returned - ticker likely doesn't exist
            return False, (
                f"\n"
                f"{'='*70}\n"
                f"ERROR: Invalid or unsupported ticker symbol '{symbol}'\n"
                f"{'='*70}\n"
                f"\n"
                f"No market data was returned for this symbol.\n"
                f"\n"
                f"Possible reasons:\n"
                f"  • The symbol doesn't exist\n"
                f"  • The symbol is delisted\n"
                f"  • The market is closed and no recent data exists\n"
                f"  • Typo in the symbol name\n"
                f"\n"
                f"Examples of valid symbols:\n"
                f"  Crypto:   BTC-USD, ETH-USD, DOGE-USD\n"
                f"  Stocks:   AAPL, TSLA, MSFT, GOOGL, NVDA\n"
                f"  ETFs:     SPY, QQQ, DIA\n"
                f"  Futures:  GC=F (Gold), CL=F (Oil)\n"
                f"\n"
                f"Tip: Verify your symbol is a valid trading symbol\n"
                f"     (e.g., check a financial website or your broker).\n"
            )

        # Ticker is valid!
        print(f"  ✓ Ticker '{symbol}' is valid")
        return True, None

    except Exception as e:
        # Any exception means the ticker is problematic
        return False, (
            f"\n"
            f"{'='*70}\n"
            f"ERROR: Could not validate ticker symbol '{symbol}'\n"
            f"{'='*70}\n"
            f"\n"
            f"Error details: {str(e)}\n"
            f"\n"
            f"This could mean:\n"
            f"  • The symbol doesn't exist\n"
            f"  • Network connectivity issues\n"
            f"  • Market data API is temporarily unavailable\n"
            f"\n"
            f"Please check your symbol and try again.\n"
        )


def clean_symbol_directory(symbol):
    """
    Selectively clean data files while preserving trained models.

    This implements the SELECTIVE DELETION pattern:
    - DELETE: Data files (regenerable via download)
    - PRESERVE: Model files (.pth) (expensive to recreate)

    This allows running download.py multiple times per day without
    losing trained models that took hours to create.

    Args:
        symbol: The ticker symbol (used to construct directory path)

    Returns:
        dict: Summary of what was deleted and preserved
    """
    symbol_dir = f"{DATASETS_DIR}/{symbol.upper()}"

    # Files to delete (regenerable)
    data_files = [
        "historical_data.json",
        "news.json",
        "news_with_price.json",
    ]

    # Files to preserve (expensive to recreate)
    preserve_extensions = [".pth"]

    result = {"deleted": [], "preserved": [], "not_found": []}

    if not os.path.exists(symbol_dir):
        print(f"  No existing data directory - starting fresh")
        return result

    print(f"  Cleaning data files (preserving trained models)...")

    # Delete only data files
    for filename in data_files:
        filepath = os.path.join(symbol_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"    ✓ Removed: {filename}")
            result["deleted"].append(filename)
        else:
            result["not_found"].append(filename)

    # Report preserved files
    for filename in os.listdir(symbol_dir):
        filepath = os.path.join(symbol_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1]
            if ext in preserve_extensions:
                print(f"    ★ Preserved: {filename} (trained model)")
                result["preserved"].append(filename)

    if not result["deleted"] and not result["preserved"]:
        print(f"  No files to clean - starting fresh")
    elif result["deleted"] and not result["preserved"]:
        print(f"  ✓ Cleaned {len(result['deleted'])} data file(s)")
    elif result["preserved"]:
        print(f"  ✓ Cleaned {len(result['deleted'])} data file(s), preserved {len(result['preserved'])} model(s)")

    return result


def ensure_file_exists(filepath, default_content):
    """Create file with default content if it doesn't exist."""
    if not os.path.exists(filepath):
        print(f"  Creating {filepath}...")
        with open(filepath, 'w') as f:
            json.dump(default_content, f, indent=4)
        return True
    return False


def init_data_files(config):
    """Initialize all required data files if they don't exist."""
    # Create symbol directory if needed
    os.makedirs(config.symbol_dir, exist_ok=True)

    created = []

    if ensure_file_exists(config.historical_data_file, {}):
        created.append(config.historical_data_file)

    if ensure_file_exists(config.news_file, []):
        created.append(config.news_file)

    if ensure_file_exists(config.training_data_file, []):
        created.append(config.training_data_file)

    if created:
        print(f"  Initialized {len(created)} new file(s)")

    return created


def download_ticker(config):
    """
    Download historical price data.
    Merges with existing data to build a larger dataset over time.
    """
    print(f"  Fetching {config.interval} price data for last {config.period}...")

    # Load existing data
    try:
        with open(config.historical_data_file, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}

    # Download new data
    data = yf.download(
        tickers=config.symbol,
        period=config.period,
        interval=config.interval
    )

    if data.empty:
        print(f"  WARNING: No price data returned for {config.symbol}")
        print(f"           Check if the symbol is valid and the market is open.")
        return

    encoded = data.to_json()
    decoded = json.loads(encoded)

    # Extract open prices (key format from yfinance)
    price_key = f"('Open', '{config.symbol}')"
    new_prices = decoded.get(price_key, {})

    # Fallback for single-ticker format (older yfinance versions)
    if not new_prices and 'Open' in decoded:
        new_prices = decoded['Open']

    # Merge with existing data (new data takes precedence)
    for timestamp in existing_data:
        if timestamp not in new_prices:
            new_prices[timestamp] = existing_data[timestamp]

    # Save merged data
    with open(config.historical_data_file, 'w') as f:
        json.dump(new_prices, f, indent=4)

    print(f"  Saved {len(new_prices)} price points to {config.historical_data_file}")


def download_news(config):
    """
    Download news articles for the symbol.
    Appends to existing news to build a larger dataset over time.
    """
    print(f"  Fetching up to {config.news_count} news articles...")

    # Load existing news
    try:
        with open(config.news_file, 'r') as f:
            news = json.load(f)
    except FileNotFoundError:
        news = []

    # Download new news (handle different yfinance versions)
    ticker = yf.Ticker(config.symbol)

    try:
        # Try new API first (yfinance >= 0.2.40)
        new_news = ticker.get_news(count=config.news_count)
    except TypeError:
        # Fall back to old API (yfinance < 0.2.40)
        try:
            new_news = ticker.get_news()
        except Exception as e:
            print(f"  WARNING: Could not fetch news: {e}")
            new_news = None
    except Exception as e:
        print(f"  WARNING: Could not fetch news: {e}")
        new_news = None

    if new_news is None:
        new_news = []

    # Append new news (may contain duplicates - could be improved)
    news += new_news

    with open(config.news_file, 'w') as f:
        json.dump(news, f, indent=4)

    print(f"  Saved {len(news)} total articles to {config.news_file}")


def prepare_data(config):
    """
    Merge news with price data to create training dataset.

    For each news article:
    1. Find the price at publication time (rounded to interval)
    2. Find the price one interval later
    3. Calculate percentage change (future - current) / current

    Positive percentage = price went UP after news
    Negative percentage = price went DOWN after news
    """
    print("  Preparing training data...")
    output = []

    # Load data
    with open(config.historical_data_file, 'r') as f:
        ticker = json.load(f)
    with open(config.news_file, 'r') as f:
        news = json.load(f)

    if not ticker:
        print("  WARNING: No price data available. Skipping data preparation.")
        return

    if not news:
        print("  WARNING: No news data available. Skipping data preparation.")
        return

    matched = 0
    skipped = 0

    for item in news:
        # Handle different news formats
        try:
            if 'content' in item:
                title = item['content']['title']
                summary = item['content'].get('summary', '')
                pubDate = item['content']['pubDate']
            else:
                title = item.get('title', '')
                summary = item.get('summary', '')
                pubDate = item.get('pubDate', item.get('providerPublishTime', ''))
        except (KeyError, TypeError):
            skipped += 1
            continue

        if not pubDate or not title:
            skipped += 1
            continue

        # Convert pubDate to unix timestamp (seconds)
        try:
            if isinstance(pubDate, str):
                pubDate_ts = int(datetime.datetime.strptime(pubDate, '%Y-%m-%dT%H:%M:%SZ').timestamp())
            else:
                pubDate_ts = int(pubDate)
        except (ValueError, TypeError):
            skipped += 1
            continue

        # Round down to nearest interval boundary
        index = pubDate_ts - (pubDate_ts % config.interval_seconds)

        # Look up prices (keys are milliseconds, so append "000")
        price = ticker.get(f"{index}000")
        future_price = ticker.get(f"{index + config.interval_seconds}000")

        if price is None or future_price is None:
            skipped += 1
            continue

        # Calculate price change (FIXED: future - current, not current - future)
        difference = future_price - price
        percentage = (difference / price) * 100

        output.append({
            'title': title,
            'index': index,
            'price': price,
            'future_price': future_price,
            'difference': difference,
            'percentage': percentage,
            'summary': summary,
            'pubDate': pubDate if isinstance(pubDate, str) else datetime.datetime.fromtimestamp(pubDate).isoformat(),
            'pubDate_ts': pubDate_ts,
        })
        matched += 1

    with open(config.training_data_file, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"  Matched {matched} articles with price data, skipped {skipped}")
    print(f"  Saved training data to {config.training_data_file}")


def main():
    """Main entry point - downloads data and prepares training set."""
    args = parse_args()

    # Lazy import data provider (allows --help to work without dependencies)
    global yf
    try:
        import yfinance as yf_module
        yf = yf_module
    except ImportError:
        sys.stderr.write("\nERROR: Required dependencies are not installed.\n")
        sys.stderr.write("\nTo install them, run:\n")
        sys.stderr.write("  pip install -r requirements.txt\n\n")
        sys.exit(1)

    print("=" * 60)
    print("MARKET DATA DOWNLOADER")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Validate ticker FIRST (fail fast on invalid symbols)
    # =========================================================================
    print("\n[1/4] VALIDATING TICKER")
    print("-" * 60)

    is_valid, error_msg = validate_ticker(args.symbol)
    if not is_valid:
        sys.stderr.write(error_msg)
        sys.exit(1)

    # =========================================================================
    # STEP 2: Clean existing data (ensures fresh start)
    # =========================================================================
    print("\n[2/4] PREPARING FRESH DIRECTORY")
    print("-" * 60)

    clean_symbol_directory(args.symbol)

    # =========================================================================
    # STEP 3: Create configuration and initialize files
    # =========================================================================
    print("\n[3/4] INITIALIZING")
    print("-" * 60)

    config = Config(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        news_count=args.news_count
    )

    print(f"  Configuration:")
    print(f"    Symbol:     {config.symbol}")
    print(f"    Period:     {config.period}")
    print(f"    Interval:   {config.interval} ({config.interval_seconds}s)")
    print(f"    News count: {config.news_count}")
    print(f"    Output dir: {config.symbol_dir}/")

    # Create fresh data files
    init_data_files(config)

    # =========================================================================
    # STEP 4: Download and prepare data
    # =========================================================================
    print("\n[4/4] DOWNLOADING DATA")
    print("-" * 60)

    download_ticker(config)
    download_news(config)
    prepare_data(config)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nFiles created in {config.symbol_dir}/:")
    print(f"  • historical_data.json   (price data)")
    print(f"  • news.json              (news articles)")
    print(f"  • news_with_price.json   (training data)")
    print(f"\nNext steps:")
    print(f"  python train.py -s {config.symbol}")
    print()


if __name__ == "__main__":
    main()

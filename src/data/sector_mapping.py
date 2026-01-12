"""
GICS (Global Industry Classification Standard) sector mappings.

This module provides mappings from ticker symbols to their GICS sectors
and sector-related utilities.
"""

from typing import Dict, List, Optional, Set

# GICS Sector Codes and Names
GICS_SECTORS = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# Common ticker to GICS sector code mapping
# This is a subset - in production you'd use a data provider API
TICKER_GICS_MAPPING: Dict[str, str] = {
    # === Information Technology (45) ===
    "AAPL": "45",
    "MSFT": "45",
    "NVDA": "45",
    "AMD": "45",
    "INTC": "45",
    "AVGO": "45",
    "QCOM": "45",
    "TXN": "45",
    "MU": "45",
    "AMAT": "45",
    "LRCX": "45",
    "ADI": "45",
    "MRVL": "45",
    "KLAC": "45",
    "NXPI": "45",
    "CRM": "45",
    "ORCL": "45",
    "IBM": "45",
    "NOW": "45",
    "ADBE": "45",
    "INTU": "45",
    "SNPS": "45",
    "CDNS": "45",
    "PANW": "45",
    "CRWD": "45",
    "FTNT": "45",

    # === Communication Services (50) ===
    "GOOGL": "50",
    "GOOG": "50",
    "META": "50",
    "NFLX": "50",
    "DIS": "50",
    "CMCSA": "50",
    "VZ": "50",
    "T": "50",
    "TMUS": "50",
    "CHTR": "50",
    "EA": "50",
    "TTWO": "50",
    "ATVI": "50",  # Now MSFT

    # === Consumer Discretionary (25) ===
    "AMZN": "25",
    "TSLA": "25",
    "HD": "25",
    "MCD": "25",
    "NKE": "25",
    "SBUX": "25",
    "LOW": "25",
    "TJX": "25",
    "BKNG": "25",
    "MAR": "25",
    "GM": "25",
    "F": "25",
    "TGT": "25",
    "ROST": "25",
    "ORLY": "25",
    "AZO": "25",
    "CMG": "25",
    "YUM": "25",
    "DPZ": "25",
    "RCL": "25",
    "CCL": "25",
    "WYNN": "25",
    "LVS": "25",
    "MGM": "25",

    # === Health Care (35) ===
    "UNH": "35",
    "JNJ": "35",
    "LLY": "35",
    "PFE": "35",
    "ABBV": "35",
    "MRK": "35",
    "TMO": "35",
    "ABT": "35",
    "DHR": "35",
    "BMY": "35",
    "AMGN": "35",
    "GILD": "35",
    "VRTX": "35",
    "REGN": "35",
    "MRNA": "35",
    "BIIB": "35",
    "ISRG": "35",
    "MDT": "35",
    "SYK": "35",
    "BSX": "35",
    "EW": "35",
    "ZTS": "35",
    "CVS": "35",
    "CI": "35",
    "HUM": "35",
    "ELV": "35",

    # === Financials (40) ===
    "BRK.A": "40",
    "BRK.B": "40",
    "JPM": "40",
    "V": "40",
    "MA": "40",
    "BAC": "40",
    "WFC": "40",
    "GS": "40",
    "MS": "40",
    "C": "40",
    "AXP": "40",
    "BLK": "40",
    "SCHW": "40",
    "CB": "40",
    "PGR": "40",
    "MMC": "40",
    "AON": "40",
    "ICE": "40",
    "CME": "40",
    "SPGI": "40",
    "MCO": "40",
    "USB": "40",
    "PNC": "40",
    "TFC": "40",
    "COF": "40",

    # === Consumer Staples (30) ===
    "PG": "30",
    "KO": "30",
    "PEP": "30",
    "COST": "30",
    "WMT": "30",
    "PM": "30",
    "MO": "30",
    "MDLZ": "30",
    "CL": "30",
    "KMB": "30",
    "EL": "30",
    "KHC": "30",
    "GIS": "30",
    "K": "30",
    "HSY": "30",
    "SJM": "30",
    "CAG": "30",
    "MKC": "30",
    "CLX": "30",
    "CHD": "30",
    "STZ": "30",
    "BF.B": "30",
    "DEO": "30",

    # === Industrials (20) ===
    "UPS": "20",
    "UNP": "20",
    "HON": "20",
    "RTX": "20",
    "BA": "20",
    "CAT": "20",
    "DE": "20",
    "LMT": "20",
    "GE": "20",
    "MMM": "20",
    "GD": "20",
    "NOC": "20",
    "WM": "20",
    "RSG": "20",
    "EMR": "20",
    "ETN": "20",
    "ITW": "20",
    "PH": "20",
    "ROK": "20",
    "FDX": "20",
    "CSX": "20",
    "NSC": "20",
    "DAL": "20",
    "UAL": "20",
    "LUV": "20",
    "AAL": "20",

    # === Energy (10) ===
    "XOM": "10",
    "CVX": "10",
    "COP": "10",
    "SLB": "10",
    "EOG": "10",
    "MPC": "10",
    "PSX": "10",
    "VLO": "10",
    "OXY": "10",
    "PXD": "10",
    "DVN": "10",
    "HAL": "10",
    "BKR": "10",
    "KMI": "10",
    "WMB": "10",
    "OKE": "10",

    # === Materials (15) ===
    "LIN": "15",
    "APD": "15",
    "SHW": "15",
    "ECL": "15",
    "DD": "15",
    "DOW": "15",
    "NEM": "15",
    "FCX": "15",
    "NUE": "15",
    "VMC": "15",
    "MLM": "15",
    "PPG": "15",
    "ALB": "15",
    "CTVA": "15",
    "CF": "15",
    "MOS": "15",

    # === Utilities (55) ===
    "NEE": "55",
    "DUK": "55",
    "SO": "55",
    "D": "55",
    "AEP": "55",
    "SRE": "55",
    "EXC": "55",
    "XEL": "55",
    "PCG": "55",
    "ED": "55",
    "WEC": "55",
    "ES": "55",
    "AWK": "55",
    "ATO": "55",

    # === Real Estate (60) ===
    "AMT": "60",
    "PLD": "60",
    "CCI": "60",
    "EQIX": "60",
    "SPG": "60",
    "PSA": "60",
    "O": "60",
    "WELL": "60",
    "DLR": "60",
    "AVB": "60",
    "EQR": "60",
    "VTR": "60",
    "ARE": "60",
    "MAA": "60",
    "UDR": "60",
    "ESS": "60",
    "SBAC": "60",

    # === Cryptocurrency (special category) ===
    "BTC-USD": "CRYPTO",
    "ETH-USD": "CRYPTO",
    "SOL-USD": "CRYPTO",
    "ADA-USD": "CRYPTO",
    "XRP-USD": "CRYPTO",
    "DOGE-USD": "CRYPTO",
    "DOT-USD": "CRYPTO",
    "AVAX-USD": "CRYPTO",
    "MATIC-USD": "CRYPTO",
    "LINK-USD": "CRYPTO",

    # === ETFs (special category) ===
    "SPY": "ETF",
    "QQQ": "ETF",
    "DIA": "ETF",
    "IWM": "ETF",
    "VTI": "ETF",
    "VOO": "ETF",
    "XLK": "ETF",
    "XLF": "ETF",
    "XLE": "ETF",
    "XLV": "ETF",
    "XLI": "ETF",
    "XLP": "ETF",
    "XLY": "ETF",
    "XLB": "ETF",
    "XLU": "ETF",
    "XLRE": "ETF",

    # === Futures (special category) ===
    "GC=F": "COMMODITY",  # Gold
    "SI=F": "COMMODITY",  # Silver
    "CL=F": "COMMODITY",  # Crude Oil
    "NG=F": "COMMODITY",  # Natural Gas
}

# Sector keywords for classification
SECTOR_KEYWORDS: Dict[str, Set[str]] = {
    "10": {"energy", "oil", "gas", "petroleum", "drilling", "refinery", "crude", "opec", "pipeline"},
    "15": {"materials", "chemicals", "metals", "mining", "steel", "aluminum", "copper", "gold", "silver"},
    "20": {"industrials", "aerospace", "defense", "airlines", "railroads", "trucking", "machinery", "construction"},
    "25": {"consumer discretionary", "retail", "automotive", "restaurants", "hotels", "leisure", "apparel", "luxury"},
    "30": {"consumer staples", "food", "beverage", "household", "personal care", "tobacco", "supermarket", "grocery"},
    "35": {"health care", "pharmaceutical", "biotech", "medical devices", "hospitals", "drug", "fda", "clinical trial"},
    "40": {"financials", "banking", "insurance", "investment", "credit", "mortgage", "lending", "asset management"},
    "45": {"technology", "software", "semiconductor", "cloud", "cybersecurity", "it services", "hardware", "chip"},
    "50": {"communication", "media", "entertainment", "telecom", "streaming", "advertising", "social media", "gaming"},
    "55": {"utilities", "electric", "power", "water", "renewable", "nuclear", "grid", "generation"},
    "60": {"real estate", "reit", "property", "commercial real estate", "residential", "office space", "retail space"},
    "CRYPTO": {"crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "defi", "nft", "web3", "altcoin"},
}


def get_sector(ticker: str) -> Optional[str]:
    """
    Get the GICS sector code for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        GICS sector code (e.g., "45") or None if unknown
    """
    return TICKER_GICS_MAPPING.get(ticker.upper())


def get_sector_name(sector_code: str) -> str:
    """
    Get the human-readable sector name from a GICS code.

    Args:
        sector_code: GICS sector code (e.g., "45")

    Returns:
        Sector name (e.g., "Information Technology")
    """
    if sector_code == "CRYPTO":
        return "Cryptocurrency"
    elif sector_code == "ETF":
        return "Exchange Traded Fund"
    elif sector_code == "COMMODITY":
        return "Commodity"
    return GICS_SECTORS.get(sector_code, "Unknown")


def get_sector_tickers(sector_code: str) -> List[str]:
    """
    Get all tickers in a given sector.

    Args:
        sector_code: GICS sector code

    Returns:
        List of tickers in that sector
    """
    return [ticker for ticker, code in TICKER_GICS_MAPPING.items() if code == sector_code]


def get_sector_peers(ticker: str, limit: int = 10) -> List[str]:
    """
    Get peer tickers in the same sector.

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of peers to return

    Returns:
        List of peer tickers (excluding the input ticker)
    """
    sector = get_sector(ticker)
    if not sector:
        return []

    peers = [t for t in get_sector_tickers(sector) if t != ticker.upper()]
    return peers[:limit]


def detect_sector_from_text(text: str) -> Optional[str]:
    """
    Detect sector from news text using keyword matching.

    Args:
        text: News headline or summary

    Returns:
        GICS sector code if detected, None otherwise
    """
    text_lower = text.lower()

    # Count matches for each sector
    sector_scores = {}
    for sector_code, keywords in SECTOR_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            sector_scores[sector_code] = score

    if not sector_scores:
        return None

    # Return sector with highest score
    return max(sector_scores.keys(), key=lambda k: sector_scores[k])


# Mapping from ticker to company name/keywords for relevance filtering
TICKER_KEYWORDS: Dict[str, List[str]] = {
    # Large Cap Tech
    "AAPL": ["Apple", "iPhone", "iPad", "Mac", "iOS", "App Store", "Tim Cook", "Apple Watch"],
    "MSFT": ["Microsoft", "Windows", "Azure", "Xbox", "Office", "Teams", "Satya Nadella", "LinkedIn"],
    "GOOGL": ["Google", "Alphabet", "Android", "YouTube", "Chrome", "Pixel", "Sundar Pichai", "Search"],
    "GOOG": ["Google", "Alphabet", "Android", "YouTube", "Chrome", "Pixel", "Sundar Pichai", "Search"],
    "AMZN": ["Amazon", "AWS", "Prime", "Alexa", "Echo", "Jeff Bezos", "Andy Jassy", "Kindle"],
    "META": ["Meta", "Facebook", "Instagram", "WhatsApp", "Messenger", "Mark Zuckerberg", "Metaverse", "Threads"],
    "NVDA": ["Nvidia", "GeForce", "CUDA", "GPU", "RTX", "Jensen Huang", "AI chip", "Graphics"],
    "TSLA": ["Tesla", "Elon Musk", "EV", "Cybertruck", "Model S", "Model 3", "Model X", "Model Y", "Supercharger"],

    # Other tech
    "AMD": ["AMD", "Ryzen", "Radeon", "EPYC", "Lisa Su", "Xilinx"],
    "INTC": ["Intel", "Core", "Xeon", "Atom", "Pat Gelsinger", "Foundry"],
    "CRM": ["Salesforce", "CRM", "Marc Benioff", "Slack", "Tableau"],
    "ORCL": ["Oracle", "Larry Ellison", "Cloud Infrastructure", "Database"],
    "IBM": ["IBM", "Watson", "Red Hat", "Mainframe", "Arvind Krishna"],
    "NFLX": ["Netflix", "Streaming", "Reed Hastings", "Subscriber", "Original content"],

    # Financials
    "JPM": ["JPMorgan", "JP Morgan", "Chase", "Jamie Dimon"],
    "BAC": ["Bank of America", "BofA", "Merrill"],
    "WFC": ["Wells Fargo"],
    "GS": ["Goldman Sachs", "Goldman"],
    "MS": ["Morgan Stanley"],
    "V": ["Visa", "Card payments"],
    "MA": ["Mastercard", "Card payments"],

    # Healthcare
    "UNH": ["UnitedHealth", "United Health", "Optum"],
    "JNJ": ["Johnson & Johnson", "J&J", "Tylenol", "Band-Aid"],
    "PFE": ["Pfizer", "Vaccine", "COVID"],
    "MRK": ["Merck", "Keytruda"],
    "LLY": ["Eli Lilly", "Lilly", "Mounjaro", "Ozempic competitor"],

    # Consumer
    "WMT": ["Walmart", "Wal-Mart"],
    "COST": ["Costco", "Membership", "Warehouse"],
    "HD": ["Home Depot"],
    "MCD": ["McDonald's", "McDonalds", "Big Mac"],
    "KO": ["Coca-Cola", "Coke"],
    "PEP": ["Pepsi", "PepsiCo", "Frito-Lay"],
    "NKE": ["Nike", "Just Do It", "Air Jordan"],

    # Crypto
    "BTC-USD": ["Bitcoin", "BTC", "crypto", "cryptocurrency", "Satoshi", "halving", "mining"],
    "ETH-USD": ["Ethereum", "ETH", "Ether", "Vitalik", "smart contract", "DeFi", "staking"],
    "SOL-USD": ["Solana", "SOL"],
    "XRP-USD": ["Ripple", "XRP"],
    "DOGE-USD": ["Dogecoin", "DOGE", "Shiba"],

    # ETFs
    "SPY": ["S&P 500", "SPY", "S&P500", "Standard & Poor"],
    "QQQ": ["Nasdaq 100", "QQQ", "Nasdaq-100", "tech stocks"],

    # Commodities
    "GC=F": ["Gold", "precious metal", "bullion", "gold price"],
    "CL=F": ["Oil", "crude", "WTI", "petroleum", "OPEC"],
}


def get_ticker_keywords(ticker: str) -> List[str]:
    """
    Get keywords associated with a ticker for relevance filtering.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of keywords associated with the ticker
    """
    return TICKER_KEYWORDS.get(ticker.upper(), [ticker.upper()])

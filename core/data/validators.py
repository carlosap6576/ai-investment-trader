/"""
Data validation utilities.

Provides validation functions for periods, intervals, tickers, and thresholds.
"""

from core.config.constants import (
    VALID_PERIODS,
    VALID_INTERVALS,
    PERIOD_TO_DAYS,
    INTERVAL_MAX_DAYS,
)


def validate_period_interval(period, interval):
    """
    Validate that the period is compatible with the interval.

    Intraday data has limits on how far back it is available.

    Args:
        period: The data period (e.g., "1mo", "1y").
        interval: The data interval (e.g., "5m", "1h").

    Returns:
        tuple: (is_valid, error_message)
            is_valid: True if combination is valid.
            error_message: None if valid, helpful error string if invalid.
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
            f"    1. Use a shorter period:  -p 1mo -i {interval}\n"
            f"    2. Use a longer interval: -p {period} -i 1d\n"
        )

    return True, None


def validate_thresholds(buy_threshold, sell_threshold):
    """
    Validate buy and sell thresholds.

    Args:
        buy_threshold: Buy threshold (must be positive).
        sell_threshold: Sell threshold (must be negative).

    Returns:
        tuple: (is_valid, error_message)
    """
    if buy_threshold <= 0:
        return False, f"Buy threshold must be positive (got {buy_threshold})"
    if sell_threshold >= 0:
        return False, f"Sell threshold must be negative (got {sell_threshold})"
    return True, None


def validate_split(split):
    """
    Validate train/test split ratio.

    Args:
        split: Split ratio (must be between 0 and 1 exclusive).

    Returns:
        tuple: (is_valid, error_message)
    """
    if split <= 0 or split >= 1:
        return False, f"Split must be between 0 and 1 (got {split})"
    return True, None


def validate_positive_int(value, name, min_value=1):
    """
    Validate that a value is a positive integer.

    Args:
        value: The value to validate.
        name: The parameter name (for error messages).
        min_value: Minimum allowed value.

    Returns:
        tuple: (is_valid, error_message)
    """
    if value < min_value:
        return False, f"{name} must be at least {min_value} (got {value})"
    return True, None


def validate_positive_float(value, name):
    """
    Validate that a value is a positive float.

    Args:
        value: The value to validate.
        name: The parameter name (for error messages).

    Returns:
        tuple: (is_valid, error_message)
    """
    if value <= 0:
        return False, f"{name} must be positive (got {value})"
    return True, None

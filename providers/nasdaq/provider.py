"""Nasdaq Data Link data providers."""

from typing import Any, Dict, List, Optional

from ..base import BaseProvider
from .config import (
    API_KEY,
    ENDPOINTS,
)


class NasdaqTimeseriesProvider(BaseProvider):
    """
    Downloads time-series data from Nasdaq Data Link using get().

    Usage:
        provider = NasdaqTimeseriesProvider("GDP", "FRED")
        provider.run()  # Downloads FRED/GDP economic data

    Output: datasets/{SYMBOL}/nasdaq_{database}_{symbol}.json
    """

    def __init__(
        self,
        symbol: str,
        database: str = "WIKI",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        super().__init__(symbol)
        self.database = database.upper()
        self.start_date = start_date
        self.end_date = end_date
        self._ndl = None

    @property
    def ndl(self):
        """Lazy load nasdaq-data-link."""
        if self._ndl is None:
            import nasdaqdatalink
            nasdaqdatalink.ApiConfig.api_key = API_KEY
            self._ndl = nasdaqdatalink
        return self._ndl

    @property
    def name(self) -> str:
        return f"Nasdaq {self.database}"

    @property
    def output_filename(self) -> str:
        template = ENDPOINTS["timeseries"]["output_file"]
        return template.format(database=self.database, symbol=self.symbol)

    def download(self) -> List[Dict[str, Any]]:
        """Download time-series data."""
        dataset = f"{self.database}/{self.symbol}"
        print(f"  Fetching {dataset}...")

        try:
            kwargs = {"dataset": dataset}
            if self.start_date:
                kwargs["start_date"] = self.start_date
            if self.end_date:
                kwargs["end_date"] = self.end_date

            df = self.ndl.get(**kwargs)
            if df is None or df.empty:
                print(f"  WARNING: No data returned for {dataset}")
                return []

            # Convert DataFrame to list of dicts
            df = df.reset_index()
            records = df.to_dict(orient="records")

            # Convert dates to strings
            for record in records:
                for key, value in record.items():
                    if hasattr(value, 'isoformat'):
                        record[key] = value.isoformat()

            return records

        except Exception as e:
            print(f"  WARNING: Could not fetch {dataset}: {e}")
            return []


class NasdaqTableProvider(BaseProvider):
    """
    Downloads table data from Nasdaq Data Link using get_table().

    Usage:
        provider = NasdaqTableProvider("AAPL", "WIKI/PRICES")
        provider.run()  # Downloads WIKI/PRICES for AAPL

    Output: datasets/{SYMBOL}/nasdaq_table_{table}.json
    """

    def __init__(
        self,
        symbol: str,
        table: str = "WIKI/PRICES",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__(symbol)
        self.table = table
        # Don't apply default dates - let the API return all available data
        self.start_date = start_date
        self.end_date = end_date
        self.columns = columns
        self._ndl = None

    @property
    def ndl(self):
        """Lazy load nasdaq-data-link."""
        if self._ndl is None:
            import nasdaqdatalink
            nasdaqdatalink.ApiConfig.api_key = API_KEY
            self._ndl = nasdaqdatalink
        return self._ndl

    @property
    def name(self) -> str:
        return f"Nasdaq {self.table}"

    @property
    def output_filename(self) -> str:
        # Replace / with _ for filename
        table_name = self.table.replace("/", "_")
        template = ENDPOINTS["table"]["output_file"]
        return template.format(table=table_name)

    def download(self) -> List[Dict[str, Any]]:
        """Download table data with pagination."""
        date_info = ""
        if self.start_date and self.end_date:
            date_info = f" ({self.start_date} to {self.end_date})"
        elif self.start_date:
            date_info = f" (from {self.start_date})"
        elif self.end_date:
            date_info = f" (until {self.end_date})"

        print(f"  Fetching {self.table} for {self.symbol}{date_info}...")

        try:
            kwargs = {
                "ticker": self.symbol,
                "paginate": True,
            }

            # Only add date filter if dates are specified
            if self.start_date or self.end_date:
                date_filter = {}
                if self.start_date:
                    date_filter["gte"] = self.start_date
                if self.end_date:
                    date_filter["lte"] = self.end_date
                kwargs["date"] = date_filter

            if self.columns:
                kwargs["qopts"] = {"columns": self.columns}

            df = self.ndl.get_table(self.table, **kwargs)
            if df is None or df.empty:
                print(f"  WARNING: No data returned for {self.table}")
                return []

            records = df.to_dict(orient="records")

            # Convert dates to strings
            for record in records:
                for key, value in record.items():
                    if hasattr(value, 'isoformat'):
                        record[key] = value.isoformat()

            return records

        except Exception as e:
            print(f"  WARNING: Could not fetch {self.table}: {e}")
            return []


class NasdaqPricesProvider(NasdaqTableProvider):
    """
    Downloads stock prices from WIKI/PRICES table.

    Usage:
        provider = NasdaqPricesProvider("AAPL")
        provider.run()

    Output: datasets/{SYMBOL}/nasdaq_prices.json
    """

    def __init__(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        super().__init__(
            symbol=symbol,
            table="WIKI/PRICES",
            start_date=start_date,
            end_date=end_date,
            columns=["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"],
        )

    @property
    def name(self) -> str:
        return "Nasdaq Prices"

    @property
    def output_filename(self) -> str:
        return "nasdaq_prices.json"


class NasdaqProvider:
    """
    Convenience wrapper to download Nasdaq Data Link data.

    Usage:
        # Stock prices from WIKI/PRICES table
        provider = NasdaqProvider("AAPL")
        provider.run()

        # Economic data from FRED
        provider = NasdaqProvider("GDP", database="FRED", use_timeseries=True)
        provider.run()
    """

    def __init__(
        self,
        symbol: str,
        database: str = "WIKI",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_timeseries: bool = False,
    ) -> None:
        self.symbol = symbol.upper()
        self.database = database.upper()
        self.use_timeseries = use_timeseries

        if use_timeseries:
            # Use get() for time-series data (FRED, etc.)
            self.provider = NasdaqTimeseriesProvider(
                symbol, database, start_date, end_date
            )
        else:
            # Use get_table() for stock data (WIKI/PRICES)
            self.provider = NasdaqPricesProvider(symbol, start_date, end_date)

    def run(self) -> Dict[str, Any]:
        """Run Nasdaq download."""
        key = "timeseries" if self.use_timeseries else "prices"
        return {
            key: self.provider.run(),
        }

"""Backward-compatible data fetch helpers.

This module maintains the legacy `fetch_stock_data` API used by some
models in the repo.

It is a small shim around `ml.data.pipeline.fetch_price`.
"""

from ml.data.pipeline import fetch_price


def fetch_stock_data(ticker: str, period: str = "3y"):
    """Fetch price history for a ticker.

    This is a thin wrapper around :func:`ml.data.pipeline.fetch_price`.
    """
    return fetch_price(ticker, period=period)

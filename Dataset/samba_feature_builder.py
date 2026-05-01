#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the 82-variable daily market feature files used by SAMBA.

SAMBA datasets contain Date, Name, and 82 feature columns for each target
market. The target market contributes primitive and technical features, while
shared external market variables are merged into each target file.

Use --mode legacy with --external-alignment reverse-position to match the
published SAMBA dataset convention. Use --mode causal with --external-alignment
date for chronological feature engineering on new data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DATE_COL = "Date"
NAME_COL = "Name"

LOCAL_FEATURES = [
    "Price",
    "Vol.",
    "weekday",
    "mom",
    "mom1",
    "mom2",
    "mom3",
    "ROC_5",
    "ROC_10",
    "ROC_15",
    "ROC_20",
    "EMA_10",
    "EMA_20",
    "EMA_50",
    "EMA_200",
]

# SAMBA dataset column order. Name replaces the target market's own external
# return column so each target still has 82 numeric features.
EXTERNAL_FEATURE_ORDER = [
    "DGS10",
    "WTI-oil",
    "FTSE-F",
    "HSI-F",
    "Gold-F",
    "NZD",
    "FCHI",
    "DGS5",
    "Brent",
    "DBAA",
    "NYSE",
    "CTB6M",
    "CTB1Y",
    "XAU",
    "S&P-F",
    "AUD",
    "AMZN",
    "RUSSELL-F",
    "CNY",
    "DTB3",
    "MSFT",
    "IXIC",
    "DTB4WK",
    "silver-F",
    "CAD",
    "DAX-F",
    "DTB6",
    "DAAA",
    "DJI-F",
    "XAG",
    "HSI",
    "XOM",
    "EUR",
    "WFC",
    "Dollar Index-F",
    "DJI",
    "GE",
    "CTB3M",
    "copper-F",
    "RUT",
    "Dollar Index",
    "JPM",
    "GAS-F",
    "JPY",
    "wheat-F",
    "GBP",
    "GSPC",
    "SSEC",
    "Nikkei-F",
    "CHF",
    "oil",
    "KOSPI-F",
    "AAPL",
    "GDAXI",
    "CAC-F",
    "NASDAQ-F",
    "JNJ",
    "FTSE",
    "TE1",
    "TE2",
    "TE3",
    "TE5",
    "TE6",
    "DE1",
    "DE2",
    "DE4",
    "DE5",
    "DE6",
]

ALL_FEATURES = LOCAL_FEATURES + EXTERNAL_FEATURE_ORDER
ALL_KNOWN_COLUMNS = set(ALL_FEATURES) | {DATE_COL, NAME_COL, "Volume"}

RATE_LEVEL_FEATURES = {
    "DTB4WK",
    "DTB3",
    "DTB6",
    "DGS5",
    "DGS10",
    "DAAA",
    "DBAA",
    "CTB3M",
    "CTB6M",
    "CTB1Y",
}

SPREAD_FORMULAS = {
    "TE1": ("DGS10", "DTB4WK"),
    "TE2": ("DGS10", "DTB3"),
    "TE3": ("DGS10", "DTB6"),
    "TE5": ("DTB3", "DTB4WK"),
    "TE6": ("DTB6", "DTB4WK"),
    # SAMBA default-spread feature.
    "DE1": ("DBAA", "DAAA"),
    "DE2": ("DBAA", "DGS10"),
    "DE4": ("DBAA", "DTB6"),
    "DE5": ("DBAA", "DTB3"),
    "DE6": ("DBAA", "DTB4WK"),
}

SPREAD_FEATURES = set(SPREAD_FORMULAS)
WORLD_INDEX_RETURN_COLUMNS = {"IXIC", "GSPC", "DJI", "NYSE", "RUT"}

MARKET_TO_RETURN_COLUMN = {
    "IXIC": "IXIC",
    "NASDAQ": "IXIC",
    "NASDAQCOMPOSITE": "IXIC",
    "DJI": "DJI",
    "DJIA": "DJI",
    "DOW": "DJI",
    "DOWJONES": "DJI",
    "NYSE": "NYSE",
    "GSPC": "GSPC",
    "SP500": "GSPC",
    "SPX": "GSPC",
    "S&P500": "GSPC",
    "RUT": "RUT",
    "RUSSELL": "RUT",
    "RUSSELL2000": "RUT",
}


def _norm_key(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


ALIASES = {
    # SAMBA date and identity aliases.
    "date": DATE_COL,
    "datetime": DATE_COL,
    "time": DATE_COL,
    "name": NAME_COL,
    "ticker": NAME_COL,
    "symbol": NAME_COL,
    # SAMBA target and raw price aliases.
    "price": "Price",
    "close": "Price",
    "adjclose": "Price",
    "adjustedclose": "Price",
    "last": "Price",
    "value": "Price",
    "volume": "Volume",
    "vol": "Volume",
    # SAMBA world-index aliases.
    "ixic": "IXIC",
    "nasdaq": "IXIC",
    "nasdaqcomposite": "IXIC",
    "nasdaqcompositeindex": "IXIC",
    "gspc": "GSPC",
    "sp500": "GSPC",
    "sandp500": "GSPC",
    "snp500": "GSPC",
    "dji": "DJI",
    "djia": "DJI",
    "dow": "DJI",
    "dowjones": "DJI",
    "dowjonesindustrialaverage": "DJI",
    "nyse": "NYSE",
    "nyacomposite": "NYSE",
    "rut": "RUT",
    "russell": "RUT",
    "russell2000": "RUT",
    "hsi": "HSI",
    "hangseng": "HSI",
    "sse": "SSEC",
    "ssec": "SSEC",
    "shanghai": "SSEC",
    "shanghaicomposite": "SSEC",
    "fchi": "FCHI",
    "cac40": "FCHI",
    "ftse": "FTSE",
    "ftse100": "FTSE",
    "gdaxi": "GDAXI",
    "dax": "GDAXI",
    # SAMBA exchange-rate aliases.
    "usdjpy": "JPY",
    "usdjy": "JPY",
    "usdy": "JPY",
    "jpy": "JPY",
    "usdgbp": "GBP",
    "gbp": "GBP",
    "usdcad": "CAD",
    "cad": "CAD",
    "usdcny": "CNY",
    "cny": "CNY",
    "usdaud": "AUD",
    "aud": "AUD",
    "usdnzd": "NZD",
    "nzd": "NZD",
    "usdchf": "CHF",
    "chf": "CHF",
    "usdeur": "EUR",
    "eur": "EUR",
    "usdx": "Dollar Index",
    "dollarindex": "Dollar Index",
    "usdollarindex": "Dollar Index",
    "dollarindexf": "Dollar Index-F",
    "usdxf": "Dollar Index-F",
    # SAMBA commodity aliases.
    "oil": "oil",
    "wti": "oil",
    "wtioil": "WTI-oil",
    "brent": "Brent",
    "gold": "Gold-F",
    "goldf": "Gold-F",
    "goldfutures": "Gold-F",
    "xau": "XAU",
    "xauusd": "XAU",
    "xag": "XAG",
    "xagusd": "XAG",
    "gas": "GAS-F",
    "naturalgas": "GAS-F",
    "gasf": "GAS-F",
    "silver": "silver-F",
    "silverf": "silver-F",
    "silverfutures": "silver-F",
    "copper": "copper-F",
    "copperf": "copper-F",
    "copperfutures": "copper-F",
    "wheat": "wheat-F",
    "wheatf": "wheat-F",
    "wheatfutures": "wheat-F",
    # SAMBA company aliases.
    "xom": "XOM",
    "jpm": "JPM",
    "aapl": "AAPL",
    "msft": "MSFT",
    "ge": "GE",
    "jnj": "JNJ",
    "wfc": "WFC",
    "amzn": "AMZN",
    # SAMBA futures aliases.
    "fchif": "CAC-F",
    "cacf": "CAC-F",
    "cac40futures": "CAC-F",
    "ftsef": "FTSE-F",
    "ftse100futures": "FTSE-F",
    "gdaxif": "DAX-F",
    "daxf": "DAX-F",
    "daxfutures": "DAX-F",
    "hsif": "HSI-F",
    "hangsengfutures": "HSI-F",
    "nikkeif": "Nikkei-F",
    "nikkeifutures": "Nikkei-F",
    "kospif": "KOSPI-F",
    "kospifutures": "KOSPI-F",
    "ixicf": "NASDAQ-F",
    "nasdaqf": "NASDAQ-F",
    "nasdaqfutures": "NASDAQ-F",
    "djif": "DJI-F",
    "djiafutures": "DJI-F",
    "spf": "S&P-F",
    "sp500f": "S&P-F",
    "sandp500f": "S&P-F",
    "sp500futures": "S&P-F",
    "russellf": "RUSSELL-F",
    "russell2000futures": "RUSSELL-F",
    # SAMBA rate and spread aliases.
    "dtb4wk": "DTB4WK",
    "dtb3": "DTB3",
    "dtb6": "DTB6",
    "dgs5": "DGS5",
    "dgs10": "DGS10",
    "daaa": "DAAA",
    "dbaa": "DBAA",
    "ctb3m": "CTB3M",
    "ctb6m": "CTB6M",
    "ctb1y": "CTB1Y",
    "te1": "TE1",
    "te2": "TE2",
    "te3": "TE3",
    "te5": "TE5",
    "te6": "TE6",
    "de1": "DE1",
    "de2": "DE2",
    "de4": "DE4",
    "de5": "DE5",
    "de6": "DE6",
}


def canonical_column_name(name: str) -> str:
    stripped = str(name).strip()
    if stripped in ALL_KNOWN_COLUMNS:
        return stripped
    return ALIASES.get(_norm_key(stripped), stripped)


def canonical_market_name(name: str) -> str:
    canonical = canonical_column_name(str(name))
    key = _norm_key(canonical).upper()
    return MARKET_TO_RETURN_COLUMN.get(key, canonical.upper())


def target_return_column(target_name: str) -> str | None:
    key = _norm_key(target_name).upper()
    if key in MARKET_TO_RETURN_COLUMN:
        return MARKET_TO_RETURN_COLUMN[key]
    canonical = canonical_column_name(target_name)
    key = _norm_key(canonical).upper()
    return MARKET_TO_RETURN_COLUMN.get(key)


def parse_numeric_series(series: pd.Series) -> pd.Series:
    """Parse numeric, comma-separated, percent, and K/M/B/T suffixed values."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    def parse_one(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
            return np.nan
        negative = text.startswith("(") and text.endswith(")")
        text = text.strip("()").replace(",", "").replace("$", "")
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1]
        multiplier = 1.0
        if text and text[-1].upper() in {"K", "M", "B", "T"}:
            multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[text[-1].upper()]
            text = text[:-1]
        try:
            number = float(text) * multiplier
        except ValueError:
            return np.nan
        if negative:
            number = -number
        if is_percent:
            number /= 100.0
        return number

    return series.map(parse_one).astype(float)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases and merge duplicate canonical columns by first non-null."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        canonical = canonical_column_name(col)
        values = df[col]
        if canonical in out.columns:
            out[canonical] = out[canonical].combine_first(values)
        else:
            out[canonical] = values
    return out


def read_source_csv(path: str | Path, date_column: str = DATE_COL) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df = canonicalize_columns(df)

    canonical_date = canonical_column_name(date_column)
    if canonical_date not in df.columns:
        date_candidates = [c for c in df.columns if canonical_column_name(c) == DATE_COL]
        if not date_candidates:
            raise ValueError(f"{path} does not contain a date column")
        canonical_date = date_candidates[0]

    if canonical_date != DATE_COL:
        df = df.rename(columns={canonical_date: DATE_COL})

    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    if dates.isna().any():
        bad = int(dates.isna().sum())
        raise ValueError(f"{path} has {bad} unparseable date values")

    try:
        dates = dates.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass

    df[DATE_COL] = dates.dt.normalize()
    df = df.sort_values(DATE_COL).drop_duplicates(DATE_COL, keep="last")
    df = df.reset_index(drop=True)
    return df


def select_value_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for candidate in ["Price", "Close", "Adj Close", "Value", "Last"]:
        canonical = canonical_column_name(candidate)
        if canonical in df.columns:
            return canonical
    numeric_candidates = [
        col
        for col in df.columns
        if col not in {DATE_COL, NAME_COL}
        and pd.to_numeric(parse_numeric_series(df[col]), errors="coerce").notna().any()
    ]
    if not numeric_candidates:
        raise ValueError("No numeric value column was found")
    return numeric_candidates[-1]


def relative_change(series: pd.Series, periods: int = 1, mode: str = "legacy") -> pd.Series:
    values = parse_numeric_series(series)
    if mode == "legacy":
        return values / values.shift(-periods) - 1.0
    if mode == "causal":
        return values / values.shift(periods) - 1.0
    raise ValueError(f"Unknown mode: {mode}")


def exponential_moving_average(series: pd.Series, span: int, mode: str = "legacy") -> pd.Series:
    values = parse_numeric_series(series)
    if mode == "legacy":
        return values.iloc[::-1].ewm(span=span, adjust=False, min_periods=0).mean().iloc[::-1]
    if mode == "causal":
        return values.ewm(span=span, adjust=False, min_periods=0).mean()
    raise ValueError(f"Unknown mode: {mode}")


def lagged_return(base_return: pd.Series, lag: int, mode: str) -> pd.Series:
    if lag == 0:
        return base_return
    if mode == "legacy":
        return base_return.shift(-lag)
    if mode == "causal":
        return base_return.shift(lag)
    raise ValueError(f"Unknown mode: {mode}")


def infer_target_name(target_df: pd.DataFrame, target_path: str | Path, explicit: str | None) -> str:
    if explicit:
        return canonical_market_name(explicit)
    if NAME_COL in target_df.columns:
        names = target_df[NAME_COL].dropna().astype(str).str.strip()
        if not names.empty:
            return canonical_market_name(names.iloc[0])
    stem = Path(target_path).stem
    for token in stem.replace("-", "_").split("_"):
        maybe = target_return_column(token)
        if maybe:
            return canonical_market_name(maybe)
    return canonical_market_name(stem)


def build_local_features(
    target_df: pd.DataFrame,
    mode: str,
    volume_mode: str = "auto",
) -> pd.DataFrame:
    if "Price" not in target_df.columns:
        raise ValueError("Target CSV must contain a close/price column")

    out = pd.DataFrame()
    out[DATE_COL] = target_df[DATE_COL]
    close = parse_numeric_series(target_df["Price"])
    out["Price"] = close
    out["weekday"] = target_df[DATE_COL].dt.dayofweek.astype(int)

    has_raw_volume = "Volume" in target_df.columns
    has_feature_volume = "Vol." in target_df.columns
    if volume_mode == "raw" or (volume_mode == "auto" and has_raw_volume):
        volume_col = "Volume" if has_raw_volume else "Vol."
        if volume_col not in target_df.columns:
            raise ValueError("volume_mode=raw requires a Volume or Vol. column")
        out["Vol."] = relative_change(target_df[volume_col], 1, mode)
    elif volume_mode == "feature" or (volume_mode == "auto" and has_feature_volume):
        out["Vol."] = parse_numeric_series(target_df["Vol."])
    else:
        out["Vol."] = np.nan

    one_day = relative_change(close, 1, mode)
    out["mom"] = lagged_return(one_day, 0, mode)
    out["mom1"] = lagged_return(one_day, 1, mode)
    out["mom2"] = lagged_return(one_day, 2, mode)
    out["mom3"] = lagged_return(one_day, 3, mode)

    for period in [5, 10, 15, 20]:
        out[f"ROC_{period}"] = relative_change(close, period, mode) * 100.0

    for span in [10, 20, 50, 200]:
        out[f"EMA_{span}"] = exponential_moving_average(close, span, mode)

    return out[[DATE_COL] + LOCAL_FEATURES]


def combine_on_date(left: pd.DataFrame | None, right: pd.DataFrame) -> pd.DataFrame:
    if left is None:
        return right.copy()
    combined = left.set_index(DATE_COL)
    incoming = right.set_index(DATE_COL)
    combined = combined.reindex(combined.index.union(incoming.index))
    incoming = incoming.reindex(combined.index)
    for col in incoming.columns:
        if col in combined.columns:
            combined[col] = combined[col].combine_first(incoming[col])
        else:
            combined[col] = incoming[col]
    return combined.reset_index()


def feature_from_raw_series(df: pd.DataFrame, feature_name: str) -> pd.Series:
    canonical = canonical_column_name(feature_name)
    value_col = canonical if canonical in df.columns else select_value_column(df)
    values = parse_numeric_series(df[value_col])
    if canonical in RATE_LEVEL_FEATURES:
        return values
    if canonical in SPREAD_FEATURES:
        raise ValueError(f"{feature_name} is derived from rate columns; provide the source rates")
    return relative_change(values, 1, "causal")


def read_raw_series_specs(specs: Iterable[str], date_column: str) -> pd.DataFrame | None:
    merged: pd.DataFrame | None = None
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Raw series spec must be FEATURE=path.csv, got: {spec}")
        feature_name, path = spec.split("=", 1)
        feature_name = canonical_column_name(feature_name)
        series_df = read_source_csv(path, date_column)
        out = pd.DataFrame({DATE_COL: series_df[DATE_COL]})
        out[feature_name] = feature_from_raw_series(series_df, feature_name)
        if merged is None:
            merged = out
        else:
            base = merged.set_index(DATE_COL)
            incoming = out.set_index(DATE_COL)
            base = base.reindex(base.index.union(incoming.index))
            incoming = incoming.reindex(base.index)
            base[feature_name] = incoming[feature_name]
            merged = base.reset_index()
    return merged


def align_external_to_target_dates(
    features: pd.DataFrame,
    target_dates: Sequence[pd.Timestamp],
    external_alignment: str,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame({DATE_COL: pd.to_datetime(target_dates)})

    features = features.sort_values(DATE_COL).drop_duplicates(DATE_COL, keep="last")
    target = pd.DataFrame({DATE_COL: pd.to_datetime(pd.Series(target_dates)).dt.normalize()})

    if external_alignment == "date":
        return target.merge(features, on=DATE_COL, how="left")

    if external_alignment != "reverse-position":
        raise ValueError(f"Unknown external_alignment: {external_alignment}")

    target_dates_set = set(target[DATE_COL])
    source = features[features[DATE_COL].isin(target_dates_set)].reset_index(drop=True)
    values = source.drop(columns=[DATE_COL]).iloc[::-1].reset_index(drop=True)
    values = values.reindex(range(len(target)))
    return pd.concat([target.reset_index(drop=True), values], axis=1)


def build_external_features(
    external_sources: list[pd.DataFrame],
    raw_series: pd.DataFrame | None,
    external_mode: str,
    target_dates: Sequence[pd.Timestamp],
    external_alignment: str,
) -> pd.DataFrame:
    external: pd.DataFrame | None = None
    for source in external_sources:
        external = combine_on_date(external, source)
    if external is None:
        external = pd.DataFrame(columns=[DATE_COL])

    external = canonicalize_columns(external)
    if DATE_COL not in external.columns:
        external[DATE_COL] = pd.to_datetime([])

    external = external.sort_values(DATE_COL).drop_duplicates(DATE_COL, keep="last")
    external = external.reset_index(drop=True)

    features = pd.DataFrame({DATE_COL: external[DATE_COL]})

    if external_mode == "features":
        for col in EXTERNAL_FEATURE_ORDER:
            if col in external.columns and col not in SPREAD_FEATURES:
                features[col] = parse_numeric_series(external[col])
            elif col in external.columns and col in SPREAD_FEATURES:
                features[col] = parse_numeric_series(external[col])
    elif external_mode == "raw":
        for col in EXTERNAL_FEATURE_ORDER:
            if col in SPREAD_FEATURES or col not in external.columns:
                continue
            values = parse_numeric_series(external[col])
            if col in RATE_LEVEL_FEATURES:
                features[col] = values
            else:
                features[col] = relative_change(values, 1, "causal")
    else:
        raise ValueError(f"Unknown external_mode: {external_mode}")

    if raw_series is not None:
        raw_series = canonicalize_columns(raw_series)
        base = features.set_index(DATE_COL)
        incoming = raw_series.set_index(DATE_COL)
        for col in incoming.columns:
            base[col] = incoming[col]
        features = base.reset_index()

    features = compute_spreads(features)
    return align_external_to_target_dates(features, target_dates, external_alignment)


def compute_spreads(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for spread, (left, right) in SPREAD_FORMULAS.items():
        if left in out.columns and right in out.columns:
            out[spread] = parse_numeric_series(out[left]) - parse_numeric_series(out[right])
        elif spread not in out.columns:
            out[spread] = np.nan
    return out


def output_columns_for_target(target_name: str) -> list[str]:
    omit_return = target_return_column(target_name)
    columns = [DATE_COL] + LOCAL_FEATURES
    inserted_name = False
    for col in EXTERNAL_FEATURE_ORDER:
        if omit_return and col == omit_return:
            columns.append(NAME_COL)
            inserted_name = True
        else:
            columns.append(col)
    if not inserted_name:
        columns.insert(1 + len(LOCAL_FEATURES), NAME_COL)
    return columns


def build_samba_features(
    target_df: pd.DataFrame,
    target_name: str,
    external_features: pd.DataFrame,
    mode: str,
    volume_mode: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    target_name = canonical_market_name(target_name)
    local = build_local_features(target_df, mode=mode, volume_mode=volume_mode)

    result = local.set_index(DATE_COL)
    ext = external_features.set_index(DATE_COL) if not external_features.empty else pd.DataFrame()
    if not ext.empty:
        result = result.join(ext, how="left", rsuffix="_external")

    result = compute_spreads(result.reset_index()).set_index(DATE_COL)
    result[NAME_COL] = target_name

    for feature in EXTERNAL_FEATURE_ORDER:
        if feature == target_return_column(target_name):
            continue
        if feature not in result.columns:
            result[feature] = np.nan

    output = result.reset_index()
    if start_date:
        start = pd.to_datetime(start_date).normalize()
        output = output[output[DATE_COL] >= start]
    if end_date:
        end = pd.to_datetime(end_date).normalize()
        output = output[output[DATE_COL] <= end]

    columns = output_columns_for_target(target_name)
    return output[columns]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SAMBA 82-feature market datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-csv", required=True, help="Target market OHLCV/price CSV.")
    parser.add_argument("--target-name", help="Target market name, e.g. IXIC, DJI, NYSE, GSPC, RUT.")
    parser.add_argument("--output", help="Output CSV path.")
    parser.add_argument(
        "--mode",
        choices=["legacy", "causal"],
        default="legacy",
        help="legacy reproduces the published SAMBA convention; causal uses chronological history.",
    )
    parser.add_argument(
        "--external-csv",
        action="append",
        default=[],
        help="External table with Date plus external feature or raw series columns. Can be repeated.",
    )
    parser.add_argument(
        "--external-mode",
        choices=["features", "raw"],
        default="features",
        help="Whether --external-csv values are already feature values or raw price/level series.",
    )
    parser.add_argument(
        "--external-alignment",
        choices=["date", "reverse-position"],
        default="date",
        help="Align shared external variables by Date or by reversed row position.",
    )
    parser.add_argument(
        "--raw-series",
        action="append",
        default=[],
        metavar="FEATURE=CSV",
        help="Raw one-series CSV to overlay, e.g. GSPC=sp500.csv or DGS10=dgs10.csv. Can be repeated.",
    )
    parser.add_argument(
        "--volume-mode",
        choices=["auto", "raw", "feature"],
        default="auto",
        help="How to treat the target volume column.",
    )
    parser.add_argument("--date-column", default=DATE_COL, help="Date column name in input CSVs.")
    parser.add_argument("--start-date", help="Filter output after computing features.")
    parser.add_argument("--end-date", help="Filter output after computing features.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    target_df = read_source_csv(args.target_csv, args.date_column)
    target_name = infer_target_name(target_df, args.target_csv, args.target_name)

    external_sources = [read_source_csv(path, args.date_column) for path in args.external_csv]
    if not external_sources:
        external_sources = [target_df]

    raw_series = read_raw_series_specs(args.raw_series, args.date_column)
    external_features = build_external_features(
        external_sources=external_sources,
        raw_series=raw_series,
        external_mode=args.external_mode,
        target_dates=target_df[DATE_COL],
        external_alignment=args.external_alignment,
    )

    output = build_samba_features(
        target_df=target_df,
        target_name=target_name,
        external_features=external_features,
        mode=args.mode,
        volume_mode=args.volume_mode,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    output_path = Path(args.output) if args.output else Path(args.target_csv).with_name(
        f"combined_dataframe_{target_name}_rebuilt.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with shape {output.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

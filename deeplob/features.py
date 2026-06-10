"""Per-snapshot feature engineering and DeepLOB label construction.

Feature blocks (concatenated in this order) per order-book snapshot:

* **OFI**      - per-level Order Flow Imbalance for the top ``n_levels`` (``n_levels``)
* **trades**   - 4 aggregated trade-tape features aligned to the snapshot

The label is the *smoothed* DeepLOB return ``l_t = (m_+ - m_-) / m_-`` thresholded
by ``alpha`` into 0=down / 1=flat / 2=up (see :func:`smoothed_label`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _level_cols(prefix: str, field: str, n_levels: int) -> list[str]:
    return [f"{prefix}_{field}_{i}" for i in range(1, n_levels + 1)]


def order_flow_imbalance(
    ob: pd.DataFrame, n_levels: int
) -> tuple[np.ndarray, list[str]]:
    """Per-level OFI (Cont et al.) for the top ``n_levels``; first row is zero-padded."""
    bp = ob[_level_cols("bid", "price", n_levels)].to_numpy(dtype=np.float64)
    bv = ob[_level_cols("bid", "volume", n_levels)].to_numpy(dtype=np.float64)
    ap = ob[_level_cols("ask", "price", n_levels)].to_numpy(dtype=np.float64)
    av = ob[_level_cols("ask", "volume", n_levels)].to_numpy(dtype=np.float64)

    cur, prev = slice(1, None), slice(0, -1)
    d_bid = np.where(
        bp[cur] > bp[prev],
        bv[cur],
        np.where(bp[cur] == bp[prev], bv[cur] - bv[prev], -bv[prev]),
    )
    d_ask = np.where(
        ap[cur] < ap[prev],
        av[cur],
        np.where(ap[cur] == ap[prev], av[cur] - av[prev], -av[prev]),
    )
    ofi = d_bid - d_ask  # (n-1, n_levels)
    ofi = np.vstack([np.zeros((1, n_levels)), ofi])
    return ofi, [f"ofi_{i}" for i in range(1, n_levels + 1)]


def trade_features(ob: pd.DataFrame, tr: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Aggregate the trade tape onto each snapshot (zero-filled when no trades occur).

    Returns ``(n, 4)``: trade-flow imbalance, log total volume, signed VWAP deviation
    from mid (bps), and trade-count imbalance.
    """
    mid = (ob["ask_price_1"] + ob["bid_price_1"]).to_numpy(dtype=np.float64) / 2.0
    t = tr.copy()
    is_buy = t["direction"].eq("buy")
    t["buy_vol"] = np.where(is_buy, t["volume"], 0.0)
    t["sell_vol"] = np.where(~is_buy, t["volume"], 0.0)
    t["n_buy"] = is_buy.astype(np.int64)
    t["n_sell"] = (~is_buy).astype(np.int64)
    t["pv"] = t["price"] * t["volume"]

    agg = t.groupby("snapshot_time").agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        n_buy=("n_buy", "sum"),
        n_sell=("n_sell", "sum"),
        pv=("pv", "sum"),
        vol=("volume", "sum"),
    )
    agg = agg.reindex(ob["time"].to_numpy())  # align to snapshots; missing -> NaN

    buy_vol = agg["buy_vol"].to_numpy()
    sell_vol = agg["sell_vol"].to_numpy()
    n_buy = agg["n_buy"].to_numpy()
    n_sell = agg["n_sell"].to_numpy()
    vol = agg["vol"].to_numpy()
    vwap = np.divide(
        agg["pv"].to_numpy(), vol, out=np.full_like(vol, np.nan), where=vol > 0
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        tfi = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        cnt_imb = (n_buy - n_sell) / (n_buy + n_sell)
        log_vol = np.log1p(np.nan_to_num(buy_vol + sell_vol))
        vwap_dev_bps = (vwap - mid) / mid * 1e4

    feats = np.column_stack(
        [
            np.nan_to_num(tfi),
            log_vol,
            np.nan_to_num(vwap_dev_bps),
            np.nan_to_num(cnt_imb),
        ]
    )
    return feats, ["trade_tfi", "trade_log_vol", "trade_vwap_dev_bps", "trade_cnt_imb"]


def build_features(
    ob: pd.DataFrame, tr: pd.DataFrame, n_levels: int
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Assemble the full feature matrix ``(n, NF)``, its column names, and the mid-price."""
    ofi, ofi_names = order_flow_imbalance(ob, n_levels)
    trd, trd_names = trade_features(ob, tr)
    feats = np.concatenate([ofi, trd], axis=1).astype(np.float32)
    mid = ((ob["ask_price_1"] + ob["bid_price_1"]) / 2.0).to_numpy(dtype=np.float64)
    return feats, ofi_names + trd_names, mid


def rolling_daily_zscore(
    feats: np.ndarray, times: np.ndarray, lookback_days: int
) -> np.ndarray:
    """Z-score each row using the mean/std of the **previous ``lookback_days`` days**.

    This is the DeepLOB normalization scheme: statistics come only from strictly
    earlier calendar days, so it is free of look-ahead (val/test days are scaled by
    past train days). The first day has no history and falls back to its own stats.
    """
    days = pd.Series(pd.to_datetime(times)).dt.normalize().to_numpy()
    unique_days = np.unique(days)
    day_rows = {d: np.where(days == d)[0] for d in unique_days}

    out = np.empty_like(feats, dtype=np.float32)
    for di, day in enumerate(unique_days):
        prior = unique_days[max(0, di - lookback_days) : di]  # up to N previous days
        ref = (
            np.concatenate([day_rows[p] for p in prior])
            if len(prior)
            else day_rows[day]
        )
        mu = feats[ref].mean(axis=0)
        sd = feats[ref].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        rows = day_rows[day]
        out[rows] = (feats[rows] - mu) / sd
    return out


def smoothed_label(mid: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """DeepLOB smoothed-return label; returns ``-1`` where the signal is undefined (edges)."""
    s = pd.Series(mid)
    m_minus = s.rolling(k).mean()
    m_plus = s.rolling(k).mean().shift(-k)
    signal = ((m_plus - m_minus) / m_minus).to_numpy()

    y = np.full(len(signal), -1, dtype=np.int64)
    valid = ~np.isnan(signal)
    y[valid] = 1  # flat
    y[valid & (signal < -alpha)] = 0  # down
    y[valid & (signal > alpha)] = 2  # up
    return y

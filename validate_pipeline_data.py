#!/usr/bin/env python3
"""
Pipeline Data Validation Script

Run this immediately after main.py to verify the HDF5 output is correct
before launching any model training. Completes in < 5 seconds.

Usage:
    python validate_pipeline_data.py

Exit codes:
    0 = all checks passed
    1 = one or more checks failed (do NOT proceed to training)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Configuration ────────────────────────────────────────────────────────────

HDF5_PATH = Path('data/processed/assets.h5')
HDF5_KEY  = 'engineered_features'

EXPECTED_FEATURES = [
    'ret_1d', 'ret_5d', 'ret_21d',
    'vol_20d', 'atr_14',
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'bb_pct', 'bb_width',
    'volume_ratio', 'sma_200_dist'
]

# These must NOT exist (removed as non-stationary)
FORBIDDEN_COLS = ['bb_mid', 'bb_upper', 'bb_lower']

REQUIRED_EXTRA_COLS = [
    'Close', 'High', 'Low', 'Open', 'Volume',
    'vix', 'vix_diff', 'vix_chg',
    'label', 'label_binary', 'days_to_barrier',
    'ret_1d_forward', 'ret_10d_forward'
]

# Optional columns (legacy, not used in training but kept for reference)
OPTIONAL_COLS = ['ret_5d_forward', 'label_5d_binary', 'label_10d_binary']

EXPECTED_TICKER = 'SPY'
DATE_MIN = '2004-01-01'
DATE_MAX = '2024-12-31'
MIN_ROWS_SPY = 4500   # ~20 years of trading days
MAX_NAN_PCT  = 5.0    # max % NaN allowed per feature column

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures = []
warnings_list = []

def check(condition, label, detail=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        failures.append(label)

def warn(condition, label, detail=""):
    if not condition:
        print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))
        warnings_list.append(label)
    else:
        print(f"  {PASS} {label}")

# ── Main validation ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PIPELINE DATA VALIDATION")
    print("=" * 70)

    # ── 1. HDF5 file exists ───────────────────────────────────────────────────
    print("\n[1] HDF5 file")
    check(HDF5_PATH.exists(), f"File exists: {HDF5_PATH}")
    if not HDF5_PATH.exists():
        print(f"\n{FAIL} Cannot continue — HDF5 not found. Run main.py first.")
        sys.exit(1)

    # ── 2. Load data ──────────────────────────────────────────────────────────
    print("\n[2] Loading data")
    try:
        df = pd.read_hdf(HDF5_PATH, key=HDF5_KEY)
        check(True, f"Loaded successfully: shape={df.shape}")
    except Exception as e:
        check(False, f"Load failed: {e}")
        sys.exit(1)

    cols = df.columns.tolist()

    # ── 3. Index structure ────────────────────────────────────────────────────
    print("\n[3] Index structure")
    check(
        isinstance(df.index, pd.MultiIndex),
        "MultiIndex present"
    )
    check(
        list(df.index.names) == ['ticker', 'date'],
        f"Index names = ['ticker', 'date']",
        f"got {df.index.names}"
    )

    # ── 4. Tickers ────────────────────────────────────────────────────────────
    print("\n[4] Tickers")
    tickers = df.index.get_level_values('ticker').unique().tolist()
    check(
        tickers == [EXPECTED_TICKER],
        f"Only SPY present",
        f"got {tickers}"
    )

    # ── 5. Date range ─────────────────────────────────────────────────────────
    print("\n[5] Date range")
    dates = df.index.get_level_values('date')
    date_min = str(dates.min().date())
    date_max = str(dates.max().date())
    check(
        date_min <= '2004-01-05',
        f"Start date within first week of 2004 (first trading day after Jan 1)",
        f"got {date_min}"
    )
    check(
        date_max >= '2024-12-01',
        f"End date ≥ 2024-12-01",
        f"got {date_max}"
    )
    print(f"       Date range: {date_min} → {date_max}")

    # ── 6. Row count ──────────────────────────────────────────────────────────
    print("\n[6] Row count (SPY)")
    spy_rows = len(df)
    check(
        spy_rows >= MIN_ROWS_SPY,
        f"≥ {MIN_ROWS_SPY} rows",
        f"got {spy_rows}"
    )
    print(f"       Total rows: {spy_rows}")

    # ── 7. Required feature columns ───────────────────────────────────────────
    print("\n[7] Expected 13 technical features")
    missing_features = [f for f in EXPECTED_FEATURES if f not in cols]
    check(
        len(missing_features) == 0,
        f"All 13 features present",
        f"missing: {missing_features}"
    )
    present = [f for f in EXPECTED_FEATURES if f in cols]
    print(f"       Present ({len(present)}/13): {present}")

    # ── 8. Forbidden columns ──────────────────────────────────────────────────
    print("\n[8] Non-stationary columns removed")
    found_forbidden = [c for c in FORBIDDEN_COLS if c in cols]
    check(
        len(found_forbidden) == 0,
        f"bb_mid / bb_upper / bb_lower absent",
        f"still present: {found_forbidden}"
    )

    # ── 9. Required extra columns ─────────────────────────────────────────────
    print("\n[9] Required extra columns (OHLCV, labels, VIX, ret_1d_forward)")
    missing_extra = [c for c in REQUIRED_EXTRA_COLS if c not in cols]
    check(
        len(missing_extra) == 0,
        "All required extra columns present",
        f"missing: {missing_extra}"
    )

    # ── 10. Label distribution ────────────────────────────────────────────────
    print("\n[10] Label distribution (label_binary)")
    if 'label_binary' in cols:
        lb = df['label_binary'].dropna()
        n_total = len(lb)
        n_pos = (lb == 1).sum()
        n_neg = (lb == -1).sum()
        n_other = n_total - n_pos - n_neg
        pct_pos = n_pos / n_total * 100 if n_total > 0 else 0
        check(
            n_other == 0,
            "label_binary contains only {-1, 1}",
            f"{n_other} unexpected values found"
        )
        check(
            30 <= pct_pos <= 70,
            f"Label balance reasonable (30–70% positive)",
            f"positive = {pct_pos:.1f}%"
        )
        print(f"       label=+1: {n_pos:5d} ({pct_pos:.1f}%)")
        print(f"       label=-1: {n_neg:5d} ({100-pct_pos:.1f}%)")

    # ── 11. NaN counts per feature ────────────────────────────────────────────
    print("\n[11] NaN counts per feature column")
    nan_issues = []
    for feat in EXPECTED_FEATURES:
        if feat not in cols:
            continue
        nan_pct = df[feat].isna().mean() * 100
        if nan_pct > MAX_NAN_PCT:
            nan_issues.append(f"{feat}: {nan_pct:.1f}%")
    check(
        len(nan_issues) == 0,
        f"All features have < {MAX_NAN_PCT}% NaN",
        f"high NaN: {nan_issues}"
    )

    # ── 12. ret_1d_forward check ──────────────────────────────────────────────
    print("\n[12] ret_1d_forward")
    if 'ret_1d_forward' in cols:
        n_nan = df['ret_1d_forward'].isna().sum()
        # Expect exactly 1 NaN per ticker (last row)
        n_tickers = len(tickers)
        check(
            n_nan == n_tickers,
            f"Exactly {n_tickers} NaN (one per ticker — last row)",
            f"got {n_nan} NaN"
        )

    # ── 13. VIX check ─────────────────────────────────────────────────────────
    print("\n[13] VIX")
    if 'vix' in cols:
        n_nan_vix = df['vix'].isna().sum()
        warn(
            n_nan_vix == 0,
            "vix has no NaN",
            f"{n_nan_vix} NaN found"
        )
        vix_min = df['vix'].min()
        vix_max = df['vix'].max()
        check(
            5 <= vix_min and vix_max <= 100,
            f"VIX range plausible (5–100)",
            f"got min={vix_min:.1f}, max={vix_max:.1f}"
        )
        print(f"       VIX range: {vix_min:.1f} – {vix_max:.1f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED — do NOT proceed to training")
        print("Failed checks:")
        for f in failures:
            print(f"  - {f}")
        if warnings_list:
            print(f"\n{len(warnings_list)} warning(s):")
            for w in warnings_list:
                print(f"  - {w}")
        print("=" * 70)
        sys.exit(1)
    else:
        if warnings_list:
            print(f"RESULT: ALL CHECKS PASSED ({len(warnings_list)} warning(s))")
            for w in warnings_list:
                print(f"  {WARN} {w}")
        else:
            print("RESULT: ALL CHECKS PASSED — ready to run training scripts")
        print("=" * 70)
        print("\nRecommended execution order:")
        print("  1. python run_walk_forward.py")
        print("  2. python run_walk_forward_regime.py")
        print("  3. python run_walk_forward_distillation.py   (slow: ~1-2h)")
        print("  4. python run_rolling_oos_evaluation.py      (slow: ~1-2h)")
        print("  5. python run_shap_analysis.py")
        print("  6. python run_backtest_distillation.py")
        sys.exit(0)


if __name__ == "__main__":
    main()

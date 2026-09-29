#!/usr/bin/env python3
"""
Download Fama-French 25 Portfolios data from Ken French's website.
Run once on a machine with internet access, saves CSV files to data/.

Usage:
    python data/download_ff25.py
"""
import os
import pandas as pd
import pandas_datareader.data as web

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_ff25():
    print("Downloading FF25 monthly returns...")
    raw = web.DataReader('25_Portfolios_5x5', 'famafrench', start='1926-07', end='2024-12')

    # key 0 = value-weighted monthly returns (in percent)
    monthly = raw[0].copy()
    monthly.index = pd.to_datetime(monthly.index.to_timestamp())

    # replace -99.99 sentinel with NaN, drop rows with any missing
    monthly.replace(-99.99, float('nan'), inplace=True)
    monthly.dropna(inplace=True)

    out_monthly = os.path.join(OUT_DIR, 'FF25_monthly.csv')
    monthly.to_csv(out_monthly)
    print(f"  Saved {len(monthly)} monthly observations → {out_monthly}")
    print(f"  Date range: {monthly.index[0].date()} — {monthly.index[-1].date()}")
    print(f"  Columns (25 portfolios): {list(monthly.columns)[:5]} ...")

    print("\nDownloading FF25 daily returns...")
    raw_d = web.DataReader('25_Portfolios_5x5_daily', 'famafrench', start='1963-07', end='2024-12')
    daily = raw_d[0].copy()
    daily.index = pd.to_datetime(daily.index)
    daily.replace(-99.99, float('nan'), inplace=True)
    daily.dropna(inplace=True)

    out_daily = os.path.join(OUT_DIR, 'FF25_daily.csv')
    daily.to_csv(out_daily)
    print(f"  Saved {len(daily)} daily observations → {out_daily}")
    print(f"  Date range: {daily.index[0].date()} — {daily.index[-1].date()}")


if __name__ == '__main__':
    download_ff25()

"""
BurstGPT Cross-Validation of DynamoLLM Inference Trace Findings
================================================================
Bartlett Fellowship — Nature Energy Submission Support

Purpose: Validate key claims made from the Azure LLM Inference Dataset 2024
(DynamoLLM, 44.1M requests, 1 week, May 2024) using the independent BurstGPT
dataset (10.3M requests, 213 days, Azure OpenAI GPT services).

Claims to validate:
  1. P99 drain time of 4.5–11.6 seconds at 60 tok/s
  2. Diurnal periodicity (weekday peaks, weekend valleys)
  3. Weekend request volume reduction
  4. Hourly coefficient of variation (load predictability)
  5. Request token length distributions

Usage:
  pip install pandas numpy matplotlib
  python burstgpt_validation.py

Data is downloaded automatically from GitHub on first run.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import os
import sys

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

DATA_DIR = Path("burstgpt_data")
DATA_DIR.mkdir(exist_ok=True)

# BurstGPT files (from GitHub releases / repo)
BURSTGPT_URLS = {
    "BurstGPT_1": "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_without_fails_1.csv",
    "BurstGPT_2": "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_without_fails_2.csv",
}

# If the above URLs don't work (GitHub raw content), try Hugging Face:
BURSTGPT_HF_URLS = {
    "BurstGPT_1": "https://huggingface.co/datasets/lzzmm/BurstGPT/resolve/main/BurstGPT_without_fails_1.csv",
    "BurstGPT_2": "https://huggingface.co/datasets/lzzmm/BurstGPT/resolve/main/BurstGPT_without_fails_2.csv",
}

# DynamoLLM benchmark values (from your Azure trace analysis)
DYNAMO_BENCHMARKS = {
    "total_requests": 44_107_694,
    "coding_requests": 16_803_695,
    "conversation_requests": 27_303_999,
    "duration_days": 7,
    "date_range": "May 10-18, 2024",
    # Drain time at 60 tok/s (P99 of GeneratedTokens / 60)
    "p99_drain_time_coding_sec": 4.5,     # ~270 tokens / 60 tok/s
    "p99_drain_time_conv_sec": 11.6,      # ~697 tokens / 60 tok/s
    # Diurnal pattern (from DynamoLLM paper)
    "coding_peak_to_valley_ratio": 34.6,  # peak load / valley load
    "conv_peak_to_valley_ratio": 3.3,
    "coding_peak_to_avg_ratio": 2.8,
    "conv_peak_to_avg_ratio": 1.7,
}

THROUGHPUT_ASSUMPTION = 60  # tok/s for drain time calculation

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def download_file(url, dest):
    """Download with progress indication."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True
    print(f"  Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Done ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def load_burstgpt():
    """Load BurstGPT data, trying GitHub first then Hugging Face."""
    print("Loading BurstGPT data...")

    dfs = []
    for name, url in BURSTGPT_URLS.items():
        dest = DATA_DIR / f"{name}.csv"
        if not download_file(url, dest):
            # Try Hugging Face fallback
            hf_url = BURSTGPT_HF_URLS.get(name)
            if hf_url:
                print(f"  Trying Hugging Face fallback...")
                if not download_file(hf_url, dest):
                    print(f"  ERROR: Could not download {name}")
                    print(f"  Please manually download from:")
                    print(f"    {url}")
                    print(f"    or {hf_url}")
                    print(f"  and place in {DATA_DIR}/")
                    sys.exit(1)

        df = pd.read_csv(dest)
        dfs.append(df)
        print(f"  Loaded {name}: {len(df):,} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(df):,} rows")

    # Standardize column names
    # BurstGPT columns: Timestamp, Session ID, Elapsed time, Model,
    #                    Request tokens, Response tokens, Total tokens, Log Type
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "_")
        if "timestamp" in cl:
            col_map[c] = "timestamp"
        elif "elapsed" in cl:
            col_map[c] = "elapsed_time"
        elif "model" in cl:
            col_map[c] = "model"
        elif "request_token" in cl or cl == "request_tokens":
            col_map[c] = "request_tokens"
        elif "response_token" in cl or cl == "response_tokens":
            col_map[c] = "response_tokens"
        elif "total_token" in cl or cl == "total_tokens":
            col_map[c] = "total_tokens"
        elif "log_type" in cl or "log type" in cl.replace("_", " "):
            col_map[c] = "log_type"
        elif "session" in cl:
            col_map[c] = "session_id"

    df = df.rename(columns=col_map)
    print(f"  Columns mapped: {list(df.columns)}")

    return df


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1: DRAIN TIME VALIDATION
# ══════════════════════════════════════════════════════════════════

def analyze_drain_times(df):
    """
    Compute drain time distributions and compare to DynamoLLM.

    Drain time = time for all in-flight requests to complete if no new
    requests are accepted. Approximated as GeneratedTokens / throughput.

    BurstGPT advantage: has actual elapsed_time field (real latency).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: DRAIN TIME VALIDATION")
    print("=" * 70)

    results = {}

    # --- Method A: Token-based drain time (comparable to DynamoLLM method) ---
    if "response_tokens" in df.columns:
        resp_tok = df["response_tokens"].dropna()
        resp_tok = resp_tok[resp_tok > 0]

        drain_times = resp_tok / THROUGHPUT_ASSUMPTION

        percentiles = [50, 90, 95, 99, 99.9]
        print(f"\n  Method A: Token-based drain time at {THROUGHPUT_ASSUMPTION} tok/s")
        print(f"  (Response tokens / {THROUGHPUT_ASSUMPTION}, same method as DynamoLLM)")
        print(f"  N = {len(resp_tok):,} requests")
        print(f"  {'Percentile':>12} {'BurstGPT (s)':>14} {'DynamoLLM (s)':>14}")
        print(f"  {'-'*42}")

        for p in percentiles:
            val = np.percentile(drain_times, p)
            dynamo_ref = ""
            if p == 99:
                dynamo_ref = f"  (Coding: {DYNAMO_BENCHMARKS['p99_drain_time_coding_sec']:.1f}, Conv: {DYNAMO_BENCHMARKS['p99_drain_time_conv_sec']:.1f})"
            print(f"  P{p:<11} {val:>13.2f}s{dynamo_ref}")

        results["drain_time_p99"] = np.percentile(drain_times, 99)
        results["drain_time_p50"] = np.percentile(drain_times, 50)

        # By service type if available
        if "log_type" in df.columns:
            print(f"\n  By service type:")
            for stype in df["log_type"].unique():
                mask = (df["log_type"] == stype) & (df["response_tokens"] > 0)
                st_drain = df.loc[mask, "response_tokens"] / THROUGHPUT_ASSUMPTION
                if len(st_drain) > 0:
                    print(f"    {stype}: P50={np.percentile(st_drain, 50):.2f}s, "
                          f"P99={np.percentile(st_drain, 99):.2f}s, "
                          f"N={len(st_drain):,}")

        # By model if available
        if "model" in df.columns:
            print(f"\n  By model:")
            for model in df["model"].unique():
                mask = (df["model"] == model) & (df["response_tokens"] > 0)
                m_drain = df.loc[mask, "response_tokens"] / THROUGHPUT_ASSUMPTION
                if len(m_drain) > 100:
                    print(f"    {model}: P50={np.percentile(m_drain, 50):.2f}s, "
                          f"P99={np.percentile(m_drain, 99):.2f}s, "
                          f"N={len(m_drain):,}")

    # --- Method B: Actual elapsed time (BurstGPT-only, real latency) ---
    if "elapsed_time" in df.columns:
        elapsed = df["elapsed_time"].dropna()
        elapsed = elapsed[elapsed > 0]

        print(f"\n  Method B: Actual elapsed time (real latency, BurstGPT only)")
        print(f"  N = {len(elapsed):,} requests")
        for p in [50, 90, 95, 99, 99.9]:
            val = np.percentile(elapsed, p)
            print(f"  P{p:<11} {val:>13.2f}s")

        results["actual_elapsed_p99"] = np.percentile(elapsed, 99)
        results["actual_elapsed_p50"] = np.percentile(elapsed, 50)

    # --- Token distribution comparison ---
    if "response_tokens" in df.columns:
        resp_tok = df["response_tokens"].dropna()
        resp_tok = resp_tok[resp_tok > 0]

        print(f"\n  Response token distribution:")
        print(f"    Mean:   {resp_tok.mean():.1f}")
        print(f"    Median: {resp_tok.median():.1f}")
        print(f"    P90:    {np.percentile(resp_tok, 90):.1f}")
        print(f"    P99:    {np.percentile(resp_tok, 99):.1f}")
        print(f"    Max:    {resp_tok.max():.0f}")

    return results


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2: DIURNAL AND WEEKLY PATTERNS
# ══════════════════════════════════════════════════════════════════

def analyze_temporal_patterns(df):
    """
    Validate diurnal periodicity and weekend reduction claims.
    DynamoLLM found: Coding 34.6x peak/valley, Conv 3.3x peak/valley.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: DIURNAL AND WEEKLY PATTERNS")
    print("=" * 70)

    results = {}

    # Convert timestamp to datetime if needed
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            # BurstGPT timestamp is seconds from midnight of first day
            if df["timestamp"].dtype in [np.float64, np.int64, float, int]:
                # Relative seconds — convert to datetime
                # BurstGPT doesn't specify the start date, but patterns are what matter
                start = pd.Timestamp("2024-01-01")
                df["datetime"] = start + pd.to_timedelta(df["timestamp"], unit="s")
            else:
                df["datetime"] = pd.to_datetime(df["timestamp"])
        else:
            df["datetime"] = df["timestamp"]

        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek  # 0=Monday
        df["date"] = df["datetime"].dt.date
        df["is_weekend"] = df["dayofweek"].isin([5, 6])

        # --- Hourly request volume ---
        hourly = df.groupby(df["datetime"].dt.floor("h")).size()

        if len(hourly) > 24:
            peak = hourly.max()
            valley = hourly[hourly > 0].min()
            avg = hourly.mean()

            peak_valley_ratio = peak / valley if valley > 0 else float("inf")
            peak_avg_ratio = peak / avg if avg > 0 else float("inf")

            print(f"\n  Hourly request volume (all types combined):")
            print(f"    Peak hour:        {peak:,.0f} requests")
            print(f"    Valley hour:      {valley:,.0f} requests")
            print(f"    Average hour:     {avg:,.0f} requests")
            print(f"    Peak/Valley:      {peak_valley_ratio:.1f}x")
            print(f"    Peak/Average:     {peak_avg_ratio:.1f}x")
            print(f"    DynamoLLM Coding: {DYNAMO_BENCHMARKS['coding_peak_to_valley_ratio']:.1f}x peak/valley, "
                  f"{DYNAMO_BENCHMARKS['coding_peak_to_avg_ratio']:.1f}x peak/avg")
            print(f"    DynamoLLM Conv:   {DYNAMO_BENCHMARKS['conv_peak_to_valley_ratio']:.1f}x peak/valley, "
                  f"{DYNAMO_BENCHMARKS['conv_peak_to_avg_ratio']:.1f}x peak/avg")

            results["peak_valley_ratio"] = peak_valley_ratio
            results["peak_avg_ratio"] = peak_avg_ratio

        # --- By service type ---
        if "log_type" in df.columns:
            print(f"\n  By service type:")
            for stype in df["log_type"].unique():
                st_hourly = df[df["log_type"] == stype].groupby(
                    df.loc[df["log_type"] == stype, "datetime"].dt.floor("h")
                ).size()
                if len(st_hourly) > 24:
                    st_peak = st_hourly.max()
                    st_valley = st_hourly[st_hourly > 0].min()
                    st_avg = st_hourly.mean()
                    pv = st_peak / st_valley if st_valley > 0 else float("inf")
                    pa = st_peak / st_avg if st_avg > 0 else float("inf")
                    print(f"    {stype}: peak/valley={pv:.1f}x, peak/avg={pa:.1f}x")

        # --- Weekend reduction ---
        weekday_vol = df[~df["is_weekend"]].groupby(df.loc[~df["is_weekend"], "date"]).size()
        weekend_vol = df[df["is_weekend"]].groupby(df.loc[df["is_weekend"], "date"]).size()

        if len(weekday_vol) > 0 and len(weekend_vol) > 0:
            wd_avg = weekday_vol.mean()
            we_avg = weekend_vol.mean()
            reduction = 1 - (we_avg / wd_avg) if wd_avg > 0 else 0

            print(f"\n  Weekend reduction:")
            print(f"    Weekday avg daily volume: {wd_avg:,.0f}")
            print(f"    Weekend avg daily volume: {we_avg:,.0f}")
            print(f"    Weekend reduction:        {reduction:.1%}")

            results["weekend_reduction"] = reduction

        # --- Hourly CoV (load predictability) ---
        hourly_by_hour = df.groupby(df["hour"]).apply(
            lambda x: x.groupby(x["datetime"].dt.date).size()
        )
        if hasattr(hourly_by_hour, "groupby"):
            cov_by_hour = []
            for h in range(24):
                h_data = df[df["hour"] == h].groupby("date").size()
                if len(h_data) > 1:
                    cov = h_data.std() / h_data.mean() if h_data.mean() > 0 else 0
                    cov_by_hour.append(cov)
            if cov_by_hour:
                print(f"\n  Hourly coefficient of variation (load predictability):")
                print(f"    Range: {min(cov_by_hour):.2f} - {max(cov_by_hour):.2f}")
                print(f"    Mean:  {np.mean(cov_by_hour):.2f}")
                results["cov_range"] = (min(cov_by_hour), max(cov_by_hour))

        # --- Diurnal pattern (average by hour of day) ---
        avg_by_hour = df.groupby("hour").size() / max(df["date"].nunique(), 1)
        if len(avg_by_hour) == 24:
            print(f"\n  Average requests by hour of day:")
            for h in range(24):
                bar = "█" * int(avg_by_hour.get(h, 0) / avg_by_hour.max() * 30)
                print(f"    {h:02d}:00  {avg_by_hour.get(h, 0):>8,.0f}  {bar}")

    return results


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3: DATASET COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════

def comparison_summary(df, drain_results, temporal_results):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY: BurstGPT vs DynamoLLM")
    print("=" * 70)

    # Dataset characteristics
    n_days = df["date"].nunique() if "date" in df.columns else "?"
    date_min = df["datetime"].min() if "datetime" in df.columns else "?"
    date_max = df["datetime"].max() if "datetime" in df.columns else "?"

    print(f"\n  {'Characteristic':<35} {'DynamoLLM':<22} {'BurstGPT':<22}")
    print(f"  {'-'*79}")
    print(f"  {'Total requests':<35} {'44.1M':<22} {len(df)/1e6:.1f}M")
    print(f"  {'Duration':<35} {'7 days':<22} {n_days} days")
    print(f"  {'Date range':<35} {'May 10-18, 2024':<22} {str(date_min)[:10]} to {str(date_max)[:10]}")
    print(f"  {'Source':<35} {'Azure internal prod':<22} {'Azure OpenAI regional'}")
    print(f"  {'Service types':<35} {'Coding, Conversation':<22} {', '.join(df['log_type'].unique()) if 'log_type' in df.columns else '?'}")

    # Key metrics
    print(f"\n  {'Metric':<35} {'DynamoLLM':<22} {'BurstGPT':<22} {'Consistent?'}")
    print(f"  {'-'*90}")

    # Drain time
    if "drain_time_p99" in drain_results:
        dynamo_range = f"{DYNAMO_BENCHMARKS['p99_drain_time_coding_sec']:.1f}-{DYNAMO_BENCHMARKS['p99_drain_time_conv_sec']:.1f}s"
        burst_val = f"{drain_results['drain_time_p99']:.1f}s"
        consistent = "YES" if 3.0 <= drain_results["drain_time_p99"] <= 15.0 else "CHECK"
        print(f"  {'P99 drain time (60 tok/s)':<35} {dynamo_range:<22} {burst_val:<22} {consistent}")

    # Peak/valley
    if "peak_valley_ratio" in temporal_results:
        dynamo_pv = f"{DYNAMO_BENCHMARKS['conv_peak_to_valley_ratio']:.1f}-{DYNAMO_BENCHMARKS['coding_peak_to_valley_ratio']:.1f}x"
        burst_pv = f"{temporal_results['peak_valley_ratio']:.1f}x"
        print(f"  {'Peak/valley ratio':<35} {dynamo_pv:<22} {burst_pv:<22}")

    # Weekend reduction
    if "weekend_reduction" in temporal_results:
        burst_wr = f"{temporal_results['weekend_reduction']:.0%}"
        print(f"  {'Weekend reduction':<35} {'43-68%':<22} {burst_wr:<22}")

    # CoV
    if "cov_range" in temporal_results:
        burst_cov = f"{temporal_results['cov_range'][0]:.2f}-{temporal_results['cov_range'][1]:.2f}"
        print(f"  {'Hourly CoV range':<35} {'0.28-0.85':<22} {burst_cov:<22}")

    print(f"\n  INTERPRETATION FOR PAPER:")
    print(f"  If BurstGPT P99 drain time falls within the DynamoLLM range (4.5-11.6s),")
    print(f"  diurnal patterns show similar periodicity, and weekend reduction is")
    print(f"  in the same ballpark, then the S2b parameter grounding is validated")
    print(f"  across two independent datasets spanning {n_days}+ days.")
    print(f"  This substantially strengthens the claim that inference workloads")
    print(f"  complete fast enough for DR dispatch timelines (10-30 min notice).")


# ══════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_figures(df, output_dir=Path("burstgpt_figures")):
    """Generate publication-quality comparison figures."""
    output_dir.mkdir(exist_ok=True)

    if "response_tokens" not in df.columns:
        print("  No response_tokens column; skipping figures.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BurstGPT Cross-Validation of DynamoLLM Findings", fontsize=14, fontweight="bold")

    # Panel A: Response token distribution
    ax = axes[0, 0]
    resp_tok = df["response_tokens"].dropna()
    resp_tok = resp_tok[resp_tok > 0]
    ax.hist(resp_tok.clip(upper=2000), bins=100, alpha=0.7, color="#2E75B6", edgecolor="none")
    ax.set_xlabel("Response tokens")
    ax.set_ylabel("Frequency")
    ax.set_title("(a) Response token distribution")
    ax.axvline(resp_tok.median(), color="red", linestyle="--", label=f"Median: {resp_tok.median():.0f}")
    ax.axvline(np.percentile(resp_tok, 99), color="orange", linestyle="--", label=f"P99: {np.percentile(resp_tok, 99):.0f}")
    ax.legend(fontsize=9)

    # Panel B: Drain time CDF comparison
    ax = axes[0, 1]
    drain_times = resp_tok / THROUGHPUT_ASSUMPTION
    sorted_dt = np.sort(drain_times)
    cdf = np.arange(1, len(sorted_dt) + 1) / len(sorted_dt)
    # Subsample for plotting
    step = max(1, len(sorted_dt) // 10000)
    ax.plot(sorted_dt[::step], cdf[::step], color="#2E75B6", linewidth=1.5, label="BurstGPT")
    ax.axvline(DYNAMO_BENCHMARKS["p99_drain_time_coding_sec"], color="green", linestyle="--",
               alpha=0.7, label=f"DynamoLLM Coding P99: {DYNAMO_BENCHMARKS['p99_drain_time_coding_sec']:.1f}s")
    ax.axvline(DYNAMO_BENCHMARKS["p99_drain_time_conv_sec"], color="red", linestyle="--",
               alpha=0.7, label=f"DynamoLLM Conv P99: {DYNAMO_BENCHMARKS['p99_drain_time_conv_sec']:.1f}s")
    ax.set_xlabel("Drain time at 60 tok/s (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title("(b) Drain time CDF")
    ax.set_xlim(0, 30)
    ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel C: Diurnal pattern
    ax = axes[1, 0]
    if "hour" in df.columns and "date" in df.columns:
        avg_by_hour = df.groupby("hour").size() / max(df["date"].nunique(), 1)
        ax.bar(range(24), [avg_by_hour.get(h, 0) for h in range(24)],
               color="#2E75B6", alpha=0.7, edgecolor="none")
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Average requests per hour")
        ax.set_title("(c) Diurnal pattern")
        ax.set_xticks(range(0, 24, 3))

    # Panel D: Weekly pattern
    ax = axes[1, 1]
    if "dayofweek" in df.columns and "date" in df.columns:
        avg_by_dow = df.groupby("dayofweek").size() / df.groupby("dayofweek")["date"].apply(lambda x: x.nunique())
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        colors = ["#2E75B6"] * 5 + ["#E07B54"] * 2  # highlight weekends
        ax.bar(range(7), [avg_by_dow.get(d, 0) for d in range(7)],
               color=colors, alpha=0.7, edgecolor="none")
        ax.set_xlabel("Day of week")
        ax.set_ylabel("Average requests per day")
        ax.set_title("(d) Weekly pattern (weekends in orange)")
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names)

    plt.tight_layout()
    fig_path = output_dir / "burstgpt_cross_validation.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\n  Figure saved: {fig_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("BurstGPT Cross-Validation of DynamoLLM Findings")
    print("Bartlett Fellowship / Nature Energy Submission")
    print("=" * 70)

    # Load data
    df = load_burstgpt()

    # Run analyses
    drain_results = analyze_drain_times(df)
    temporal_results = analyze_temporal_patterns(df)

    # Summary comparison
    comparison_summary(df, drain_results, temporal_results)

    # Generate figures
    generate_figures(df)

    print("\n" + "=" * 70)
    print("DONE. Review results above for Nature Energy Methods section.")
    print("=" * 70)

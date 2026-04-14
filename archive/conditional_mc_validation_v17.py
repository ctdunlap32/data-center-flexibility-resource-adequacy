# ══════════════════════════════════════════════════════════════════════════════
# CELL 18: CONDITIONAL MONTE CARLO — Joint Feasibility with Risk Profile (v17)
# ══════════════════════════════════════════════════════════════════════════════
#
# v17: Restructured from v16.1. Key changes:
#   - Old compat × (1 - util) decomposed into D2 × D3 × D4 × D5
#     D2 = utilization headroom (1 - util)
#     D3 = hardware compatibility (CUDA/TensorRT, without pre-staging)
#     D4 = inference workload share at destination
#     D5 = operational readiness / pre-staging (weakest layer)
#   - Each drawn independently from Uniform distributions per
#     compatible_fraction_reference.md and TGV-corrected ranges.
#   - Commitment formula already corrected in v16.1:
#     commit = spatial_frac + DVFS * (S1 - spatial_frac)
#
# D1 is OBSERVED (which destinations are unstressed each hour).
# D2, D3, D4, D5 are DRAWN from distributions.
# S1, S2a, S2b, E1, E2 are held fixed at central values.
#
# Requires: stress_df, price_wide, meta, dest_results (from Cells 1-6)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────

SOURCE_ZONE = 'PJM_COMED'
DVFS = 0.25
N_DRAWS = 2000  # MC draws per stress hour (200 hours × 2000 = 400,000 total)

# Fixed cascade parameters (genuinely independent of grid state)
S1  = 0.70
S2b = 0.90
E1  = 0.997
E2  = 0.995

# v16.1: DVFS is measured at GPU cluster (Colangelo et al. 2026).
# Facility-relative DVFS floor = DVFS × S1 (DVFS only operates on shiftable compute)
DVFS_FLOOR_FACILITY = DVFS * S1  # 0.175

# S2a: held at central for now (partially dependent, noted as limitation)
S2A_CENTRAL = 0.80

# ── v17: Distribution specifications for D2, D3, D4, D5 ─────────────────────
# These replace the old util × compat formulation.

# D2: Utilization headroom = 1 - utilization.
#   Central util ~0.67 per SemiAnalysis/MIT. Range bridges literature disagreement:
#   SemiAnalysis/MIT GPU util 0.50-0.90 → headroom 0.10-0.50
#   TGV recommends util 0.20-0.50 → headroom 0.50-0.80
#   Wide range [0.10, 0.80] captures both; Phase 4 sensitivity surface will resolve.
D2_LO, D2_HI = 0.10, 0.80

# D3: Hardware compatibility (CUDA backward compat, TensorRT).
#   compatible_fraction_reference Layer 1: 0.80-0.92.
#   Does NOT include pre-staging (that is D5).
D3_LO, D3_HI = 0.80, 0.92

# D4: Inference workload share at destination.
#   compatible_fraction_reference Layer 2: 0.33-0.67.
#   Deloitte Nov 2025: ~50% in 2025. McKinsey: 30-40% by 2030.
D4_LO, D4_HI = 0.33, 0.67

# D5: Operational readiness / pre-staging.
#   compatible_fraction_reference Layer 3: 0.50-0.80.
#   WEAKEST LAYER per Pillar 1 TGV. Lower bound 0.50 is reference conservative.
D5_LO, D5_HI = 0.50, 0.80

# v17: Cascade reference for comparison. Computed from central values:
# product = S1*S2a*S2b*D1*D2*D3*D4*D5*E1*E2 = 0.0467
# commit  = 0.0467 + 0.25*(0.70 - 0.0467) = 0.210
CASCADE_COMMITMENT = 0.210

print('=' * 90)
print('CONDITIONAL MONTE CARLO — Joint Feasibility with Risk Profile (v17)')
print('=' * 90)
print()
print(f'  N draws per stress hour:     {N_DRAWS:,}')
print(f'  D2 (util headroom):          Uniform[{D2_LO}, {D2_HI}]')
print(f'  D3 (HW compat):             Uniform[{D3_LO}, {D3_HI}]')
print(f'  D4 (inference share):        Uniform[{D4_LO}, {D4_HI}]')
print(f'  D5 (pre-staging):           Uniform[{D5_LO}, {D5_HI}]')
print(f'  S2a:                         Fixed at {S2A_CENTRAL} (noted as simplification)')
print(f'  Cascade reference (v17):     {CASCADE_COMMITMENT:.1%}')
print()

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Identify source stress hours and destination infrastructure
# ══════════════════════════════════════════════════════════════════════════════

source_rto = meta.loc[SOURCE_ZONE, 'rto']

# Cross-RTO destination zones and their DC capacity
# v16.1 fix: also filter by migration_role to exclude MISO_ILLINOIS, which is
# tagged as 'intra_rto_control' because it is geographically adjacent to ComEd
# and shares weather. Using it as a migration destination would be physically
# incorrect (same cold snap would stress both zones simultaneously).
# Also excludes WECC non-ISO hubs (Mid-C, Palo Verde) which are absent from
# the hourly stress_df due to daily-resolution source data. See Methods.
dest_zones = [z for z in stress_df.columns
              if z != SOURCE_ZONE
              and z in meta.index
              and meta.loc[z, 'rto'] != source_rto
              and meta.loc[z, 'migration_role'] == 'cross_ba_destination']

dest_mw = {z: float(meta.loc[z, 'dc_capacity_mw']) for z in dest_zones
           if 'dc_capacity_mw' in meta.columns}

# PJM source zones and their DC capacity (for fleet calculation)
pjm_zones = [z for z in stress_df.columns
             if z in meta.index and meta.loc[z, 'rto'] == 'PJM']
pjm_dc_mw = {z: float(meta.loc[z, 'dc_capacity_mw']) for z in pjm_zones
             if 'dc_capacity_mw' in meta.columns}

source_stressed = stress_df[SOURCE_ZONE].fillna(False).astype(bool)
source_stress_idx = stress_df.index[source_stressed]
n_stress_hours = len(source_stress_idx)

print(f'Source: {SOURCE_ZONE} ({pjm_dc_mw.get(SOURCE_ZONE, 0):,.0f} MW DC)')
print(f'Stress hours: {n_stress_hours}')
print(f'Cross-RTO destinations: {len(dest_zones)} zones, {sum(dest_mw.values()):,.0f} MW total')
print(f'PJM source zones: {len(pjm_zones)} ({sum(pjm_dc_mw.values()):,.0f} MW total PJM DC)')
print()

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Precompute per-hour destination availability
# ══════════════════════════════════════════════════════════════════════════════

# For each stress hour, compute available destination MW (unstressed zones)
# This is D1 REALIZED — observed, not drawn

hour_dest_available_mw = []  # List of dicts: {zone: mw} for unstressed zones

for hour in source_stress_idx:
    available = {}
    for z in dest_zones:
        if z in stress_df.columns:
            z_stressed = stress_df.loc[hour, z]
            if pd.isna(z_stressed) or not z_stressed:
                available[z] = dest_mw.get(z, 0)
    hour_dest_available_mw.append(available)

# Also compute per-hour PJM co-stress (for fleet calculation)
hour_pjm_co_stressed_mw = []
for hour in source_stress_idx:
    co_stressed_mw = 0
    for z in pjm_zones:
        if z in stress_df.columns:
            z_stressed = stress_df.loc[hour, z]
            if not pd.isna(z_stressed) and z_stressed:
                co_stressed_mw += pjm_dc_mw.get(z, 0)
    hour_pjm_co_stressed_mw.append(co_stressed_mw)

print('Per-hour destination availability (summary):')
avail_mw_per_hour = [sum(h.values()) for h in hour_dest_available_mw]
print(f'  Min available dest MW:  {min(avail_mw_per_hour):,.0f}')
print(f'  Mean available dest MW: {np.mean(avail_mw_per_hour):,.0f}')
print(f'  Max available dest MW:  {max(avail_mw_per_hour):,.0f}')
print()
print('Per-hour PJM co-stress (fleet migrating simultaneously):')
print(f'  Min co-stressed PJM MW:  {min(hour_pjm_co_stressed_mw):,.0f}')
print(f'  Mean co-stressed PJM MW: {np.mean(hour_pjm_co_stressed_mw):,.0f}')
print(f'  Max co-stressed PJM MW:  {max(hour_pjm_co_stressed_mw):,.0f}')
print()

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Conditional Monte Carlo — Single Facility
# ══════════════════════════════════════════════════════════════════════════════

print('CONDITIONAL MONTE CARLO — SINGLE FACILITY (500 MW) [v17: 4-draw decomposition]')
print('=' * 90)

SOURCE_FAC_MW = 500
migrating_base = SOURCE_FAC_MW * S1 * S2A_CENTRAL * S2b  # What wants to move

# v17: Draw D2, D3, D4, D5 separately (replacing old util × compat)
D2_draws = np.random.uniform(D2_LO, D2_HI, size=N_DRAWS)
D3_draws = np.random.uniform(D3_LO, D3_HI, size=N_DRAWS)
D4_draws = np.random.uniform(D4_LO, D4_HI, size=N_DRAWS)
D5_draws = np.random.uniform(D5_LO, D5_HI, size=N_DRAWS)

# For each hour × each draw, compute commitment depth
all_commit_depths = np.zeros((n_stress_hours, N_DRAWS))
all_headroom_ratios = np.zeros((n_stress_hours, N_DRAWS))

for h_idx in range(n_stress_hours):
    available_zones = hour_dest_available_mw[h_idx]
    total_available_mw = sum(available_zones.values())

    for d_idx in range(N_DRAWS):
        D2 = D2_draws[d_idx]  # utilization headroom
        D3 = D3_draws[d_idx]  # hardware compatibility
        D4 = D4_draws[d_idx]  # inference workload share
        D5 = D5_draws[d_idx]  # pre-staging readiness

        # v17: effective absorbable destination MW = raw_available × D2 × D3 × D4 × D5
        headroom = total_available_mw * D2 * D3 * D4 * D5

        # Feasible migration (capped, with execution reliability)
        feasible = min(migrating_base, headroom) * E1 * E2
        spatial_frac = feasible / SOURCE_FAC_MW
        # v16.1: DVFS operates on the shiftable-compute residual (S1 - spatial_frac),
        # not the facility residual. Colangelo 25% is GPU-cluster measured.
        commit = spatial_frac + DVFS * (S1 - spatial_frac)

        all_commit_depths[h_idx, d_idx] = commit
        all_headroom_ratios[h_idx, d_idx] = headroom / migrating_base if migrating_base > 0 else np.inf

# ── Aggregate statistics ─────────────────────────────────────────────────────

# Mean commitment across all hours and draws
overall_mean = np.mean(all_commit_depths)
overall_median = np.median(all_commit_depths)
overall_p5 = np.percentile(all_commit_depths, 5)
overall_p95 = np.percentile(all_commit_depths, 95)

# Per-hour means (average across draws for each hour)
hour_means = np.mean(all_commit_depths, axis=1)
hour_p5s = np.percentile(all_commit_depths, 5, axis=1)

# CVaR: expected commitment depth in the worst 5% of outcomes
flat_commits = all_commit_depths.flatten()
var_5 = np.percentile(flat_commits, 5)
cvar_5 = np.mean(flat_commits[flat_commits <= var_5])

# Cascade comparison — CASCADE_COMMITMENT set in configuration section (v17: 0.210)

print(f'\n  Source facility:           {SOURCE_FAC_MW} MW')
print(f'  Migrating (S1×S2a×S2b):   {migrating_base:.0f} MW')
print(f'  Total MC samples:         {n_stress_hours * N_DRAWS:,}')
print()
print(f'  ┌─────────────────────────────────────────────────────┐')
print(f'  │  CONDITIONAL MC RESULTS                             │')
print(f'  │  Mean commitment depth:     {overall_mean:.1%}                  │')
print(f'  │  Median:                    {overall_median:.1%}                  │')
print(f'  │  P5-P95 range:              [{overall_p5:.1%}, {overall_p95:.1%}]          │')
print(f'  │                                                     │')
print(f'  │  CASCADE (independence):    {CASCADE_COMMITMENT:.1%}                  │')
print(f'  │  Delta (MC minus cascade):  {overall_mean - CASCADE_COMMITMENT:+.1%}                  │')
print(f'  │                                                     │')
print(f'  │  VaR (5%):                  {var_5:.1%}                  │')
print(f'  │  CVaR (5%):                 {cvar_5:.1%}                  │')
print(f'  │  DVFS-only floor (fac):     {DVFS_FLOOR_FACILITY:.1%}                │')
print(f'  └─────────────────────────────────────────────────────┘')
print()

# ── Reliability curve: committable MW at varying confidence levels ───────────
# v16.1: This is the operator-facing view. An operator writing a binding
# capacity commitment cannot deliver at the mean; they must deliver at a
# confidence level consistent with penalty exposure. These percentiles tell
# them what MW they could safely commit at each confidence level.
SOURCE_FAC_MW_FOR_CURVE = SOURCE_FAC_MW  # 500 MW
confidence_levels = [50, 75, 90, 95, 99, 99.5, 99.9]
print('  RELIABILITY CURVE — SINGLE FACILITY (500 MW)')
print('  Operator-facing: what can be firmly committed at each confidence level')
print('  ' + '─' * 70)
print(f'  {"Confidence":<15} {"Committable depth":<22} {"Committable MW":<20}')
print('  ' + '─' * 70)
for c in confidence_levels:
    q = 100 - c  # P5 = 5th percentile = 95% confidence
    committable_frac = np.percentile(flat_commits, q)
    committable_mw = committable_frac * SOURCE_FAC_MW_FOR_CURVE
    print(f'  {c:>6.1f}%        {committable_frac:>6.1%}                 {committable_mw:>6.1f} MW')
print('  ' + '─' * 70)
print(f'  For reference: mean = {overall_mean:.1%} ({overall_mean*SOURCE_FAC_MW_FOR_CURVE:.0f} MW), '
      f'DVFS floor = {DVFS_FLOOR_FACILITY:.1%} ({DVFS_FLOOR_FACILITY*SOURCE_FAC_MW_FOR_CURVE:.0f} MW)')
print()

# Worst hours
print('5 worst hours (lowest mean commitment across MC draws):')
worst_hour_idx = np.argsort(hour_means)[:5]
for idx in worst_hour_idx:
    hour = source_stress_idx[idx]
    month = hour.month
    season = 'winter' if month in [12, 1, 2] else ('summer' if month in [6, 7, 8] else 'shoulder')
    avail = sum(hour_dest_available_mw[idx].values())
    co_stress = hour_pjm_co_stressed_mw[idx]
    print(f'  {str(hour):<22}  mean={hour_means[idx]:.1%}  P5={hour_p5s[idx]:.1%}  '
          f'dest_avail={avail:,.0f}MW  pjm_costress={co_stress:,.0f}MW  {season}')

print()

# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Conditional Monte Carlo — Geographically Distributed Fleet
# ══════════════════════════════════════════════════════════════════════════════

print('CONDITIONAL MONTE CARLO — GEOGRAPHICALLY DISTRIBUTED FLEET')
print('=' * 90)
print()
print('For each stress hour, the migrating fleet is NOT all of PJM.')
print('It is only the PJM zones that are SIMULTANEOUSLY stressed.')
print('DOM (9,431 MW) is co-stressed with ComEd only 36% of the time.')
print()

# For each hour × each draw, compute fleet-level commitment depth
fleet_commit_depths = np.zeros((n_stress_hours, N_DRAWS))
fleet_migrating_mw = np.zeros(n_stress_hours)
fleet_constrained = np.zeros((n_stress_hours, N_DRAWS), dtype=bool)

for h_idx in range(n_stress_hours):
    # The migrating fleet this hour = sum of DC MW in co-stressed PJM zones
    co_stressed_mw = hour_pjm_co_stressed_mw[h_idx]
    fleet_mig = co_stressed_mw * S1 * S2A_CENTRAL * S2b
    fleet_migrating_mw[h_idx] = fleet_mig

    available_zones = hour_dest_available_mw[h_idx]
    total_available_mw = sum(available_zones.values())

    for d_idx in range(N_DRAWS):
        D2 = D2_draws[d_idx]
        D3 = D3_draws[d_idx]
        D4 = D4_draws[d_idx]
        D5 = D5_draws[d_idx]

        # v17: effective absorbable destination MW = raw_available × D2 × D3 × D4 × D5
        headroom = total_available_mw * D2 * D3 * D4 * D5

        feasible = min(fleet_mig, headroom) * E1 * E2 if fleet_mig > 0 else 0

        # Commitment depth relative to the co-stressed fleet
        spatial_frac = feasible / co_stressed_mw if co_stressed_mw > 0 else 0
        # v16.1: DVFS operates on shiftable-compute residual (S1 - spatial_frac)
        commit = spatial_frac + DVFS * (S1 - spatial_frac)

        fleet_commit_depths[h_idx, d_idx] = commit
        fleet_constrained[h_idx, d_idx] = headroom < fleet_mig

# ── Fleet statistics ─────────────────────────────────────────────────────────
fleet_mean = np.mean(fleet_commit_depths)
fleet_median = np.median(fleet_commit_depths)
fleet_p5 = np.percentile(fleet_commit_depths, 5)
fleet_p95 = np.percentile(fleet_commit_depths, 95)
fleet_flat = fleet_commit_depths.flatten()
fleet_var5 = np.percentile(fleet_flat, 5)
fleet_cvar5 = np.mean(fleet_flat[fleet_flat <= fleet_var5])
fleet_pct_constrained = np.mean(fleet_constrained) * 100

print(f'  PJM DC capacity by zone:')
for z in sorted(pjm_dc_mw.keys(), key=lambda x: -pjm_dc_mw[x]):
    overlap_with_source = 0
    if z != SOURCE_ZONE and z in stress_df.columns:
        both = (stress_df[SOURCE_ZONE].fillna(False) & stress_df[z].fillna(False)).sum()
        overlap_with_source = both / n_stress_hours * 100
    elif z == SOURCE_ZONE:
        overlap_with_source = 100.0
    print(f'    {z:<15} {pjm_dc_mw[z]:>8,.0f} MW   '
          f'co-stress with ComEd: {overlap_with_source:.0f}%')

print()
print(f'  Migrating fleet per hour (MW):')
print(f'    Min:    {fleet_migrating_mw.min():>8,.0f} MW')
print(f'    Mean:   {fleet_migrating_mw.mean():>8,.0f} MW')
print(f'    Max:    {fleet_migrating_mw.max():>8,.0f} MW')
print(f'    (vs. 10 GW ref × S1 × S2a × S2b = {10000 * S1 * S2A_CENTRAL * S2b:,.0f} MW if all 10 GW migrated)')
print()

print(f'  ┌─────────────────────────────────────────────────────┐')
print(f'  │  FLEET CONDITIONAL MC RESULTS                       │')
print(f'  │  Mean commitment depth:     {fleet_mean:.1%}                  │')
print(f'  │  Median:                    {fleet_median:.1%}                  │')
print(f'  │  P5-P95 range:              [{fleet_p5:.1%}, {fleet_p95:.1%}]          │')
print(f'  │                                                     │')
print(f'  │  CASCADE (independence):    {CASCADE_COMMITMENT:.1%}                  │')
print(f'  │  Delta (MC minus cascade):  {fleet_mean - CASCADE_COMMITMENT:+.1%}                  │')
print(f'  │                                                     │')
print(f'  │  VaR (5%):                  {fleet_var5:.1%}                  │')
print(f'  │  CVaR (5%):                 {fleet_cvar5:.1%}                  │')
print(f'  │  % draws constrained:       {fleet_pct_constrained:.1f}%                 │')
print(f'  │  DVFS-only floor (fac):     {DVFS_FLOOR_FACILITY:.1%}                │')
print(f'  └─────────────────────────────────────────────────────┘')
print()

# ── Reliability curve: fleet (empirical PJM co-stress) ───────────────────────
# v17 Phase 3: This uses the empirical PJM co-stress fleet (variable size per
# hour). For parametric fleet-size analysis, see the per-GW sweep (Part 4b).
# The reference MW here is illustrative — depth percentiles are the key output.
FLEET_REF_MW = 10000  # 10 GW reference fleet (for MW illustration only)
print('  RELIABILITY CURVE — FLEET (empirical PJM co-stress)')
print('  Based on actual co-stressed PJM DC MW per hour (variable fleet size)')
print('  MW column uses 10 GW reference for illustration')
print('  ' + '─' * 70)
print(f'  {"Confidence":<15} {"Committable depth":<22} {"Committable MW @10GW":<20}')
print('  ' + '─' * 70)
for c in confidence_levels:
    q = 100 - c
    committable_frac = np.percentile(fleet_flat, q)
    committable_mw = committable_frac * FLEET_REF_MW
    print(f'  {c:>6.1f}%        {committable_frac:>6.1%}                 {committable_mw:>6,.0f} MW')
print('  ' + '─' * 70)
print(f'  For reference: mean = {fleet_mean:.1%} ({fleet_mean*FLEET_REF_MW:,.0f} MW @10GW), '
      f'DVFS floor = {DVFS_FLOOR_FACILITY:.1%} ({DVFS_FLOOR_FACILITY*FLEET_REF_MW:,.0f} MW @10GW)')
print()

# ── Gap analysis: population mean vs. individually-committable ───────────────
# v16.1: This is the framing finding. The gap between the aggregate resource
# adequacy contribution (population mean) and what any individual operator
# could firmly commit at 95% confidence is the "coordination gap" — the share
# of physical flexibility that exists but cannot be captured without a
# mechanism that aggregates across operators and insures against tail hours.
print('  COORDINATION GAP ANALYSIS')
print('  ' + '─' * 70)
single_p95_commit = np.percentile(flat_commits, 5)   # 95% confidence
fleet_p95_commit = np.percentile(fleet_flat, 5)
single_gap = overall_mean - single_p95_commit
fleet_gap = fleet_mean - fleet_p95_commit
print(f'  Single facility (500 MW):')
print(f'    Population mean:       {overall_mean:.1%}  ({overall_mean*SOURCE_FAC_MW_FOR_CURVE:.0f} MW)')
print(f'    Committable @ 95%:     {single_p95_commit:.1%}  ({single_p95_commit*SOURCE_FAC_MW_FOR_CURVE:.0f} MW)')
print(f'    Coordination gap:      {single_gap:+.1%}  ({single_gap*SOURCE_FAC_MW_FOR_CURVE:+.0f} MW)')
print()
print(f'  Fleet (empirical co-stress, 10 GW ref for MW):')
print(f'    Population mean:       {fleet_mean:.1%}  ({fleet_mean*FLEET_REF_MW:,.0f} MW @10GW)')
print(f'    Committable @ 95%:     {fleet_p95_commit:.1%}  ({fleet_p95_commit*FLEET_REF_MW:,.0f} MW @10GW)')
print(f'    Coordination gap:      {fleet_gap:+.1%}  ({fleet_gap*FLEET_REF_MW:+,.0f} MW @10GW)')
print('  ' + '─' * 70)
print()

# Worst fleet hours
fleet_hour_means = np.mean(fleet_commit_depths, axis=1)
print('5 worst fleet hours:')
worst_fleet_idx = np.argsort(fleet_hour_means)[:5]
for idx in worst_fleet_idx:
    hour = source_stress_idx[idx]
    month = hour.month
    season = 'winter' if month in [12, 1, 2] else ('summer' if month in [6, 7, 8] else 'shoulder')
    avail = sum(hour_dest_available_mw[idx].values())
    co_stress = hour_pjm_co_stressed_mw[idx]
    fleet_mig = fleet_migrating_mw[idx]
    print(f'  {str(hour):<22}  mean={fleet_hour_means[idx]:.1%}  '
          f'fleet_mig={fleet_mig:,.0f}MW  dest_avail={avail:,.0f}MW  '
          f'pjm_costress={co_stress:,.0f}MW  {season}')

print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 4b: Per-GW Fleet Sweep — Commitment Depth as Function of Fleet Size
# ══════════════════════════════════════════════════════════════════════════════
#
# Phase 3 "per-GW framing": instead of a single fleet size, sweep across
# fleet MW from 0.5 GW to 15 GW. At each size, the migrating budget is
# fleet_mw × S1 × S2a × S2b (applied uniformly every stress hour — this
# is the parametric fleet, not the empirical PJM co-stress fleet).
# Destination availability is still observed per-hour (D1 realized).
# D2/D3/D4/D5 are drawn from the same distributions as Parts 3-4.
#
# For each fleet size we report 5 metrics:
#   mean, median, P5 (=committable@95%), CVaR(5%), % constrained
#
# This produces the contention curve: commitment depth ≈ flat at small fleet
# sizes (headroom >> demand) and declining at larger sizes (headroom binds).

FLEET_SWEEP_GW = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]

print('=' * 90)
print('PER-GW FLEET SWEEP — Commitment Depth vs Fleet Size')
print('=' * 90)
print()
print(f'Fleet sizes: {FLEET_SWEEP_GW} GW')
print(f'Migrating budget at each size: fleet_mw × S1({S1}) × S2a({S2A_CENTRAL}) × S2b({S2b})')
print(f'N draws: {N_DRAWS}, stress hours: {n_stress_hours}')
print()

# Reuse the same D2/D3/D4/D5 draws from Parts 3-4 for consistency
# (same random seed, same draw vectors)

sweep_results = []

for fleet_gw in FLEET_SWEEP_GW:
    fleet_mw = fleet_gw * 1000
    fleet_mig_mw = fleet_mw * S1 * S2A_CENTRAL * S2b  # fixed migrating budget

    gw_commit_depths = np.zeros((n_stress_hours, N_DRAWS))
    gw_constrained = np.zeros((n_stress_hours, N_DRAWS), dtype=bool)

    for h_idx in range(n_stress_hours):
        available_zones = hour_dest_available_mw[h_idx]
        total_available_mw = sum(available_zones.values())

        for d_idx in range(N_DRAWS):
            D2 = D2_draws[d_idx]
            D3 = D3_draws[d_idx]
            D4 = D4_draws[d_idx]
            D5 = D5_draws[d_idx]

            headroom = total_available_mw * D2 * D3 * D4 * D5
            feasible = min(fleet_mig_mw, headroom) * E1 * E2 if fleet_mig_mw > 0 else 0

            spatial_frac = feasible / fleet_mw if fleet_mw > 0 else 0
            commit = spatial_frac + DVFS * (S1 - spatial_frac)

            gw_commit_depths[h_idx, d_idx] = commit
            gw_constrained[h_idx, d_idx] = headroom < fleet_mig_mw

    gw_flat = gw_commit_depths.flatten()
    gw_var5 = np.percentile(gw_flat, 5)
    gw_cvar5 = np.mean(gw_flat[gw_flat <= gw_var5]) if np.any(gw_flat <= gw_var5) else gw_var5

    result = {
        'fleet_gw': fleet_gw,
        'fleet_mw': fleet_mw,
        'migrating_mw': fleet_mig_mw,
        'mean': np.mean(gw_commit_depths),
        'median': np.median(gw_commit_depths),
        'p5': gw_var5,
        'cvar5': gw_cvar5,
        'pct_constrained': np.mean(gw_constrained) * 100,
    }
    sweep_results.append(result)
    print(f'  {fleet_gw:>5.1f} GW  mean={result["mean"]:.1%}  median={result["median"]:.1%}  '
          f'P5={result["p5"]:.1%}  CVaR5={result["cvar5"]:.1%}  '
          f'constrained={result["pct_constrained"]:.1f}%')

# ── Summary table ────────────────────────────────────────────────────────────
print()
print('  PER-GW SWEEP SUMMARY TABLE')
print('  ' + '─' * 95)
print(f'  {"Fleet GW":>10} | {"Mig MW":>8} | {"Mean":>7} | {"Median":>7} | '
      f'{"P5":>7} | {"CVaR5":>7} | {"% Constr":>9} | {"Commit MW @P5":>14}')
print('  ' + '─' * 95)
for r in sweep_results:
    commit_mw_p5 = r['p5'] * r['fleet_mw']
    print(f'  {r["fleet_gw"]:>8.1f} GW | {r["migrating_mw"]:>7,.0f} | {r["mean"]:>6.1%} | '
          f'{r["median"]:>6.1%} | {r["p5"]:>6.1%} | {r["cvar5"]:>6.1%} | '
          f'{r["pct_constrained"]:>8.1f}% | {commit_mw_p5:>12,.0f} MW')
print('  ' + '─' * 95)
print()

# ── Identify contention onset ────────────────────────────────────────────────
# Find the smallest fleet size where % constrained exceeds 50%
contention_onset_gw = None
for r in sweep_results:
    if r['pct_constrained'] > 50.0:
        contention_onset_gw = r['fleet_gw']
        break

if contention_onset_gw is not None:
    print(f'  Contention onset (>50% constrained): {contention_onset_gw} GW')
else:
    print(f'  Contention onset (>50% constrained): beyond {FLEET_SWEEP_GW[-1]} GW')

# Mean depth drop from smallest to largest fleet
smallest = sweep_results[0]
largest = sweep_results[-1]
print(f'  Mean depth range: {smallest["mean"]:.1%} ({smallest["fleet_gw"]} GW) → '
      f'{largest["mean"]:.1%} ({largest["fleet_gw"]} GW)')
print(f'  P5 range: {smallest["p5"]:.1%} ({smallest["fleet_gw"]} GW) → '
      f'{largest["p5"]:.1%} ({largest["fleet_gw"]} GW)')

# ── Reference cases for downstream: 1 GW and 10 GW ──────────────────────────
ref_1gw = next(r for r in sweep_results if r['fleet_gw'] == 1.0)
ref_10gw = next(r for r in sweep_results if r['fleet_gw'] == 10.0)

print()
print(f'  REFERENCE CASES:')
print(f'    1 GW (individual facility scale):')
print(f'      Mean={ref_1gw["mean"]:.1%}, P5={ref_1gw["p5"]:.1%}, '
      f'CVaR5={ref_1gw["cvar5"]:.1%}, constrained={ref_1gw["pct_constrained"]:.1f}%')
print(f'    10 GW (fleet scale):')
print(f'      Mean={ref_10gw["mean"]:.1%}, P5={ref_10gw["p5"]:.1%}, '
      f'CVaR5={ref_10gw["cvar5"]:.1%}, constrained={ref_10gw["pct_constrained"]:.1f}%')
print()

# Store sweep as DataFrame for Phase 4 sensitivity surface input
sweep_df = pd.DataFrame(sweep_results)
print('  sweep_df stored in namespace for Phase 4.')
print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: Figure — Four-Panel Conditional MC Results
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Conditional Monte Carlo Validation of Cascade Independence Assumption',
             fontsize=14, fontweight='bold')

# Panel A: Single-facility commitment depth distribution
ax = axes[0, 0]
ax.hist(flat_commits, bins=80, density=True, color='#2E75B6', alpha=0.7, edgecolor='none')
ax.axvline(CASCADE_COMMITMENT, color='red', linestyle='--', linewidth=2,
           label=f'Cascade (independent): {CASCADE_COMMITMENT:.1%}')
ax.axvline(overall_mean, color='black', linestyle='-', linewidth=2,
           label=f'Conditional MC mean: {overall_mean:.1%}')
ax.axvline(var_5, color='orange', linestyle=':', linewidth=2,
           label=f'VaR 5%: {var_5:.1%}')
ax.axvline(DVFS_FLOOR_FACILITY, color='gray', linestyle=':', alpha=0.5,
           label=f'DVFS floor: {DVFS_FLOOR_FACILITY:.1%}')
ax.set_xlabel('Commitment Depth (facility-relative)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(a) Single Facility (500 MW)\nCommitment Depth Distribution', fontsize=11)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0.10, 0.60)  # v17: extended to 60% to show the spike at ~55%

# Panel B: Fleet commitment depth distribution
ax = axes[0, 1]
ax.hist(fleet_flat, bins=80, density=True, color='#E07B54', alpha=0.7, edgecolor='none')
ax.axvline(CASCADE_COMMITMENT, color='red', linestyle='--', linewidth=2,
           label=f'Cascade (independent): {CASCADE_COMMITMENT:.1%}')
ax.axvline(fleet_mean, color='black', linestyle='-', linewidth=2,
           label=f'Fleet MC mean: {fleet_mean:.1%}')
ax.axvline(fleet_var5, color='orange', linestyle=':', linewidth=2,
           label=f'VaR 5%: {fleet_var5:.1%}')
ax.axvline(DVFS_FLOOR_FACILITY, color='gray', linestyle=':', alpha=0.5,
           label=f'DVFS floor: {DVFS_FLOOR_FACILITY:.1%}')
ax.set_xlabel('Commitment Depth (facility-relative)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(b) Geographically Distributed Fleet\nCommitment Depth Distribution', fontsize=11)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0.10, 0.50)  # v17: narrower range

# Panel C: Per-hour commitment depth (time series of hour means)
ax = axes[1, 0]
# Sort by calendar date for visual
sort_idx = np.argsort(source_stress_idx)

# Color by season
colors_single = []
colors_fleet = []
for idx in sort_idx:
    month = source_stress_idx[idx].month
    if month in [6, 7, 8]:
        colors_single.append('#2E75B6')
        colors_fleet.append('#E07B54')
    elif month in [12, 1, 2]:
        colors_single.append('#1A4A7A')
        colors_fleet.append('#A85030')
    else:
        colors_single.append('#88B8D8')
        colors_fleet.append('#F0A888')

ax.scatter(range(n_stress_hours), hour_means[sort_idx],
           c=colors_single, s=15, alpha=0.8, label='Single facility (500 MW)')
ax.scatter(range(n_stress_hours), fleet_hour_means[sort_idx],
           c=colors_fleet, s=15, alpha=0.8, marker='s', label='Fleet (distributed)')
ax.axhline(CASCADE_COMMITMENT, color='red', linestyle='--', linewidth=1.5,
           label=f'Cascade: {CASCADE_COMMITMENT:.1%}')
ax.axhline(DVFS, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Stress Hour (chronological)', fontsize=11)
ax.set_ylabel('Mean Commitment Depth', fontsize=11)
ax.set_title('(c) Per-Hour Mean Commitment\n(darker = winter, lighter = summer)', fontsize=11)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0.10, 0.60)  # v17: extended to match panel (a) range
ax.legend(fontsize=8, loc='lower left')

# Panel D: Fleet migrating MW vs available headroom (at median draw)
ax = axes[1, 1]
# v17: use median of each decomposed parameter
median_D2 = np.median(D2_draws)
median_D3 = np.median(D3_draws)
median_D4 = np.median(D4_draws)
median_D5 = np.median(D5_draws)
fleet_headrooms = []
for h_idx in range(n_stress_hours):
    avail = sum(hour_dest_available_mw[h_idx].values())
    hr = avail * median_D2 * median_D3 * median_D4 * median_D5
    fleet_headrooms.append(hr)

fleet_headrooms = np.array(fleet_headrooms)
ax.scatter(fleet_migrating_mw[sort_idx], fleet_headrooms[sort_idx],
           c=[colors_single[i] for i in range(len(sort_idx))],
           s=25, alpha=0.7)
# 1:1 line
max_val = max(fleet_migrating_mw.max(), max(fleet_headrooms)) * 1.1
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Headroom = Demand (1:1)')
ax.set_xlabel('Fleet Migrating MW (PJM co-stressed zones)', fontsize=11)
ax.set_ylabel(f'Aggregate Dest Headroom MW\n(at median D2={median_D2:.0%}, D3={median_D3:.0%}, D4={median_D4:.0%}, D5={median_D5:.0%})',
              fontsize=9)
ax.set_title('(d) Fleet Demand vs Destination Capacity\nPoints above line = unconstrained', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)

plt.tight_layout()
fig_path = 'conditional_mc_validation.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f'Figure saved: {fig_path}')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# PART 6: Summary for Paper
# ══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 90)
print('SUMMARY FOR NATURE ENERGY')
print('=' * 90)
print()
print('The cascade decomposes spatial migration feasibility into independently')
print('estimable parameters. In reality, these parameters are correlated through')
print('the physical state of the grid during stress events. We validate the')
print('independence approximation via conditional Monte Carlo simulation.')
print()
print(f'SINGLE FACILITY (500 MW):')
print(f'  The cascade UNDERESTIMATES commitment depth by {overall_mean - CASCADE_COMMITMENT:+.1%}.')
print(f'  Favorable correlations dominate: when destinations are available,')
print(f'  their aggregate headroom far exceeds single-facility demand.')
print(f'  VaR(5%) = {var_5:.1%}, CVaR(5%) = {cvar_5:.1%}: even in the worst 5% of')
print(f'  outcomes, commitment depth remains {cvar_5 - DVFS_FLOOR_FACILITY:+.1%} above the DVFS floor.')
print()
print(f'GEOGRAPHICALLY DISTRIBUTED FLEET (per-GW framing):')
print(f'  The fleet MC accounts for the fact that PJM DC capacity is spread')
print(f'  across zones with imperfectly correlated stress. During a typical')
print(f'  ComEd stress hour, {np.mean(fleet_migrating_mw):,.0f} MW of PJM fleet is')
print(f'  co-stressed (vs {sum(pjm_dc_mw.values()):,.0f} MW total PJM DC).')
print(f'  Empirical fleet commitment depth: mean={fleet_mean:.1%}, VaR(5%)={fleet_var5:.1%}.')
print(f'  Delta from cascade: {fleet_mean - CASCADE_COMMITMENT:+.1%}.')
print(f'  {fleet_pct_constrained:.1f}% of fleet-hour-draw combinations are capacity-constrained.')
print()
print(f'  PER-GW REFERENCE CASES:')
print(f'    1 GW:  mean={ref_1gw["mean"]:.1%}, P5={ref_1gw["p5"]:.1%}, '
      f'CVaR5={ref_1gw["cvar5"]:.1%}, constrained={ref_1gw["pct_constrained"]:.1f}%')
print(f'    10 GW: mean={ref_10gw["mean"]:.1%}, P5={ref_10gw["p5"]:.1%}, '
      f'CVaR5={ref_10gw["cvar5"]:.1%}, constrained={ref_10gw["pct_constrained"]:.1f}%')
print()
print('KEY FINDING:')
print(f'  The independence assumption introduces a bias of {overall_mean - CASCADE_COMMITMENT:+.1%}')
print(f'  (single facility) to {fleet_mean - CASCADE_COMMITMENT:+.1%} (distributed fleet).')
delta_text = 'conservative (understates depth)' if overall_mean > CASCADE_COMMITMENT else 'optimistic (overstates depth)'
print(f'  Direction: the cascade is {delta_text} for single operators.')
fleet_delta_text = 'conservative' if fleet_mean > CASCADE_COMMITMENT else 'optimistic'
print(f'  At fleet level: {fleet_delta_text}.')
print()
print('LANGUAGE FOR PAPER (v17 per-GW framing):')
print('  "We validate the cascade\'s independence approximation using conditional')
print('   Monte Carlo simulation (N=400,000) that conditions on observed per-hour')
print('   destination availability and draws utilization headroom, hardware')
print('   compatibility, inference workload share, and pre-staging readiness')
print('   from distributions calibrated to published estimates.')
print(f'   For single-facility migration, the scenario-based estimate exceeds')
print(f'   the cascade by {overall_mean - CASCADE_COMMITMENT:+.1%}, confirming that favorable correlations')
print(f'   (abundant aggregate headroom relative to individual facility demand)')
print(f'   dominate. Commitment depth varies with fleet size: at 1 GW')
print(f'   (individual facility scale), mean depth is {ref_1gw["mean"]:.1%} with')
print(f'   {ref_1gw["pct_constrained"]:.0f}% of outcomes constrained by destination capacity;')
print(f'   at 10 GW (fleet scale), mean depth declines to {ref_10gw["mean"]:.1%} with')
print(f'   {ref_10gw["pct_constrained"]:.0f}% constrained. The 5% conditional value-at-risk')
print(f'   ranges from {ref_1gw["cvar5"]:.1%} (1 GW) to {ref_10gw["cvar5"]:.1%} (10 GW)."')

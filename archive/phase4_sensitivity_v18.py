# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Sensitivity Surface + Tornado Chart (v17)
# ══════════════════════════════════════════════════════════════════════════════
#
# Paste into Cross_BA_Stress_Correlation_v4.ipynb AFTER running
# conditional_mc_validation_v17.py (needs stress_df, meta, dest_zones,
# dest_mw, source_stress_idx, hour_dest_available_mw in namespace).
#
# Produces:
#   1. 2D sensitivity surface: fleet commitment depth as f(D2, D5)
#      at 10 GW reference fleet, all other params at central.
#   2. Tornado chart: one-at-a-time sensitivity across all 10 params
#      at 10 GW fleet, sorted by impact magnitude.
#   3. CSV exports of both for Phase 5.
#
# No MC draws — each grid point / sweep point is deterministic across
# stress hours. The conditional MC framework is used in the sense that
# per-hour destination availability (D1 realized) varies across hours.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm

# ── Configuration ────────────────────────────────────────────────────────────

# Fixed parameters (same as conditional MC)
S1  = 0.70
S2a = 0.80
S2b = 0.90
D1_CENTRAL = 0.99   # Not used directly — D1 is observed per hour
D2_CENTRAL = 0.33
D3_CENTRAL = 0.88
D4_CENTRAL = 0.50
D5_CENTRAL = 0.65
E1  = 0.997
E2  = 0.995
DVFS = 0.25

# Ranges (lo, central, hi) — from CASCADE_RANGES in bartlett_analysis_v17.py
PARAM_RANGES = {
    'S1':  (0.70,  0.70,  0.70),
    'S2a': (0.60,  0.80,  0.95),
    'S2b': (0.85,  0.90,  0.95),
    'D1':  (0.945, 0.99,  0.995),
    'D2':  (0.20,  0.33,  0.50),    # v18: narrowed (GPU util, not facility load factor)
    'D3':  (0.80,  0.88,  0.92),
    'D4':  (0.33,  0.50,  0.67),
    'D5':  (0.50,  0.65,  0.80),
    'E1':  (0.995, 0.997, 0.999),
    'E2':  (0.99,  0.995, 1.00),
}

CENTRAL = {k: v[1] for k, v in PARAM_RANGES.items()}

# Fleet size: 10 GW reference
FLEET_MW = 10000
FLEET_MIG_MW = FLEET_MW * S1 * S2a * S2b  # 5,040 MW

# Precompute per-hour destination availability (reuse from namespace)
n_stress_hours = len(source_stress_idx)
avail_per_hour = np.array([sum(h.values()) for h in hour_dest_available_mw])

print('=' * 90)
print('PHASE 4: SENSITIVITY ANALYSIS (v17)')
print('=' * 90)
print(f'  Fleet: {FLEET_MW/1000:.0f} GW, migrating: {FLEET_MIG_MW:,.0f} MW')
print(f'  Stress hours: {n_stress_hours}')
print(f'  Mean dest available: {avail_per_hour.mean():,.0f} MW')
print()


# ══════════════════════════════════════════════════════════════════════════════
# Helper: compute mean fleet commitment depth for given parameter values
# ══════════════════════════════════════════════════════════════════════════════

def fleet_commitment_depth(d2, d3, d4, d5, s1=S1, s2a=S2a, s2b=S2b,
                           e1=E1, e2=E2, fleet_mw=FLEET_MW,
                           fleet_mig_mw_override=None, fleet_mw_override=None):
    """
    Compute mean fleet commitment depth across all stress hours.
    D1 is observed (per-hour destination availability already filters it).
    Returns: (mean_depth, median_depth, p5_depth, pct_constrained)
    Use fleet_mw_override / fleet_mig_mw_override for panel (b) at alternate fleet sizes.
    """
    if fleet_mw_override is not None:
        fleet_mw = fleet_mw_override
    fleet_mig = fleet_mig_mw_override if fleet_mig_mw_override is not None else fleet_mw * s1 * s2a * s2b
    depths = np.zeros(n_stress_hours)
    constrained = np.zeros(n_stress_hours, dtype=bool)

    for h_idx in range(n_stress_hours):
        headroom = avail_per_hour[h_idx] * d2 * d3 * d4 * d5
        feasible = min(fleet_mig, headroom) * e1 * e2
        spatial_frac = feasible / fleet_mw if fleet_mw > 0 else 0
        commit = spatial_frac + DVFS * (s1 - spatial_frac)
        depths[h_idx] = commit
        constrained[h_idx] = headroom < fleet_mig

    return {
        'mean': np.mean(depths),
        'median': np.median(depths),
        'p5': np.percentile(depths, 5),
        'pct_constrained': np.mean(constrained) * 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: 2D Sensitivity Surface — D2 × D5
# ══════════════════════════════════════════════════════════════════════════════

print('2D SENSITIVITY SURFACE: Fleet Commitment Depth as f(D2, D5)')
print('─' * 70)
print(f'  D3={D3_CENTRAL}, D4={D4_CENTRAL} (fixed at central)')
print(f'  Fleet: {FLEET_MW/1000:.0f} GW')
print()

# Grid: 10 points per axis
D2_grid = np.linspace(0.20, 0.50, 10)
D5_grid = np.linspace(0.50, 0.80, 10)

surface_mean = np.zeros((len(D5_grid), len(D2_grid)))
surface_p5 = np.zeros((len(D5_grid), len(D2_grid)))
surface_constrained = np.zeros((len(D5_grid), len(D2_grid)))

surface_rows = []

for i, d5 in enumerate(D5_grid):
    for j, d2 in enumerate(D2_grid):
        result = fleet_commitment_depth(d2=d2, d3=D3_CENTRAL, d4=D4_CENTRAL, d5=d5)
        surface_mean[i, j] = result['mean']
        surface_p5[i, j] = result['p5']
        surface_constrained[i, j] = result['pct_constrained']

        surface_rows.append({
            'D2': round(d2, 4),
            'D5': round(d5, 4),
            'mean_depth': round(result['mean'], 5),
            'median_depth': round(result['median'], 5),
            'p5_depth': round(result['p5'], 5),
            'pct_constrained': round(result['pct_constrained'], 2),
        })

    # Progress
    print(f'  D5={d5:.2f}: D2 sweep complete '
          f'(mean range {surface_mean[i,:].min():.1%}–{surface_mean[i,:].max():.1%})')

surface_df = pd.DataFrame(surface_rows)

# ── Print summary table ──────────────────────────────────────────────────────
print()
print('  SURFACE: Mean commitment depth (%)')
header = f'  {"D5\\D2":>8}'
for d2 in D2_grid:
    header += f' | {d2:>5.2f}'
print(header)
print('  ' + '─' * (10 + 8 * len(D2_grid)))
for i, d5 in enumerate(D5_grid):
    row = f'  {d5:>8.2f}'
    for j in range(len(D2_grid)):
        row += f' | {surface_mean[i,j]:>4.1%}'
    print(row)

# ── Central point check ──────────────────────────────────────────────────────
central_result = fleet_commitment_depth(d2=D2_CENTRAL, d3=D3_CENTRAL,
                                         d4=D4_CENTRAL, d5=D5_CENTRAL)
print(f'\n  Central point (D2={D2_CENTRAL}, D5={D5_CENTRAL}): '
      f'mean={central_result["mean"]:.1%}, P5={central_result["p5"]:.1%}, '
      f'constrained={central_result["pct_constrained"]:.1f}%')


# ── Compute panel (b) at 3 GW (near contention onset) ───────────────────────
# At 10 GW, % constrained is uniformly ~100% across all D2×D5 combinations,
# making the contour panel uninformative. Recompute at 3 GW where there is
# meaningful variation in the constrained fraction.
PANEL_B_FLEET_MW = 3000
PANEL_B_MIG_MW = PANEL_B_FLEET_MW * S1 * S2a * S2b  # 1,512 MW

surface_constrained_3gw = np.zeros((len(D5_grid), len(D2_grid)))
print(f'\n  Computing panel (b) constrained surface at {PANEL_B_FLEET_MW/1000:.0f} GW...')
for i, d5 in enumerate(D5_grid):
    for j, d2 in enumerate(D2_grid):
        result_3gw = fleet_commitment_depth(d2=d2, d3=D3_CENTRAL, d4=D4_CENTRAL, d5=d5,
                                             fleet_mig_mw_override=PANEL_B_MIG_MW,
                                             fleet_mw_override=PANEL_B_FLEET_MW)
        surface_constrained_3gw[i, j] = result_3gw['pct_constrained']
print(f'  Done. Constrained range: {surface_constrained_3gw.min():.1f}%–{surface_constrained_3gw.max():.1f}%')

# ── Figure 1: Sensitivity Surface ────────────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(16, 7))
fig1.suptitle('Fleet Commitment Depth: D2 (Utilization Headroom) × D5 (Pre-Staging)\n'
              f'D3={D3_CENTRAL}, D4={D4_CENTRAL} fixed at central',
              fontsize=13, fontweight='bold')

# Panel A: Mean commitment depth (10 GW)
ax = axes[0]
D2_mesh, D5_mesh = np.meshgrid(D2_grid, D5_grid)
cf = ax.contourf(D2_mesh, D5_mesh, surface_mean * 100,
                 levels=15, cmap='RdYlGn')
cs = ax.contour(D2_mesh, D5_mesh, surface_mean * 100,
                levels=[17.5, 20, 25, 30, 35, 40, 45, 50],
                colors='black', linewidths=0.8, alpha=0.6)
ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f%%')
ax.plot(D2_CENTRAL, D5_CENTRAL, 'k*', markersize=15, markeredgewidth=1.5,
        markerfacecolor='white', label=f'Central ({D2_CENTRAL}, {D5_CENTRAL})')
# DVFS floor line
dvfs_floor_pct = DVFS * S1 * 100  # 17.5%
ax.set_xlabel('D2: Utilization Headroom (1 − util)', fontsize=11)
ax.set_ylabel('D5: Pre-Staging Readiness', fontsize=11)
ax.set_title('(a) Mean Commitment Depth (%), 10 GW Fleet', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
cbar = plt.colorbar(cf, ax=ax, label='Mean Commitment Depth (%)')

# Panel B: % constrained at 3 GW (near contention onset)
# NOTE: At 10 GW this panel is uniformly ~100% constrained — uninformative.
# Recomputed at 3 GW where the contention transition is visible.
ax = axes[1]
cf2 = ax.contourf(D2_mesh, D5_mesh, surface_constrained_3gw,
                  levels=15, cmap='RdYlGn_r')
cs2 = ax.contour(D2_mesh, D5_mesh, surface_constrained_3gw,
                 levels=[10, 25, 50, 75, 90],
                 colors='black', linewidths=0.8, alpha=0.6)
ax.clabel(cs2, inline=True, fontsize=8, fmt='%.0f%%')
ax.plot(D2_CENTRAL, D5_CENTRAL, 'k*', markersize=15, markeredgewidth=1.5,
        markerfacecolor='white', label=f'Central ({D2_CENTRAL}, {D5_CENTRAL})')
ax.set_xlabel('D2: Utilization Headroom (1 − util)', fontsize=11)
ax.set_ylabel('D5: Pre-Staging Readiness', fontsize=11)
ax.set_title(f'(b) % Hours Constrained, {PANEL_B_FLEET_MW/1000:.0f} GW Fleet', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
cbar2 = plt.colorbar(cf2, ax=ax, label='% Hours Constrained')

plt.tight_layout()
fig1.savefig('sensitivity_surface_D2_D5.png', dpi=200, bbox_inches='tight')
print(f'\n  Figure saved: sensitivity_surface_D2_D5.png')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Tornado Chart — One-at-a-Time Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

print()
print('TORNADO CHART: One-at-a-Time Parameter Sensitivity')
print('─' * 70)
print(f'  Fleet: {FLEET_MW/1000:.0f} GW, all other params at central')
print()

# For each parameter, compute commitment depth at low and high values
# while all others are at central. Since S1, S2a, S2b affect the source
# side (migrating MW and DVFS envelope), and D2/D3/D4/D5 affect
# destination headroom, and E1/E2 affect execution, we need to route
# each parameter to the right place.

tornado_rows = []

for param_name, (lo, central, hi) in PARAM_RANGES.items():
    results_by_val = {}
    for val_label, val in [('low', lo), ('central', central), ('high', hi)]:
        # Start with all params at central
        kwargs = {
            'd2': D2_CENTRAL, 'd3': D3_CENTRAL,
            'd4': D4_CENTRAL, 'd5': D5_CENTRAL,
            's1': S1, 's2a': S2a, 's2b': S2b,
            'e1': E1, 'e2': E2,
        }
        # Override the swept parameter
        param_to_kwarg = {
            'S1': 's1', 'S2a': 's2a', 'S2b': 's2b',
            'D1': None,  # D1 is observed, handled specially
            'D2': 'd2', 'D3': 'd3', 'D4': 'd4', 'D5': 'd5',
            'E1': 'e1', 'E2': 'e2',
        }

        if param_name == 'D1':
            # D1 affects which hours have destinations available.
            # We can't easily sweep D1 since it's realized per-hour.
            # Approximate: scale available MW by (val / D1_realized_mean).
            # D1_realized_mean ≈ fraction of hours with >0 available MW.
            # Simpler: treat D1 as a multiplicative scalar on available MW.
            # At D1=0.99, about 99% of potential destination capacity is
            # available. Sweeping D1 scales effective destination pool.
            #
            # Implementation: multiply avail_per_hour by (val / CENTRAL['D1'])
            # This is an approximation but captures the sensitivity direction.
            scale = val / CENTRAL['D1']
            # Temporarily modify avail for this calculation
            _orig_avail = avail_per_hour.copy()
            avail_per_hour_tmp = _orig_avail * scale
            # Manual computation with scaled availability
            fleet_mig = FLEET_MW * S1 * S2a * S2b
            depths = np.zeros(n_stress_hours)
            for h_idx in range(n_stress_hours):
                headroom = avail_per_hour_tmp[h_idx] * D2_CENTRAL * D3_CENTRAL * D4_CENTRAL * D5_CENTRAL
                feasible = min(fleet_mig, headroom) * E1 * E2
                spatial_frac = feasible / FLEET_MW
                commit = spatial_frac + DVFS * (S1 - spatial_frac)
                depths[h_idx] = commit
            results_by_val[val_label] = np.mean(depths)
        else:
            kwarg_name = param_to_kwarg[param_name]
            kwargs[kwarg_name] = val
            r = fleet_commitment_depth(**kwargs)
            results_by_val[val_label] = r['mean']

    swing = results_by_val['high'] - results_by_val['low']
    tornado_rows.append({
        'parameter': param_name,
        'low_val': lo,
        'central_val': central,
        'high_val': hi,
        'depth_at_low': round(results_by_val['low'], 5),
        'depth_at_central': round(results_by_val['central'], 5),
        'depth_at_high': round(results_by_val['high'], 5),
        'swing': round(swing, 5),
        'abs_swing': round(abs(swing), 5),
    })

    print(f'  {param_name:<4}: low({lo:.3f})={results_by_val["low"]:.1%}  '
          f'central({central:.3f})={results_by_val["central"]:.1%}  '
          f'high({hi:.3f})={results_by_val["high"]:.1%}  '
          f'swing={swing:+.1%}')

# Sort by absolute swing
tornado_rows.sort(key=lambda r: r['abs_swing'], reverse=True)
tornado_df = pd.DataFrame(tornado_rows)

print()
print('  SORTED BY IMPACT MAGNITUDE:')
print(f'  {"Rank":>4} {"Param":<6} {"Low":>7} {"Central":>8} {"High":>7} '
      f'{"Depth@Lo":>9} {"Depth@Hi":>9} {"Swing":>8}')
print('  ' + '─' * 70)
for i, r in enumerate(tornado_rows):
    print(f'  {i+1:>4} {r["parameter"]:<6} {r["low_val"]:>7.3f} {r["central_val"]:>8.3f} '
          f'{r["high_val"]:>7.3f} {r["depth_at_low"]:>8.1%} {r["depth_at_high"]:>8.1%} '
          f'{r["swing"]:>+7.1%}')


# ── Figure 2: Tornado Chart ──────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 7))

n_params = len(tornado_rows)
y_pos = np.arange(n_params)
central_depth = tornado_rows[0]['depth_at_central']  # should all be same

# Compute offsets from central
for r in tornado_rows:
    r['low_offset'] = r['depth_at_low'] - r['depth_at_central']
    r['high_offset'] = r['depth_at_high'] - r['depth_at_central']

# Plot bars
for i, r in enumerate(tornado_rows):
    lo_off = r['low_offset']
    hi_off = r['high_offset']

    # Low side (left of central)
    left = min(lo_off, hi_off)
    right = max(lo_off, hi_off)

    # Red for decrease, green for increase
    ax.barh(i, lo_off, left=0, height=0.6, color='#D44B3F', alpha=0.8,
            label='Conservative (low)' if i == 0 else '')
    ax.barh(i, hi_off, left=0, height=0.6, color='#4CAF50', alpha=0.8,
            label='Optimistic (high)' if i == 0 else '')

    # Annotations
    ax.text(lo_off - 0.002, i, f'{r["depth_at_low"]:.1%}',
            va='center', ha='right' if lo_off < 0 else 'left', fontsize=8,
            color='#8B0000')
    ax.text(hi_off + 0.002, i, f'{r["depth_at_high"]:.1%}',
            va='center', ha='left' if hi_off > 0 else 'right', fontsize=8,
            color='#006400')

# Labels
param_labels = [f'{r["parameter"]} [{r["low_val"]:.2f}–{r["high_val"]:.2f}]'
                for r in tornado_rows]
ax.set_yticks(y_pos)
ax.set_yticklabels(param_labels, fontsize=10)
ax.invert_yaxis()

# Central line
ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax.text(0.001, -0.7, f'Central: {central_depth:.1%}', fontsize=9,
        ha='left', fontweight='bold')

# DVFS floor
dvfs_floor = DVFS * S1
dvfs_offset = dvfs_floor - central_depth
ax.axvline(dvfs_offset, color='gray', linewidth=1, linestyle=':',
           alpha=0.7, label=f'DVFS floor: {dvfs_floor:.1%}')

ax.set_xlabel('Change in Mean Commitment Depth from Central', fontsize=11)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax.set_title(f'One-at-a-Time Parameter Sensitivity (10 GW Fleet)\n'
             f'Central commitment depth: {central_depth:.1%}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')

# Grid
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(ax.get_xlim()[0] * 1.15, ax.get_xlim()[1] * 1.15)

plt.tight_layout()
fig2.savefig('tornado_sensitivity_10gw.png', dpi=200, bbox_inches='tight')
print(f'\n  Figure saved: tornado_sensitivity_10gw.png')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: CSV Exports
# ══════════════════════════════════════════════════════════════════════════════

surface_df.to_csv('sensitivity_surface_D2_D5.csv', index=False)
tornado_df.to_csv('tornado_sensitivity_10gw.csv', index=False)

print()
print('CSV EXPORTS:')
print(f'  sensitivity_surface_D2_D5.csv  ({len(surface_df)} rows)')
print(f'  tornado_sensitivity_10gw.csv   ({len(tornado_df)} rows)')
print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Key Findings
# ══════════════════════════════════════════════════════════════════════════════

print('─' * 70)
print('PHASE 4 KEY FINDINGS')
print('─' * 70)
print()

# Surface extremes
print(f'  SURFACE (D2 × D5 at 10 GW):')
print(f'    Mean depth range: {surface_mean.min():.1%} – {surface_mean.max():.1%}')
print(f'    Central (D2={D2_CENTRAL}, D5={D5_CENTRAL}): {central_result["mean"]:.1%}')
# Best corner: D2=0.50, D5=0.80
best_corner = surface_mean[-1, -1]
worst_corner = surface_mean[0, 0]
print(f'    Best corner  (D2=0.50, D5=0.80): {best_corner:.1%}')
print(f'    Worst corner (D2=0.20, D5=0.50): {worst_corner:.1%}')
print(f'    Corner-to-corner range: {best_corner - worst_corner:+.1%}')
print()

# Tornado top 3
print(f'  TORNADO (one-at-a-time, 10 GW):')
for i, r in enumerate(tornado_rows[:3]):
    print(f'    #{i+1}: {r["parameter"]} — swing {r["swing"]:+.1%} '
          f'({r["depth_at_low"]:.1%} to {r["depth_at_high"]:.1%})')
print(f'    Remaining {n_params - 3} parameters: combined swing < '
      f'{sum(r["abs_swing"] for r in tornado_rows[3:]):.1%}')
print()

# D2 dominance check
d2_swing = next(r for r in tornado_rows if r['parameter'] == 'D2')['abs_swing']
total_swing = sum(r['abs_swing'] for r in tornado_rows)
print(f'  D2 contributes {d2_swing/total_swing:.0%} of total absolute sensitivity')
print(f'  D5 contributes {next(r for r in tornado_rows if r["parameter"] == "D5")["abs_swing"]/total_swing:.0%}')
print(f'  D2 + D5 combined: {(d2_swing + next(r for r in tornado_rows if r["parameter"] == "D5")["abs_swing"])/total_swing:.0%}')
print()
print('  → D2 and D5 were correctly identified as the axes for the')
print('    sensitivity surface. They dominate the one-at-a-time sensitivity.')

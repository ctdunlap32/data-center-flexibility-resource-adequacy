# Inter-Notebook Contract Schemas

Notebooks communicate through JSON and parquet files in `outputs/contracts/`.
These files are regenerated each time the upstream notebook runs. Do not
hand-edit them.

---

## Notebook 01 → Notebook 02

### stress_correlation_results.json

**Producer:** Notebook 01, Cell 4-1
**Consumer:** Notebook 02, Cell 0-2

```json
{
  "metadata": {
    "version": "1.0",
    "source_zone": "PJM_COMED",
    "top_n": 50,
    "years": [2022, 2023, 2024, 2025],
    "n_destination_zones": 17,
    "total_destination_mw": 17200.4,
    "produced_by": "notebook 01 — empirical_evidence"
  },
  "headline": {
    "capacity_weighted_overlap_pct": float,
    "dynamic_availability_pct": float,
    "all_stressed_pct": float,
    "avg_intra_pjm_overlap_pct": float,
    "avg_cross_ba_overlap_pct": float
  },
  "empirical_destination_lmps": {
    "capacity_weighted_mean": float,
    "capacity_weighted_median": float,
    "note": string
  },
  "per_zone": {
    "<ZONE_ID>": {
      "zone": string,
      "rto": string,
      "interconnection": string,
      "dc_capacity_mw": float,
      "overlap_pct": float,
      "overlap_hours": int,
      "dest_mean_during_source_stress": float,
      "dest_median_during_source_stress": float,
      "dest_p90_during_source_stress": float,
      "dest_max_during_source_stress": float,
      "dest_overall_mean": float,
      "stress_premium": float
    }
  },
  "yearly": {
    "<YEAR>": {
      "source_stress_hours": int,
      "intra_pjm_avg": float,
      "all_dest_stressed": float
    }
  }
}
```

**Zone IDs (17 zones):** ERCOT_LZ_NORTH, ERCOT_LZ_SOUTH, ERCOT_LZ_WEST,
ERCOT_LZ_HOUSTON, CAISO_NP15, CAISO_SP15, MISO_MINN_HUB, MISO_INDIANA_HUB,
MISO_ILLINOIS_HUB, MISO_MICHIGAN_HUB, MISO_ARKANSAS_HUB, MISO_LOUISIANA_HUB,
NYISO_ZONE_J, NYISO_ZONE_F, NYISO_ZONE_A, NYISO_ZONE_G, NYISO_ZONE_K

---

### per_hour_destination_availability.parquet

**Producer:** Notebook 01, Cell 4-2
**Consumer:** Notebook 02, Cell 0-2 / Part 3

DataFrame with one row per ComEd stress hour. 200 rows × 19 columns.

| Column | Type | Description |
|---|---|---|
| timestamp | datetime64[ns] | Stress hour UTC timestamp |
| ERCOT_LZ_NORTH | float64 | Available MW (0 if co-stressed) |
| ERCOT_LZ_SOUTH | float64 | " |
| ERCOT_LZ_WEST | float64 | " |
| ERCOT_LZ_HOUSTON | float64 | " |
| CAISO_NP15 | float64 | " |
| CAISO_SP15 | float64 | " |
| MISO_MINN_HUB | float64 | " |
| MISO_INDIANA_HUB | float64 | " |
| MISO_ILLINOIS_HUB | float64 | " |
| MISO_MICHIGAN_HUB | float64 | " |
| MISO_ARKANSAS_HUB | float64 | " |
| MISO_LOUISIANA_HUB | float64 | " |
| NYISO_ZONE_J | float64 | " |
| NYISO_ZONE_F | float64 | " |
| NYISO_ZONE_A | float64 | " |
| NYISO_ZONE_G | float64 | " |
| NYISO_ZONE_K | float64 | " |
| pjm_co_stressed_mw | float64 | Total PJM MW simultaneously stressed |

Each destination column contains the zone's operational DC capacity (MW) when
unstressed, or 0.0 when co-stressed with the source zone. Notebook 02 Part 3
reads these values directly as realized D1 availability per hour.

---

### workload_parameters.json

**Producer:** Notebook 01, Cell 4-3
**Consumer:** Notebook 02, Cell 0-2

```json
{
  "version": "1.0",
  "produced_by": "notebook 01 — empirical_evidence",
  "throughput_assumption_tok_per_sec": 60,
  "dynamolm": {
    "available": bool,
    "n_requests": int,
    "date_range_source": string,
    "p99_drain_time_coding_sec": float,
    "p99_drain_time_conv_sec": float,
    "citation": string
  },
  "burstgpt": {
    "available": bool,
    "n_requests": int,
    "date_range": [string, string],
    "date_range_source": string,
    "p99_drain_time_sec": float,
    "citation": string
  },
  "s3_parameterization": {
    "value": 0.90,
    "justification": string,
    "pjm_dispatch_window_seconds": 600
  }
}
```

---

## Notebook 02 → Notebook 03

### cascade_parameters.json

**Producer:** Notebook 02, Cell 5-1
**Consumer:** Notebook 03, Cell 0-2

```json
{
  "version": "1.0",
  "produced_by": "notebook 02 — cascade_simulation",
  "parameters": {
    "<PARAM>": {
      "central": float,
      "range": [float, float, float],
      "grounding": string
    }
  },
  "cascade_product": {
    "conservative": float,
    "central": float,
    "optimistic": float
  },
  "commitment_depth_baseline": float,
  "dvfs_floor_facility": float,
  "flex_frac": float,
  "variance_decomposition_eta_squared": {
    "<PARAM>": float
  },
  "notes": string
}
```

**Parameter IDs:** S1, S2, S3, D1, D2, D3, D4, D5, E1, E2

**Grounding values:** STRUCTURAL, DATA, ESTIMATED-WEAK, ESTIMATED-MODERATE, ESTIMATED

**Range format:** [conservative, central, optimistic]

---

### conditional_mc_results.json

**Producer:** Notebook 02, Cell 5-2
**Consumer:** Notebook 03, Cell 0-2 / Parts 1-2

```json
{
  "version": "1.0",
  "produced_by": "notebook 02 — cascade_simulation",
  "n_draws_per_hour": 2000,
  "n_stress_hours": 200,
  "source_zone": "PJM_COMED",
  "single_facility_500mw": {
    "facility_mw": int,
    "mean_commitment_depth": float,
    "median": float,
    "p5": float,
    "p95": float,
    "cvar5": float,
    "pct_constrained": float
  },
  "empirical_fleet": {
    "mean_commitment_depth": float,
    "median": float,
    "p5": float,
    "cvar5": float,
    "pct_constrained": float,
    "mean_migrating_mw": float,
    "max_migrating_mw": float
  },
  "per_gw_sweep": [
    {
      "fleet_gw": float,
      "fleet_mw": float,
      "migrating_mw": float,
      "mean": float,
      "median": float,
      "p5": float,
      "p95": float,
      "cvar5": float,
      "pct_constrained": float
    }
  ],
  "contention_onset_gw": float,
  "sensitivity": {
    "surface_fleet_mw": int,
    "surface_depth_range": [float, float],
    "tornado_fleet_mw": int,
    "tornado_baseline_depth": float
  }
}
```

---

## Notebook 03 → (terminal)

### final_results.json

**Producer:** Notebook 03, Cell 4-2
**Consumer:** None (reference artifact)

```json
{
  "version": "1.0",
  "produced_by": "notebook 03 — system_implications",
  "elcc": 0.92,
  "e3_ct_levelized_cost": 180000,
  "e3_ct_levelized_ceja": 205000,
  "reference_cases": [
    {
      "fleet_gw": float,
      "fleet_mw": float,
      "mean_depth": float,
      "committed_mw": float,
      "accredited_mw": float,
      "annual_avoided_base": float,
      "annual_avoided_ceja": float
    }
  ],
  "per_gw_curve": [
    {
      "fleet_gw": float,
      "fleet_mw": float,
      "mean": float,
      "committed_mw": float,
      "accredited_mw": float,
      "annual_avoided_base": float
    }
  ],
  "ix_acceleration": {
    "facility_gw": float,
    "gpu_per_mw_grid": int,
    "gpu_rate_hr": float,
    "annual_compute_rev": float,
    "wacc": float,
    "npv_conservative_2yr": float,
    "npv_central_3yr": float,
    "npv_optimistic_4yr": float,
    "dominance_ratio": float
  },
  "contention_onset_gw": float
}
```

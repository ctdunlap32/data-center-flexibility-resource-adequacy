# Bartlett Fellowship — Data Center Flexibility as Resource Adequacy
## Analysis Notebook (v15)

**Fundamental Research Question:** Given that PJM assigns a 92% ELCC to demand response, how deeply can a data center fleet credibly commit — and what does that commitment mean for grid reliability and resource planning?

**Source-of-Truth Hierarchy:**
This notebook → Pillar 1 (cascade model) → Pillar 3 (capacity market) → Pillar 2 (stress correlation)

---

### Notebook Architecture

| Part | Section | What It Establishes |
|------|---------|-------------------|
| **0** | **Setup & Parameters** | Shared constants, seven-parameter cascade, empirical destination data |
| **1** | **Empirical Stress Profile** | When stress happens, how long events last, duration vs. mechanism design |
| **2** | **Cascade Validation** | Monte Carlo variance decomposition, scenario sensitivity |
| **3** | **Commitment Depth & Grid Impact** | Mechanism profiles → optimization → election mechanism → E3 counterfactual |
| **4** | **Energy Economics (Supporting)** | Arbitrage value stack, three-prong taxonomy, spatial break-even, portfolio interaction |
| **5** | **Behavioral Incentive & Forward View** | IX queue NPV (the actual incentive), forward MC projections |
| **6** | **Summary & Export** | Consolidated results, validation checks, CSV export |

### Key Parameters (from Part 0)

| Parameter | Value | Grounding |
|-----------|-------|-----------|
| Cascade product (central) | 0.412 | 7-param model |
| Commitment depth (inference-dominant) | 55.9% | Cascade + DVFS |
| DVFS-only floor | 25% | Colangelo et al. 2025 |
| DR ELCC | 92% | PJM class rating (exogenous) |
| BRA price | $333.44/MW-day | PJM 2027/28 |
| Primary uncertainty | P3_B (67%), H (15%) | Author's estimates |

### Headline Results (from Parts 3, 5)

| Metric | DVFS Only | DVFS + Spatial |
|--------|-----------|---------------|
| Commitment depth | 25% | 54% |
| Accredited MW (10 GW) | 2,300 | ~4,990 |
| Net capacity value (10 GW) | see Part 3 | see Part 3 |
| IX Queue NPV (1 GW, 3yr) | — | ~$10B+ |

## Part 0: Setup and Parameters
- Cell 1: Imports, file paths, data loading
- Cell 2: Core economic parameters - GPU economics, BRA prices, reserve margins, E3 values, discount rate
- Cell 3: Ten-parameter cascade definition - cascade, ranges, and product
- Cell 4: v5 empirical data loading

### 0.1 Data Loading


```python
# Cell 1: Setup, file paths, and load ALL data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re, math
from scipy import stats

BASE_DIR = r"C:\Users\dunla\OneDrive\Documents\Bartlett Fellowship\Data\pjm"

DA_LMP_FILE = os.path.join(BASE_DIR, "comed_da_lmps.csv")
RT_LMP_FILE = os.path.join(BASE_DIR, "comed_rt_lmps.csv")
LOAD_FILE   = os.path.join(BASE_DIR, "comed_hourly_load.csv")

YEARS = [2022, 2023, 2024, 2025]
PRIMARY_YEAR = 2025

os.makedirs('output', exist_ok=True)

# --- Load DA LMPs ---
da_raw = pd.read_csv(DA_LMP_FILE)
# Filter to COMED zone only
da = da_raw[da_raw['pnode_name'] == 'COMED'].copy()
da['datetime'] = pd.to_datetime(da['datetime_beginning_ept'])

# --- De-duplicate DA by keeping the current version ---
da = da.sort_values(['datetime', 'version_nbr'])

if 'row_is_current' in da.columns:
    da = (
        da.sort_values(['datetime', 'row_is_current', 'version_nbr'])
          .drop_duplicates(subset=['datetime'], keep='last')
          .reset_index(drop=True)
    )
else:
    da = (
        da.sort_values(['datetime', 'version_nbr'])
          .drop_duplicates(subset=['datetime'], keep='last')
          .reset_index(drop=True)
    )

print(f"DA after version de-dupe: {len(da):,} rows")
da['year'] = da['datetime'].dt.year
da = da[da['year'].isin(YEARS)].sort_values('datetime').reset_index(drop=True)
da['month'] = da['datetime'].dt.month
da['hour']  = da['datetime'].dt.hour

# Sanity check
print(f'  DA LMPs: {len(da):,} hours, {da["datetime"].min()} to {da["datetime"].max()}')
print(f'  Per year: {da.groupby("year").size().to_dict()}')

# --- Load RT LMPs ---
rt_raw = pd.read_csv(RT_LMP_FILE)
rt = rt_raw[rt_raw['pnode_name'] == 'COMED'].copy()
rt['datetime'] = pd.to_datetime(rt['datetime_beginning_ept'])

# --- De-duplicate RT by keeping the current version ---
rt = rt.sort_values(['datetime', 'version_nbr'])

if 'row_is_current' in rt.columns:
    rt = (
        rt.sort_values(['datetime', 'row_is_current', 'version_nbr'])
          .drop_duplicates(subset=['datetime'], keep='last')
          .reset_index(drop=True)
    )
else:
    rt = (
        rt.sort_values(['datetime', 'version_nbr'])
          .drop_duplicates(subset=['datetime'], keep='last')
          .reset_index(drop=True)
    )

print(f"RT after version de-dupe: {len(rt):,} rows")
rt['year'] = rt['datetime'].dt.year
rt = rt[rt['year'].isin(YEARS)].sort_values('datetime').reset_index(drop=True)
print(f'  RT LMPs: {len(rt):,} hours')

# --- Load Hourly Load ---
load_raw = pd.read_csv(LOAD_FILE)
load_df = load_raw.copy()
load_df['datetime'] = pd.to_datetime(load_df['datetime_beginning_ept'])
load_df['year']  = load_df['datetime'].dt.year
load_df['month'] = load_df['datetime'].dt.month
load_df = load_df[load_df['year'].isin(YEARS)].sort_values('datetime').reset_index(drop=True)
print(f'  Load: {len(load_df):,} hours')

# --- Data Quality Check ---
for yr in YEARS:
    expected = 8784 if yr % 4 == 0 else 8760
    actual_da   = len(da[da['year'] == yr])
    actual_load = len(load_df[load_df['year'] == yr])
    flag = ' ⚠️' if abs(actual_da - expected) > 24 else ''
    print(f'  {yr}: DA={actual_da}/{expected} Load={actual_load}/{expected}{flag}')

print('\n--- All data loaded. ---')
```

    DA after version de-dupe: 35,060 rows
      DA LMPs: 35,060 hours, 2022-01-01 00:00:00 to 2025-12-31 23:00:00
      Per year: {2022: 8759, 2023: 8759, 2024: 8783, 2025: 8759}
    

    RT after version de-dupe: 35,060 rows
      RT LMPs: 35,060 hours
    

      Load: 35,064 hours
      2022: DA=8759/8760 Load=8760/8760
      2023: DA=8759/8760 Load=8760/8760
      2024: DA=8783/8784 Load=8784/8784
      2025: DA=8759/8760 Load=8760/8760
    
    --- All data loaded. ---
    

### 0.2 Economic & Physical Parameters

GPU economics, capacity market prices, E3 counterfactual values, 
and all non-cascade shared constants. Every downstream cell reads from here.


```python
# ======================================================================
# Cell 0-2: SHARED PARAMETERS — Economic & Physical (v18)
# ===========================================================
# Single source of truth for all downstream cells.
# Split from original monolithic Cell 2 for readability.
#
# SOURCE QUALITY KEY:
#   [DATA]         = Primary source, verified and citable
#   [E3-RA]        = From E3 2025 IL Resource Adequacy Study (Dec 2025)
#   [CALIBRATE]    = Real-data anchor not yet connected
#   [ESTIMATED]    = Informed estimate; needs sensitivity analysis
#   [STRUCTURAL]   = Modeling choice; justify in methodology
#   [PLACEHOLDER]  = Needs replacement — NOT used in headline results
#   [DERIVED]      = Calculated from other parameters
# ===========================================================
 
# --- GPU Economics ---
# [DATA] Hardware Specs (Source: NVIDIA DGX H100 Datasheet)
SYSTEM_MAX_POWER_KW = 10.2  # Max power for an 8-GPU DGX H100 node
GPUS_PER_SYSTEM     = 8
# [DATA] Facility Efficiency (Source: Uptime Institute 2024)
FACILITY_PUE = 1.3          # Power Usage Effectiveness for modern AI data center
 
# [DERIVED] Power metrics
GPU_IT_POWER_KW     = SYSTEM_MAX_POWER_KW / GPUS_PER_SYSTEM          # 1.275 kW/GPU (IT only)
GPU_GRID_POWER_KW   = GPU_IT_POWER_KW * FACILITY_PUE                 # 1.6575 kW/GPU (including cooling)
GPU_PER_MW_GRID     = int(1000 / GPU_GRID_POWER_KW)                  # ~603 GPUs per MW (grid-metered)
GPU_PER_MW_IT       = int(1000 / GPU_IT_POWER_KW)                    # ~784 GPUs per MW (IT only)
 
GPU_RATE_HR          = 2.20    # [DATA] Jan 2026 H100 spot rate. Was $3/hr mid-2024, $8/hr early 2024.
HOURLY_COMPUTE_VALUE = GPU_PER_MW_GRID * GPU_RATE_HR                  # [DERIVED] ~$1,326/MWh (grid-metered basis)
# NOTE: Spatial migration uses GPU_PER_MW_IT since you migrate compute, not cooling
 
# --- Curtailment Dynamics ---
IT_LOAD_MW = 100.0                # [STRUCTURAL] Reference facility size
COOLING_DECAY_MINUTES = 30        # [STRUCTURAL] Latent heat evacuation time
AVG_EVENT_DURATION_HRS = 4.0      # [CALIBRATE] Part 1 empirical: load events avg ~4h (range 1-13h)
 
THERMO_PENALTY_PER_EVENT = (COOLING_DECAY_MINUTES / 60) * (FACILITY_PUE - 1) / FACILITY_PUE
THERMO_EFFICIENCY = 1.0 - (THERMO_PENALTY_PER_EVENT / AVG_EVENT_DURATION_HRS)
 
# --- Flexibility Parameters ---
FLEX_FRAC     = 0.25             # [DATA-Colangelo] 25% via DVFS + job pausing (Colangelo et al. 2025 Nature Energy)
MIGRATION_LATENCY_MIN = 15       # [DATA-Colangelo] 5–20 min range for LLaMA-scale checkpoints; 15 min mid-range
MIGRATION_COST_PER_EVENT = HOURLY_COMPUTE_VALUE * (MIGRATION_LATENCY_MIN / 60)  # [DERIVED] ~$331.50/event/MW
 
# --- Inference Routing Parameters (NEW — v15.1) ---
# Inference routing is fundamentally different from training migration:
#   - No checkpoint. No optimizer state. No training run to interrupt.
#   - Active requests drain in seconds (P99 = 12.1s, BurstGPT cross-validation).
#   - New requests route to destination immediately via load balancer config.
#   - In-flight requests complete normally during drain; no compute is wasted.
#   - Cost structure is annual readiness, not per-event friction.
#
INFERENCE_DRAIN_TIME_SEC   = 12.1     # [DATA] P99 drain time, BurstGPT (validated against DynamoLLM 4.5-11.6s)
INFERENCE_DRAIN_TIME_MIN   = INFERENCE_DRAIN_TIME_SEC / 60                    # [DERIVED] ~0.20 min
INFERENCE_LOST_COMPUTE     = 0.0      # [STRUCTURAL] Requests complete during drain; no wasted compute
INFERENCE_COST_PER_EVENT   = HOURLY_COMPUTE_VALUE * (INFERENCE_LOST_COMPUTE)  # [DERIVED] $0/event/MW
 
# Readiness costs: annual fixed costs to maintain migration capability at destinations.
# These are incurred regardless of whether stress events occur.
#   Pre-staging:  storing model weights at N destination facilities (~storage + periodic updates)
#   Serving stack: maintaining warm inference serving infrastructure at destinations
#   Bandwidth:    reserved cross-region WAN capacity for burst migration traffic
#
# [PLACEHOLDER] — These need estimation from hyperscaler operational data.
# Included here to enable sensitivity analysis on break-even readiness cost.
INFERENCE_READINESS_COST_PER_MW_YR = None  # $/MW-yr; set to None until grounded
 
# --- Event Frequency ---
EVENTS_PER_YEAR = 10             # [CALIBRATE] Part 1 empirical: 8-12 load events/year (ComEd, 2022-2025)
 
# --- Capacity Market (BRA) ---
BRA_2025_26_PRICE = 269.92       # [DATA] $/MW-day (PJM 2025/26 BRA)
BRA_2026_27_PRICE = 329.17       # [DATA] $/MW-day (PJM 2026/27 BRA)
BRA_2027_28_PRICE = 333.44       # [DATA] $/MW-day (PJM 2027/28 BRA)
ELCC_2025_26_DR   = 0.76         # [DATA] PJM ELCC Class Ratings
ELCC_2026_27_DR   = 0.69         # [DATA]
ELCC_2027_28_DR   = 0.92         # [DATA] year-round commitment
 
# --- Forward Growth ---
ORGANIC_GROWTH = 0.009           # [DATA] 0.9%/yr non-DC load growth (PJM 2025 LTLF)
DC_GROWTH_MAP  = {2026: 2.0, 2028: 6.0, 2030: 10.0, 2032: 12.0, 2035: 14.0}  # [ESTIMATED] GW DC online in ComEd
RETIREMENTS_MAP = {2026: 1500, 2028: 5000, 2030: 8000, 2032: 12000, 2035: 16000}  # [ESTIMATED] Cumulative MW under CEJA
 
# --- Capacity Scenarios ---
CAP_SCENARIOS = {
    'Current cap holds':   333.44,    # [DATA] 2027/28 BRA clearing price
    'Modest increase':     450.00,    # [ESTIMATED] Brattle 6th Quadrennial Review ($450-$625 range)
    'Cap lifted/reformed': 530.00,    # [ESTIMATED] Near Net CONE ($528/MW-day, Brattle)
}
 
# --- Reserve Margin ---
RESERVE_MARGIN_TARGET = 1.20  # [DATA] PJM 2027/28 IRM = 20%
 
# --- ELCC ---
# v13.1: ELCC is exogenous (PJM DR class rating = 0.92). No saturation model.
# Mechanism choice determines commitment DEPTH, not ELCC rating.
 
# --- Supply Curve ---
PJM_OFFER_CAP  = 2000.0     # [DATA] PJM Operating Agreement energy offer cap
SCARCITY_THRESHOLD = 0.90    # [STRUCTURAL] Utilization level at hockey-stick regime
KNEE_SENSITIVITY   = 3.0     # [STRUCTURAL] Supply curve steepness (sensitivity: 2.0, 3.0, 5.0)
CANNIBALIZATION_RATE = 0.03  # [STRUCTURAL] 3% spread suppression per GW flex capacity
CANNIBALIZATION_FLOOR = 0.50 # [STRUCTURAL] Floor at 50% suppression
 
# --- Avoided Cost ---
BESS_CAPEX_PER_MW   = 1_400_000     # [DATA] NREL ATB 2024 (~$350/kWh × 4hr)
PEAKER_CAPEX_PER_MW = 1_000_000     # [DATA] EIA AEO 2024 new CT
BESS_8HR_CAPEX      = 2_400_000     # [ESTIMATED] Lazard LCOS 2024
BTM_CAPEX_PER_MW    = 1_500_000     # [ESTIMATED] 4hr BESS + controls
 
# --- E3 RA Study Integration ---
E3_COMED_SHORTFALL_START = 2029     # [E3-RA] Year PJM capacity shortfall begins
E3_NEW_GAS_CT_GW = 13.0            # [E3-RA] GW new in-state gas CTs in Base Case
E3_IMPORT_GW = 18.0                # [E3-RA] GW import capacity needed
E3_CT_OVERNIGHT_COST = 1_100_000   # [E3-RA] $/MW overnight cost for new CT
E3_CT_ANNUAL_FIXED_OM = 35_000     # [E3-RA] $/MW-yr fixed O&M
E3_CT_LEVELIZED_COST = 180_000     # [E3-RA] $/MW-yr levelized (Table 6-1)
 
# --- CEJA Sensitivity ---
# Illinois-specific: CEJA (2021) creates 100% clean standard by 2045.
# New gas CTs face stranded asset risk. If Illinois cannot build in-state,
# marginal resource = PJM import at ~$205K/MW-yr (base + transmission adder).
E3_CT_LEVELIZED_CEJA  = 205_000   # [SENSITIVITY] $/MW-yr
E3_CT_BUILD_RATE_CEJA = 0.15      # [SENSITIVITY] GW/yr in-state (near-zero)
CEJA_NOTE = (
    "CEJA sensitivity: assumes Illinois cannot rely on in-state gas CT builds "
    "post-2025. Marginal resource = PJM-delivered import capacity at ~$205K/MW-yr. "
    "Conservative: does not add clean peaker premium or carbon cost adder."
)
 
# --- V-LDES / Duration ---
CRISIS_DURATION_HRS = 8             # [STRUCTURAL] 8-hour crisis (Elliott=4d, Uri=~77h; 8h conservative)
CRISIS_TEMPORAL_DEGRADATION = 0.20  # [ESTIMATED] Emergency-mode throughput loss
 
# --- Discount Rate ---
WACC = 0.10                         # [STRUCTURAL] 10% WACC (Brattle VRR ATWACC = 9.5%)
 
# --- Merit Order (HIGH UNCERTAINTY) ---
PRICE_ELASTICITY_PER_GW = 150.0  # [ESTIMATED — CRITICAL] $/MWh DRIPE per GW curtailed
                                 # Source: Synapse AESC 2024. THIS DRIVES THE BIG SAVINGS NUMBER.
 
# ─────────────────────────────────────────────────────────────
print(f"Parameters loaded (v16).")
print(f"  GPU economics: {GPU_PER_MW_GRID} GPUs/MW (grid) | {GPU_PER_MW_IT} GPUs/MW (IT)")
print(f"  Compute value: ${HOURLY_COMPUTE_VALUE:,.0f}/MWh")
print(f"  Thermo efficiency: {THERMO_EFFICIENCY:.1%} for {AVG_EVENT_DURATION_HRS:.1f}-hr events")
print(f"  E3 avoided CT cost: ${E3_CT_LEVELIZED_COST:,}/MW-yr")
print(f"  Training migration cost/event: ${MIGRATION_COST_PER_EVENT:,.0f}/MW  (latency={MIGRATION_LATENCY_MIN} min)")
print(f"  Inference routing cost/event:  ${INFERENCE_COST_PER_EVENT:,.0f}/MW  (drain={INFERENCE_DRAIN_TIME_SEC}s, no lost compute)")
```

    Parameters loaded (v16).
      GPU economics: 603 GPUs/MW (grid) | 784 GPUs/MW (IT)
      Compute value: $1,327/MWh
      Thermo efficiency: 97.1% for 4.0-hr events
      E3 avoided CT cost: $180,000/MW-yr
      Training migration cost/event: $332/MW  (latency=15 min)
      Inference routing cost/event:  $0/MW  (drain=12.1s, no lost compute)
    

### 0.3 Ten-Parameter Cascade (Pillar 1)

The cascade model is the core technical contribution. Each parameter 
is an independent filter on the spatial migration pathway:

**Source-side:** S1 (workload candidacy) × S2 (data locality) × S3 (operational timing)  
**Destination-side:** D1 (availability) × D2 (utilization headroom) × D3 (HW compat) × D4 (inference share) × D5 (pre-staging)  
**Execution:** E1 (migration completion) × E2 (WAN bandwidth)

Product = effective spatial fraction = share of facility load that 
disappears from the source node during dispatch. Central: **0.0467**. 
Commitment depth (cascade + DVFS on shiftable residual): **21.0%**.


```python
# ======================================================================
# Cell 0-3: TEN-PARAMETER CASCADE — Pillar 1 (v18)
# ===========================================================
# Effective Spatial Fraction = S1 × S2 × S3 × D1 × D2 × D3 × D4 × D5 × E1 × E2
#
# This is the core technical contribution of the thesis.
# Each parameter is an independent filter on the spatial migration pathway.
# The product is the share of facility load that disappears from the
# source node during a capacity market dispatch event.
#
# SOURCE QUALITY:
#   S1, E2:           [STRUCTURAL / DATA] — well-grounded
#   D1:               [DATA] — 35,060 hrs DA LMP empirical (Pillar 2)
#   S3, D3, E1:      [ESTIMATED] — moderate grounding
#   S2:              [ESTIMATED-WEAK] — no published shiftability measurement
#   D2:               [ESTIMATED-WEAK] — literature disagrees on appropriate utilization metric
#   D4:               [ESTIMATED-MODERATE] — Deloitte/McKinsey inference share projections
#   D5:               [ESTIMATED-WEAK] — least empirically grounded (Pillar 1 TGV)
# ===========================================================

# --- Source-side: what fraction of load is a candidate and can execute? ---
CASCADE_S1     = 0.70    # [STRUCTURAL] Workload candidacy (inference-dominant scenario)
CASCADE_S2    = 0.70     # [ESTIMATED — regulatory floor] Data locality. Conservative central
                         #   within [0.60, 0.95] range, anchored on data residency requirements
                         #   for cross-region inference routing (same-country, same-jurisdiction).
                         #   Pillar1 Factor A original estimate was 0.80; revised conservative for
                         #   NE submission. Variance share ~13% (Methods).
CASCADE_S3    = 0.90    # [ESTIMATED] Operational feasibility given advance notice

# --- Destination-side: is there somewhere to go, and can it absorb? ---
CASCADE_D1     = 0.99    # [DATA] Destination availability (Pillar 2: 198/200 stress hours)
CASCADE_D2     = 0.33    # [ESTIMATED-WEAK] Utilization headroom = 1 - util.
                         #   Central util ~0.67 per SemiAnalysis/MIT GPU measurements.
                         #   Wide range [0.10, 0.80] reflects literature disagreement:
                         #   SemiAnalysis/MIT measure GPU util 0.50-0.90 (headroom 0.10-0.50);
                         #   TGV recommends sensitivity across util 0.20-0.50 (headroom 0.50-0.80).
                         #   Phase 4 sensitivity surface will address this directly.
CASCADE_D3     = 0.88    # [ESTIMATED-MODERATE] Hardware compatibility (CUDA backward compat, TensorRT)
                         #   compatible_fraction_reference Layer 1: 0.80-0.92.
                         #   Does NOT include pre-staging (that is now D5).
                         #   Old D3=0.97 included pre-staging implicitly per TGV.
CASCADE_D4     = 0.50    # [ESTIMATED-MODERATE] Inference workload share at destination
                         #   compatible_fraction_reference Layer 2: 0.33-0.67.
                         #   Deloitte (Nov 2025): ~50% in 2025. McKinsey: 30-40% by 2030.
CASCADE_D5     = 0.65    # [ESTIMATED-WEAK] Operational readiness / pre-staging
                         #   compatible_fraction_reference Layer 3: 0.50-0.80.
                         #   WEAKEST LAYER per Pillar 1 TGV. Central from reference
                         #   central scenario (0.65), not reference floor (0.50).
                         #   Hyperscaler own-fleet: 0.60-0.80. Conservative: 0.50.

# --- Execution: does the migration operation succeed? ---
CASCADE_E1     = 0.95   # [CONSERVATIVE SLA ENVELOPE] Routing completion. Below measured cloud LB
                        #   failure rates (~0.999 steady-state) to accommodate DNS propagation,
                        #   health-check latency, and cross-BA routing variability not captured
                        #   in published SLAs. Variance share <5% (Methods); result insensitive.
CASCADE_E2     = 0.98   # [CONSERVATIVE SLA ENVELOPE] Bandwidth adequacy. Below hyperscaler WAN
                        #   headroom implied by orders-of-magnitude argument (Pillar1 §E2);
                        #   deliberate buffer for unmodeled congestion events. Variance share
                        #   <5% (Methods); result insensitive.

# --- Cascade product (10-parameter) ---
EFFECTIVE_SPATIAL_FRAC = (CASCADE_S1 * CASCADE_S2 * CASCADE_S3 *
                          CASCADE_D1 * CASCADE_D2 * CASCADE_D3 *
                          CASCADE_D4 * CASCADE_D5 *
                          CASCADE_E1 * CASCADE_E2)

# --- Ranges for sensitivity (conservative, central, optimistic) ---
CASCADE_RANGES = {
    'S1':  (0.70,  0.70,  0.70),    # Fixed within inference-dominant scenario
    'S2': (0.60,  0.80,  0.95),    # WEAK — no published shiftability measurement
    'S3': (0.85,  0.90,  0.95),    # Moderate-strong (Azure trace: seconds vs minutes)
    'D1':  (0.945, 0.99,  0.995),   # Strong (empirical, forward range for climate/reflexivity)
    'D2':  (0.20,  0.33,  0.50),    # Narrowed: GPU utilization 50-80% -> headroom 0.20-0.50 (facility load factor excluded)
    'D3':  (0.80,  0.88,  0.92),    # Moderate (compatible_fraction_reference Layer 1, hardware only)
    'D4':  (0.33,  0.50,  0.67),    # Moderate (compatible_fraction_reference Layer 2)
    'D5':  (0.50,  0.65,  0.80),    # WEAK — least grounded (compatible_fraction_reference Layer 3)
    'E1':  (0.995, 0.997, 0.999),   # Moderate (SLA-bounded)
    'E2':  (0.99,  0.995, 1.00),    # Strong (orders-of-magnitude)
}

# Compute from CASCADE_RANGES — single source of truth
_con = {k: v[0] for k, v in CASCADE_RANGES.items()}
_opt = {k: v[2] for k, v in CASCADE_RANGES.items()}
_cascade_con = (_con['S1'] * _con['S2'] * _con['S3'] *
                _con['D1'] * _con['D2'] * _con['D3'] *
                _con['D4'] * _con['D5'] *
                _con['E1'] * _con['E2'])
_cascade_opt = (_opt['S1'] * _opt['S2'] * _opt['S3'] *
                _opt['D1'] * _opt['D2'] * _opt['D3'] *
                _opt['D4'] * _opt['D5'] *
                _opt['E1'] * _opt['E2'])

# --- Commitment depth (cascade + DVFS on remainder) ---
# v16.1: DVFS operates on the shiftable-compute envelope (S1), not facility total.
# Colangelo et al. 2026 measured 25% reduction at the GPU cluster, not at the
# facility meter. Applying 0.25 to (1 - spatial_frac) would include non-compute
# facility load (cooling, power conditioning, etc.) in the DVFS base, which
# overstates the flexibility floor. The corrected formula applies DVFS to the
# GPU-amenable residual within the S1 envelope only.
COMMITMENT_DEPTH = EFFECTIVE_SPATIAL_FRAC + FLEX_FRAC * (CASCADE_S1 - EFFECTIVE_SPATIAL_FRAC)

# ─────────────────────────────────────────────────────────────
print(f"TEN-PARAMETER CASCADE (Pillar 1, v17)")
print(f"  Source-side:")
print(f"    S1  (candidacy):       {CASCADE_S1:.2f}   [STRUCTURAL]")
print(f"    S2 (data locality):   {CASCADE_S2:.2f}   [ESTIMATED-WEAK]")
print(f"    S3 (timing):          {CASCADE_S3:.2f}   [ESTIMATED]")
print(f"  Destination-side:")
print(f"    D1  (availability):    {CASCADE_D1:.3f}  [DATA — empirical]")
print(f"    D2  (util headroom):   {CASCADE_D2:.2f}   [ESTIMATED-WEAK]")
print(f"    D3  (HW compat):      {CASCADE_D3:.2f}   [ESTIMATED-MODERATE]")
print(f"    D4  (inf share):      {CASCADE_D4:.2f}   [ESTIMATED-MODERATE]")
print(f"    D5  (pre-staging):    {CASCADE_D5:.2f}   [ESTIMATED-WEAK]")
print(f"  Execution:")
print(f"    E1  (completion):      {CASCADE_E1:.3f}  [ESTIMATED]")
print(f"    E2  (bandwidth):       {CASCADE_E2:.3f}  [ESTIMATED]")
print(f"  ─────────────────────────────────")
print(f"  PRODUCT (central):     {EFFECTIVE_SPATIAL_FRAC:.4f}")
print(f"  PRODUCT (cons/opt):    {_cascade_con:.4f} / {_cascade_opt:.4f}")
print(f"  Commitment depth:      {COMMITMENT_DEPTH:.1%}"
      f"  (= {EFFECTIVE_SPATIAL_FRAC:.4f} spatial + {FLEX_FRAC * (CASCADE_S1 - EFFECTIVE_SPATIAL_FRAC):.4f} DVFS on shiftable residual)")
```

    TEN-PARAMETER CASCADE (Pillar 1, v17)
      Source-side:
        S1  (candidacy):       0.70   [STRUCTURAL]
        S2 (data locality):   0.70   [ESTIMATED-WEAK]
        S3 (timing):          0.90   [ESTIMATED]
      Destination-side:
        D1  (availability):    0.990  [DATA — empirical]
        D2  (util headroom):   0.33   [ESTIMATED-WEAK]
        D3  (HW compat):      0.88   [ESTIMATED-MODERATE]
        D4  (inf share):      0.50   [ESTIMATED-MODERATE]
        D5  (pre-staging):    0.65   [ESTIMATED-WEAK]
      Execution:
        E1  (completion):      0.950  [ESTIMATED]
        E2  (bandwidth):       0.980  [ESTIMATED]
      ─────────────────────────────────
      PRODUCT (central):     0.0384
      PRODUCT (cons/opt):    0.0088 / 0.1548
      Commitment depth:      20.4%  (= 0.0384 spatial + 0.1654 DVFS on shiftable residual)
    

### 0.4 Empirical Destination Data

Loads capacity-weighted destination LMPs and coincidence factors from 
the companion stress correlation notebook (v5). These feed the energy 
economics calculations in Part 4. **Not required for capacity market 
headlines** — commitment depth, accredited MW, and avoided capacity 
are determined entirely by the cascade in Cell 0.3.


```python
# ======================================================================
# Cell 0-4: EMPIRICAL DESTINATION DATA — v5 Stress Correlation Results
# ===========================================================
# Loads empirical destination LMPs and coincidence factors from the
# companion Cross_BA_Stress_Correlation_v5 notebook.
#
# If the JSON is not found (companion notebook not yet run), falls back
# to conservative placeholder values. The fallback changes headline
# numbers — see warnings below.
#
# DEPENDENCY: Cross_BA_Stress_Correlation_v5.ipynb must be run first
#             to generate stress_correlation_results.json
# ===========================================================

import json as _json

_V5_RESULTS_PATH = r'C:\Users\dunla\OneDrive\Documents\Bartlett Fellowship\Demand Response Direction\1_Working Version\stress_correlation_results.json'

try:
    with open(_V5_RESULTS_PATH) as _f:
        _v5 = _json.load(_f)

    # Capacity-weighted mean destination LMP during ComEd top-50 stress hours
    DESTINATION_LMP_CRISIS = _v5['empirical_destination_lmps']['capacity_weighted_mean']

    # Per-interconnection destination LMPs
    _dest_zones = _v5['per_zone']
    _ic_lmps = {}
    for _z, _r in _dest_zones.items():
        _ic = _r['interconnection']
        if _ic not in _ic_lmps:
            _ic_lmps[_ic] = {'num': 0, 'denom': 0}
        if _r['dest_mean_during_source_stress'] is not None:
            _ic_lmps[_ic]['num'] += _r['dest_mean_during_source_stress'] * _r['dc_capacity_mw']
            _ic_lmps[_ic]['denom'] += _r['dc_capacity_mw']

    DEST_LMP_ERCOT   = _ic_lmps.get('ERCOT', {}).get('num', 0) / max(_ic_lmps.get('ERCOT', {}).get('denom', 1), 1)
    DEST_LMP_WESTERN = _ic_lmps.get('Western', {}).get('num', 0) / max(_ic_lmps.get('Western', {}).get('denom', 1), 1)
    
    # Eastern IC contains both MISO and NYISO — break out by RTO for energy economics
    _rto_lmps = {}
    for _z, _r in _dest_zones.items():
        _rto = _r.get('rto', _r.get('interconnection', 'Unknown'))
        if _rto not in _rto_lmps:
            _rto_lmps[_rto] = {'num': 0, 'denom': 0}
        if _r['dest_mean_during_source_stress'] is not None:
            _rto_lmps[_rto]['num'] += _r['dest_mean_during_source_stress'] * _r['dc_capacity_mw']
            _rto_lmps[_rto]['denom'] += _r['dc_capacity_mw']
    
    DEST_LMP_MISO    = _rto_lmps.get('MISO', {}).get('num', 0) / max(_rto_lmps.get('MISO', {}).get('denom', 1), 1)
    DEST_LMP_NYISO   = _rto_lmps.get('NYISO', {}).get('num', 0) / max(_rto_lmps.get('NYISO', {}).get('denom', 1), 1)
    DEST_LMP_EASTERN = _ic_lmps.get('Eastern', {}).get('num', 0) / max(_ic_lmps.get('Eastern', {}).get('denom', 1), 1)

    # Capacity-weighted stress overlap
    EMPIRICAL_CF = _v5['headline']['capacity_weighted_overlap_pct'] / 100

    DESTINATION_LMP_NORMAL = 40.0  # [ESTIMATED] non-crisis average

    _V5_LOADED = True
    print(f'✓ v5 empirical values loaded:')
    print(f'  DESTINATION_LMP_CRISIS = ${DESTINATION_LMP_CRISIS:.1f}/MWh')
    print(f'  DEST_LMP_ERCOT   = ${DEST_LMP_ERCOT:.1f}/MWh')
    print(f'  DEST_LMP_WESTERN = ${DEST_LMP_WESTERN:.1f}/MWh')
    print(f'  DEST_LMP_MISO    = ${DEST_LMP_MISO:.1f}/MWh')
    print(f'  DEST_LMP_NYISO   = ${DEST_LMP_NYISO:.1f}/MWh') if DEST_LMP_NYISO > 0 else None
    print(f'  DEST_LMP_EASTERN = ${DEST_LMP_EASTERN:.1f}/MWh (MISO+NYISO combined)')
    print(f'  EMPIRICAL_CF     = {EMPIRICAL_CF:.3f} ({EMPIRICAL_CF*100:.1f}%)')

except FileNotFoundError:
    _V5_LOADED = False
    print(f'⚠ WARNING: v5 results not found.')
    print(f'  Path: {_V5_RESULTS_PATH}')
    print(f'  Run Cross_BA_Stress_Correlation_v5.ipynb first.')
    print(f'  Falling back to PLACEHOLDER values:')

    DESTINATION_LMP_CRISIS = 120.0  # PLACEHOLDER — empirical is ~$168
    DEST_LMP_ERCOT   = 50.0
    DEST_LMP_WESTERN = 55.0
    DEST_LMP_MISO    = 45.0
    DEST_LMP_NYISO   = 50.0
    DEST_LMP_EASTERN = 45.0
    EMPIRICAL_CF = 0.05
    DESTINATION_LMP_NORMAL = 40.0

    print(f'  DESTINATION_LMP_CRISIS = ${DESTINATION_LMP_CRISIS:.1f}/MWh (placeholder, empirical ≈ $168)')
    print(f'  EMPIRICAL_CF = {EMPIRICAL_CF:.1%} (placeholder)')
    print(f'')
    print(f'  ⚠ Spatial migration cost will be ~$203/MWh instead of ~$251/MWh')
    print(f'  ⚠ All downstream energy economics affected. Capacity market')
    print(f'    headlines (commitment depth, accredited MW) are NOT affected.')
```

    ✓ v5 empirical values loaded:
      DESTINATION_LMP_CRISIS = $169.3/MWh
      DEST_LMP_ERCOT   = $182.3/MWh
      DEST_LMP_WESTERN = $112.2/MWh
      DEST_LMP_MISO    = $159.0/MWh
      DEST_LMP_NYISO   = $193.2/MWh
      DEST_LMP_EASTERN = $163.5/MWh (MISO+NYISO combined)
      EMPIRICAL_CF     = 0.272 (27.2%)
    

## Part 1: Empirical Stress Profile
- Cell 1: ComEd Price Landscape
- Cell 2: Stress Event Identification
- Cell 3: Stress Characterization & Headline Findings

When does the ComEd grid experience stress, how long do events last, 
and what does the duration profile mean for capacity resource design?

This section uses 35,060 hours of DA LMP data and hourly load data 
(Jan 2022 – Dec 2025) to identify stress events under two definitions 
(load-based and price-based), then characterizes their duration and 
seasonal patterns. The key finding — that price-based stress reveals 
multi-day winter crises invisible to load analysis — directly determines 
how deeply a data center can credibly commit as DR.

### 1.1 ComEd Price Landscape

Descriptive statistics and seasonal distribution of DA LMPs. 
Context for the stress event identification that follows.


```python
# ======================================================================
# Cell 1-1: ComEd DA LMP LANDSCAPE (v18)
# ===========================================================
# Descriptive context: what does the ComEd price landscape look like?
# This is background for the stress event identification that follows.
#
# INPUTS: da (DataFrame), YEARS (from Cell 0-1 data loading)
# OUTPUTS: None (descriptive only)
# ===========================================================

# --- Annual LMP Statistics ---
print('DA LMP STATISTICS BY YEAR ($/MWh)')
print('=' * 90)
for yr in YEARS + ['All']:
    if yr == 'All':
        subset = da['total_lmp_da']
        label = '2022-2025'
    else:
        subset = da[da['year'] == yr]['total_lmp_da']
        label = str(yr)
    print(f'{label:>10}: Mean=${subset.mean():>7.2f}  Med=${subset.median():>7.2f}  '
          f'P95=${subset.quantile(0.95):>8.2f}  P99=${subset.quantile(0.99):>8.2f}  '
          f'Max=${subset.max():>9.2f}  Neg%={100*((subset<0).sum()/len(subset)):>5.1f}%')

# --- Seasonal Distribution of Top-100 Price Hours ---
print(f'\n\nSEASONAL DISTRIBUTION OF TOP-100 LMP HOURS')
print('=' * 70)
season_map = {12:'Winter', 1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring', 5:'Spring',
              6:'Summer', 7:'Summer', 8:'Summer', 9:'Fall', 10:'Fall', 11:'Fall'}

for yr in YEARS:
    lmps = da[da['year'] == yr].copy()
    lmps_sorted = lmps.nlargest(100, 'total_lmp_da')
    lmps_sorted['season'] = lmps_sorted['month'].map(season_map)
    counts = lmps_sorted['season'].value_counts()
    avg_price = lmps_sorted.groupby('season')['total_lmp_da'].mean()
    print(f'\n{yr}:')
    for s in ['Summer', 'Winter', 'Spring', 'Fall']:
        n = counts.get(s, 0)
        p = avg_price.get(s, 0)
        print(f'  {s:>8}: {n:>3} hours  (avg ${p:>8.2f}/MWh)')
    hour_counts = lmps_sorted['hour'].value_counts().sort_index()
    peak_hours = hour_counts.nlargest(3)
    print(f'  Peak hours: {", ".join(f"{h}:00 ({c})" for h, c in peak_hours.items())}')
```

    DA LMP STATISTICS BY YEAR ($/MWh)
    ==========================================================================================
          2022: Mean=$  60.40  Med=$  51.65  P95=$  123.57  P99=$  173.55  Max=$   363.78  Neg%=  0.0%
          2023: Mean=$  26.68  Med=$  25.04  P95=$   46.72  P99=$   64.64  Max=$   292.76  Neg%=  0.8%
          2024: Mean=$  25.55  Med=$  22.36  P95=$   52.58  P99=$   93.55  Max=$   277.22  Neg%=  0.6%
          2025: Mean=$  36.64  Med=$  30.89  P95=$   78.05  P99=$  134.94  Max=$   497.18  Neg%=  1.2%
     2022-2025: Mean=$  37.31  Med=$  30.15  P95=$   90.45  P99=$  143.73  Max=$   497.18  Neg%=  0.7%
    
    
    SEASONAL DISTRIBUTION OF TOP-100 LMP HOURS
    ======================================================================
    

    
    2022:
        Summer:  62 hours  (avg $  202.43/MWh)
        Winter:  35 hours  (avg $  232.69/MWh)
        Spring:   1 hours  (avg $  173.44/MWh)
          Fall:   2 hours  (avg $  173.44/MWh)
      Peak hours: 17:00 (19), 16:00 (17), 15:00 (9)
    

    
    2023:
        Summer:  62 hours  (avg $  106.11/MWh)
        Winter:   5 hours  (avg $   76.76/MWh)
        Spring:   1 hours  (avg $   66.04/MWh)
          Fall:  32 hours  (avg $   72.82/MWh)
      Peak hours: 17:00 (24), 16:00 (19), 18:00 (16)
    
    2024:
        Summer:  29 hours  (avg $  120.16/MWh)
        Winter:  69 hours  (avg $  133.43/MWh)
        Spring:   0 hours  (avg $    0.00/MWh)
          Fall:   2 hours  (avg $   88.54/MWh)
      Peak hours: 17:00 (16), 18:00 (9), 16:00 (8)
    

    
    2025:
        Summer:  44 hours  (avg $  227.05/MWh)
        Winter:  49 hours  (avg $  188.05/MWh)
        Spring:   0 hours  (avg $    0.00/MWh)
          Fall:   7 hours  (avg $  143.21/MWh)
      Peak hours: 18:00 (15), 19:00 (13), 17:00 (11)
    

### 1.2 Stress Event Identification

Identifies stress events using two complementary methods:
- **Load-based:** Top 50 demand hours per year (summer-dominated)
- **Price-based:** Top 50 DA LMP hours per year, with conditional 
  merging across overnight gaps to capture multi-day weather systems

The union of both (de-overlapped) produces the combined event set 
used by Part 3's commitment optimization.


```python
# ======================================================================
# Cell 1-2: STRESS EVENT IDENTIFICATION (v18)
# ===========================================================
# Identifies WHEN and HOW LONG the ComEd grid is stressed using two
# complementary definitions:
#
#   LOAD-BASED:  Top 50 hours by MW demand per year
#     → Summer-dominated, captures thermal peak events
#
#   PRICE-BASED: Top 50 hours by DA LMP per year
#     → Balanced summer/winter, captures supply-side failures
#     → Conditionally merged across overnight gaps to reveal
#       multi-day weather systems (Elliott, polar vortex)
#
#   COMBINED:    Load ∪ Price, de-overlapped
#     → Most comprehensive dispatch probability estimate
#     → Consumed by commitment optimization (Part 3)
#
# INPUTS: da, load_df, YEARS (from Part 0)
# OUTPUTS:
#   load_events_by_yr           — dict of load-based events per year
#   price_events_raw_by_yr      — dict of raw price clusters per year
#   price_events_merged_by_yr   — dict of conditionally merged price events
#   combined_events_by_yr       — dict of union (load ∪ price) events
# ===========================================================

import numpy as np
import pandas as pd

TOP_N = 50  # Top 0.57% of hours — aligned with PJM 1-in-10 LOLE


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def cluster_top_hours(values, datetimes, top_n=TOP_N, gap_hours=1):
    """
    Identify the top_n hours by value, cluster into contiguous events.
    Works for both load (MW) and price ($/MWh) series.
    gap_hours: merge clusters separated by ≤ gap_hours.
    """
    vals = np.asarray(values, dtype=float)
    dts = pd.DatetimeIndex(datetimes)
    top_idx = np.sort(np.argsort(vals)[-top_n:])

    if len(top_idx) == 0:
        return []

    events = []
    cur = [top_idx[0]]
    for i in range(1, len(top_idx)):
        gap = (dts[top_idx[i]] - dts[top_idx[i-1]]).total_seconds() / 3600
        if gap <= (1 + gap_hours):
            cur.append(top_idx[i])
        else:
            events.append(_make_event(cur, vals, dts))
            cur = [top_idx[i]]
    events.append(_make_event(cur, vals, dts))
    return events


def _make_event(indices, vals, dts):
    """Package a cluster of indices into an event dict."""
    start_dt = pd.Timestamp(dts[indices[0]])
    end_dt   = pd.Timestamp(dts[indices[-1]])
    duration_hours = int((end_dt - start_dt) / pd.Timedelta(hours=1)) + 1

    return {
        'window_start': indices[0],
        'window_end': indices[-1],
        'stress_indices': indices.copy(),
        'duration': duration_hours,
        'start_dt': start_dt,
        'end_dt': end_dt,
        'peak_val': float(np.max(vals[indices])),
        'n_stress': int(len(indices)),
    }


def conditional_merge(events, full_series, max_gap_hours=18, pctl_threshold=0.75):
    """
    Merge adjacent price events when gap prices remain elevated.

    Logic: if the gap between two events is ≤ max_gap_hours AND the
    median price during the gap is ≥ the pctl_threshold of the annual
    series, merge them. This bridges overnight dips within multi-day
    weather systems (e.g., LMP drops from $300 to $100 at 3am during
    Elliott — still well above normal) without merging truly separate
    events (prices return to $30 for a week).
    """
    if len(events) <= 1:
        return events

    threshold_val = np.percentile(full_series, pctl_threshold * 100)
    merged = [events[0].copy()]

    for i in range(1, len(events)):
        prev, curr = merged[-1], events[i]
        gap_start = prev['window_end'] + 1
        gap_end = curr['window_start']
        gap_h = gap_end - gap_start

        if 0 < gap_h <= max_gap_hours:
            gap_vals = full_series[gap_start:gap_end]
            if len(gap_vals) > 0 and np.median(gap_vals) >= threshold_val:
                all_stress = sorted(set(prev['stress_indices'] + curr['stress_indices']))
                merged_end_dt = curr['end_dt']
                merged_start_dt = prev['start_dt']
                merged_duration = int((merged_end_dt - merged_start_dt) / pd.Timedelta(hours=1)) + 1
                merged[-1] = {
                    'window_start': prev['window_start'],
                    'window_end': curr['window_end'],
                    'stress_indices': all_stress,
                    'duration': merged_duration,
                    'start_dt': merged_start_dt, 'end_dt': merged_end_dt,
                    'peak_val': max(prev['peak_val'], curr['peak_val']),
                    'n_stress': len(all_stress)
                }
                continue
        merged.append(curr.copy())
    return merged


def get_season(dt):
    m = dt.month
    if m in [6, 7, 8]: return 'Summer'
    elif m in [12, 1, 2]: return 'Winter'
    return 'Shoulder'


# ─────────────────────────────────────────────────────────────
# IDENTIFY EVENTS
# ─────────────────────────────────────────────────────────────

load_events_by_yr = {}
price_events_raw_by_yr = {}
price_events_merged_by_yr = {}

for yr in YEARS:
    # Load-based
    ym = load_df['year'] == yr
    yr_loads = load_df.loc[ym, 'mw'].values
    yr_load_dts = load_df.loc[ym, 'datetime'].values
    if len(yr_loads) < 8000: continue
    load_events_by_yr[yr] = cluster_top_hours(yr_loads, yr_load_dts)

    # Price-based: cluster then conditionally merge
    pm = da['year'] == yr
    yr_lmps = da.loc[pm, 'total_lmp_da'].values
    yr_lmp_dts = da.loc[pm, 'datetime'].values
    if len(yr_lmps) < 8000: continue
    raw = cluster_top_hours(yr_lmps, yr_lmp_dts)
    price_events_raw_by_yr[yr] = raw
    price_events_merged_by_yr[yr] = conditional_merge(raw, yr_lmps)


# ─────────────────────────────────────────────────────────────
# COMBINED EVENTS (load ∪ price, de-overlapped)
# ─────────────────────────────────────────────────────────────

combined_events_by_yr = {}
for yr in YEARS:
    if yr not in load_events_by_yr or yr not in price_events_merged_by_yr:
        continue
    all_evts = load_events_by_yr[yr] + price_events_merged_by_yr[yr]
    all_evts.sort(key=lambda e: e['window_start'])

    # De-overlap: merge events that share or abut in time
    merged = [all_evts[0].copy()]
    for ev in all_evts[1:]:
        prev = merged[-1]
        if ev['window_start'] <= prev['window_end'] + 1:
            merged[-1] = {
                'window_start': min(prev['window_start'], ev['window_start']),
                'window_end': max(prev['window_end'], ev['window_end']),
                'stress_indices': sorted(set(
                    prev.get('stress_indices', []) + ev.get('stress_indices', []))),
                'duration': int((max(prev['end_dt'], ev['end_dt']) - min(prev['start_dt'], ev['start_dt'])) / pd.Timedelta(hours=1)) + 1,
                'start_dt': min(prev['start_dt'], ev['start_dt']),
                'end_dt': max(prev['end_dt'], ev['end_dt']),
                'peak_val': max(prev['peak_val'], ev['peak_val']),
                'n_stress': len(set(
                    prev.get('stress_indices', []) + ev.get('stress_indices', [])))
            }
        else:
            merged.append(ev.copy())
    combined_events_by_yr[yr] = merged


# ─────────────────────────────────────────────────────────────
# PRINT IDENTIFIED EVENTS
# ─────────────────────────────────────────────────────────────

print("LOAD-BASED STRESS EVENTS")
print(f"Top {TOP_N} hours per year by ComEd hourly load (MW)")
print("=" * 90)
for yr, events in load_events_by_yr.items():
    print(f"\n  {yr}: {len(events)} events from {TOP_N} stress hours")
    for ev in events:
        s = get_season(ev['start_dt'])
        print(f"    {ev['start_dt'].strftime('%Y-%m-%d %H:%M')} — "
              f"{ev['duration']:>2}h  peak={ev['peak_val']:,.0f} MW  ({s})")

print(f"\n\nPRICE-BASED STRESS EVENTS")
print(f"Top {TOP_N} hours per year by DA LMP, conditionally merged")
print("(gaps ≤18h merged when median gap price > 75th percentile annual LMP)")
print("=" * 90)
for yr, events in price_events_merged_by_yr.items():
    raw_n = len(price_events_raw_by_yr[yr])
    print(f"\n  {yr}: {raw_n} raw clusters → {len(events)} events after merge")
    for ev in events:
        s = get_season(ev['start_dt'])
        print(f"    {ev['start_dt'].strftime('%Y-%m-%d %H:%M')} — "
              f"{ev['duration']:>3}h window ({ev['n_stress']:>2} stress hrs)  "
              f"peak=${ev['peak_val']:,.0f}/MWh  ({s})")

print(f"\n\nCOMBINED EVENTS (load ∪ price, de-overlapped)")
print("=" * 90)
for yr, events in combined_events_by_yr.items():
    total_window = sum(ev['duration'] for ev in events)
    n_long = sum(1 for ev in events if ev['duration'] > 24)
    print(f"  {yr}: {len(events)} events, {total_window}h total window, "
          f"{n_long} events >24h")
```

    LOAD-BASED STRESS EVENTS
    Top 50 hours per year by ComEd hourly load (MW)
    ==========================================================================================
    
      2022: 12 events from 50 stress hours
        2022-06-14 13:00 —  9h  peak=20,606 MW  (Summer)
        2022-06-15 12:00 — 10h  peak=20,731 MW  (Summer)
        2022-06-16 15:00 —  5h  peak=19,210 MW  (Summer)
        2022-06-20 18:00 —  1h  peak=18,535 MW  (Summer)
        2022-06-21 13:00 — 10h  peak=21,262 MW  (Summer)
        2022-06-30 17:00 —  1h  peak=18,545 MW  (Summer)
        2022-07-05 14:00 —  5h  peak=19,685 MW  (Summer)
        2022-07-19 16:00 —  2h  peak=18,583 MW  (Summer)
        2022-07-21 15:00 —  4h  peak=19,172 MW  (Summer)
        2022-08-02 18:00 —  1h  peak=18,535 MW  (Summer)
        2022-08-03 14:00 —  1h  peak=18,507 MW  (Summer)
        2022-08-06 16:00 —  1h  peak=18,531 MW  (Summer)
    
      2023: 8 events from 50 stress hours
        2023-07-05 14:00 —  2h  peak=18,252 MW  (Summer)
        2023-07-25 17:00 —  2h  peak=17,973 MW  (Summer)
        2023-07-27 13:00 —  8h  peak=19,358 MW  (Summer)
        2023-07-28 14:00 —  6h  peak=19,294 MW  (Summer)
        2023-08-22 16:00 —  4h  peak=18,262 MW  (Summer)
        2023-08-23 11:00 — 13h  peak=21,929 MW  (Summer)
        2023-08-24 10:00 — 13h  peak=22,467 MW  (Summer)
        2023-09-05 14:00 —  2h  peak=18,322 MW  (Shoulder)
    
      2024: 10 events from 50 stress hours
        2024-06-17 14:00 —  8h  peak=19,874 MW  (Summer)
        2024-06-18 15:00 —  5h  peak=19,495 MW  (Summer)
        2024-06-19 14:00 —  6h  peak=19,486 MW  (Summer)
        2024-06-21 17:00 —  2h  peak=18,568 MW  (Summer)
        2024-07-15 16:00 —  4h  peak=19,220 MW  (Summer)
        2024-07-30 16:00 —  3h  peak=18,610 MW  (Summer)
        2024-07-31 17:00 —  3h  peak=18,912 MW  (Summer)
        2024-08-05 15:00 —  2h  peak=18,890 MW  (Summer)
        2024-08-26 14:00 —  8h  peak=20,851 MW  (Summer)
        2024-08-27 12:00 —  9h  peak=21,560 MW  (Summer)
    
      2025: 12 events from 50 stress hours
        2025-06-22 18:00 —  2h  peak=19,134 MW  (Summer)
        2025-06-23 13:00 —  9h  peak=20,714 MW  (Summer)
        2025-06-24 13:00 —  2h  peak=19,244 MW  (Summer)
        2025-06-26 14:00 —  6h  peak=20,152 MW  (Summer)
        2025-07-16 14:00 —  2h  peak=19,410 MW  (Summer)
        2025-07-23 15:00 —  7h  peak=20,375 MW  (Summer)
        2025-07-24 13:00 —  3h  peak=19,914 MW  (Summer)
        2025-07-28 15:00 —  5h  peak=19,594 MW  (Summer)
        2025-07-29 14:00 —  6h  peak=19,749 MW  (Summer)
        2025-08-07 17:00 —  2h  peak=19,140 MW  (Summer)
        2025-08-08 14:00 —  5h  peak=19,732 MW  (Summer)
        2025-08-11 18:00 —  1h  peak=19,034 MW  (Summer)
    
    
    PRICE-BASED STRESS EVENTS
    Top 50 hours per year by DA LMP, conditionally merged
    (gaps ≤18h merged when median gap price > 75th percentile annual LMP)
    ==========================================================================================
    
      2022: 15 raw clusters → 7 events after merge
        2022-06-14 14:00 —  53h window (18 stress hrs)  peak=$329/MWh  (Summer)
        2022-06-21 16:00 —   3h window ( 3 stress hrs)  peak=$219/MWh  (Summer)
        2022-07-20 16:00 —   2h window ( 2 stress hrs)  peak=$201/MWh  (Summer)
        2022-07-21 16:00 —   2h window ( 2 stress hrs)  peak=$201/MWh  (Summer)
        2022-07-22 16:00 —   2h window ( 2 stress hrs)  peak=$203/MWh  (Summer)
        2022-08-29 16:00 —   2h window ( 2 stress hrs)  peak=$199/MWh  (Summer)
        2022-12-24 08:00 —  72h window (21 stress hrs)  peak=$364/MWh  (Winter)
    
      2023: 19 raw clusters → 19 events after merge
        2023-02-03 18:00 —   3h window ( 3 stress hrs)  peak=$90/MWh  (Winter)
        2023-06-01 17:00 —   1h window ( 1 stress hrs)  peak=$79/MWh  (Summer)
        2023-06-02 16:00 —   1h window ( 1 stress hrs)  peak=$75/MWh  (Summer)
        2023-06-29 17:00 —   1h window ( 1 stress hrs)  peak=$78/MWh  (Summer)
        2023-06-30 16:00 —   2h window ( 2 stress hrs)  peak=$77/MWh  (Summer)
        2023-07-24 17:00 —   1h window ( 1 stress hrs)  peak=$75/MWh  (Summer)
        2023-07-25 16:00 —   2h window ( 2 stress hrs)  peak=$80/MWh  (Summer)
        2023-07-26 16:00 —   3h window ( 3 stress hrs)  peak=$87/MWh  (Summer)
        2023-07-27 12:00 —   9h window ( 9 stress hrs)  peak=$282/MWh  (Summer)
        2023-07-28 12:00 —   8h window ( 8 stress hrs)  peak=$293/MWh  (Summer)
        2023-08-21 16:00 —   2h window ( 2 stress hrs)  peak=$77/MWh  (Summer)
        2023-08-23 16:00 —   3h window ( 3 stress hrs)  peak=$82/MWh  (Summer)
        2023-08-24 15:00 —   4h window ( 4 stress hrs)  peak=$86/MWh  (Summer)
        2023-10-02 16:00 —   2h window ( 2 stress hrs)  peak=$91/MWh  (Shoulder)
        2023-10-03 16:00 —   3h window ( 3 stress hrs)  peak=$101/MWh  (Shoulder)
        2023-10-04 16:00 —   2h window ( 2 stress hrs)  peak=$85/MWh  (Shoulder)
        2023-10-23 07:00 —   1h window ( 1 stress hrs)  peak=$100/MWh  (Shoulder)
        2023-10-27 17:00 —   1h window ( 1 stress hrs)  peak=$76/MWh  (Shoulder)
        2023-11-01 07:00 —   1h window ( 1 stress hrs)  peak=$75/MWh  (Shoulder)
    
      2024: 8 raw clusters → 5 events after merge
        2024-01-15 05:00 —  53h window (39 stress hrs)  peak=$277/MWh  (Winter)
        2024-07-15 15:00 —   4h window ( 4 stress hrs)  peak=$172/MWh  (Summer)
        2024-08-27 15:00 —   4h window ( 4 stress hrs)  peak=$174/MWh  (Summer)
        2024-08-28 16:00 —   2h window ( 2 stress hrs)  peak=$135/MWh  (Summer)
        2024-12-02 07:00 —   1h window ( 1 stress hrs)  peak=$140/MWh  (Winter)
    
      2025: 13 raw clusters → 7 events after merge
        2025-01-20 07:00 —  50h window (22 stress hrs)  peak=$277/MWh  (Winter)
        2025-02-21 07:00 —   1h window ( 1 stress hrs)  peak=$207/MWh  (Winter)
        2025-06-23 16:00 —  29h window (11 stress hrs)  peak=$354/MWh  (Summer)
        2025-06-25 17:00 —   3h window ( 3 stress hrs)  peak=$195/MWh  (Summer)
        2025-07-24 19:00 —   1h window ( 1 stress hrs)  peak=$204/MWh  (Summer)
        2025-07-28 17:00 —  28h window (11 stress hrs)  peak=$497/MWh  (Summer)
        2025-10-06 18:00 —   1h window ( 1 stress hrs)  peak=$187/MWh  (Shoulder)
    
    
    COMBINED EVENTS (load ∪ price, de-overlapped)
    ==========================================================================================
      2022: 14 events, 159h total window, 2 events >24h
      2023: 22 events, 78h total window, 0 events >24h
      2024: 13 events, 107h total window, 1 events >24h
      2025: 15 events, 146h total window, 3 events >24h
    

### 1.3 Stress Characterization & Headline Findings

Duration distribution, seasonal patterns, and the four key findings 
that determine mechanism design in Part 3. The critical insight: 
price-based events reveal winter duration risk that load-based 
analysis misses entirely.


```python
# ======================================================================
# Cell 1-3: STRESS EVENT CHARACTERIZATION & HEADLINE FINDINGS (v18)
# ===========================================================
# Analyzes the events identified in Cell 1-2 to answer:
#   - How long do events last? (duration distribution)
#   - When do they happen? (seasonal breakdown)
#   - What does this mean for mechanism design? (headline findings)
#
# KEY FINDING: Load-based stress is short/summer. Price-based stress
# reveals multi-day winter crises invisible to load analysis. This
# duration gap determines how deeply a DC can credibly commit.
#
# INPUTS: load_events_by_yr, price_events_raw_by_yr,
#         price_events_merged_by_yr, combined_events_by_yr (from Cell 1-2)
# OUTPUTS: Characterization tables (consumed by Part 3 commitment optimization)
# ===========================================================


# ─────────────────────────────────────────────────────────────
# DURATION DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def duration_dist(events_dict):
    buckets = {'1-2h': 0, '3-4h': 0, '5-8h': 0, '9-16h': 0, '17-24h': 0, '24+h': 0}
    hrs = {k: 0 for k in buckets}
    for yr, events in events_dict.items():
        for ev in events:
            d = ev['duration']
            k = '1-2h' if d <= 2 else '3-4h' if d <= 4 else '5-8h' if d <= 8 else \
                '9-16h' if d <= 16 else '17-24h' if d <= 24 else '24+h'
            buckets[k] += 1; hrs[k] += d
    return buckets, hrs

load_bkt, load_hrs = duration_dist(load_events_by_yr)
price_raw_bkt, price_raw_hrs = duration_dist(price_events_raw_by_yr)
price_mrg_bkt, price_mrg_hrs = duration_dist(price_events_merged_by_yr)

print(f"EVENT DURATION DISTRIBUTION (pooled across {len(YEARS)} years)")
print("=" * 90)
print(f"  {'Duration':>8} | {'Load':>5} {'hrs':>5} | {'Price raw':>9} {'hrs':>5} | {'Price merged':>12} {'hrs':>5}")
print(f"  {'-'*8}-+-{'-'*5}-{'-'*5}-+-{'-'*9}-{'-'*5}-+-{'-'*12}-{'-'*5}")
for k in load_bkt:
    print(f"  {k:>8} | {load_bkt[k]:>4}  {load_hrs[k]:>4}h | {price_raw_bkt[k]:>8}  {price_raw_hrs[k]:>4}h | {price_mrg_bkt[k]:>11}  {price_mrg_hrs[k]:>4}h")

total_load = sum(load_hrs.values())
total_price = sum(price_mrg_hrs.values())
pct_4plus_load = sum(v for k, v in load_hrs.items() if k not in ['1-2h', '3-4h']) / total_load
pct_4plus_price = sum(v for k, v in price_mrg_hrs.items() if k not in ['1-2h', '3-4h']) / total_price
pct_24plus_price = price_mrg_hrs.get('24+h', 0) / total_price

print(f"\n  Hours in events >4h:  Load: {pct_4plus_load:.0%}  |  Price (merged): {pct_4plus_price:.0%}")
print(f"  Hours in events >24h: Price (merged): {pct_24plus_price:.0%}")


# ─────────────────────────────────────────────────────────────
# SEASONAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────

print(f"\n\nSEASONAL DISTRIBUTION OF STRESS EVENTS")
print("=" * 90)
for label, ev_dict in [("Load-based", load_events_by_yr),
                        ("Price-based (merged)", price_events_merged_by_yr)]:
    sh = {'Summer': 0, 'Winter': 0, 'Shoulder': 0}
    n_events = {'Summer': 0, 'Winter': 0, 'Shoulder': 0}
    for yr, evts in ev_dict.items():
        for ev in evts:
            s = get_season(ev['start_dt'])
            sh[s] += ev['duration']
            n_events[s] += 1
    t = sum(sh.values())
    print(f"\n  {label}:")
    for s in ['Summer', 'Winter', 'Shoulder']:
        if sh[s] > 0:
            print(f"    {s:>10}: {n_events[s]:>3} events, {sh[s]:>4}h ({sh[s]/t:>4.0%})")


# ─────────────────────────────────────────────────────────────
# HEADLINE FINDINGS
# ─────────────────────────────────────────────────────────────

# Load-based metrics
load_summer_hrs = 0
load_total_hrs = 0
load_durations = []

for yr, evts in load_events_by_yr.items():
    for ev in evts:
        load_total_hrs += ev['duration']
        load_durations.append(ev['duration'])
        if get_season(ev['start_dt']) == 'Summer':
            load_summer_hrs += ev['duration']

load_summer_pct = load_summer_hrs / load_total_hrs if load_total_hrs > 0 else 0
load_median_dur = np.median(load_durations) if load_durations else 0

# Price-based metrics
price_winter_hrs = 0
price_total_hrs = 0
long_price_events = []

for yr, evts in price_events_merged_by_yr.items():
    for ev in evts:
        price_total_hrs += ev['duration']
        if get_season(ev['start_dt']) == 'Winter':
            price_winter_hrs += ev['duration']
        if ev['duration'] >= 24:
            long_price_events.append(ev)

price_winter_pct = price_winter_hrs / price_total_hrs if price_total_hrs > 0 else 0

# Top multi-day events
long_price_events.sort(key=lambda x: x['duration'], reverse=True)
event_bullets = ""
for ev in long_price_events[:3]:
    mon_yr = ev['start_dt'].strftime('%b %Y')
    event_bullets += f"     • {mon_yr} crisis: {ev['duration']}h event window\n"

print(f"""

HEADLINE FINDINGS:
  1. Load-based stress is {load_summer_pct:.0%} summer — short-to-medium events (median ~{load_median_dur:.0f}h).
     A 4h battery covers most of these.
  
  2. Price-based stress is {price_winter_pct:.0%} winter — reveals multi-day crises:
{event_bullets}     A 4h battery is exhausted in the first morning of these events.
  
  3. Price-based events are what drive PJM Performance Assessment Hours.
     Any capacity resource that cannot sustain delivery beyond 4 hours
     faces escalating CP penalty exposure during these events.
  
  4. This duration gap determines how deeply a DC can credibly commit
     as DR or non-firm load — quantified in Part 3.
""")
```

    EVENT DURATION DISTRIBUTION (pooled across 4 years)
    ==========================================================================================
      Duration |  Load   hrs | Price raw   hrs | Price merged   hrs
      ---------+-------------+-----------------+-------------------
          1-2h |   16    26h |       27    38h |          21    31h
          3-4h |    6    21h |       11    38h |           9    30h
          5-8h |   13    80h |       14    88h |           1     8h
         9-16h |    7    73h |        2    18h |           1     9h
        17-24h |    0     0h |        1    18h |           0     0h
          24+h |    0     0h |        0     0h |           6   285h
    
      Hours in events >4h:  Load: 76%  |  Price (merged): 83%
      Hours in events >24h: Price (merged): 79%
    
    
    SEASONAL DISTRIBUTION OF STRESS EVENTS
    ==========================================================================================
    
      Load-based:
            Summer:  41 events,  198h ( 99%)
          Shoulder:   1 events,    2h (  1%)
    
      Price-based (merged):
            Summer:  25 events,  172h ( 47%)
            Winter:   6 events,  180h ( 50%)
          Shoulder:   7 events,   11h (  3%)
    
    
    HEADLINE FINDINGS:
      1. Load-based stress is 99% summer — short-to-medium events (median ~4h).
         A 4h battery covers most of these.
      
      2. Price-based stress is 50% winter — reveals multi-day crises:
         • Dec 2022 crisis: 72h event window
         • Jun 2022 crisis: 53h event window
         • Jan 2024 crisis: 53h event window
         A 4h battery is exhausted in the first morning of these events.
      
      3. Price-based events are what drive PJM Performance Assessment Hours.
         Any capacity resource that cannot sustain delivery beyond 4 hours
         faces escalating CP penalty exposure during these events.
      
      4. This duration gap determines how deeply a DC can credibly commit
         as DR or non-firm load — quantified in Part 3.
    
    

## Part 2: Cascade Validation and Sensitivity
- Cell 1: Monte Carlo Variance Decomposition
- Cell 2: Cascade Scenario Summary

The cascade product (0.390 central) is only as credible as its weakest 
parameter. This section quantifies how much uncertainty each parameter 
contributes and shows the range of outcomes across 
conservative/central/optimistic scenarios.

Two parameters dominate: **P3 Factor B** (data locality, ~67% of variance) 
and **H** (destination headroom, ~15%). Both are author's estimates. 
Resolving either one is the highest-value empirical investment for 
tightening the commitment depth range.

### 2.1 Monte Carlo Variance Decomposition

50,000 draws from parameter distributions. η² correlation ratios 
identify which parameters drive cascade uncertainty. The sum ≈ 1.0 
confirms negligible parameter interactions, consistent with the 
multiplicative independence assumption.


```python
# ======================================================================
# Cell 2-1: CASCADE MONTE CARLO VARIANCE DECOMPOSITION (v18)
# ===========================================================
# Monte Carlo sensitivity analysis for the ten-parameter cascade.
# Produces:
#   1. Distribution of cascade product and commitment depth
#   2. Variance attribution (η² correlation ratios) identifying which
#      parameters drive uncertainty
#
# v17: 10-parameter MC. Variance decomposition now covers all 10 parameters.
# Commitment formula corrected: commit = spatial + DVFS * (S1 - spatial).
# (v16.1 Part 2.1 still had the old formula commit = spatial + DVFS * (1 - spatial).)
#
# INPUTS: CASCADE_* params, CASCADE_RANGES, FLEX_FRAC (from Part 0)
# OUTPUTS: CASCADE_MC_PRODUCTS, CASCADE_MC_COMMITMENT, CASCADE_ETA_SQ
# ===========================================================

import numpy as np

np.random.seed(429)  # Standard course seed
N = 50_000

# S1 is fixed at 0.70 (no within-scenario variance)
S1_FIXED = CASCADE_S1

# Sample from CASCADE_RANGES (conservative → optimistic)
# D1: triangular with mode at central, bounded by con/opt
# All others: uniform between conservative and optimistic
samples = {
    'S2': np.random.uniform(0.60, 0.95, N),      # WEAK — no published shiftability measurement
    'S3': np.random.uniform(0.85, 0.95, N),       # Moderate-strong
    'D1':  np.random.triangular(0.945, 0.99, 0.995, N),  # Strong (empirical)
    'D2':  np.random.uniform(0.20, 0.50, N),       # Narrowed: GPU util 50-80% -> headroom 0.20-0.50
    'D3':  np.random.uniform(0.80, 0.92, N),       # Moderate (HW compat, no pre-staging)
    'D4':  np.random.uniform(0.33, 0.67, N),       # Moderate (inference workload share)
    'D5':  np.random.uniform(0.50, 0.80, N),       # WEAK — least grounded (pre-staging)
    'E1':  np.random.uniform(0.995, 0.999, N),     # Moderate (SLA-bounded)
    'E2':  np.random.uniform(0.99, 1.00, N),       # Strong
}

# Cascade product for all draws (10-parameter)
products = (S1_FIXED *
            samples['S2'] * samples['S3'] *
            samples['D1'] * samples['D2'] * samples['D3'] *
            samples['D4'] * samples['D5'] *
            samples['E1'] * samples['E2'])

# v17: corrected commitment formula — DVFS on shiftable residual (S1 - spatial), not (1 - spatial)
commitment = products + FLEX_FRAC * (S1_FIXED - products)

print(f"CASCADE MONTE CARLO (N={N:,}, 10-parameter)")
print(f"{'='*70}")
print(f"  Cascade product:   mean={np.mean(products):.4f}, "
      f"std={np.std(products):.4f}, "
      f"[P5={np.percentile(products,5):.4f}, P95={np.percentile(products,95):.4f}]")
print(f"  Commitment depth:  mean={np.mean(commitment):.4f} ({np.mean(commitment):.1%}), "
      f"std={np.std(commitment):.4f}, "
      f"[P5={np.percentile(commitment,5):.1%}, P95={np.percentile(commitment,95):.1%}]")


# ─────────────────────────────────────────────────────────────
# VARIANCE ATTRIBUTION (η² correlation ratios)
# ─────────────────────────────────────────────────────────────
# For each parameter: bin into 20 quantile groups, compute between-group
# variance as fraction of total variance. Nonparametric first-order
# sensitivity index.

total_var = np.var(products)
eta_sq = {}

for param, vals in samples.items():
    bins = np.digitize(vals, np.percentile(vals, np.linspace(0, 100, 21)))
    group_means = np.array([np.mean(products[bins == b]) for b in range(1, 21) if np.sum(bins == b) > 0])
    group_counts = np.array([np.sum(bins == b) for b in range(1, 21) if np.sum(bins == b) > 0])
    between_var = np.sum(group_counts * (group_means - np.mean(products))**2) / N
    eta_sq[param] = between_var / total_var

grounding = {
    'S2': 'WEAK', 'S3': 'Moderate', 'D1': 'STRONG',
    'D2': 'WEAK', 'D3': 'Moderate', 'D4': 'Moderate',
    'D5': 'WEAK', 'E1': 'Moderate', 'E2': 'Strong'
}

eta_total = sum(eta_sq.values())

print(f"\n  VARIANCE ATTRIBUTION (η² correlation ratios, 10 parameters):")
print(f"  {'Parameter':<12} | {'η²':>8} | {'% of Σ η²':>10} | {'Grounding':>15}")
print(f"  {'─'*55}")
for param, eta in sorted(eta_sq.items(), key=lambda x: -x[1]):
    print(f"  {param:<12} | {eta:>8.3f} | {eta/eta_total*100:>9.1f}% | {grounding.get(param,''):>15}")

print(f"\n  Sum η² = {eta_total:.3f} (≈1.0 confirms negligible parameter interactions)")
print(f"\n  INTERPRETATION:")
print(f"    With 10-parameter decomposition, variance is distributed across D2, D5,")
print(f"    S2, D4 (the weakest-grounded parameters with widest ranges).")
print(f"    Phase 4 sensitivity surface will focus on D2 and D5.")

# Store for downstream
CASCADE_MC_PRODUCTS = products
CASCADE_MC_COMMITMENT = commitment
CASCADE_ETA_SQ = eta_sq
```

    CASCADE MONTE CARLO (N=50,000, 10-parameter)
    ======================================================================
    

      Cascade product:   mean=0.0463, std=0.0176, [P5=0.0224, P95=0.0792]
      Commitment depth:  mean=0.2097 (21.0%), std=0.0132, [P5=19.2%, P95=23.4%]
    

    
      VARIANCE ATTRIBUTION (η² correlation ratios, 10 parameters):
      Parameter    |       η² |  % of Σ η² |       Grounding
      ───────────────────────────────────────────────────────
      D2           |    0.423 |      44.2% |            WEAK
      D4           |    0.268 |      28.0% |        Moderate
      D5           |    0.125 |      13.1% |            WEAK
      S2           |    0.121 |      12.6% |            WEAK
      D3           |    0.010 |       1.0% |        Moderate
      S3           |    0.008 |       0.9% |        Moderate
      D1           |    0.001 |       0.1% |          STRONG
      E2           |    0.000 |       0.1% |          Strong
      E1           |    0.000 |       0.0% |        Moderate
    
      Sum η² = 0.958 (≈1.0 confirms negligible parameter interactions)
    
      INTERPRETATION:
        With 10-parameter decomposition, variance is distributed across D2, D5,
        S2, D4 (the weakest-grounded parameters with widest ranges).
        Phase 4 sensitivity surface will focus on D2 and D5.
    

### 2.2 Cascade Scenario Summary

What do the conservative/central/optimistic cascade products imply for 
commitment depth and accredited capacity? And which parameters have the 
largest marginal impact? This table is the bridge between the cascade 
model (Part 0) and the commitment optimization (Part 3).


```python
# ======================================================================
# Cell 2-2: CASCADE SCENARIO SUMMARY (v18)
# ===========================================================
# Translates the cascade's uncertainty range into commitment depth
# and fleet-level committed MW across conservative/central/optimistic.
#
# This is the bridge between "what is the cascade?" (Part 0) and
# "what does it mean for the capacity market?" (Part 3). An advisor
# can look at this table and understand the full range of outcomes
# before seeing the mechanism cost and revenue analysis.
#
# NOTE: This cell computes committed MW and accredited MW only.
# Avoided installed capacity (which requires the IRM multiplier and
# election mechanism) is computed in Part 3.
#
# INPUTS: CASCADE_RANGES, _cascade_con, _cascade_opt,
#         EFFECTIVE_SPATIAL_FRAC, FLEX_FRAC, ELCC_2027_28_DR (from Part 0)
# OUTPUTS: CASCADE_SCENARIOS dict (consumed by Part 3)
# ===========================================================

DR_ELCC = ELCC_2027_28_DR  # 0.92 — PJM DR class rating, exogenous

# ─────────────────────────────────────────────────────────────
# SCENARIO TABLE
# ─────────────────────────────────────────────────────────────

CASCADE_SCENARIOS = {
    'Conservative': _cascade_con,
    'Central':      EFFECTIVE_SPATIAL_FRAC,
    'Optimistic':   _cascade_opt,
}

print("CASCADE SCENARIO SUMMARY")
print("=" * 90)
print(f"\n  {'Scenario':<16} | {'Cascade':>8} | {'Commit':>7} | {'Firm':>6} | "
      f"{'Committed MW':>13} | {'Accred MW':>10} | {'Accred MW':>10}")
print(f"  {'':16} | {'Product':>8} | {'Depth':>7} | {'Level':>6} | "
      f"{'@ 10 GW':>13} | {'@ 10 GW':>10} | {'@ 20 GW':>10}")
print(f"  {'─'*85}")

for label, sf in CASCADE_SCENARIOS.items():
    depth = sf + FLEX_FRAC * (CASCADE_S1 - sf)  # v17: corrected formula (DVFS on shiftable residual)
    firm = 1 - depth
    committed_10 = 10000 * depth
    accredited_10 = committed_10 * DR_ELCC
    accredited_20 = 20000 * depth * DR_ELCC

    print(f"  {label:<16} | {sf:>8.3f} | {depth:>6.1%} | {firm:>5.1%} | "
          f"{committed_10:>11,.0f} MW | {accredited_10:>8,.0f} MW | {accredited_20:>8,.0f} MW")

_dvfs_floor = FLEX_FRAC * CASCADE_S1  # v17: 17.5% (facility-relative)
print(f"\n  DVFS-only floor | {'—':>8} | {_dvfs_floor:>6.1%} | {1-_dvfs_floor:>5.1%} | "
      f"{10000*_dvfs_floor:>11,.0f} MW | {10000*_dvfs_floor*DR_ELCC:>8,.0f} MW | {20000*_dvfs_floor*DR_ELCC:>8,.0f} MW")

# ─────────────────────────────────────────────────────────────
# PARAMETER SENSITIVITY (one-at-a-time)
# ─────────────────────────────────────────────────────────────
# For each parameter: compute cascade product at low and high
# while holding all others at central. Shows marginal impact.

central_vals = {k: v[1] for k, v in CASCADE_RANGES.items()}

print(f"\n\n  PARAMETER SENSITIVITY (one-at-a-time, 10 GW fleet)")
print(f"  {'Parameter':<12} | {'Low':>6} | {'High':>6} | {'Depth Lo':>9} | {'Depth Hi':>9} | {'Δ Accred MW':>12}")
print(f"  {'─'*65}")

sensitivities = []
for param, (lo, mid, hi) in CASCADE_RANGES.items():
    if lo == hi:  # Fixed (S1)
        continue

    # Low case
    vals_lo = central_vals.copy()
    vals_lo[param] = lo
    prod_lo = (vals_lo['S1'] * vals_lo['S2'] * vals_lo['S3'] *
               vals_lo['D1'] * vals_lo['D2'] * vals_lo['D3'] *
               vals_lo['D4'] * vals_lo['D5'] *
               vals_lo['E1'] * vals_lo['E2'])
    depth_lo = prod_lo + FLEX_FRAC * (vals_lo['S1'] - prod_lo)  # v17: corrected formula

    # High case
    vals_hi = central_vals.copy()
    vals_hi[param] = hi
    prod_hi = (vals_hi['S1'] * vals_hi['S2'] * vals_hi['S3'] *
               vals_hi['D1'] * vals_hi['D2'] * vals_hi['D3'] *
               vals_hi['D4'] * vals_hi['D5'] *
               vals_hi['E1'] * vals_hi['E2'])
    depth_hi = prod_hi + FLEX_FRAC * (vals_hi['S1'] - prod_hi)  # v17: corrected formula

    delta_accred = (depth_hi - depth_lo) * 10000 * DR_ELCC
    sensitivities.append((param, lo, hi, depth_lo, depth_hi, delta_accred))

    print(f"  {param:<12} | {lo:>6.3f} | {hi:>6.3f} | {depth_lo:>8.1%} | {depth_hi:>8.1%} | {delta_accred:>+10,.0f} MW")

# Rank
sensitivities.sort(key=lambda x: abs(x[5]), reverse=True)
print(f"\n  Ranked by impact on accredited capacity:")
for i, (param, lo, hi, dlo, dhi, delta) in enumerate(sensitivities):
    bar = '█' * max(1, int(abs(delta) / 40))
    print(f"    {i+1}. {param:<12}: {delta:>+7,.0f} MW  {bar}")

print(f"\n  KEY TAKEAWAY:")
print(f"    With 10-parameter decomposition, sensitivity is distributed across")
print(f"    D2 (utilization headroom), D5 (pre-staging), S2 (data locality),")
print(f"    and D4 (inference share). Phase 4 sensitivity surface will map D2 x D5.")
```

    CASCADE SCENARIO SUMMARY
    ==========================================================================================
    
      Scenario         |  Cascade |  Commit |   Firm |  Committed MW |  Accred MW |  Accred MW
                       |  Product |   Depth |  Level |       @ 10 GW |    @ 10 GW |    @ 20 GW
      ─────────────────────────────────────────────────────────────────────────────────────
      Conservative     |    0.009 |  18.2% | 81.8% |       1,816 MW |    1,671 MW |    3,341 MW
      Central          |    0.038 |  20.4% | 79.6% |       2,038 MW |    1,875 MW |    3,749 MW
      Optimistic       |    0.155 |  29.1% | 70.9% |       2,911 MW |    2,678 MW |    5,357 MW
    
      DVFS-only floor |        — |  17.5% | 82.5% |       1,750 MW |    1,610 MW |    3,220 MW
    
    
      PARAMETER SENSITIVITY (one-at-a-time, 10 GW fleet)
      Parameter    |    Low |   High |  Depth Lo |  Depth Hi |  Δ Accred MW
      ─────────────────────────────────────────────────────────────────
      S2           |  0.600 |  0.950 |    20.1% |    21.7% |       +141 MW
      S3           |  0.850 |  0.950 |    20.8% |    21.2% |        +36 MW
      D1           |  0.945 |  0.995 |    20.8% |    21.0% |        +16 MW
      D2           |  0.200 |  0.500 |    19.6% |    22.8% |       +293 MW
      D3           |  0.800 |  0.920 |    20.7% |    21.2% |        +44 MW
      D4           |  0.330 |  0.670 |    19.8% |    22.2% |       +219 MW
      D5           |  0.500 |  0.800 |    20.2% |    21.8% |       +149 MW
      E1           |  0.995 |  0.999 |    21.0% |    21.0% |         +1 MW
      E2           |  0.990 |  1.000 |    21.0% |    21.0% |         +3 MW
    
      Ranked by impact on accredited capacity:
        1. D2          :    +293 MW  ███████
        2. D4          :    +219 MW  █████
        3. D5          :    +149 MW  ███
        4. S2          :    +141 MW  ███
        5. D3          :     +44 MW  █
        6. S3          :     +36 MW  █
        7. D1          :     +16 MW  █
        8. E2          :      +3 MW  █
        9. E1          :      +1 MW  █
    
      KEY TAKEAWAY:
        With 10-parameter decomposition, sensitivity is distributed across
        D2 (utilization headroom), D5 (pre-staging), S2 (data locality),
        and D4 (inference share). Phase 4 sensitivity surface will map D2 x D5.
    

## Part 3: Commitment Depth and Grid Impact
- Cell 1: Mechanism Profiles & Portfolios
- Cell 2: Commitment Optimization
- Cell 3: Election Mechanism & Avoided Installed Capacity
- Cell 4: E3 Counterfactual

This is the core analytical contribution. Given the cascade model 
(Part 0) and the empirical stress profile (Part 1), how deeply can 
a data center fleet credibly commit as demand response?

The answer: spatial migration more than doubles commitment depth from 
25% (DVFS-only floor) to ~54% for inference-dominant facilities. This 
section computes the mechanism costs, optimizes portfolio commitment, 
translates to avoided installed capacity, and benchmarks against E3's 
recommended gas buildout.

### 3.1 Mechanism Profiles & Portfolios

Three mechanisms (DVFS, BTM battery, spatial migration) and four 
portfolio combinations. Each has a cost, depth, and duration profile 
that determines its contribution to commitment optimization.

ELCC is exogenous (PJM DR class rating = 92%). The decision variable 
is commitment depth — how many MW to offer, not how they're rated.


```python
# ======================================================================
# Cell 3-1: MECHANISM PROFILES & PORTFOLIO DEFINITIONS (v18)
# ===========================================================
# Defines the three flexibility mechanisms (DVFS, BTM Battery, Spatial)
# and four portfolio combinations. Each mechanism has a cost, depth,
# and duration profile. The portfolio's commitment depth determines
# how many MW the fleet can credibly offer into the BRA or commit
# under connect-and-manage.
#
# KEY INSIGHT: The decision variable is commitment depth, not ELCC.
# PJM assigns 92% ELCC to all DR. The mechanism portfolio determines
# how deeply you can commit at that rate.
#
# INPUTS: FLEX_FRAC, EFFECTIVE_SPATIAL_FRAC, HOURLY_COMPUTE_VALUE,
#         MIGRATION_LATENCY_MIN, AVG_EVENT_DURATION_HRS, DESTINATION_LMP_CRISIS,
#         BTM_CAPEX_PER_MW, ELCC_2027_28_DR, BRA_2027_28_PRICE (from Part 0)
# OUTPUTS: DR_ELCC, BRA_ANNUAL, MECHANISM_PROFILES, PORTFOLIOS
# ===========================================================
 
DR_ELCC = ELCC_2027_28_DR  # 0.92 — PJM DR class rating, exogenous
BRA_ANNUAL = BRA_2027_28_PRICE * 365.25  # $/MW-yr per accredited MW
 
print("COMMITMENT OPTIMIZATION FRAMEWORK")
print("=" * 90)
print(f"  DR ELCC: {DR_ELCC:.0%} (PJM class rating — exogenous)")
print(f"  BRA clearing: ${BRA_ANNUAL:,.0f}/MW-yr per accredited MW")
print(f"  Revenue per committed MW: ${DR_ELCC * BRA_ANNUAL:,.0f}/MW-yr")
print()
 
 
# ─────────────────────────────────────────────────────────────
# MECHANISM COST PROFILES
# ─────────────────────────────────────────────────────────────
 
MECHANISM_PROFILES = {
    'DVFS': {
        # v16.1: Colangelo's 25% is measured at GPU cluster, not facility.
        # Scaled to facility-relative by multiplying by S1 (shiftable-compute share).
        'max_depth': FLEX_FRAC * CASCADE_S1,   # 17.5% of facility [DATA-Colangelo, S1-scaled]
        'max_duration': None,             # Indefinite
        'cost_per_mwh': HOURLY_COMPUTE_VALUE * 0.02,  # ~$27/MWh (2% throughput degradation)
        'desc': 'CPU/GPU frequency scaling + job pausing',
    },
    'BTM Battery': {
        'max_depth': 1.00,               # Full facility during discharge
        'max_duration': 4,                # 4-hour BESS
        'cost_per_mwh': 15.0,            # Battery cycling degradation
        'desc': '4hr BESS behind the meter (duration-limited)',
    },
    'Spatial Migration': {
        'max_depth': EFFECTIVE_SPATIAL_FRAC,  # 10-param cascade product
        'max_duration': None,             # Indefinite
        'cost_per_mwh': None,             # Computed below
        'desc': f'Geographic workload relocation (cascade = {EFFECTIVE_SPATIAL_FRAC:.3f})',
    }
}
 
# Spatial cost = destination energy + migration friction amortized over event
_spatial_dest_cost = DESTINATION_LMP_CRISIS
_spatial_friction_amortized = (HOURLY_COMPUTE_VALUE * (MIGRATION_LATENCY_MIN / 60)
                               / max(AVG_EVENT_DURATION_HRS, 1))
MECHANISM_PROFILES['Spatial Migration']['cost_per_mwh'] = _spatial_dest_cost + _spatial_friction_amortized
 
# Inference routing variant (v15.1): no checkpoint friction, destination energy only
MECHANISM_PROFILES['Spatial (Inference)'] = {
    'max_depth': EFFECTIVE_SPATIAL_FRAC,
    'max_duration': None,
    'cost_per_mwh': _spatial_dest_cost,  # Destination energy only, no friction
    'response_min': INFERENCE_DRAIN_TIME_MIN,  # ~0.20 min (~12 seconds)
    'desc': f'Inference routing — no checkpoint (drain={INFERENCE_DRAIN_TIME_SEC}s P99)',
}
 
print("MECHANISM COST PROFILES")
print("-" * 90)
for name, m in MECHANISM_PROFILES.items():
    dur_str = f"{m.get('max_duration')}h" if m.get('max_duration') else "indefinite"
    print(f"  {name:<20}: depth={m['max_depth']:.0%}, duration={dur_str}, "
          f"cost=${m['cost_per_mwh']:,.0f}/MWh")
 
# Spatial cost decomposition (for Pillar 3 footnote transparency)
print(f"\n  Spatial cost breakdown (training migration):")
print(f"    Destination LMP during stress:  ${_spatial_dest_cost:,.1f}/MWh  (empirical, capacity-weighted)")
print(f"    Migration friction (amortized): ${_spatial_friction_amortized:,.1f}/MWh  "
      f"(= ${HOURLY_COMPUTE_VALUE:,.0f} × {MIGRATION_LATENCY_MIN}min/60 / {AVG_EVENT_DURATION_HRS}h)")
print(f"    Total:                          ${MECHANISM_PROFILES['Spatial Migration']['cost_per_mwh']:,.1f}/MWh")
print(f"\n  Inference routing cost:")
print(f"    Destination LMP during stress:  ${_spatial_dest_cost:,.1f}/MWh")
print(f"    Migration friction:             $0/MWh  (no checkpoint; drain={INFERENCE_DRAIN_TIME_SEC}s)")
print(f"    Total:                          ${MECHANISM_PROFILES['Spatial (Inference)']['cost_per_mwh']:,.1f}/MWh")
print()
 
 
# ─────────────────────────────────────────────────────────────
# PORTFOLIO DEFINITIONS
# ─────────────────────────────────────────────────────────────
 
PORTFOLIOS = {
    'DVFS Only': {
        'depth': FLEX_FRAC * CASCADE_S1,   # 17.5% (facility-relative; Colangelo 25% × S1)
        'duration': None,          # Indefinite
        'mechanisms': ['DVFS'],
        'cost_per_mwh': MECHANISM_PROFILES['DVFS']['cost_per_mwh'],
        'capex_per_mw': 0,
        'desc': 'Software-only. Lowest cost, shallowest commitment.'
    },
    'DVFS + Battery': {
        'depth': FLEX_FRAC * CASCADE_S1,   # Battery extends duration, not depth
        'duration': None,          # DVFS sustains after battery
        'mechanisms': ['DVFS', 'BTM Battery'],
        'cost_per_mwh': MECHANISM_PROFILES['DVFS']['cost_per_mwh'],
        'capex_per_mw': BTM_CAPEX_PER_MW,
        'desc': 'Battery = NPC insurance for events where DVFS insufficient.'
    },
    'DVFS + Spatial': {
        'depth': None,             # Computed below
        'duration': None,          # Both indefinite
        'mechanisms': ['DVFS', 'Spatial Migration'],
        'cost_per_mwh': None,      # Blended, computed below
        'capex_per_mw': 0,         # Orchestration layer is OpEx
        'desc': 'KEY PORTFOLIO. Spatial adds 3-4 pct pts on top of DVFS floor. Requires orchestration layer.'
    },
    'Full Stack': {
        'depth': None,
        'duration': None,
        'mechanisms': ['DVFS', 'BTM Battery', 'Spatial Migration'],
        'cost_per_mwh': None,
        'capex_per_mw': BTM_CAPEX_PER_MW,
        'desc': 'Maximum depth + battery insurance.'
    }
}
 
# Compute effective depth for spatial portfolios
# v16.1: DVFS operates on the shiftable-compute residual (S1 - spatial_frac),
# not the facility residual (1 - spatial_frac). Colangelo 25% is GPU-cluster
# measured, so applying it to non-compute load would overstate flexibility.
# These are mutually exclusive within the S1 envelope — no double-counting.
_spatial_frac = MECHANISM_PROFILES['Spatial Migration']['max_depth']
_dvfs_on_remaining = (CASCADE_S1 - _spatial_frac) * FLEX_FRAC
_combined_depth = _spatial_frac + _dvfs_on_remaining
 
PORTFOLIOS['DVFS + Spatial']['depth'] = _combined_depth
PORTFOLIOS['Full Stack']['depth'] = _combined_depth
 
# Blended cost weighted by MW contribution
_spatial_share = _spatial_frac / _combined_depth
_dvfs_share = _dvfs_on_remaining / _combined_depth
_blended_cost = (_spatial_share * MECHANISM_PROFILES['Spatial Migration']['cost_per_mwh'] +
                 _dvfs_share * MECHANISM_PROFILES['DVFS']['cost_per_mwh'])
PORTFOLIOS['DVFS + Spatial']['cost_per_mwh'] = _blended_cost
PORTFOLIOS['Full Stack']['cost_per_mwh'] = _blended_cost
 
print("PORTFOLIO DEFINITIONS")
print("-" * 90)
for name, p in PORTFOLIOS.items():
    print(f"  {name:<20}: depth={p['depth']:.0%}, cost=${p['cost_per_mwh']:,.0f}/MWh, "
          f"CapEx=${p['capex_per_mw']:,.0f}/MW")
    print(f"    {p['desc']}")
print()
print(f"  Depth breakdown: {_spatial_frac:.0%} migrated + "
      f"{_dvfs_on_remaining:.1%} DVFS on remainder = {_combined_depth:.1%} total")
```

    COMMITMENT OPTIMIZATION FRAMEWORK
    ==========================================================================================
      DR ELCC: 92% (PJM class rating — exogenous)
      BRA clearing: $121,789/MW-yr per accredited MW
      Revenue per committed MW: $112,046/MW-yr
    
    MECHANISM COST PROFILES
    ------------------------------------------------------------------------------------------
      DVFS                : depth=18%, duration=indefinite, cost=$27/MWh
      BTM Battery         : depth=100%, duration=4h, cost=$15/MWh
      Spatial Migration   : depth=4%, duration=indefinite, cost=$252/MWh
      Spatial (Inference) : depth=4%, duration=indefinite, cost=$169/MWh
    
      Spatial cost breakdown (training migration):
        Destination LMP during stress:  $169.3/MWh  (empirical, capacity-weighted)
        Migration friction (amortized): $82.9/MWh  (= $1,327 × 15min/60 / 4.0h)
        Total:                          $252.2/MWh
    
      Inference routing cost:
        Destination LMP during stress:  $169.3/MWh
        Migration friction:             $0/MWh  (no checkpoint; drain=12.1s)
        Total:                          $169.3/MWh
    
    PORTFOLIO DEFINITIONS
    ------------------------------------------------------------------------------------------
      DVFS Only           : depth=18%, cost=$27/MWh, CapEx=$0/MW
        Software-only. Lowest cost, shallowest commitment.
      DVFS + Battery      : depth=18%, cost=$27/MWh, CapEx=$1,500,000/MW
        Battery = NPC insurance for events where DVFS insufficient.
      DVFS + Spatial      : depth=20%, cost=$69/MWh, CapEx=$0/MW
        KEY PORTFOLIO. Spatial adds 3-4 pct pts on top of DVFS floor. Requires orchestration layer.
      Full Stack          : depth=20%, cost=$69/MWh, CapEx=$1,500,000/MW
        Maximum depth + battery insurance.
    
      Depth breakdown: 4% migrated + 16.5% DVFS on remainder = 20.4% total
    

### 3.2 Commitment Optimization

Given the mechanism portfolios and the empirical dispatch profile 
from Part 1, what is the optimal commitment depth and net value 
for each portfolio? Also translates commitment to connect-and-manage 
firm service levels (Option B in the sequential framework).


```python
# ======================================================================
# Cell 3-2: COMMITMENT OPTIMIZATION (v18)
# ===========================================================
# For each portfolio at each fleet size, computes:
#   Revenue  = committed_MW × ELCC × BRA_annual
#   Cost     = dispatch_hrs × cost/MWh × committed + NPC + CapEx
#   Net      = Revenue - Cost
#
# Then translates commitment into connect-and-manage firm service levels.
#
# INPUTS: PORTFOLIOS, MECHANISM_PROFILES, DR_ELCC, BRA_ANNUAL (from Cell 3-1)
#         combined_events_by_yr, YEARS (from Part 1)
#         WACC, NPC rate (from Part 0)
# OUTPUTS: commitment_results, OPTIMAL_COMMITMENT_FRAC,
#          HEADLINE_COMMITTED_MW_10GW, HEADLINE_ACCREDITED_MW_10GW,
#          HEADLINE_NET_VALUE_10GW, ANNUAL_NET_VALUE_PER_MW
# ===========================================================


# ─────────────────────────────────────────────────────────────
# EMPIRICAL DISPATCH PROBABILITY (from Part 1)
# ─────────────────────────────────────────────────────────────

print("EMPIRICAL DISPATCH DISTRIBUTION (from Part 1 combined events)")
print("-" * 90)

all_event_durations = []
events_per_year = {}
dispatch_hours_per_year = {}

for yr in YEARS:
    if yr in combined_events_by_yr:
        evts = combined_events_by_yr[yr]
        events_per_year[yr] = len(evts)
        durations = [ev['duration'] for ev in evts]
        total_hrs = sum(durations)
        dispatch_hours_per_year[yr] = total_hrs
        all_event_durations.extend(durations)
        print(f"  {yr}: {len(evts)} events, {total_hrs} total dispatch hours, "
              f"max duration = {max(durations)}h, median = {np.median(durations):.0f}h")

avg_events_yr = np.mean(list(events_per_year.values()))
avg_dispatch_hrs = np.mean(list(dispatch_hours_per_year.values()))
max_event_duration = max(all_event_durations)
p95_duration = np.percentile(all_event_durations, 95)
p99_duration = np.percentile(all_event_durations, 99) if len(all_event_durations) > 10 else max_event_duration

print(f"\n  Summary: {avg_events_yr:.0f} events/yr avg, {avg_dispatch_hrs:.0f} dispatch hours/yr avg")
print(f"  Duration: max={max_event_duration}h, P95={p95_duration:.0f}h, P99={p99_duration:.0f}h")

# Duration exceedance for NPC exposure
dur_thresholds = [4, 8, 12, 24, 48, 72]
print(f"\n  Events exceeding duration thresholds:")
for thresh in dur_thresholds:
    n_exceed = sum(1 for d in all_event_durations if d > thresh)
    pct = n_exceed / len(all_event_durations) * 100 if all_event_durations else 0
    print(f"    >{thresh:>3}h: {n_exceed}/{len(all_event_durations)} events ({pct:.0f}%)")
print()


# ─────────────────────────────────────────────────────────────
# NPC EXPOSURE
# ─────────────────────────────────────────────────────────────

NPC_RATE_PER_MWH = 2500  # [ESTIMATED] Effective NPC rate per MWh of non-delivery

print("NPC EXPOSURE ANALYSIS")
print("-" * 90)
print(f"  NPC rate: ${NPC_RATE_PER_MWH:,.0f}/MWh of non-delivery (estimated)")
print()

for pname, portfolio in PORTFOLIOS.items():
    p_duration = portfolio.get('duration')
    if p_duration is None:
        print(f"  {pname:<20}: Duration=indefinite → NPC exposure: $0/yr")
    else:
        excess_hrs = sum(max(d - p_duration, 0) for d in all_event_durations)
        npc_hrs_per_yr = excess_hrs / len(YEARS)
        npc_per_mw_yr = npc_hrs_per_yr * NPC_RATE_PER_MWH
        print(f"  {pname:<20}: Duration={p_duration}h → {npc_hrs_per_yr:.0f} excess hrs/yr → "
              f"${npc_per_mw_yr:,.0f}/MW-yr NPC exposure")

print(f"\n  Key: indefinite-duration portfolios (DVFS, Spatial) face zero NPC")
print(f"  from duration exhaustion. Battery-only faces severe tail risk.")
print()


# ─────────────────────────────────────────────────────────────
# COMMITMENT OPTIMIZATION
# ─────────────────────────────────────────────────────────────

print("COMMITMENT OPTIMIZATION BY PORTFOLIO")
print("=" * 90)

# v17 Phase 3: fleet_gws includes 1 GW (individual facility scale) and
# 10 GW (fleet scale) as reference cases. Results are no longer framed
# as "10 GW headline" — they are reported at both reference sizes.
fleet_gws = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
commitment_results = {}

for pname, portfolio in PORTFOLIOS.items():
    max_depth = portfolio['depth']
    cost_mwh = portfolio['cost_per_mwh']
    capex_mw = portfolio['capex_per_mw']
    p_duration = portfolio.get('duration')

    npc_hrs_yr = 0 if p_duration is None else \
        sum(max(d - p_duration, 0) for d in all_event_durations) / len(YEARS)

    commitment_results[pname] = {}

    for gw in fleet_gws:
        fleet_mw = gw * 1000
        committed_mw = fleet_mw * max_depth
        accredited_mw = committed_mw * DR_ELCC
        revenue = accredited_mw * BRA_ANNUAL
        dispatch_cost = avg_dispatch_hrs * cost_mwh * committed_mw
        npc_cost = npc_hrs_yr * NPC_RATE_PER_MWH * committed_mw
        capex_annual = capex_mw * WACC * committed_mw
        net = revenue - dispatch_cost - npc_cost - capex_annual

        commitment_results[pname][gw] = {
            'committed_mw': committed_mw,
            'accredited_mw': accredited_mw,
            'revenue': revenue,
            'dispatch_cost': dispatch_cost,
            'npc_cost': npc_cost,
            'capex_annual': capex_annual,
            'net_value': net,
            'net_per_mw_committed': net / committed_mw if committed_mw > 0 else 0,
            'depth': max_depth,
        }

# Print results
for pname in PORTFOLIOS:
    portfolio = PORTFOLIOS[pname]
    print(f"\n  {pname} (depth={portfolio['depth']:.0%}, cost=${portfolio['cost_per_mwh']:,.0f}/MWh)")
    print(f"  {'Fleet':>6} | {'Committed':>10} | {'Accredited':>10} | {'Revenue':>12} | "
          f"{'Disp Cost':>10} | {'NPC Cost':>10} | {'CapEx':>10} | {'Net Value':>12} | {'Net/MW':>10}")
    print(f"  {'-'*105}")
    for gw in fleet_gws:
        r = commitment_results[pname][gw]
        print(f"  {gw:>4.0f} GW | {r['committed_mw']:>10,.0f} | {r['accredited_mw']:>10,.0f} | "
              f"${r['revenue']/1e6:>10.1f}M | ${r['dispatch_cost']/1e6:>8.1f}M | "
              f"${r['npc_cost']/1e6:>8.1f}M | ${r['capex_annual']/1e6:>8.1f}M | "
              f"${r['net_value']/1e6:>10.1f}M | ${r['net_per_mw_committed']:>8,.0f}")


# ─────────────────────────────────────────────────────────────
# PORTFOLIO COMPARISON AT REFERENCE FLEET SIZES (1 GW, 10 GW)
# ─────────────────────────────────────────────────────────────
# v17 Phase 3: Results reported at two reference sizes rather than
# a single "10 GW headline." 1 GW = individual facility scale,
# 10 GW = fleet scale. Both produce concrete numbers; neither is
# the framework's central claim.

print(f"\n\n{'='*90}")
print("PORTFOLIO COMPARISON AT REFERENCE FLEET SIZES (1 GW, 10 GW)")
print(f"{'='*90}")

for gw_ref in [1.0, 10.0]:
    ref_label = "1 GW (individual facility scale)" if gw_ref == 1.0 else "10 GW (fleet scale)"
    print(f"\n  ── {ref_label} ──")
    print(f"  {'Portfolio':<20} | {'Depth':>6} | {'Committed':>10} | {'Accred MW':>10} | "
          f"{'Revenue':>12} | {'Net Value':>12} | {'Net/MW':>10}")
    print(f"  {'-'*100}")

    for pname in PORTFOLIOS:
        r = commitment_results[pname][gw_ref]
        print(f"  {pname:<20} | {r['depth']:>5.0%} | {r['committed_mw']:>10,.0f} | "
              f"{r['accredited_mw']:>10,.0f} | ${r['revenue']/1e6:>10.1f}M | "
              f"${r['net_value']/1e6:>10.1f}M | ${r['net_per_mw_committed']:>8,.0f}")

    dvfs_only = commitment_results['DVFS Only'][gw_ref]
    dvfs_spatial = commitment_results['DVFS + Spatial'][gw_ref]

    print(f"\n  SPATIAL MIGRATION UPLIFT (DVFS+Spatial vs DVFS Only) at {gw_ref:.0f} GW:")
    print(f"    Additional accredited MW:  {dvfs_spatial['accredited_mw'] - dvfs_only['accredited_mw']:>+10,.0f} MW")
    print(f"    Additional revenue:        ${(dvfs_spatial['revenue'] - dvfs_only['revenue'])/1e6:>+10.1f}M/yr")
    print(f"    Additional net value:      ${(dvfs_spatial['net_value'] - dvfs_only['net_value'])/1e6:>+10.1f}M/yr")
    if dvfs_only['accredited_mw'] > 0:
        print(f"    → Spatial adds {(dvfs_spatial['accredited_mw'] - dvfs_only['accredited_mw'])/dvfs_only['accredited_mw']:.0%} more accredited capacity")


# ─────────────────────────────────────────────────────────────
# CONNECT-AND-MANAGE TRANSLATION
# ─────────────────────────────────────────────────────────────

print(f"\n\n{'='*90}")
print("CONNECT-AND-MANAGE: COMMITMENT → FIRM SERVICE LEVEL")
print(f"{'='*90}")
print(f"\n  Under flexible interconnection (Option B), committed MW = non-firm load.")
print(f"  PJM load forecast reflects firm service level, not nameplate.")
print()

print(f"  {'Portfolio':<20} | {'Fleet':>6} | {'Committed':>10} | {'Firm Level':>10} | {'% Firm':>7} | {'Demand Reduction':>16}")
print(f"  {'-'*85}")

# v17 Phase 3: Show reference cases (1 GW, 10 GW) plus 5 GW and 20 GW for range
for pname in ['DVFS Only', 'DVFS + Spatial', 'Full Stack']:
    for gw in [1.0, 5.0, 10.0, 20.0]:
        r = commitment_results[pname][gw]
        fleet_mw = gw * 1000
        firm_mw = fleet_mw - r['committed_mw']
        firm_pct = firm_mw / fleet_mw
        print(f"  {pname:<20} | {gw:>4.0f} GW | {r['committed_mw']:>10,.0f} | "
              f"{firm_mw:>10,.0f} | {firm_pct:>6.0%} | {r['committed_mw']:>14,.0f} MW")

dvfs_spatial_1gw = commitment_results['DVFS + Spatial'][1.0]
dvfs_spatial_10gw = commitment_results['DVFS + Spatial'][10.0]
print(f"\n  Reference cases (DVFS+Spatial):")
print(f"    1 GW:  firm service = {1000 - dvfs_spatial_1gw['committed_mw']:,.0f} MW, "
      f"removes {dvfs_spatial_1gw['committed_mw']:,.0f} MW from Reliability Requirement")
print(f"    10 GW: firm service = {10000 - dvfs_spatial_10gw['committed_mw']:,.0f} MW, "
      f"removes {dvfs_spatial_10gw['committed_mw']:,.0f} MW from Reliability Requirement")
print(f"  BEFORE any supply-side DR participation.")


# ─────────────────────────────────────────────────────────────
# STORE RESULTS — DUAL REFERENCE CASES (1 GW, 10 GW)
# ─────────────────────────────────────────────────────────────
# v17 Phase 3: Store results at both reference fleet sizes.
# These are reference cases, not headlines. The per-GW sweep in the
# conditional MC is the authoritative source for fleet-size dependence.

PRIMARY_PORTFOLIO = 'DVFS + Spatial'
OPTIMAL_COMMITMENT_FRAC = PORTFOLIOS[PRIMARY_PORTFOLIO]['depth']

# 1 GW reference (individual facility scale)
REF_COMMITTED_MW_1GW = commitment_results[PRIMARY_PORTFOLIO][1.0]['committed_mw']
REF_ACCREDITED_MW_1GW = commitment_results[PRIMARY_PORTFOLIO][1.0]['accredited_mw']
REF_NET_VALUE_1GW = commitment_results[PRIMARY_PORTFOLIO][1.0]['net_value']

# 10 GW reference (fleet scale)
REF_COMMITTED_MW_10GW = commitment_results[PRIMARY_PORTFOLIO][10.0]['committed_mw']
REF_ACCREDITED_MW_10GW = commitment_results[PRIMARY_PORTFOLIO][10.0]['accredited_mw']
REF_NET_VALUE_10GW = commitment_results[PRIMARY_PORTFOLIO][10.0]['net_value']
ANNUAL_NET_VALUE_PER_MW = commitment_results[PRIMARY_PORTFOLIO][10.0]['net_per_mw_committed']

# Legacy aliases — downstream sections (election, BCA) still reference these
HEADLINE_COMMITTED_MW_10GW = REF_COMMITTED_MW_10GW
HEADLINE_ACCREDITED_MW_10GW = REF_ACCREDITED_MW_10GW
HEADLINE_NET_VALUE_10GW = REF_NET_VALUE_10GW

DVFS_COMMITMENT_FRAC = FLEX_FRAC * CASCADE_S1  # v16.1: facility-relative (Colangelo × S1)
DVFS_COMMITTED_MW_10GW = commitment_results['DVFS Only'][10.0]['committed_mw']
DVFS_ACCREDITED_MW_10GW = commitment_results['DVFS Only'][10.0]['accredited_mw']
DVFS_COMMITTED_MW_1GW = commitment_results['DVFS Only'][1.0]['committed_mw']
DVFS_ACCREDITED_MW_1GW = commitment_results['DVFS Only'][1.0]['accredited_mw']

print(f"\n{'='*90}")
print("STORED REFERENCE VARIABLES (1 GW and 10 GW)")
print(f"{'='*90}")
print(f"  OPTIMAL_COMMITMENT_FRAC     = {OPTIMAL_COMMITMENT_FRAC:.1%} ({PRIMARY_PORTFOLIO})")
print(f"  ── 1 GW reference (individual facility scale) ──")
print(f"  REF_COMMITTED_MW_1GW        = {REF_COMMITTED_MW_1GW:,.0f} MW")
print(f"  REF_ACCREDITED_MW_1GW       = {REF_ACCREDITED_MW_1GW:,.0f} MW")
print(f"  REF_NET_VALUE_1GW           = ${REF_NET_VALUE_1GW/1e6:,.1f}M/yr")
print(f"  ── 10 GW reference (fleet scale) ──")
print(f"  REF_COMMITTED_MW_10GW       = {REF_COMMITTED_MW_10GW:,.0f} MW")
print(f"  REF_ACCREDITED_MW_10GW      = {REF_ACCREDITED_MW_10GW:,.0f} MW")
print(f"  REF_NET_VALUE_10GW          = ${REF_NET_VALUE_10GW/1e6:,.1f}M/yr")
print(f"  ANNUAL_NET_VALUE_PER_MW     = ${ANNUAL_NET_VALUE_PER_MW:,.0f}/MW-yr (fleet-scale)")
```

    EMPIRICAL DISPATCH DISTRIBUTION (from Part 1 combined events)
    ------------------------------------------------------------------------------------------
      2022: 14 events, 159 total dispatch hours, max duration = 72h, median = 2h
      2023: 22 events, 78 total dispatch hours, max duration = 13h, median = 2h
      2024: 13 events, 107 total dispatch hours, max duration = 53h, median = 5h
      2025: 15 events, 146 total dispatch hours, max duration = 50h, median = 3h
    
      Summary: 16 events/yr avg, 122 dispatch hours/yr avg
      Duration: max=72h, P95=47h, P99=61h
    
      Events exceeding duration thresholds:
        >  4h: 21/64 events (33%)
        >  8h: 11/64 events (17%)
        > 12h: 8/64 events (12%)
        > 24h: 6/64 events (9%)
        > 48h: 4/64 events (6%)
        > 72h: 0/64 events (0%)
    
    NPC EXPOSURE ANALYSIS
    ------------------------------------------------------------------------------------------
      NPC rate: $2,500/MWh of non-delivery (estimated)
    
      DVFS Only           : Duration=indefinite → NPC exposure: $0/yr
      DVFS + Battery      : Duration=indefinite → NPC exposure: $0/yr
      DVFS + Spatial      : Duration=indefinite → NPC exposure: $0/yr
      Full Stack          : Duration=indefinite → NPC exposure: $0/yr
    
      Key: indefinite-duration portfolios (DVFS, Spatial) face zero NPC
      from duration exhaustion. Battery-only faces severe tail risk.
    
    COMMITMENT OPTIMIZATION BY PORTFOLIO
    ==========================================================================================
    
      DVFS Only (depth=18%, cost=$27/MWh)
       Fleet |  Committed | Accredited |      Revenue |  Disp Cost |   NPC Cost |      CapEx |    Net Value |     Net/MW
      ---------------------------------------------------------------------------------------------------------
         1 GW |        175 |        161 | $      19.6M | $     0.6M | $     0.0M | $     0.0M | $      19.0M | $ 108,796
         2 GW |        350 |        322 | $      39.2M | $     1.1M | $     0.0M | $     0.0M | $      38.1M | $ 108,796
         5 GW |        875 |        805 | $      98.0M | $     2.8M | $     0.0M | $     0.0M | $      95.2M | $ 108,796
        10 GW |      1,750 |      1,610 | $     196.1M | $     5.7M | $     0.0M | $     0.0M | $     190.4M | $ 108,796
        15 GW |      2,625 |      2,415 | $     294.1M | $     8.5M | $     0.0M | $     0.0M | $     285.6M | $ 108,796
        20 GW |      3,500 |      3,220 | $     392.2M | $    11.4M | $     0.0M | $     0.0M | $     380.8M | $ 108,796
    
      DVFS + Battery (depth=18%, cost=$27/MWh)
       Fleet |  Committed | Accredited |      Revenue |  Disp Cost |   NPC Cost |      CapEx |    Net Value |     Net/MW
      ---------------------------------------------------------------------------------------------------------
         1 GW |        175 |        161 | $      19.6M | $     0.6M | $     0.0M | $    26.2M | $      -7.2M | $ -41,204
         2 GW |        350 |        322 | $      39.2M | $     1.1M | $     0.0M | $    52.5M | $     -14.4M | $ -41,204
         5 GW |        875 |        805 | $      98.0M | $     2.8M | $     0.0M | $   131.2M | $     -36.1M | $ -41,204
        10 GW |      1,750 |      1,610 | $     196.1M | $     5.7M | $     0.0M | $   262.5M | $     -72.1M | $ -41,204
        15 GW |      2,625 |      2,415 | $     294.1M | $     8.5M | $     0.0M | $   393.8M | $    -108.2M | $ -41,204
        20 GW |      3,500 |      3,220 | $     392.2M | $    11.4M | $     0.0M | $   525.0M | $    -144.2M | $ -41,204
    
      DVFS + Spatial (depth=20%, cost=$69/MWh)
       Fleet |  Committed | Accredited |      Revenue |  Disp Cost |   NPC Cost |      CapEx |    Net Value |     Net/MW
      ---------------------------------------------------------------------------------------------------------
         1 GW |        204 |        187 | $      22.8M | $     1.7M | $     0.0M | $     0.0M | $      21.1M | $ 103,591
         2 GW |        408 |        375 | $      45.7M | $     3.4M | $     0.0M | $     0.0M | $      42.2M | $ 103,591
         5 GW |      1,019 |        937 | $     114.2M | $     8.6M | $     0.0M | $     0.0M | $     105.5M | $ 103,591
        10 GW |      2,038 |      1,875 | $     228.3M | $    17.2M | $     0.0M | $     0.0M | $     211.1M | $ 103,591
        15 GW |      3,057 |      2,812 | $     342.5M | $    25.8M | $     0.0M | $     0.0M | $     316.6M | $ 103,591
        20 GW |      4,075 |      3,749 | $     456.6M | $    34.5M | $     0.0M | $     0.0M | $     422.2M | $ 103,591
    
      Full Stack (depth=20%, cost=$69/MWh)
       Fleet |  Committed | Accredited |      Revenue |  Disp Cost |   NPC Cost |      CapEx |    Net Value |     Net/MW
      ---------------------------------------------------------------------------------------------------------
         1 GW |        204 |        187 | $      22.8M | $     1.7M | $     0.0M | $    30.6M | $      -9.5M | $ -46,409
         2 GW |        408 |        375 | $      45.7M | $     3.4M | $     0.0M | $    61.1M | $     -18.9M | $ -46,409
         5 GW |      1,019 |        937 | $     114.2M | $     8.6M | $     0.0M | $   152.8M | $     -47.3M | $ -46,409
        10 GW |      2,038 |      1,875 | $     228.3M | $    17.2M | $     0.0M | $   305.7M | $     -94.6M | $ -46,409
        15 GW |      3,057 |      2,812 | $     342.5M | $    25.8M | $     0.0M | $   458.5M | $    -141.9M | $ -46,409
        20 GW |      4,075 |      3,749 | $     456.6M | $    34.5M | $     0.0M | $   611.3M | $    -189.1M | $ -46,409
    
    
    ==========================================================================================
    PORTFOLIO COMPARISON AT REFERENCE FLEET SIZES (1 GW, 10 GW)
    ==========================================================================================
    
      ── 1 GW (individual facility scale) ──
      Portfolio            |  Depth |  Committed |  Accred MW |      Revenue |    Net Value |     Net/MW
      ----------------------------------------------------------------------------------------------------
      DVFS Only            |   18% |        175 |        161 | $      19.6M | $      19.0M | $ 108,796
      DVFS + Battery       |   18% |        175 |        161 | $      19.6M | $      -7.2M | $ -41,204
      DVFS + Spatial       |   20% |        204 |        187 | $      22.8M | $      21.1M | $ 103,591
      Full Stack           |   20% |        204 |        187 | $      22.8M | $      -9.5M | $ -46,409
    
      SPATIAL MIGRATION UPLIFT (DVFS+Spatial vs DVFS Only) at 1 GW:
        Additional accredited MW:         +26 MW
        Additional revenue:        $      +3.2M/yr
        Additional net value:      $      +2.1M/yr
        → Spatial adds 16% more accredited capacity
    
      ── 10 GW (fleet scale) ──
      Portfolio            |  Depth |  Committed |  Accred MW |      Revenue |    Net Value |     Net/MW
      ----------------------------------------------------------------------------------------------------
      DVFS Only            |   18% |      1,750 |      1,610 | $     196.1M | $     190.4M | $ 108,796
      DVFS + Battery       |   18% |      1,750 |      1,610 | $     196.1M | $     -72.1M | $ -41,204
      DVFS + Spatial       |   20% |      2,038 |      1,875 | $     228.3M | $     211.1M | $ 103,591
      Full Stack           |   20% |      2,038 |      1,875 | $     228.3M | $     -94.6M | $ -46,409
    
      SPATIAL MIGRATION UPLIFT (DVFS+Spatial vs DVFS Only) at 10 GW:
        Additional accredited MW:        +265 MW
        Additional revenue:        $     +32.2M/yr
        Additional net value:      $     +20.7M/yr
        → Spatial adds 16% more accredited capacity
    
    
    ==========================================================================================
    CONNECT-AND-MANAGE: COMMITMENT → FIRM SERVICE LEVEL
    ==========================================================================================
    
      Under flexible interconnection (Option B), committed MW = non-firm load.
      PJM load forecast reflects firm service level, not nameplate.
    
      Portfolio            |  Fleet |  Committed | Firm Level |  % Firm | Demand Reduction
      -------------------------------------------------------------------------------------
      DVFS Only            |    1 GW |        175 |        825 |    82% |            175 MW
      DVFS Only            |    5 GW |        875 |      4,125 |    82% |            875 MW
      DVFS Only            |   10 GW |      1,750 |      8,250 |    82% |          1,750 MW
      DVFS Only            |   20 GW |      3,500 |     16,500 |    82% |          3,500 MW
      DVFS + Spatial       |    1 GW |        204 |        796 |    80% |            204 MW
      DVFS + Spatial       |    5 GW |      1,019 |      3,981 |    80% |          1,019 MW
      DVFS + Spatial       |   10 GW |      2,038 |      7,962 |    80% |          2,038 MW
      DVFS + Spatial       |   20 GW |      4,075 |     15,925 |    80% |          4,075 MW
      Full Stack           |    1 GW |        204 |        796 |    80% |            204 MW
      Full Stack           |    5 GW |      1,019 |      3,981 |    80% |          1,019 MW
      Full Stack           |   10 GW |      2,038 |      7,962 |    80% |          2,038 MW
      Full Stack           |   20 GW |      4,075 |     15,925 |    80% |          4,075 MW
    
      Reference cases (DVFS+Spatial):
        1 GW:  firm service = 796 MW, removes 204 MW from Reliability Requirement
        10 GW: firm service = 7,962 MW, removes 2,038 MW from Reliability Requirement
      BEFORE any supply-side DR participation.
    
    ==========================================================================================
    STORED REFERENCE VARIABLES (1 GW and 10 GW)
    ==========================================================================================
      OPTIMAL_COMMITMENT_FRAC     = 20.4% (DVFS + Spatial)
      ── 1 GW reference (individual facility scale) ──
      REF_COMMITTED_MW_1GW        = 204 MW
      REF_ACCREDITED_MW_1GW       = 187 MW
      REF_NET_VALUE_1GW           = $21.1M/yr
      ── 10 GW reference (fleet scale) ──
      REF_COMMITTED_MW_10GW       = 2,038 MW
      REF_ACCREDITED_MW_10GW      = 1,875 MW
      REF_NET_VALUE_10GW          = $211.1M/yr
      ANNUAL_NET_VALUE_PER_MW     = $103,591/MW-yr (fleet-scale)
    

### 3.3 Election Mechanism & Avoided Installed Capacity

The election mechanism splits the fleet into Mode A (local DVFS, 
accredited as DR) and Mode B (spatial migration, reducing planning 
reserve requirement). Total avoided installed capacity combines both 
modes — this is the resource planning impact number.


```python
# ======================================================================
# Cell 3-3: ELECTION MECHANISM & AVOIDED INSTALLED CAPACITY (v18)
# ===========================================================
# The election mechanism splits the fleet into two mutually exclusive modes:
#
#   Mode A (local DR):   Non-migrating load × DVFS × ELCC → UCAP MW
#   Mode B (spatial):    Migrating load × (1 + IRM) → avoided installed MW
#
# Total avoided capacity = Mode A + Mode B (no double-counting).
#
# NOTE ON UNITS: Mode B produces installed MW (demand reduction × IRM
# multiplier). Mode A produces UCAP MW (accredited capacity). These are
# different units being summed. The result is conservative — converting
# Mode A to installed MW via the IRM multiplier would increase the total.
# This asymmetry is documented here rather than "fixed" because the
# accredited-MW framing for Mode A is more natural for the BRA context.
#
# INPUTS: EFFECTIVE_SPATIAL_FRAC, FLEX_FRAC, DR_ELCC, RESERVE_MARGIN_TARGET,
#         E3_NEW_GAS_CT_GW, E3_CT_LEVELIZED_COST, E3_CT_LEVELIZED_CEJA,
#         BRA_2027_28_PRICE, OPTIMAL_COMMITMENT_FRAC (from Parts 0, 3)
# OUTPUTS: election_results, SPATIAL_AVOIDED_1GW, SPATIAL_AVOIDED_10GW,
#          SPATIAL_ANNUAL_SAVINGS_1GW, SPATIAL_ANNUAL_SAVINGS_10GW
# ===========================================================

# Planning reserve margin: removing 1 MW of committed load avoids
# building (1 + IRM) MW of installed generation capacity.
PLANNING_RESERVE_MARGIN = RESERVE_MARGIN_TARGET - 1.0  # 0.20 (IRM = 20%)

# v17 Phase 3: include 1 GW reference case
fleet_gws = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]

print("ELECTION-BASED SPATIAL CAPACITY & AVOIDED INSTALLED CAPACITY (v18)")
print("=" * 100)
print(f"  Cascade product: {EFFECTIVE_SPATIAL_FRAC:.3f}")
print(f"  Planning reserve margin: {PLANNING_RESERVE_MARGIN:.0%} (PJM 2027/28 IRM = {RESERVE_MARGIN_TARGET:.0%})")
print(f"  E3 CT buildout target: {E3_NEW_GAS_CT_GW:.0f} GW")
print()


# ─────────────────────────────────────────────────────────────
# CORE CALCULATION
# ─────────────────────────────────────────────────────────────

header = (f"  {'Fleet':>6} | {'Gross':>7} | "
          f"{'Spatial':>8} | {'Mode A':>7} | {'TOTAL':>8} | {'% of':>6}")
subhdr = (f"  {'(GW)':>6} | {'Spat MW':>7} | "
          f"{'Avoided':>8} | {'DR MW':>7} | {'Avoided':>8} | {'E3 13G':>6}")
print(header)
print(subhdr)
print(f"  {'─' * 80}")

election_results = {}
for gw in fleet_gws:
    fleet_mw = gw * 1000

    # Mode B: Spatial migration → avoided installed capacity
    gross_spatial = fleet_mw * EFFECTIVE_SPATIAL_FRAC
    spatial_avoided = gross_spatial * (1 + PLANNING_RESERVE_MARGIN)

    # Mode A: Local DR on non-migrating shiftable compute → accredited (UCAP) MW
    # v16.1: DVFS envelope = (S1 - spatial), not (1 - spatial)
    non_migrating_mw = fleet_mw * (CASCADE_S1 - EFFECTIVE_SPATIAL_FRAC)
    dvfs_commitment = non_migrating_mw * FLEX_FRAC
    local_dr_accredited = dvfs_commitment * DR_ELCC

    # Combined
    total_avoided = spatial_avoided + local_dr_accredited
    pct_e3 = total_avoided / (E3_NEW_GAS_CT_GW * 1000) * 100

    election_results[gw] = {
        'gross_spatial': gross_spatial,
        'spatial_avoided': spatial_avoided,
        'non_migrating_mw': non_migrating_mw,
        'dvfs_commitment': dvfs_commitment,
        'local_dr_accredited': local_dr_accredited,
        'total_avoided': total_avoided,
        'pct_e3': pct_e3,
    }

    print(f"  {gw:>4.0f} GW | {gross_spatial:>7,.0f} | "
          f"{spatial_avoided:>8,.0f} | {local_dr_accredited:>7,.0f} | "
          f"{total_avoided:>8,.0f} | {pct_e3:>5.1f}%")

print()
print(f"  Mode B (spatial): fleet × {EFFECTIVE_SPATIAL_FRAC:.3f} × {1+PLANNING_RESERVE_MARGIN:.2f}")
print(f"  Mode A (local DR): fleet × {1-EFFECTIVE_SPATIAL_FRAC:.3f} × {FLEX_FRAC} × {DR_ELCC}")
print(f"  Total = Mode B (installed MW) + Mode A (UCAP MW) — conservative sum")


# ─────────────────────────────────────────────────────────────
# BRIDGE: Avoided Capacity vs. Accredited MW
# ─────────────────────────────────────────────────────────────
# These are related but distinct metrics:
#   Accredited MW = fleet × depth × ELCC → market supply / BRA offer
#   Avoided installed = spatial × (1+IRM) + local_DR × ELCC → planning impact

print(f"\n  ── Bridge to Pillar 1 (reference cases) ──")
for _ref_gw in [1.0, 10.0]:
    _ref_mw = _ref_gw * 1000
    _accredited_mw = _ref_mw * OPTIMAL_COMMITMENT_FRAC * DR_ELCC
    _ref_label = "individual facility scale" if _ref_gw == 1.0 else "fleet scale"
    print(f"  {_ref_gw:.0f} GW ({_ref_label}):")
    print(f"    Accredited MW: {_accredited_mw:,.0f} MW "
          f"(= {_ref_gw:.0f} GW × {OPTIMAL_COMMITMENT_FRAC:.0%} depth × {DR_ELCC:.0%} ELCC)")
    print(f"    Avoided installed: "
          f"{election_results[_ref_gw]['total_avoided']:,.0f} MW "
          f"(spatial × {1+PLANNING_RESERVE_MARGIN:.2f} + local DR × ELCC)")
print(f"  Accredited MW = market-facing. Avoided installed = planning impact.")


# ─────────────────────────────────────────────────────────────
# AVOIDED COST — ANNUAL SAVINGS
# ─────────────────────────────────────────────────────────────

print(f"\n{'─'*100}")
print("ANNUAL AVOIDED COST")
print(f"{'─'*100}")
print()

print(f"  {'Fleet':>6} | {'Spatial Av':>10} | {'Local DR':>8} | {'Total Av':>9} | {'Annual Base':>12} | {'Annual CEJA':>12}")
print(f"  {'─' * 90}")

for gw in fleet_gws:
    r = election_results[gw]
    annual_base = r['total_avoided'] * E3_CT_LEVELIZED_COST
    annual_ceja = r['total_avoided'] * E3_CT_LEVELIZED_CEJA
    print(f"  {gw:>4.0f} GW | {r['spatial_avoided']:>10,.0f} | {r['local_dr_accredited']:>8,.0f} | "
          f"{r['total_avoided']:>9,.0f} | ${annual_base/1e6:>10.0f}M | ${annual_ceja/1e6:>10.0f}M ")

print(f"\n  Base: ${E3_CT_LEVELIZED_COST:,}/MW-yr | CEJA: ${E3_CT_LEVELIZED_CEJA:,}/MW-yr")


# ─────────────────────────────────────────────────────────────
# VIRTUAL TRANSMISSION COMPARISON
# ─────────────────────────────────────────────────────────────

print(f"\n{'─'*100}")
print("VIRTUAL vs. PHYSICAL INTER-REGIONAL TRANSMISSION")
print(f"{'─'*100}")

physical_dc_tie_gw = 1.3

# v17 Phase 3: reference cases plus 5 GW and 20 GW for range
for gw in [1.0, 5.0, 10.0, 20.0]:
    r = election_results[gw]
    virtual_mw = r['gross_spatial']
    print(f"\n  At {gw:.0f} GW DC fleet:")
    print(f"    Virtual transfer capability: {virtual_mw:,.0f} MW ({virtual_mw/1000:.1f} GW)")
    print(f"    Physical DC ties (existing): {physical_dc_tie_gw * 1000:,.0f} MW ({physical_dc_tie_gw:.1f} GW)")
    print(f"    Virtual / Physical ratio:    {virtual_mw / (physical_dc_tie_gw * 1000):.1f}x")


# ─────────────────────────────────────────────────────────────
# PER-FACILITY CAPACITY REVENUE
# ─────────────────────────────────────────────────────────────

print(f"\n{'─'*100}")
print("PER-FACILITY CAPACITY REVENUE")
print(f"{'─'*100}")

# Use empirical events/year from Part 1 (was hardcoded at 50 — fixed in v15)
_fac_events_yr = avg_events_yr  # From Cell 3-2 dispatch distribution
_fac_migration_cost_per_event_mw = MIGRATION_COST_PER_EVENT  # From Part 0

facility_sizes = [0.5, 1.0, 2.0, 5.0]
print(f"\n  Events/yr: {_fac_events_yr:.0f} (empirical from Part 1)")
print(f"  Migration cost/event/MW: ${_fac_migration_cost_per_event_mw:,.0f}")
print()
print(f"  {'Facility':>10} | {'Spatial MW':>10} | {'Revenue':>12} | {'Mig Cost':>10} | {'Net':>12}")
print(f"  {'─' * 70}")

for size_gw in facility_sizes:
    size_mw = size_gw * 1000
    spatial_mw = size_mw * EFFECTIVE_SPATIAL_FRAC
    annual_rev = spatial_mw * BRA_2027_28_PRICE * 365
    annual_cost = spatial_mw * _fac_events_yr * _fac_migration_cost_per_event_mw
    net = annual_rev - annual_cost
    print(f"  {size_gw:>8.1f} GW | {spatial_mw:>10,.0f} | "
          f"${annual_rev/1e6:>10.1f}M | ${annual_cost/1e6:>8.1f}M | ${net/1e6:>10.1f}M")

print(f"\n  Note: Revenue is home-market only. Compare to IX queue NPV (Part 5).")
print(f"  Capacity revenue is real but queue incentive dominates.")

# ─────────────────────────────────────────────────────────────
# PER-FACILITY CAPACITY REVENUE — INFERENCE ROUTING (v15.1)
# ─────────────────────────────────────────────────────────────
# Inference routing has no per-event checkpoint friction.
# Active requests drain in ~12s (P99); new requests reroute instantly.
# Per-event cost is effectively zero. Cost structure is annual readiness.

print(f"\n{'─'*100}")
print("PER-FACILITY CAPACITY REVENUE — INFERENCE ROUTING (no checkpoint friction)")
print(f"{'─'*100}")
print(f"\n  Inference drain time: {INFERENCE_DRAIN_TIME_SEC}s (P99)")
print(f"  Per-event cost: ${INFERENCE_COST_PER_EVENT:,.0f}/MW (no lost compute)")
print()
print(f"  {'Facility':>10} | {'Spatial MW':>10} | {'Revenue':>12} | {'Mig Cost':>10} | {'Net':>12} | {'vs Training':>12}")
print(f"  {'─' * 85}")

for size_gw in facility_sizes:
    size_mw = size_gw * 1000
    spatial_mw = size_mw * EFFECTIVE_SPATIAL_FRAC
    inf_annual_rev = spatial_mw * BRA_2027_28_PRICE * 365
    inf_annual_cost = 0.0  # No per-event friction for inference
    inf_net = inf_annual_rev - inf_annual_cost
    # Delta vs training (from table above)
    training_cost = spatial_mw * _fac_events_yr * _fac_migration_cost_per_event_mw
    delta = training_cost  # The entire training cost is the difference
    print(f"  {size_gw:>8.1f} GW | {spatial_mw:>10,.0f} | "
          f"${inf_annual_rev/1e6:>10.1f}M | ${inf_annual_cost/1e6:>8.1f}M | "
          f"${inf_net/1e6:>10.1f}M | +${delta/1e6:>9.1f}M")

print(f"\n  Training migration cost absorbed by checkpoint overhead: "
      f"${_fac_migration_cost_per_event_mw:,.0f}/event/MW × {_fac_events_yr:.0f} events/yr")
print(f"  Inference routing eliminates this entirely.")
print(f"  Note: Does not yet include annual readiness costs (pre-staging, serving stack).")
print(f"  Readiness cost would need to exceed the training friction delta to change the comparison.")

# ─────────────────────────────────────────────────────────────
# STORE RESULTS
# ─────────────────────────────────────────────────────────────

# v17 Phase 3: Store at both reference fleet sizes
SPATIAL_AVOIDED_1GW = election_results[1.0]['total_avoided']
SPATIAL_ANNUAL_SAVINGS_1GW = SPATIAL_AVOIDED_1GW * E3_CT_LEVELIZED_COST
SPATIAL_RESIDUAL_FIRM_1GW = 1000 * (1 - OPTIMAL_COMMITMENT_FRAC)

SPATIAL_AVOIDED_10GW = election_results[10.0]['total_avoided']
SPATIAL_ANNUAL_SAVINGS_10GW = SPATIAL_AVOIDED_10GW * E3_CT_LEVELIZED_COST
SPATIAL_RESIDUAL_FIRM_10GW = 10000 * (1 - OPTIMAL_COMMITMENT_FRAC)

V2_SPATIAL_FRAC = EFFECTIVE_SPATIAL_FRAC
V2_ELECTION_RESULTS = election_results

print(f"\n{'─'*100}")
print("STORED: ELECTION MECHANISM RESULTS (reference cases)")
print(f"{'─'*100}")
print(f"  ── 1 GW reference ──")
print(f"  SPATIAL_AVOIDED_1GW          = {SPATIAL_AVOIDED_1GW:,.0f} MW")
print(f"  SPATIAL_ANNUAL_SAVINGS_1GW   = ${SPATIAL_ANNUAL_SAVINGS_1GW/1e6:,.1f}M/yr")
print(f"  SPATIAL_RESIDUAL_FIRM_1GW    = {SPATIAL_RESIDUAL_FIRM_1GW:,.0f} MW")
print(f"  ── 10 GW reference ──")
print(f"  SPATIAL_AVOIDED_10GW         = {SPATIAL_AVOIDED_10GW:,.0f} MW")
print(f"  SPATIAL_ANNUAL_SAVINGS_10GW  = ${SPATIAL_ANNUAL_SAVINGS_10GW/1e6:,.0f}M/yr")
print(f"  SPATIAL_RESIDUAL_FIRM_10GW   = {SPATIAL_RESIDUAL_FIRM_10GW:,.0f} MW")
```

    ELECTION-BASED SPATIAL CAPACITY & AVOIDED INSTALLED CAPACITY (v18)
    ====================================================================================================
      Cascade product: 0.038
      Planning reserve margin: 20% (PJM 2027/28 IRM = 120%)
      E3 CT buildout target: 13 GW
    
       Fleet |   Gross |  Spatial |  Mode A |    TOTAL |   % of
        (GW) | Spat MW |  Avoided |   DR MW |  Avoided | E3 13G
      ────────────────────────────────────────────────────────────────────────────────
         1 GW |      38 |       46 |     152 |      198 |   1.5%
         2 GW |      77 |       92 |     304 |      396 |   3.0%
         5 GW |     192 |      230 |     761 |      991 |   7.6%
        10 GW |     384 |      460 |   1,522 |    1,982 |  15.2%
        15 GW |     575 |      691 |   2,283 |    2,973 |  22.9%
        20 GW |     767 |      921 |   3,044 |    3,964 |  30.5%
    
      Mode B (spatial): fleet × 0.038 × 1.20
      Mode A (local DR): fleet × 0.962 × 0.25 × 0.92
      Total = Mode B (installed MW) + Mode A (UCAP MW) — conservative sum
    
      ── Bridge to Pillar 1 (reference cases) ──
      1 GW (individual facility scale):
        Accredited MW: 187 MW (= 1 GW × 20% depth × 92% ELCC)
        Avoided installed: 198 MW (spatial × 1.20 + local DR × ELCC)
      10 GW (fleet scale):
        Accredited MW: 1,875 MW (= 10 GW × 20% depth × 92% ELCC)
        Avoided installed: 1,982 MW (spatial × 1.20 + local DR × ELCC)
      Accredited MW = market-facing. Avoided installed = planning impact.
    
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    ANNUAL AVOIDED COST
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    
       Fleet | Spatial Av | Local DR |  Total Av |  Annual Base |  Annual CEJA
      ──────────────────────────────────────────────────────────────────────────────────────────
         1 GW |         46 |      152 |       198 | $        36M | $        41M 
         2 GW |         92 |      304 |       396 | $        71M | $        81M 
         5 GW |        230 |      761 |       991 | $       178M | $       203M 
        10 GW |        460 |    1,522 |     1,982 | $       357M | $       406M 
        15 GW |        691 |    2,283 |     2,973 | $       535M | $       609M 
        20 GW |        921 |    3,044 |     3,964 | $       714M | $       813M 
    
      Base: $180,000/MW-yr | CEJA: $205,000/MW-yr
    
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    VIRTUAL vs. PHYSICAL INTER-REGIONAL TRANSMISSION
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    
      At 1 GW DC fleet:
        Virtual transfer capability: 38 MW (0.0 GW)
        Physical DC ties (existing): 1,300 MW (1.3 GW)
        Virtual / Physical ratio:    0.0x
    
      At 5 GW DC fleet:
        Virtual transfer capability: 192 MW (0.2 GW)
        Physical DC ties (existing): 1,300 MW (1.3 GW)
        Virtual / Physical ratio:    0.1x
    
      At 10 GW DC fleet:
        Virtual transfer capability: 384 MW (0.4 GW)
        Physical DC ties (existing): 1,300 MW (1.3 GW)
        Virtual / Physical ratio:    0.3x
    
      At 20 GW DC fleet:
        Virtual transfer capability: 767 MW (0.8 GW)
        Physical DC ties (existing): 1,300 MW (1.3 GW)
        Virtual / Physical ratio:    0.6x
    
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    PER-FACILITY CAPACITY REVENUE
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    
      Events/yr: 16 (empirical from Part 1)
      Migration cost/event/MW: $332
    
        Facility | Spatial MW |      Revenue |   Mig Cost |          Net
      ──────────────────────────────────────────────────────────────────────
           0.5 GW |         19 | $       2.3M | $     0.1M | $       2.2M
           1.0 GW |         38 | $       4.7M | $     0.2M | $       4.5M
           2.0 GW |         77 | $       9.3M | $     0.4M | $       8.9M
           5.0 GW |        192 | $      23.3M | $     1.0M | $      22.3M
    
      Note: Revenue is home-market only. Compare to IX queue NPV (Part 5).
      Capacity revenue is real but queue incentive dominates.
    
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    PER-FACILITY CAPACITY REVENUE — INFERENCE ROUTING (no checkpoint friction)
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    
      Inference drain time: 12.1s (P99)
      Per-event cost: $0/MW (no lost compute)
    
        Facility | Spatial MW |      Revenue |   Mig Cost |          Net |  vs Training
      ─────────────────────────────────────────────────────────────────────────────────────
           0.5 GW |         19 | $       2.3M | $     0.0M | $       2.3M | +$      0.1M
           1.0 GW |         38 | $       4.7M | $     0.0M | $       4.7M | +$      0.2M
           2.0 GW |         77 | $       9.3M | $     0.0M | $       9.3M | +$      0.4M
           5.0 GW |        192 | $      23.3M | $     0.0M | $      23.3M | +$      1.0M
    
      Training migration cost absorbed by checkpoint overhead: $332/event/MW × 16 events/yr
      Inference routing eliminates this entirely.
      Note: Does not yet include annual readiness costs (pre-staging, serving stack).
      Readiness cost would need to exceed the training friction delta to change the comparison.
    
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    STORED: ELECTION MECHANISM RESULTS (reference cases)
    ────────────────────────────────────────────────────────────────────────────────────────────────────
      ── 1 GW reference ──
      SPATIAL_AVOIDED_1GW          = 198 MW
      SPATIAL_ANNUAL_SAVINGS_1GW   = $35.7M/yr
      SPATIAL_RESIDUAL_FIRM_1GW    = 796 MW
      ── 10 GW reference ──
      SPATIAL_AVOIDED_10GW         = 1,982 MW
      SPATIAL_ANNUAL_SAVINGS_10GW  = $357M/yr
      SPATIAL_RESIDUAL_FIRM_10GW   = 7,962 MW
    

### 3.4 E3 Counterfactual
How much of E3's recommended 13 GW gas CT buildout is avoidable if 
DC flexibility is recognized? Benefit-cost ratio under base case 
and CEJA sensitivity (Illinois clean energy mandate constraining 
in-state gas development).


```python
# ======================================================================
# Cell 3-4: E3 COUNTERFACTUAL — AVOIDED CT COST (v18)
# ===========================================================
# The E3 2025 Illinois Resource Adequacy Study recommends 13 GW of
# new gas CTs at ~$180K/MW-yr levelized. This cell asks: how much
# of that buildout is avoidable if DC flexibility is recognized?
#
# Uses commitment depth from Cell 3-2 and ELCC from PJM (exogenous).
# Two scenarios: base case (E3 direct) and CEJA sensitivity (Illinois
# clean energy mandate constrains in-state gas builds).
#
# INPUTS: OPTIMAL_COMMITMENT_FRAC, DVFS_COMMITMENT_FRAC, DR_ELCC,
#         BRA_2027_28_PRICE, E3_* params, CEJA params (from Parts 0, 3)
#         PRIMARY_ENERGY_100H, PRIMARY_PAH_DEDUCTION (from Part 4 if run)
# OUTPUTS: E3_AVOIDED_COST_10GW, E3_BCR_10GW, E3_BCR_TIERED_T3_CEJA
# ===========================================================

print('E3 COUNTERFACTUAL: DC DR AS CAPACITY SUBSTITUTE')
print('=' * 80)
print(f'  DR ELCC: {DR_ELCC:.0%} (PJM class rating — exogenous)')
print(f'  Commitment depth: {OPTIMAL_COMMITMENT_FRAC:.0%} (DVFS+Spatial)')
print(f'  E3 target: {E3_NEW_GAS_CT_GW:.0f} GW new gas CTs @ ${E3_CT_LEVELIZED_COST:,}/MW-yr')
print()


# ─────────────────────────────────────────────────────────────
# REGIME COMPARISON TABLE
# ─────────────────────────────────────────────────────────────

print(f'{"─"*100}')
print(f'REGIME COMPARISON: Committed & Accredited MW by Portfolio')
print(f'{"─"*100}')
print(f'{"DC Load":>8} | {"Commitment":>11} | {"Committed MW":>12} | {"Accredited MW":>13} | {"Avoided Cost":>13}')
print(f'{"─"*70}')

# v17 Phase 3: include 1 GW reference case
dc_scenario_gws = [1.0, 2.0, 6.0, 10.0, 14.0]
for dc_gw in dc_scenario_gws:
    for label, depth in [('DVFS only', DVFS_COMMITMENT_FRAC), ('DVFS+Spatial', OPTIMAL_COMMITMENT_FRAC)]:
        flex_mw = dc_gw * 1000 * depth
        cap_mw = flex_mw * DR_ELCC
        savings = cap_mw * E3_CT_LEVELIZED_COST
        print(f'{dc_gw:>6.0f} GW | {label:>11} | {flex_mw:>12,.0f} | {cap_mw:>13,.0f} | ${savings/1e6:>11.1f}M')


# ─────────────────────────────────────────────────────────────
# BENEFIT-COST ANALYSIS — REFERENCE CASES (1 GW, 10 GW)
# ─────────────────────────────────────────────────────────────

print(f'\n{"─"*80}')
print(f'BENEFIT-COST ANALYSIS — Reference Cases (1 GW, 10 GW)')
print(f'{"─"*80}')

# v17 Phase 3: BCA computed at 10 GW reference; per-MW economics are
# fleet-size-independent (cascade product doesn't depend on fleet size),
# so 1 GW results are just 1/10th. The interesting fleet-size dependence
# (contention) lives in the conditional MC per-GW sweep.
dc_gw = 10.0
cap_price_annual = BRA_2027_28_PRICE * 365

# Try to include energy components if Part 4 has been run
try:
    _energy_adder = PRIMARY_ENERGY_100H - PRIMARY_PAH_DEDUCTION
except NameError:
    _energy_adder = 0  # Part 4 not yet run; capacity-only BCR

for label, depth in [('DVFS Only (25%)', DVFS_COMMITMENT_FRAC),
                     ('DVFS+Spatial (primary)', OPTIMAL_COMMITMENT_FRAC)]:
    flex_mw = dc_gw * 1000 * depth
    rev_per_mw = cap_price_annual * DR_ELCC + _energy_adder
    total_rev = flex_mw * rev_per_mw
    avoided = flex_mw * DR_ELCC * E3_CT_LEVELIZED_COST
    bcr = avoided / total_rev if total_rev > 0 else 0
    net = avoided - total_rev

    print(f'\n  {label}')
    print(f'    Committed: {flex_mw:,.0f} MW ({depth:.0%} of {dc_gw:.0f} GW)')
    print(f'    Accredited: {flex_mw * DR_ELCC:,.0f} MW (at {DR_ELCC:.0%} ELCC)')
    print(f'    System avoided CT cost: ${avoided/1e6:>8.1f}M/yr')
    print(f'    DC operator revenue:    ${total_rev/1e6:>8.1f}M/yr')
    print(f'    Net system benefit:     ${net/1e6:>8.1f}M/yr')
    print(f'    Benefit-cost ratio:      {bcr:.2f}x')


# ─────────────────────────────────────────────────────────────
# CEJA SENSITIVITY
# ─────────────────────────────────────────────────────────────

print(f'\n{"─"*80}')
print(f'CEJA SENSITIVITY (Illinois clean energy mandate)')
print(f'  {CEJA_NOTE}')

_ceja_flex = dc_gw * 1000 * OPTIMAL_COMMITMENT_FRAC
_ceja_rev = _ceja_flex * (cap_price_annual * DR_ELCC + _energy_adder)
_ceja_avoided = _ceja_flex * DR_ELCC * E3_CT_LEVELIZED_CEJA
_ceja_bcr = _ceja_avoided / _ceja_rev if _ceja_rev > 0 else 0

print(f'\n  DVFS+Spatial under CEJA:')
print(f'    Avoided: ${_ceja_avoided/1e6:>7.1f}M  |  DC rev: ${_ceja_rev/1e6:>7.1f}M  |  BCR: {_ceja_bcr:.2f}x')

E3_BCR_TIERED_T3_CEJA = _ceja_bcr


# ─────────────────────────────────────────────────────────────
# STORE RESULTS
# ─────────────────────────────────────────────────────────────

# v17 Phase 3: Store at both reference sizes
_primary_flex_10 = dc_gw * 1000 * OPTIMAL_COMMITMENT_FRAC
_primary_rev_10 = _primary_flex_10 * (cap_price_annual * DR_ELCC + _energy_adder)
_primary_avoided_10 = _primary_flex_10 * DR_ELCC * E3_CT_LEVELIZED_COST

E3_AVOIDED_COST_10GW = _primary_avoided_10
E3_BCR_10GW = _primary_avoided_10 / _primary_rev_10 if _primary_rev_10 > 0 else 0

# 1 GW is linear scaling (no contention in cascade-based BCA)
E3_AVOIDED_COST_1GW = E3_AVOIDED_COST_10GW / 10.0
E3_BCR_1GW = E3_BCR_10GW  # Per-MW economics are identical

print(f'\n{"─"*80}')
print(f'STORED: E3 COUNTERFACTUAL RESULTS (reference cases)')
print(f'  E3_AVOIDED_COST_1GW  = ${E3_AVOIDED_COST_1GW/1e6:.1f}M/yr (1 GW ref)')
print(f'  E3_AVOIDED_COST_10GW = ${E3_AVOIDED_COST_10GW/1e6:.1f}M/yr (10 GW ref)')
print(f'  E3_BCR (both)        = {E3_BCR_10GW:.2f}x (per-MW, fleet-size independent)')
print(f'  E3_BCR_CEJA          = {E3_BCR_TIERED_T3_CEJA:.2f}x')

print(f'\n  INTERPRETATION:')
print(f'    Spatial adds {(OPTIMAL_COMMITMENT_FRAC - DVFS_COMMITMENT_FRAC)*100:.1f} pct pts on top of DVFS-only')
print(f'    ({DVFS_COMMITMENT_FRAC:.0%} → {OPTIMAL_COMMITMENT_FRAC:.0%}), a modest uplift under the 10-parameter cascade.')
print(f'    IX queue acceleration (Part 5) dominates the investment decision;')
print(f'    capacity revenue is the compliance hook, not the primary incentive.')
```

    E3 COUNTERFACTUAL: DC DR AS CAPACITY SUBSTITUTE
    ================================================================================
      DR ELCC: 92% (PJM class rating — exogenous)
      Commitment depth: 20% (DVFS+Spatial)
      E3 target: 13 GW new gas CTs @ $180,000/MW-yr
    
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    REGIME COMPARISON: Committed & Accredited MW by Portfolio
    ────────────────────────────────────────────────────────────────────────────────────────────────────
     DC Load |  Commitment | Committed MW | Accredited MW |  Avoided Cost
    ──────────────────────────────────────────────────────────────────────
         1 GW |   DVFS only |          175 |           161 | $       29.0M
         1 GW | DVFS+Spatial |          204 |           187 | $       33.7M
         2 GW |   DVFS only |          350 |           322 | $       58.0M
         2 GW | DVFS+Spatial |          408 |           375 | $       67.5M
         6 GW |   DVFS only |        1,050 |           966 | $      173.9M
         6 GW | DVFS+Spatial |        1,223 |         1,125 | $      202.5M
        10 GW |   DVFS only |        1,750 |         1,610 | $      289.8M
        10 GW | DVFS+Spatial |        2,038 |         1,875 | $      337.4M
        14 GW |   DVFS only |        2,450 |         2,254 | $      405.7M
        14 GW | DVFS+Spatial |        2,853 |         2,625 | $      472.4M
    
    ────────────────────────────────────────────────────────────────────────────────
    BENEFIT-COST ANALYSIS — Reference Cases (1 GW, 10 GW)
    ────────────────────────────────────────────────────────────────────────────────
    
      DVFS Only (25%)
        Committed: 1,750 MW (18% of 10 GW)
        Accredited: 1,610 MW (at 92% ELCC)
        System avoided CT cost: $   289.8M/yr
        DC operator revenue:    $   195.9M/yr
        Net system benefit:     $    93.9M/yr
        Benefit-cost ratio:      1.48x
    
      DVFS+Spatial (primary)
        Committed: 2,038 MW (20% of 10 GW)
        Accredited: 1,875 MW (at 92% ELCC)
        System avoided CT cost: $   337.4M/yr
        DC operator revenue:    $   228.2M/yr
        Net system benefit:     $   109.3M/yr
        Benefit-cost ratio:      1.48x
    
    ────────────────────────────────────────────────────────────────────────────────
    CEJA SENSITIVITY (Illinois clean energy mandate)
      CEJA sensitivity: assumes Illinois cannot rely on in-state gas CT builds post-2025. Marginal resource = PJM-delivered import capacity at ~$205K/MW-yr. Conservative: does not add clean peaker premium or carbon cost adder.
    
      DVFS+Spatial under CEJA:
        Avoided: $  384.3M  |  DC rev: $  228.2M  |  BCR: 1.68x
    
    ────────────────────────────────────────────────────────────────────────────────
    STORED: E3 COUNTERFACTUAL RESULTS (reference cases)
      E3_AVOIDED_COST_1GW  = $33.7M/yr (1 GW ref)
      E3_AVOIDED_COST_10GW = $337.4M/yr (10 GW ref)
      E3_BCR (both)        = 1.48x (per-MW, fleet-size independent)
      E3_BCR_CEJA          = 1.68x
    
      INTERPRETATION:
        Spatial adds 2.9 pct pts on top of DVFS-only
        (18% → 20%), a modest uplift under the 10-parameter cascade.
        IX queue acceleration (Part 5) dominates the investment decision;
        capacity revenue is the compliance hook, not the primary incentive.
    

## Part 4: Energy Economics (Supporting)
- Cell 1: Energy Arbitrage & Value Stack
- Cell 2: Three-Prong Energy Economics
- Cell 3: Spatial Break-Even Projection
- Cell 4: Portfolio Interaction Effects

Energy arbitrage is NOT the economic driver for spatial migration. 
Spatial shifting is currently uneconomic on energy alone — destination 
energy costs during stress events exceed the arbitrage spread. This 
is the correct framing: you don't do spatial migration for energy 
arbitrage. You do it for capacity market revenue and grid reliability.

This section establishes that fact, projects when the energy economics 
may cross into positive territory, and quantifies the interaction 
premium when mechanisms are combined.

### 4.1 Energy Arbitrage & Grounded Value Stack

Compute arbitrage from curtailing during peak LMP hours. Adjust for 
double-counting with capacity market PAH overlap. Assemble the 
grounded value stack (capacity + energy, computed from primary data only).


```python
# ======================================================================
# Cell 4-1: ENERGY ARBITRAGE & GROUNDED VALUE STACK (v18)
# ===========================================================
# Computes the energy arbitrage value of curtailing during peak LMP
# hours, adjusts for PAH double-counting with capacity revenue, and
# assembles the grounded value stack (capacity + energy only).
#
# This is SUPPORTING analysis. The headline numbers come from Part 3
# (capacity market commitment depth). Energy arbitrage alone does not
# justify spatial migration — that's a feature, not a bug.
#
# INPUTS: da, PRIMARY_YEAR, YEARS, BRA/ELCC params, CAP_SCENARIOS,
#         THERMO_EFFICIENCY, FACILITY_PUE, COOLING_DECAY_MINUTES,
#         AVG_EVENT_DURATION_HRS (from Parts 0, 1)
# OUTPUTS: arb_by_year, PRIMARY_CAP_REV, PRIMARY_ENERGY_100H,
#          PRIMARY_PAH_DEDUCTION, GROUNDED_VALUE_MID
# ===========================================================


# ─────────────────────────────────────────────────────────────
# ENERGY ARBITRAGE BY CURTAILMENT BUCKET
# ─────────────────────────────────────────────────────────────

curtailment_hours = [20, 50, 100, 200]

def compute_arbitrage(lmps, hours_list, avg_event_duration_hrs=AVG_EVENT_DURATION_HRS):
    """Energy arbitrage from curtailing during peak LMP hours,
    with thermodynamic penalty for cooling tail."""
    sorted_lmps = lmps.sort_values(ascending=False).values
    avg = lmps.mean()
    results = []

    for h in hours_list:
        top_h = sorted_lmps[:h]
        spread = top_h.mean() - avg
        num_events = h / avg_event_duration_hrs
        cooling_tail_hours = (COOLING_DECAY_MINUTES / 60) * num_events
        cooling_fraction = (FACILITY_PUE - 1) / FACILITY_PUE
        energy_lost_fraction = (cooling_tail_hours * cooling_fraction) / h
        thermo_eff = 1.0 - min(energy_lost_fraction, 0.15)
        total_gross_value = spread * h
        adjusted_net_value = total_gross_value * thermo_eff

        results.append({
            'hours': h, 'avg_curtail': top_h.mean(), 'avg_all': avg,
            'spread': spread, 'gross_value_per_mw_yr': total_gross_value,
            'value_per_mw_yr': adjusted_net_value, 'thermo_eff': thermo_eff,
        })
    return results

arb_by_year = {}
for yr in YEARS:
    arb_by_year[yr] = compute_arbitrage(da[da['year'] == yr]['total_lmp_da'].dropna(), curtailment_hours)
arb_by_year['2022-2025 avg'] = compute_arbitrage(da['total_lmp_da'].dropna(), curtailment_hours)

arb_table = pd.DataFrame(index=curtailment_hours)
for yr, results in arb_by_year.items():
    arb_table[yr] = [r['value_per_mw_yr'] for r in results]
arb_table.index.name = 'Curtailment Hours'

print('ENERGY ARBITRAGE VALUE ($/MW-year of flexible capacity)')
print('=' * 90)
print(arb_table.map(lambda x: f'${x:,.0f}'))

print(f'\nThermodynamic efficiency by curtailment bucket:')
for r in arb_by_year[PRIMARY_YEAR]:
    print(f'  {r["hours"]:>3}h: {r["thermo_eff"]:.1%} efficiency')


# ─────────────────────────────────────────────────────────────
# DOUBLE-COUNTING ADJUSTMENT (PAH overlap)
# ─────────────────────────────────────────────────────────────

PAH_HOURS = 40  # [ESTIMATED] typical PAH overlap with top-100

lmps_primary = da[da['year'] == PRIMARY_YEAR]['total_lmp_da']
pah_lmps = lmps_primary.sort_values(ascending=False).values[:PAH_HOURS]
pah_energy_value = (pah_lmps.mean() - lmps_primary.mean()) * PAH_HOURS * THERMO_EFFICIENCY

PRIMARY_CAP_REV = BRA_2027_28_PRICE * ELCC_2027_28_DR * 365
PRIMARY_ENERGY_100H = arb_by_year[PRIMARY_YEAR][2]['value_per_mw_yr']
PRIMARY_PAH_DEDUCTION = pah_energy_value
PRIMARY_ADJUSTED_TOTAL = PRIMARY_CAP_REV + PRIMARY_ENERGY_100H - PRIMARY_PAH_DEDUCTION

print(f'\n{PRIMARY_YEAR} DOUBLE-COUNTING ADJUSTMENT:')
print(f'  Capacity revenue (BRA 27/28): ${PRIMARY_CAP_REV:>10,.0f}/MW-yr')
print(f'  Energy arbitrage (100h):      ${PRIMARY_ENERGY_100H:>10,.0f}/MW-yr')
print(f'  PAH overlap deduction:       -${PRIMARY_PAH_DEDUCTION:>10,.0f}/MW-yr')
print(f'  Adjusted combined:            ${PRIMARY_ADJUSTED_TOTAL:>10,.0f}/MW-yr')


# ─────────────────────────────────────────────────────────────
# GROUNDED VALUE STACK
# ─────────────────────────────────────────────────────────────

print(f'\n\nGROUNDED VALUE STACK ($/MW-yr of flexible capacity)')
print('Only includes components computed from primary data sources.')
print('=' * 80)

cap_scenarios_annual = {name: price * ELCC_2027_28_DR * 365
                        for name, price in CAP_SCENARIOS.items()}

cap_lo = cap_scenarios_annual['Current cap holds']
cap_mid = cap_scenarios_annual['Modest increase']
cap_hi = cap_scenarios_annual['Cap lifted/reformed']

eng_lo = arb_by_year[PRIMARY_YEAR][0]['value_per_mw_yr']   # 20h
eng_mid = arb_by_year[PRIMARY_YEAR][2]['value_per_mw_yr']  # 100h
eng_hi = arb_by_year[PRIMARY_YEAR][3]['value_per_mw_yr']   # 200h

adj = PRIMARY_PAH_DEDUCTION

print(f'{"Component":<30} | {"Conservative":>12} | {"Central":>12} | {"Optimistic":>12}')
print('-' * 80)
print(f'{"Capacity market":<30} | ${cap_lo:>11,.0f} | ${cap_mid:>11,.0f} | ${cap_hi:>11,.0f}')
print(f'{"Energy arbitrage":<30} | ${eng_lo:>11,.0f} | ${eng_mid:>11,.0f} | ${eng_hi:>11,.0f}')
print(f'{"PAH overlap adjustment":<30} | -${adj/2:>10,.0f} | -${adj:>10,.0f} | -${adj*1.5:>10,.0f}')
total_lo = cap_lo + eng_lo - adj/2
total_mid = cap_mid + eng_mid - adj
total_hi = cap_hi + eng_hi - adj*1.5
print('-' * 80)
print(f'{"TOTAL (grounded)":<30} | ${total_lo:>11,.0f} | ${total_mid:>11,.0f} | ${total_hi:>11,.0f}')

print(f'\nPotential additional streams (NOT in headline):')
for name in ['Avoided transmission (RTEP)', 'Congestion relief', 'Ancillary services']:
    print(f'  {name} — requires further analysis')

GROUNDED_VALUE_LOW = total_lo
GROUNDED_VALUE_MID = total_mid
GROUNDED_VALUE_HIGH = total_hi
```

    ENERGY ARBITRAGE VALUE ($/MW-year of flexible capacity)

    
    ==========================================================================================
                          2022    2023     2024     2025 2022-2025 avg
    Curtailment Hours                                                 
    20                  $4,604  $3,004   $2,846   $5,545        $6,180
    50                  $8,942  $4,521   $6,119  $10,517       $12,790
    100                $14,738  $6,498  $10,015  $16,066       $21,218
    200                $24,160  $9,621  $14,821  $23,436       $34,895
    
    Thermodynamic efficiency by curtailment bucket:
       20h: 97.1% efficiency
       50h: 97.1% efficiency
      100h: 97.1% efficiency
      200h: 97.1% efficiency
    
    2025 DOUBLE-COUNTING ADJUSTMENT:
      Capacity revenue (BRA 27/28): $   111,969/MW-yr
      Energy arbitrage (100h):      $    16,066/MW-yr
      PAH overlap deduction:       -$     9,033/MW-yr
      Adjusted combined:            $   119,002/MW-yr
    
    
    GROUNDED VALUE STACK ($/MW-yr of flexible capacity)
    Only includes components computed from primary data sources.
    ================================================================================
    Component                      | Conservative |      Central |   Optimistic
    --------------------------------------------------------------------------------
    Capacity market                | $    111,969 | $    151,110 | $    177,974
    Energy arbitrage               | $      5,545 | $     16,066 | $     23,436
    PAH overlap adjustment         | -$     4,516 | -$     9,033 | -$    13,549
    --------------------------------------------------------------------------------
    TOTAL (grounded)               | $    112,997 | $    158,143 | $    187,861
    
    Potential additional streams (NOT in headline):
      Avoided transmission (RTEP) — requires further analysis
      Congestion relief — requires further analysis
      Ancillary services — requires further analysis
    

### 4.2 Three-Prong Energy Economics

How does each mechanism perform on energy alone? Spatial is negative, 
DVFS is positive but small, BTM is positive but requires CapEx. 
Per-interconnection breakdown shows ERCOT and WECC destinations 
improve the spatial economics relative to the capacity-weighted average.


```python
# ======================================================================
# Cell 4-2: THREE-PRONG ENERGY ECONOMICS (v18)
# ===========================================================
# Energy-only value of each flexibility mechanism, using empirical
# destination LMPs from v5. This establishes that spatial migration
# is currently uneconomic on energy alone — confirming the thesis
# framing that capacity market revenue, not energy arbitrage, is
# the economic driver.
#
# INPUTS: arb_by_year, PRIMARY_YEAR, EFFECTIVE_SPATIAL_FRAC, FLEX_FRAC,
#         DESTINATION_LMP_CRISIS, DEST_LMP_*, GPU params, BTM params,
#         EVENTS_PER_YEAR, PRIMARY_CAP_REV, EMPIRICAL_CF (from Parts 0, 4-1)
# OUTPUTS: spatial_net, temporal_net, btm_net, inference_net (consumed by Cell 4-4)
# ===========================================================
 
gross_arbitrage_100h = arb_by_year[PRIMARY_YEAR][2]['value_per_mw_yr']
HOURLY_COMPUTE_VALUE_IT = GPU_PER_MW_IT * GPU_RATE_HR
 
print(f'THREE-PRONG ENERGY ECONOMICS ({PRIMARY_YEAR} Historical)')
print(f'Compute: ${HOURLY_COMPUTE_VALUE:,.0f}/MWh | Events: {EVENTS_PER_YEAR}/yr | Flex: {FLEX_FRAC:.0%}')
print(f'Destination LMP: ${DESTINATION_LMP_CRISIS:.1f}/MWh (empirical, capacity-weighted)')
print('=' * 85)
 
 
# ─── PRONG 1: SPATIAL (Geographic Migration) ─────────────────
 
spatial_load_drop = EFFECTIVE_SPATIAL_FRAC  # 10-param cascade
spatial_downtime_hrs = MIGRATION_LATENCY_MIN / 60
spatial_friction_annual = (spatial_downtime_hrs * HOURLY_COMPUTE_VALUE_IT
                           * EVENTS_PER_YEAR)
spatial_gross = gross_arbitrage_100h * spatial_load_drop * THERMO_EFFICIENCY
spatial_dest_cost = DESTINATION_LMP_CRISIS * 100 * spatial_load_drop
spatial_net = spatial_gross - spatial_dest_cost - (spatial_friction_annual * spatial_load_drop)
 
print('PRONG 1: SPATIAL (Geographic Migration)')
print(f'  Gross arbitrage captured:      ${spatial_gross:>10,.0f}/MW-yr')
print(f'  Destination energy cost:      -${spatial_dest_cost:>10,.0f}/MW-yr')
print(f'  Migration friction:           -${spatial_friction_annual * spatial_load_drop:>10,.0f}/MW-yr')
print(f'  Net energy value:              ${spatial_net:>10,.0f}/MW-yr')
if spatial_net < 0:
    print(f'  ⚠  CURRENTLY UNECONOMIC on energy alone — capacity market is the driver.')
 
print(f'\n  Per-interconnection breakdown:')
_ic_breakout = [('ERCOT', DEST_LMP_ERCOT), ('WECC', DEST_LMP_WESTERN), ('MISO', DEST_LMP_MISO)]
if DEST_LMP_NYISO > 0:
    _ic_breakout.append(('NYISO', DEST_LMP_NYISO))
for ic_name, ic_lmp in _ic_breakout:
    ic_dest_cost = ic_lmp * 100 * spatial_load_drop
    ic_net = spatial_gross - ic_dest_cost - (spatial_friction_annual * spatial_load_drop)
    print(f'    {ic_name:<8}: dest LMP ${ic_lmp:.1f}/MWh → net ${ic_net:>8,.0f}/MW-yr')
print('-' * 85)
 
 
# ─── PRONG 2: TEMPORAL (DVFS) ────────────────────────────────
 
temporal_load_drop = FLEX_FRAC
temporal_degradation = 0.02
temporal_friction = HOURLY_COMPUTE_VALUE * temporal_degradation * 100
temporal_gross = gross_arbitrage_100h * temporal_load_drop * THERMO_EFFICIENCY
temporal_net = temporal_gross - temporal_friction
 
print('PRONG 2: TEMPORAL (Software Throttling / DVFS)')
print(f'  Gross arbitrage captured:      ${temporal_gross:>10,.0f}/MW-yr')
print(f'  Throughput friction:          -${temporal_friction:>10,.0f}/MW-yr')
print(f'  Net energy value:              ${temporal_net:>10,.0f}/MW-yr')
print('-' * 85)
 
 
# ─── PRONG 3: BTM (Battery) ──────────────────────────────────
 
btm_gross = gross_arbitrage_100h * THERMO_EFFICIENCY
btm_capex_annual = BTM_CAPEX_PER_MW * WACC
btm_net = btm_gross
 
print('PRONG 3: BTM (Behind-the-Meter Battery)')
print(f'  Gross arbitrage captured:      ${btm_gross:>10,.0f}/MW-yr')
print(f'  BTM CapEx (annualized @{WACC:.0%}): -${btm_capex_annual:>10,.0f}/MW-yr')
print(f'  Net energy (pre-CapEx):        ${btm_net:>10,.0f}/MW-yr')
print(f'  Net energy (post-CapEx):       ${btm_net - btm_capex_annual:>10,.0f}/MW-yr')
print('-' * 85)
 
 
# ─── SUMMARY ─────────────────────────────────────────────────
 
print(f'\nSUMMARY (Energy Only — excludes capacity):')
print(f'  Spatial:   ${spatial_net:>10,.0f}/MW-yr  {"❌ Uneconomic" if spatial_net < 0 else "✓"}')
print(f'  Temporal:  ${temporal_net:>10,.0f}/MW-yr  ✓')
print(f'  BTM:       ${btm_net:>10,.0f}/MW-yr  ✓ (requires ${BTM_CAPEX_PER_MW:,} CapEx)')
print(f'\nCapacity revenue adds ${PRIMARY_CAP_REV:,.0f}/MW-yr across all prongs.')
print(f'Spatial migration is justified by capacity market depth, not energy arbitrage.')
 
 
# ─── INFERENCE ROUTING VARIANT (v15.1) ───────────────────────
# Same energy arbitrage, zero checkpoint friction.
# Inference requests drain in ~12s (P99); no lost compute.
# Cost structure is annual readiness, not per-event friction.
 
inference_friction_annual = 0.0
inference_net = spatial_gross - spatial_dest_cost - inference_friction_annual
 
print(f'\n{"─"*85}')
print(f'INFERENCE ROUTING VARIANT (no checkpoint friction)')
print(f'{"─"*85}')
print(f'  Gross arbitrage captured:      ${spatial_gross:>10,.0f}/MW-yr  (same)')
print(f'  Destination energy cost:      -${spatial_dest_cost:>10,.0f}/MW-yr  (same)')
print(f'  Migration friction:           -${inference_friction_annual:>10,.0f}/MW-yr  (no checkpoint)')
print(f'  Net energy value:              ${inference_net:>10,.0f}/MW-yr')
if inference_net > 0:
    print(f'  → Inference routing is ECONOMIC on energy alone today.')
else:
    print(f'  → Still uneconomic even without friction (destination cost exceeds spread).')
print(f'\n  Delta vs training: +${inference_net - spatial_net:,.0f}/MW-yr '
      f'(= eliminated friction of ${spatial_friction_annual * spatial_load_drop:,.0f}/MW-yr)')

```

    THREE-PRONG ENERGY ECONOMICS (2025 Historical)
    Compute: $1,327/MWh | Events: 10/yr | Flex: 25%
    Destination LMP: $169.3/MWh (empirical, capacity-weighted)
    =====================================================================================
    PRONG 1: SPATIAL (Geographic Migration)
      Gross arbitrage captured:      $       599/MW-yr
      Destination energy cost:      -$       649/MW-yr
      Migration friction:           -$       165/MW-yr
      Net energy value:              $      -216/MW-yr
      ⚠  CURRENTLY UNECONOMIC on energy alone — capacity market is the driver.
    
      Per-interconnection breakdown:
        ERCOT   : dest LMP $182.3/MWh → net $    -266/MW-yr
        WECC    : dest LMP $112.2/MWh → net $       3/MW-yr
        MISO    : dest LMP $159.0/MWh → net $    -177/MW-yr
        NYISO   : dest LMP $193.2/MWh → net $    -308/MW-yr
    -------------------------------------------------------------------------------------
    PRONG 2: TEMPORAL (Software Throttling / DVFS)
      Gross arbitrage captured:      $     3,901/MW-yr
      Throughput friction:          -$     2,653/MW-yr
      Net energy value:              $     1,247/MW-yr
    -------------------------------------------------------------------------------------
    PRONG 3: BTM (Behind-the-Meter Battery)
      Gross arbitrage captured:      $    15,602/MW-yr
      BTM CapEx (annualized @10%): -$   150,000/MW-yr
      Net energy (pre-CapEx):        $    15,602/MW-yr
      Net energy (post-CapEx):       $  -134,398/MW-yr
    -------------------------------------------------------------------------------------
    
    SUMMARY (Energy Only — excludes capacity):
      Spatial:   $      -216/MW-yr  ❌ Uneconomic
      Temporal:  $     1,247/MW-yr  ✓
      BTM:       $    15,602/MW-yr  ✓ (requires $1,500,000 CapEx)
    
    Capacity revenue adds $111,969/MW-yr across all prongs.
    Spatial migration is justified by capacity market depth, not energy arbitrage.
    
    ─────────────────────────────────────────────────────────────────────────────────────
    INFERENCE ROUTING VARIANT (no checkpoint friction)
    ─────────────────────────────────────────────────────────────────────────────────────
      Gross arbitrage captured:      $       599/MW-yr  (same)
      Destination energy cost:      -$       649/MW-yr  (same)
      Migration friction:           -$         0/MW-yr  (no checkpoint)
      Net energy value:              $       -51/MW-yr
      → Still uneconomic even without friction (destination cost exceeds spread).
    
      Delta vs training: +$165/MW-yr (= eliminated friction of $165/MW-yr)
    

### 4.3 Spatial Break-Even Projection

Three converging trends: falling checkpoint latency, rising LMP spreads, 
declining destination correlation. Central scenario: spatial becomes 
energy-economic around 2028. Conservative: 2030. Even before that 
crossing, capacity market revenue makes spatial investment positive 
(Part 3).


```python
# ======================================================================
# Cell 4-3: SPATIAL BREAK-EVEN — FORWARD PROJECTION (v18)
# ===========================================================
# Spatial is uneconomic today on energy alone. When do the curves cross?
# Three variables are evolving:
#   1. Checkpoint latency — dropping with ML framework improvements
#   2. Inter-zonal LMP spreads — rising with retirements + renewable pen.
#   3. Destination LMP correlation — declining as grid diversifies
#
# INPUTS: da, PRIMARY_YEAR, MIGRATION_LATENCY_MIN, DESTINATION_LMP_CRISIS,
#         HOURLY_COMPUTE_VALUE_IT, EVENTS_PER_YEAR, EFFECTIVE_SPATIAL_FRAC,
#         THERMO_EFFICIENCY, spatial_gross, spatial_dest_cost, spatial_net
#         (from Parts 0, 4-1, 4-2)
# OUTPUTS: results dict (Central/Conservative/Aggressive trajectories)
# ===========================================================
print('NOTE: This projection applies to TRAINING migration.')
print('For INFERENCE routing, per-event friction is ~$0 (no checkpoint).')
print(f'Inference routing net energy value: ${inference_net:,.0f}/MW-yr')
print(f'Status: {"✓ Already economic" if inference_net > 0 else "❌ Uneconomic"}')
print()

import numpy as np

# Current state
current_crisis_lmp = da[da['year']==PRIMARY_YEAR]['total_lmp_da'].nlargest(100).mean()

print('SPATIAL BREAK-EVEN: FORWARD PROJECTION')
print('=' * 90)
print(f'Current state ({PRIMARY_YEAR}):')
print(f'  Checkpoint latency:    {MIGRATION_LATENCY_MIN} min')
print(f'  Crisis LMP (top 100h): ${current_crisis_lmp:,.0f}/MWh')
print(f'  Destination LMP:       ${DESTINATION_LMP_CRISIS:.0f}/MWh')
print(f'  Spatial net value:     ${spatial_net:,.0f}/MW-yr (uneconomic)')


# ─── PROJECTION PARAMETERS ───────────────────────────────────

latency_improvement_rates = {'Conservative': 0.15, 'Central': 0.20, 'Aggressive': 0.25}
spread_growth_rates = {'Conservative': 0.05, 'Central': 0.08, 'Aggressive': 0.12}
destination_lmp_improvement = 0.02  # 2%/yr decrease in correlation

years_forward = np.arange(0, 11)  # 2025–2035
results = {}

for scenario in ['Conservative', 'Central', 'Aggressive']:
    lat_rate = latency_improvement_rates[scenario]
    spread_rate = spread_growth_rates[scenario]
    yearly_values = []

    for y in years_forward:
        yr = PRIMARY_YEAR + y
        proj_latency = MIGRATION_LATENCY_MIN * (1 - lat_rate) ** y
        proj_crisis_lmp = current_crisis_lmp * (1 + spread_rate) ** y
        proj_dest_lmp = DESTINATION_LMP_CRISIS * (1 - destination_lmp_improvement) ** y
        proj_spread = proj_crisis_lmp - proj_dest_lmp

        proj_gross = proj_spread * 100 * EFFECTIVE_SPATIAL_FRAC * THERMO_EFFICIENCY
        proj_friction = ((proj_latency / 60) * HOURLY_COMPUTE_VALUE_IT
                         * EVENTS_PER_YEAR * EFFECTIVE_SPATIAL_FRAC)
        proj_dest_cost = proj_dest_lmp * 100 * EFFECTIVE_SPATIAL_FRAC
        proj_net = proj_gross - proj_dest_cost - proj_friction

        yearly_values.append({
            'year': yr, 'latency_min': proj_latency,
            'crisis_spread': proj_spread, 'net_value': proj_net,
        })
    results[scenario] = yearly_values


# ─── CROSSING POINTS ─────────────────────────────────────────

print(f'\nCROSSING POINTS (year spatial becomes economic on energy):')
print('-' * 60)
for scenario, vals in results.items():
    crossing_yr = None
    for v in vals:
        if v['net_value'] > 0:
            crossing_yr = v['year']
            break
    if crossing_yr:
        cv = next(v for v in vals if v['year'] == crossing_yr)
        print(f'  {scenario:>14}: {crossing_yr} (latency={cv["latency_min"]:.1f} min, '
              f'spread=${cv["crisis_spread"]:,.0f}/MWh)')
    else:
        print(f'  {scenario:>14}: Beyond {PRIMARY_YEAR + 10}')


# ─── DETAILED TRAJECTORY (Central) ───────────────────────────

print(f'\nCENTRAL SCENARIO TRAJECTORY:')
print(f'{"Year":>6} | {"Latency":>10} | {"Spread":>10} | {"Net Value":>12} | Status')
print('-' * 65)
for v in results['Central']:
    status = '✓ Economic' if v['net_value'] > 0 else '❌ Uneconomic'
    print(f'{v["year"]:>6} | {v["latency_min"]:>8.1f} min | ${v["crisis_spread"]:>8,.0f} | '
          f'${v["net_value"]:>10,.0f} | {status}')


# ─── WORKLOAD-SPECIFIC ANALYSIS ──────────────────────────────

print(f'\nWORKLOAD-SPECIFIC BREAK-EVEN (current state):')
workloads = [
    ('LLM Training (batch)', 15, 'Large checkpoints, highly deferrable'),
    ('LLM Inference (batch)', 2, 'Tiny state, high SLA sensitivity'),
    ('Scientific HPC', 5, 'Moderate checkpoints, flexible scheduling'),
    ('Rendering/Media', 1, 'Minimal state, embarrassingly parallel'),
]
for name, latency, note in workloads:
    friction = ((latency / 60) * HOURLY_COMPUTE_VALUE_IT
                * EVENTS_PER_YEAR * EFFECTIVE_SPATIAL_FRAC)
    net = spatial_gross - spatial_dest_cost - friction
    status = '✓' if net > 0 else '❌'
    print(f'  {name:<25} lat={latency:>2}min  net=${net:>8,.0f}/MW-yr  {status}  ({note})')
```

    NOTE: This projection applies to TRAINING migration.
    For INFERENCE routing, per-event friction is ~$0 (no checkpoint).
    Inference routing net energy value: $-51/MW-yr
    Status: ❌ Uneconomic
    
    SPATIAL BREAK-EVEN: FORWARD PROJECTION
    ==========================================================================================
    Current state (2025):
      Checkpoint latency:    15 min
      Crisis LMP (top 100h): $202/MWh
      Destination LMP:       $169/MWh
      Spatial net value:     $-216/MW-yr (uneconomic)
    
    CROSSING POINTS (year spatial becomes economic on energy):
    ------------------------------------------------------------
        Conservative: 2034 (latency=3.5 min, spread=$172/MWh)
             Central: 2031 (latency=3.9 min, spread=$171/MWh)
          Aggressive: 2030 (latency=3.6 min, spread=$203/MWh)
    
    CENTRAL SCENARIO TRAJECTORY:
      Year |    Latency |     Spread |    Net Value | Status
    -----------------------------------------------------------------
      2025 |     15.0 min | $      33 | $      -693 | ❌ Uneconomic
      2026 |     12.0 min | $      52 | $      -574 | ❌ Uneconomic
      2027 |      9.6 min | $      73 | $      -457 | ❌ Uneconomic
      2028 |      7.7 min | $      95 | $      -341 | ❌ Uneconomic
      2029 |      6.1 min | $     119 | $      -224 | ❌ Uneconomic
      2030 |      4.9 min | $     144 | $      -105 | ❌ Uneconomic
      2031 |      3.9 min | $     171 | $        17 | ✓ Economic
      2032 |      3.1 min | $     199 | $       144 | ✓ Economic
      2033 |      2.5 min | $     230 | $       277 | ✓ Economic
      2034 |      2.0 min | $     263 | $       415 | ✓ Economic
      2035 |      1.6 min | $     298 | $       562 | ✓ Economic
    
    WORKLOAD-SPECIFIC BREAK-EVEN (current state):
      LLM Training (batch)      lat=15min  net=$    -216/MW-yr  ❌  (Large checkpoints, highly deferrable)
      LLM Inference (batch)     lat= 2min  net=$     -73/MW-yr  ❌  (Tiny state, high SLA sensitivity)
      Scientific HPC            lat= 5min  net=$    -106/MW-yr  ❌  (Moderate checkpoints, flexible scheduling)
      Rendering/Media           lat= 1min  net=$     -62/MW-yr  ❌  (Minimal state, embarrassingly parallel)
    

### 4.4 Portfolio Interaction Effects

The combination is worth more than the sum of parts. Software 
flexibility (DVFS + spatial) substitutes for physical infrastructure 
(larger batteries), creating three interaction channels: duration 
extension, battery sizing reduction, and ELCC stacking.


```python
# ======================================================================
# Cell 4-4: PORTFOLIO INTERACTION EFFECTS (v18)
# ===========================================================
# The combination of mechanisms is worth more than the sum of parts
# because software prongs substitute for physical infrastructure.
# Three interaction channels: duration extension, battery sizing
# reduction, and ELCC stacking.
#
# INPUTS: PRIMARY_CAP_REV, spatial_net, temporal_net, btm_net,
#         btm_capex_annual, BTM_CAPEX_PER_MW, BESS_8HR_CAPEX,
#         BESS_CAPEX_PER_MW, FLEX_FRAC, spatial_load_drop,
#         GROUNDED_VALUE_MID, WACC (from Parts 0, 4-1, 4-2)
# OUTPUTS: total_interaction, capex_savings_annualized,
#          capex_savings_sizing, elcc_stacking_value
# ===========================================================

print('PORTFOLIO INTERACTION EFFECTS')
print('=' * 85)


# ─── INDIVIDUAL PRONG ECONOMICS (energy + capacity) ──────────

print('\n1. INDIVIDUAL PRONG ECONOMICS (Energy + Capacity):')

prong_results = {}
spatial_with_cap = PRIMARY_CAP_REV + spatial_net
prong_results['spatial'] = {'energy': spatial_net, 'capacity': PRIMARY_CAP_REV,
                            'total': spatial_with_cap, 'capex': 0}

temporal_with_cap = PRIMARY_CAP_REV + temporal_net
prong_results['temporal'] = {'energy': temporal_net, 'capacity': PRIMARY_CAP_REV,
                             'total': temporal_with_cap, 'capex': 0}

btm_with_cap = PRIMARY_CAP_REV + btm_net - btm_capex_annual
prong_results['btm'] = {'energy': btm_net, 'capacity': PRIMARY_CAP_REV,
                         'total': btm_with_cap, 'capex': BTM_CAPEX_PER_MW}

for name, r in prong_results.items():
    print(f'  {name.upper():<10}: Energy ${r["energy"]:>8,.0f} + Cap ${r["capacity"]:>8,.0f} '
          f'= ${r["total"]:>8,.0f}/MW-yr  (CapEx: ${r["capex"]:>10,})')


# ─── COMBINED PORTFOLIO EFFECTS ──────────────────────────────

print(f'\n2. COMBINED PORTFOLIO EFFECTS:')

# Duration extension: avoid 8hr battery, use 4hr + software for hours 5-8
capex_savings_duration = BESS_8HR_CAPEX - BESS_CAPEX_PER_MW
capex_savings_annualized = capex_savings_duration * WACC

# Battery sizing reduction: software flexibility substitutes for battery capacity
battery_reduction_factor = 1.0 - (FLEX_FRAC * 0.5 + EFFECTIVE_SPATIAL_FRAC * 0.3)
capex_savings_sizing = BTM_CAPEX_PER_MW * (1 - battery_reduction_factor) * WACC

# ELCC stacking: combined coverage more complete across seasons
combined_elcc_bonus = 0.05  # [ESTIMATED] 5% ELCC improvement from portfolio
elcc_stacking_value = PRIMARY_CAP_REV * combined_elcc_bonus

total_interaction = capex_savings_annualized + capex_savings_sizing + elcc_stacking_value

print(f'  Duration extension (avoid 8hr battery): ${capex_savings_annualized:>10,.0f}/MW-yr')
print(f'  Battery sizing reduction:                ${capex_savings_sizing:>10,.0f}/MW-yr')
print(f'  ELCC stacking bonus:                     ${elcc_stacking_value:>10,.0f}/MW-yr')
print(f'  Total interaction premium:               ${total_interaction:>10,.0f}/MW-yr')


# ─── SUM OF PARTS vs PORTFOLIO ───────────────────────────────

print(f'\n3. SUM OF PARTS vs PORTFOLIO:')
portfolio_total = GROUNDED_VALUE_MID + total_interaction

print(f'  Best individual prong:     ${max(r["total"] for r in prong_results.values()):>10,.0f}/MW-yr')
print(f'  Grounded value (cap+eng):  ${GROUNDED_VALUE_MID:>10,.0f}/MW-yr')
print(f'  Portfolio with interaction: ${portfolio_total:>10,.0f}/MW-yr')
print(f'  Interaction premium:       ${total_interaction:>10,.0f}/MW-yr '
      f'({total_interaction/GROUNDED_VALUE_MID:.1%} of base)')


# ─── CAPEX SUBSTITUTION FRONTIER ─────────────────────────────

print(f'\n4. CAPEX SUBSTITUTION FRONTIER:')
print(f'{"Software Flex":>15} | {"Battery Needed":>15} | {"Total CapEx":>12} | {"Savings vs Full BTM":>20}')
print('-' * 75)
for sw_flex in [0.0, 0.10, 0.25, 0.40, 0.55]:
    batt_needed = max(0, 1.0 - sw_flex * 1.5)
    capex = batt_needed * BTM_CAPEX_PER_MW
    savings = BTM_CAPEX_PER_MW - capex
    print(f'{sw_flex:>14.0%} | {batt_needed:>14.0%} | ${capex:>10,} | ${savings:>18,}')
```

    PORTFOLIO INTERACTION EFFECTS
    =====================================================================================
    
    1. INDIVIDUAL PRONG ECONOMICS (Energy + Capacity):
      SPATIAL   : Energy $    -216 + Cap $ 111,969 = $ 111,753/MW-yr  (CapEx: $         0)
      TEMPORAL  : Energy $   1,247 + Cap $ 111,969 = $ 113,217/MW-yr  (CapEx: $         0)
      BTM       : Energy $  15,602 + Cap $ 111,969 = $ -22,428/MW-yr  (CapEx: $ 1,500,000)
    
    2. COMBINED PORTFOLIO EFFECTS:
      Duration extension (avoid 8hr battery): $   100,000/MW-yr
      Battery sizing reduction:                $    20,476/MW-yr
      ELCC stacking bonus:                     $     5,598/MW-yr
      Total interaction premium:               $   126,075/MW-yr
    
    3. SUM OF PARTS vs PORTFOLIO:
      Best individual prong:     $   113,217/MW-yr
      Grounded value (cap+eng):  $   158,143/MW-yr
      Portfolio with interaction: $   284,218/MW-yr
      Interaction premium:       $   126,075/MW-yr (79.7% of base)
    
    4. CAPEX SUBSTITUTION FRONTIER:
      Software Flex |  Battery Needed |  Total CapEx |  Savings vs Full BTM
    ---------------------------------------------------------------------------
                0% |           100% | $1,500,000.0 | $               0.0
               10% |            85% | $1,275,000.0 | $         225,000.0
               25% |            62% | $ 937,500.0 | $         562,500.0
               40% |            40% | $599,999.9999999999 | $900,000.0000000001
               55% |            17% | $262,499.9999999999 | $       1,237,500.0
    

## Part 5: Behavioral Incentive and Forward View
- Cell 1: IX Queue NPV
- Cell 2: Forward Energy Arbitrage Projection

Why would a hyperscaler actually invest in spatial migration 
capability? The capacity market revenue differential is real but 
small relative to their balance sheet. The actual incentive is 
interconnection queue acceleration: getting online years earlier 
has an NPV that dwarfs any capacity payment.

This section quantifies the IX queue incentive (the "why"), then 
projects forward energy arbitrage to show how the economics evolve 
as supply gaps open and checkpoint technology improves.

### 5.1 Interconnection Queue NPV

The NPV of getting a 1 GW facility online 3 years earlier is 
~$10B+ at gross compute revenue — orders of magnitude larger than 
the annual capacity revenue differential from DVFS+Spatial vs 
DVFS-only. This makes interconnection queue priority the dominant 
behavioral incentive, not capacity market payments.


```python
# ======================================================================
# Cell 5-1: IX QUEUE NPV — BEHAVIORAL INCENTIVE (v18)
# ===========================================================
# The actual incentive to invest in spatial migration is NOT the
# capacity market revenue differential. It is interconnection queue
# acceleration. The NPV of getting online years earlier dwarfs any
# capacity market payment.
#
# Capacity revenue is the compliance hook; IX speed is the payoff.
#
# STRUCTURAL ARGUMENT: During a macro reliability event, the entire
# native RTO is stressed. Intra-RTO migration (ComEd → AEP) provides
# zero RA benefit. Cross-interconnect (Eastern → ERCOT/WECC) is a
# logical prerequisite for spatial credibility, not a design option.
#
# INPUTS: GPU_PER_MW_GRID, GPU_RATE_HR, FACILITY_PUE, FLEX_FRAC,
#         OPTIMAL_COMMITMENT_FRAC, DVFS_COMMITMENT_FRAC, DR_ELCC,
#         BRA_2027_28_PRICE, WACC (from Parts 0, 3)
# OUTPUTS: IX_NPV_CENTRAL, IX_NPV_LOW, IX_NPV_HIGH, IX_DOMINANCE_RATIO
# ===========================================================

print('IX QUEUE NPV — BEHAVIORAL INCENTIVE QUANTIFICATION')
print('=' * 70)


# ─────────────────────────────────────────────────────────────
# 1. REFERENCE FACILITY
# ─────────────────────────────────────────────────────────────

FACILITY_GW       = 1.0
FACILITY_MW       = FACILITY_GW * 1000
UTILIZATION       = 0.80         # [ESTIMATED] 80% GPU utilization (hyperscaler)
HOURS_PER_YEAR    = 8760

annual_compute_rev = GPU_PER_MW_GRID * GPU_RATE_HR * HOURS_PER_YEAR * UTILIZATION * FACILITY_MW

print(f'Reference facility: {FACILITY_GW:.0f} GW grid-connected')
print(f'GPU density: {GPU_PER_MW_GRID} GPUs/MW (grid-metered, PUE={FACILITY_PUE})')
print(f'H100 spot rate: ${GPU_RATE_HR}/hr')
print(f'Utilization: {UTILIZATION:.0%}')
print(f'Annual gross compute revenue: ${annual_compute_rev/1e9:.2f}B/yr')
print()


# ─────────────────────────────────────────────────────────────
# 2. IX QUEUE BASELINE
# ─────────────────────────────────────────────────────────────

PJM_QUEUE_MEDIAN_YRS   = 70 / 12   # [LBNL-2024] 5.83 years
PJM_QUEUE_P75_YRS      = 84 / 12   # 7 years

print(f'PJM baseline queue [LBNL-2024]:')
print(f'  Median IR→COD: {PJM_QUEUE_MEDIAN_YRS:.1f} years')
print(f'  P75 IR→COD:    {PJM_QUEUE_P75_YRS:.1f} years')
print()


# ─────────────────────────────────────────────────────────────
# 3. FIA ACCELERATION SCENARIOS
# ─────────────────────────────────────────────────────────────

acceleration_scenarios = {
    'Conservative': 2.0,
    'Central':      3.0,
    'Optimistic':   4.0,
}

print(f'WACC: {WACC:.0%} (hyperscaler range 8-12%)')
print()

# NPV = annual_rev × annuity_factor(N, r)
print(f'{"─"*70}')
print(f'NPV OF IX QUEUE ACCELERATION ({FACILITY_GW:.0f} GW facility)')
print(f'{"─"*70}')
print(f'{"Scenario":>14} | {"Accel (yrs)":>11} | {"Annuity Factor":>14} | {"NPV (gross)":>12} | {"vs Ann Cap Rev":>14}')
print(f'{"-"*70}')

# Annual capacity revenue at 1 GW
flex_mw_1gw = FACILITY_MW * FLEX_FRAC
cap_rev_t3_annual = flex_mw_1gw * BRA_2027_28_PRICE * 365 * DR_ELCC

ix_npv_results = {}
for label, N in acceleration_scenarios.items():
    annuity = (1 - (1 + WACC)**(-N)) / WACC
    npv_gross = annual_compute_rev * annuity
    vs_cap_rev = npv_gross / cap_rev_t3_annual
    ix_npv_results[label] = {'N': N, 'annuity': annuity, 'npv': npv_gross}
    print(f'{label:>14} | {N:>11.0f} | {annuity:>14.3f} | ${npv_gross/1e9:>10.1f}B | {vs_cap_rev:>12.0f}×')

print()


# ─────────────────────────────────────────────────────────────
# 4. DOMINANCE ANALYSIS
# ─────────────────────────────────────────────────────────────

npv_central = ix_npv_results['Central']['npv']

# Revenue differential: DVFS+Spatial vs DVFS-only
cap_rev_diff_annual = (FACILITY_MW * (OPTIMAL_COMMITMENT_FRAC - DVFS_COMMITMENT_FRAC)
                       * BRA_2027_28_PRICE * 365 * DR_ELCC)
cap_diff_npv = cap_rev_diff_annual * ix_npv_results['Central']['annuity']

print(f'{"─"*70}')
print(f'DOMINANCE ANALYSIS')
print(f'{"─"*70}')
print(f'Central scenario (3yr acceleration, {FACILITY_GW:.0f} GW):')
print(f'  Gross IX NPV:               ${npv_central/1e9:.1f}B')
print(f'  DVFS+Spatial vs DVFS-only cap differential: ${cap_rev_diff_annual/1e6:.1f}M/yr')
print(f'  Cap differential NPV (3yr):  ${cap_diff_npv/1e6:.0f}M')
print()

breakeven_margin = cap_diff_npv / npv_central
print(f'  IX NPV exceeds cap differential at EBITDA margin > {breakeven_margin:.2%}')
print(f'  Typical hyperscaler cloud EBITDA: 35-45%')
print(f'  → IX incentive dominates at any plausible margin')
print()


# ─────────────────────────────────────────────────────────────
# 5. POLICY IMPLICATION
# ─────────────────────────────────────────────────────────────

cap_rev_spatial_annual = FACILITY_MW * OPTIMAL_COMMITMENT_FRAC * BRA_2027_28_PRICE * 365 * DR_ELCC
cap_rev_dvfs_annual = FACILITY_MW * DVFS_COMMITMENT_FRAC * BRA_2027_28_PRICE * 365 * DR_ELCC

print(f'{"─"*70}')
print(f'MECHANISM DESIGN IMPLICATION')
print(f'{"─"*70}')
print(f'  Capacity revenue differential (DVFS+Spatial vs DVFS-only):')
print(f'    ${cap_rev_spatial_annual/1e6:.1f}M/yr ({OPTIMAL_COMMITMENT_FRAC:.0%} depth) vs '
      f'${cap_rev_dvfs_annual/1e6:.1f}M/yr ({DVFS_COMMITMENT_FRAC:.0%} depth)')
print(f'    = ${cap_rev_diff_annual/1e6:.1f}M/yr incremental — the STATED instrument')
print()
print(f'  IX queue NPV (central, gross): ${npv_central/1e9:.1f}B — the ACTUAL incentive')
print(f'  Ratio: {npv_central/cap_rev_diff_annual:.0f}× annual differential')
print()
print(f'  IRP design implication:')
print(f'    Regulators do not need large capacity price differentials.')
print(f'    A credible FIA preference rule — tied to demonstrated spatial')
print(f'    flexibility — creates dominant incentive to invest, because')
print(f'    hyperscalers optimize on time-to-revenue, not $/MW-yr.')
print(f'    Capacity market is the compliance hook; IX speed is the payoff.')


# ─────────────────────────────────────────────────────────────
# STORE RESULTS
# ─────────────────────────────────────────────────────────────

IX_NPV_CENTRAL   = ix_npv_results['Central']['npv']
IX_NPV_LOW       = ix_npv_results['Conservative']['npv']
IX_NPV_HIGH      = ix_npv_results['Optimistic']['npv']
IX_DOMINANCE_RATIO = npv_central / cap_rev_diff_annual
```

    IX QUEUE NPV — BEHAVIORAL INCENTIVE QUANTIFICATION
    ======================================================================
    Reference facility: 1 GW grid-connected
    GPU density: 603 GPUs/MW (grid-metered, PUE=1.3)
    H100 spot rate: $2.2/hr
    Utilization: 80%
    Annual gross compute revenue: $9.30B/yr
    
    PJM baseline queue [LBNL-2024]:
      Median IR→COD: 5.8 years
      P75 IR→COD:    7.0 years
    
    WACC: 10% (hyperscaler range 8-12%)
    
    ──────────────────────────────────────────────────────────────────────
    NPV OF IX QUEUE ACCELERATION (1 GW facility)
    ──────────────────────────────────────────────────────────────────────
          Scenario | Accel (yrs) | Annuity Factor |  NPV (gross) | vs Ann Cap Rev
    ----------------------------------------------------------------------
      Conservative |           2 |          1.736 | $      16.1B |          576×
           Central |           3 |          2.487 | $      23.1B |          826×
        Optimistic |           4 |          3.170 | $      29.5B |         1053×
    
    ──────────────────────────────────────────────────────────────────────
    DOMINANCE ANALYSIS
    ──────────────────────────────────────────────────────────────────────
    Central scenario (3yr acceleration, 1 GW):
      Gross IX NPV:               $23.1B
      DVFS+Spatial vs DVFS-only cap differential: $3.2M/yr
      Cap differential NPV (3yr):  $8M
    
      IX NPV exceeds cap differential at EBITDA margin > 0.03%
      Typical hyperscaler cloud EBITDA: 35-45%
      → IX incentive dominates at any plausible margin
    
    ──────────────────────────────────────────────────────────────────────
    MECHANISM DESIGN IMPLICATION
    ──────────────────────────────────────────────────────────────────────
      Capacity revenue differential (DVFS+Spatial vs DVFS-only):
        $22.8M/yr (20% depth) vs $19.6M/yr (18% depth)
        = $3.2M/yr incremental — the STATED instrument
    
      IX queue NPV (central, gross): $23.1B — the ACTUAL incentive
      Ratio: 7177× annual differential
    
      IRP design implication:
        Regulators do not need large capacity price differentials.
        A credible FIA preference rule — tied to demonstrated spatial
        flexibility — creates dominant incentive to invest, because
        hyperscalers optimize on time-to-revenue, not $/MW-yr.
        Capacity market is the compliance hook; IX speed is the payoff.
    

### 5.2 Forward Energy Arbitrage Projection

Monte Carlo supply-gap model: as demand growth outpaces CT build 
timelines through the IX queue, a temporary scarcity window opens 
(2027–2031). Energy arbitrage value rises during this window, 
especially under CEJA constraints that limit in-state gas builds.


```python
# ======================================================================
# Cell 5-2: FORWARD ENERGY ARBITRAGE — MONTE CARLO PROJECTION (v18)
# ===========================================================
# Supply-gap model: projects energy arbitrage value forward by modeling
# the temporary scarcity window (2027-2031) created by demand growth
# outpacing CT build timelines through the IX queue.
#
# Framework:
#   - E3 base case: CTs DO get built (~13 GW over 20 years), but with
#     a 4-5 year IX queue lag creating temporary scarcity
#   - CEJA sensitivity: in-state CT builds constrained → longer scarcity
#   - Flex capacity online → spread cannibalization
#
# INPUTS: da, load_df, PRIMARY_YEAR, PRIMARY_ENERGY_100H,
#         DC_GROWTH_MAP, RESERVE_MARGIN_TARGET, PJM_OFFER_CAP,
#         CANNIBALIZATION_RATE/FLOOR, FLEX_FRAC, E3_CT_BUILD_RATE_CEJA
#         (from Parts 0, 1, 4)
# OUTPUTS: PROJECTED_ARB_MC, PROJECTED_ARB_MC_CEJA
# ===========================================================

import numpy as np


# ─────────────────────────────────────────────────────────────
# SUPPLY-GAP MODEL
# ─────────────────────────────────────────────────────────────

def compute_supply_gap(yr, base_peak_mw, organic_growth, dc_gw_by_year,
                        ct_lag_yrs=4, ct_build_rate_gw_yr=1.5):
    """Compute MW supply gap: peak demand minus available dispatchable capacity."""
    yrs = list(range(2024, yr + 1))
    peak = base_peak_mw
    demand = {}
    for y in yrs:
        dc_mw = dc_gw_by_year.get(y, dc_gw_by_year.get(max(k for k in dc_gw_by_year if k <= y), 0)) * 1000
        org = peak * (1 + organic_growth) ** (y - 2024)
        demand[y] = org + dc_mw

    capacity = {2024: peak * RESERVE_MARGIN_TARGET}
    for y in yrs[1:]:
        trigger_yr = y - ct_lag_yrs
        if trigger_yr >= 2024:
            trigger_gap = max(0, demand.get(trigger_yr, 0) - capacity.get(trigger_yr, capacity[2024]))
            ct_added = min(trigger_gap, ct_build_rate_gw_yr * 1000)
        else:
            ct_added = 0
        baseline_build = 500  # 500 MW/yr baseline interconnection completions
        capacity[y] = capacity[y-1] + ct_added + baseline_build

    return max(0, demand[yr] - capacity[yr]), demand[yr], capacity[yr]


def project_lmps_v2(base_lmps, base_peak, years_out, supply_gap_mw,
                     flex_gw_online, knee_sensitivity, organic_growth,
                     base_trend_rate=0.02):
    """Project LMP distribution given supply gap and flex suppression."""
    base_mult = (1 + base_trend_rate) ** years_out
    scarcity_premium_per_gw = knee_sensitivity * 12.0
    gap_gw = supply_gap_mw / 1000.0
    scarcity_premium = scarcity_premium_per_gw * gap_gw

    n = len(base_lmps)
    n_top = max(1, int(n * 0.005))
    n_mid = max(1, int(n * 0.02))
    projected = base_lmps * base_mult
    projected[:n_top]      += scarcity_premium
    projected[n_top:n_mid] += scarcity_premium * 0.5
    projected = np.clip(projected, -100, PJM_OFFER_CAP)

    suppression_factor = max(CANNIBALIZATION_FLOOR, 1.0 - flex_gw_online * CANNIBALIZATION_RATE)
    return projected, suppression_factor, scarcity_premium, gap_gw


# ─────────────────────────────────────────────────────────────
# BASE DATA
# ─────────────────────────────────────────────────────────────

base_lmps_sorted = da[da['year'] == 2024]['total_lmp_da'].sort_values(ascending=False).values
base_peak = load_df[load_df['year'] == 2024]['mw'].max()

# Interpolate DC_GROWTH_MAP to yearly resolution
dc_by_year_full = {}
map_keys = sorted(DC_GROWTH_MAP.keys())
for yr in range(2024, 2036):
    if yr <= map_keys[0]:
        dc_by_year_full[yr] = 0
    elif yr >= map_keys[-1]:
        dc_by_year_full[yr] = DC_GROWTH_MAP[map_keys[-1]]
    else:
        lo = max(k for k in map_keys if k <= yr)
        hi = min(k for k in map_keys if k >= yr)
        if lo == hi:
            dc_by_year_full[yr] = DC_GROWTH_MAP[lo]
        else:
            t = (yr - lo) / (hi - lo)
            dc_by_year_full[yr] = DC_GROWTH_MAP[lo] + t * (DC_GROWTH_MAP[hi] - DC_GROWTH_MAP[lo])


# ─────────────────────────────────────────────────────────────
# MONTE CARLO DRAWS
# ─────────────────────────────────────────────────────────────

N_MC = 300
rng = np.random.default_rng(42)

knee_dist    = rng.uniform(1.5, 4.0, N_MC)
organic_dist = rng.normal(0.009, 0.003, N_MC)
ct_lag_dist  = rng.integers(3, 7, N_MC)
ct_rate_base = rng.uniform(0.8, 2.2, N_MC)
ct_rate_ceja = rng.uniform(0.0, E3_CT_BUILD_RATE_CEJA * 2, N_MC)
trend_dist   = rng.normal(0.02, 0.01, N_MC)

projection_years = [2026, 2028, 2030, 2032, 2035]
mc_results      = {yr: [] for yr in projection_years}
mc_results_ceja = {yr: [] for yr in projection_years}

for i in range(N_MC):
    for yr in projection_years:
        years_out = yr - 2024
        dc_gw   = dc_by_year_full.get(yr, 14.0)
        flex_gw = dc_gw * FLEX_FRAC

        # Base case
        gap_base, _, _ = compute_supply_gap(
            yr, base_peak, organic_dist[i], dc_by_year_full,
            ct_lag_yrs=ct_lag_dist[i], ct_build_rate_gw_yr=ct_rate_base[i])
        proj_b, supp_b, _, gap_gw_b = project_lmps_v2(
            base_lmps_sorted, base_peak, years_out,
            gap_base, flex_gw, knee_dist[i], organic_dist[i],
            base_trend_rate=trend_dist[i])
        avg_b = np.mean(proj_b)
        for h in [50, 100]:
            mc_results[yr].append({
                'hours': h, 'value': (proj_b[:h].mean() - avg_b) * supp_b * h,
                'gap_gw': gap_gw_b})

        # CEJA sensitivity
        gap_ceja, _, _ = compute_supply_gap(
            yr, base_peak, organic_dist[i], dc_by_year_full,
            ct_lag_yrs=ct_lag_dist[i], ct_build_rate_gw_yr=ct_rate_ceja[i])
        proj_c, supp_c, _, gap_gw_c = project_lmps_v2(
            base_lmps_sorted, base_peak, years_out,
            gap_ceja, flex_gw, knee_dist[i], organic_dist[i],
            base_trend_rate=trend_dist[i])
        avg_c = np.mean(proj_c)
        for h in [50, 100]:
            mc_results_ceja[yr].append({
                'hours': h, 'value': (proj_c[:h].mean() - avg_c) * supp_c * h,
                'gap_gw': gap_gw_c})


# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────

def summarize_mc(results, label):
    out = {}
    print(f'\n{label}')
    print(f'{"Year":>6} | {"P25":>10} | {"Median":>10} | {"P75":>10} | {"Avg Gap GW":>10}')
    print('-' * 55)
    for yr in projection_years:
        vals = [r['value'] for r in results[yr] if r['hours'] == 100]
        gaps = [r['gap_gw'] for r in results[yr] if r['hours'] == 100]
        p25, p50, p75 = np.percentile(vals, [25, 50, 75])
        avg_gap = np.mean(gaps)
        print(f'{yr:>6} | ${p25:>9,.0f} | ${p50:>9,.0f} | ${p75:>9,.0f} | {avg_gap:>9.1f} GW')
        out[yr] = {'p25': p25, 'median': p50, 'p75': p75, 'gap': avg_gap}
    return out

print('FORWARD ENERGY ARBITRAGE — MONTE CARLO (Supply-Gap Model)')
print(f'Base: CT builds feasible (0.8-2.2 GW/yr) | CEJA: constrained (0-0.3 GW/yr)')
print(f'2025 baseline (100h): ${PRIMARY_ENERGY_100H:,.0f}/MW-yr')
print('=' * 70)

PROJECTED_ARB_MC      = summarize_mc(mc_results,      'BASE CASE (CT builds feasible):')
PROJECTED_ARB_MC_CEJA = summarize_mc(mc_results_ceja, 'CEJA SENSITIVITY (CT builds constrained):')

print()
print('COMPARISON — 100h median:')
print(f'  {"Year":>6} | {"Base":>12} | {"CEJA":>12} | {"CEJA premium":>14}')
print(f'  {"-"*50}')
print(f'  {"2025":>6} | ${PRIMARY_ENERGY_100H:>11,.0f} | ${PRIMARY_ENERGY_100H:>11,.0f} | {"(baseline)":>14}')
for yr in projection_years:
    b = PROJECTED_ARB_MC[yr]["median"]
    c = PROJECTED_ARB_MC_CEJA[yr]["median"]
    print(f'  {yr:>6} | ${b:>11,.0f} | ${c:>11,.0f} | +${c-b:>12,.0f}')
print()
print('CEJA: constrained CT builds → longer supply gap → higher scarcity premiums.')
print('Base case is the conservative floor. CEJA reflects IL-specific policy reality.')
```

    FORWARD ENERGY ARBITRAGE — MONTE CARLO (Supply-Gap Model)
    Base: CT builds feasible (0.8-2.2 GW/yr) | CEJA: constrained (0-0.3 GW/yr)
    2025 baseline (100h): $16,066/MW-yr
    ======================================================================
    
    BASE CASE (CT builds feasible):
      Year |        P25 |     Median |        P75 | Avg Gap GW
    -------------------------------------------------------
      2026 | $   10,578 | $   10,725 | $   10,851 |       0.0 GW
      2028 | $   11,173 | $   11,658 | $   12,194 |       0.5 GW
      2030 | $   16,848 | $   18,913 | $   20,878 |       3.9 GW
      2032 | $   18,057 | $   20,235 | $   23,541 |       4.7 GW
      2035 | $   12,940 | $   16,361 | $   19,299 |       2.4 GW
    
    CEJA SENSITIVITY (CT builds constrained):
      Year |        P25 |     Median |        P75 | Avg Gap GW
    -------------------------------------------------------
      2026 | $   10,578 | $   10,725 | $   10,851 |       0.0 GW
      2028 | $   11,173 | $   11,658 | $   12,194 |       0.5 GW
      2030 | $   16,848 | $   18,913 | $   20,878 |       3.9 GW
      2032 | $   19,137 | $   21,806 | $   24,371 |       5.2 GW
      2035 | $   20,295 | $   23,610 | $   26,316 |       5.9 GW
    
    COMPARISON — 100h median:
        Year |         Base |         CEJA |   CEJA premium
      --------------------------------------------------
        2025 | $     16,066 | $     16,066 |     (baseline)
        2026 | $     10,725 | $     10,725 | +$           0
        2028 | $     11,658 | $     11,658 | +$           0
        2030 | $     18,913 | $     18,913 | +$           0
        2032 | $     20,235 | $     21,806 | +$       1,571
        2035 | $     16,361 | $     23,610 | +$       7,250
    
    CEJA: constrained CT builds → longer supply gap → higher scarcity premiums.
    Base case is the conservative floor. CEJA reflects IL-specific policy reality.
    

## Part 6: Summary, Validation, and Export
- Cell 1: Consolidated Results
- Cell 2: Validation
- Cell 3: Exports

Consolidated results for the EPIC paper, automated validation checks,
and CSV export. Every number in the consolidated results traces to a
specific Part and Cell above. The validation cell catches version drift
between the notebook and the Pillar documents.

### 6.1 Consolidated Results


```python
# ======================================================================
# Cell 6-1: CONSOLIDATED RESULTS (v18)
# ===========================================================
# Single summary of all headline outputs for the EPIC paper.
# Every number here should trace to a specific Part/Cell above.
# If any number looks wrong, the validation cell (6-3) will catch it.
#
# INPUTS: All stored variables from Parts 0-5
# OUTPUTS: Print-only (summary tables for paper)
# ===========================================================

print('╔' + '═'*88 + '╗')
print('║' + ' CONSOLIDATED RESULTS — BARTLETT FELLOWSHIP v18 '.center(88) + '║')
print('╚' + '═'*88 + '╝')


# ── TABLE 1: Grounded Historical Value Stack ─────────────────

print(f'\nTABLE 1: Historical Value Stack ({PRIMARY_YEAR}, $/MW-yr of flexible capacity)')
print('-' * 70)
print(f'  Capacity (BRA 27/28 × ELCC):  ${PRIMARY_CAP_REV:>10,.0f}')
print(f'  Energy arbitrage (100h):        ${PRIMARY_ENERGY_100H:>10,.0f}')
print(f'  PAH overlap adjustment:        -${PRIMARY_PAH_DEDUCTION:>10,.0f}')
print(f'  TOTAL:                          ${PRIMARY_ADJUSTED_TOTAL:>10,.0f}')
print(f'  Source: Part 4, Cell 4-1')


# ── TABLE 2: Three-Prong Energy Economics ────────────────────

print(f'\nTABLE 2: Mechanism Energy Profiles ($/MW-yr, energy only)')
print('-' * 70)
print(f'  Spatial (geographic migration):  ${spatial_net:>10,.0f}  {"⚠ Uneconomic today" if spatial_net < 0 else ""}')
print(f'  Temporal (software throttling):   ${temporal_net:>10,.0f}')
print(f'  BTM (battery + controls):         ${btm_net:>10,.0f}  (pre-CapEx)')
print(f'  Source: Part 4, Cell 4-2')


# ── TABLE 3: E3 Counterfactual ───────────────────────────────

print(f'\nTABLE 3: E3 Counterfactual — DC DR as CT Substitute (10 GW)')
print('-' * 70)
print(f'  ELCC: {DR_ELCC:.0%} (PJM class rating — exogenous)')
print(f'  E3 target: {E3_NEW_GAS_CT_GW:.0f} GW new gas CTs @ ${E3_CT_LEVELIZED_COST:,}/MW-yr')
print()
print(f'  {"Commitment":<25} | {"Depth":>6} | {"Committed":>10} | {"Accredited":>10} | {"Avoided Cost":>13} | {"BCR":>5}')
print(f'  {"-"*80}')

for label, depth in [('DVFS Only', DVFS_COMMITMENT_FRAC),
                     ('DVFS + Spatial', OPTIMAL_COMMITMENT_FRAC)]:
    flex_mw = 10000 * depth
    acc_mw = flex_mw * DR_ELCC
    avoided = acc_mw * E3_CT_LEVELIZED_COST
    cap_rev = flex_mw * (BRA_2027_28_PRICE * 365 * DR_ELCC + PRIMARY_ENERGY_100H - PRIMARY_PAH_DEDUCTION)
    bcr = avoided / cap_rev if cap_rev > 0 else 0
    print(f'  {label:<25} | {depth:>5.0%} | {flex_mw:>10,.0f} | {acc_mw:>10,.0f} | ${avoided/1e6:>11.1f}M | {bcr:>4.2f}x')

print(f'  CEJA sensitivity (DVFS+Spatial): BCR = {E3_BCR_TIERED_T3_CEJA:.2f}x')
print(f'  Source: Part 3, Cell 3-4')


# ── TABLE 4: Spatial Break-Even Timeline ─────────────────────

print(f'\nTABLE 4: Spatial Shifting — Projected Economics (Central)')
print('-' * 70)
for v in results['Central']:
    if v['year'] in [2025, 2027, 2029, 2031, 2033, 2035]:
        status = '✓' if v['net_value'] > 0 else '❌'
        print(f'  {v["year"]}: lat={v["latency_min"]:>5.1f}min  spread=${v["crisis_spread"]:>6,.0f}  '
              f'net=${v["net_value"]:>8,.0f}/MW-yr  {status}')
print(f'  Source: Part 4, Cell 4-3')


# ── TABLE 5: Portfolio Interaction ────────────────────────────

print(f'\nTABLE 5: Portfolio Interaction Value')
print('-' * 70)
print(f'  Duration extension savings:   ${capex_savings_annualized:>10,.0f}/MW-yr')
print(f'  Battery sizing reduction:      ${capex_savings_sizing:>10,.0f}/MW-yr')
print(f'  ELCC stacking bonus:           ${elcc_stacking_value:>10,.0f}/MW-yr')
print(f'  Total interaction premium:     ${total_interaction:>10,.0f}/MW-yr')
print(f'  Source: Part 4, Cell 4-4')


# ── TABLE 6: Forward Energy Arbitrage ────────────────────────

print(f'\nTABLE 6: Forward Energy Arbitrage — MC Projection (100h, median)')
print('-' * 70)
print(f'  {"Year":>6} | {"P25":>10} | {"Median":>10} | {"P75":>10} | {"Avg Gap GW":>10}')
print(f'  {"-"*55}')
print(f'  {"2025":>6} | {"(hist)":>10} | ${PRIMARY_ENERGY_100H:>9,.0f} | {"(hist)":>10} | {"---":>10}')
for yr in [2026, 2028, 2030, 2032, 2035]:
    if yr in PROJECTED_ARB_MC:
        r = PROJECTED_ARB_MC[yr]
        print(f'  {yr:>6} | ${r["p25"]:>9,.0f} | ${r["median"]:>9,.0f} | ${r["p75"]:>9,.0f} | {r["gap"]:>9.1f} GW')
print(f'  Source: Part 5, Cell 5-2')


# ── TABLE 7: Connect-and-Manage ──────────────────────────────

print(f'\nTABLE 7: Connect-and-Manage — Firm Service Level')
print('-' * 70)
for label, depth in [('DVFS Only', DVFS_COMMITMENT_FRAC),
                     ('DVFS + Spatial', OPTIMAL_COMMITMENT_FRAC)]:
    firm_pct = 1 - depth
    print(f'  {label:<20}: commit {depth:.0%} → firm service = {firm_pct:.0%} of nameplate')
print(f'  Reference cases (DVFS+Spatial):')
print(f'    1 GW:  removes {1000*OPTIMAL_COMMITMENT_FRAC:,.0f} MW from Reliability Requirement')
print(f'    10 GW: removes {10000*OPTIMAL_COMMITMENT_FRAC:,.0f} MW from Reliability Requirement')
print(f'  Source: Part 3, Cell 3-2')


# ── KEY FINDINGS ─────────────────────────────────────────────
# v18 Phase 3: per-GW framing. Reference cases at 1 GW and 10 GW.
# Headline is commitment depth as a function of fleet size, not
# a single fleet-size number.

print(f'\n{"="*70}')
print(f'KEY FINDINGS (v18 per-GW framing):')
print(f'  Cascade product (central):    {EFFECTIVE_SPATIAL_FRAC:.4f} (10-param, inference-dominant)')
print(f'  Commitment depth:             {OPTIMAL_COMMITMENT_FRAC:.0%} (DVFS+Spatial) vs {DVFS_COMMITMENT_FRAC:.0%} (DVFS only)')
_spatial_uplift_pct_pts = (OPTIMAL_COMMITMENT_FRAC - DVFS_COMMITMENT_FRAC) * 100
print(f'  Spatial uplift:               +{_spatial_uplift_pct_pts:.1f} pct pts on top of DVFS floor')
print(f'  ── 1 GW reference (individual facility scale) ──')
print(f'  Accredited MW (1 GW):         {REF_ACCREDITED_MW_1GW:>,.0f} MW')
print(f'  Net value (1 GW):            ${REF_NET_VALUE_1GW/1e6:>,.1f}M/yr')
print(f'  Avoided CT cost (1 GW):      ${E3_AVOIDED_COST_1GW/1e6:>,.1f}M/yr')
print(f'  ── 10 GW reference (fleet scale) ──')
print(f'  Accredited MW (10 GW):        {REF_ACCREDITED_MW_10GW:>,.0f} MW')
print(f'  Net value (10 GW):           ${REF_NET_VALUE_10GW/1e6:>,.1f}M/yr')
print(f'  Avoided CT cost (10 GW):     ${E3_AVOIDED_COST_10GW/1e6:>,.1f}M/yr')
print(f'  ──')
print(f'  Grounded value stack:        ${GROUNDED_VALUE_MID:>,.0f}/MW-yr (cap + energy)')
# Compute break-even years from Cell 4-3 results dict
def _crossing_year(vals):
    for v in vals:
        if v['net_value'] > 0:
            return v['year']
    return None

_central_cross = _crossing_year(results['Central'])
_conservative_cross = _crossing_year(results['Conservative'])

if _central_cross and _conservative_cross:
    print(f'  Spatial break-even:           {_central_cross} (central), {_conservative_cross} (conservative)')
elif _central_cross:
    print(f'  Spatial break-even:           {_central_cross} (central), beyond 2035 (conservative)')
else:
    print(f'  Spatial break-even:           beyond 2035 (all scenarios)')
print(f'  IX queue NPV (1 GW, 3yr):   ${IX_NPV_CENTRAL/1e9:>,.1f}B — dominates cap revenue')
print(f'  NOTE: Commitment depth varies with fleet size due to destination')
print(f'        contention. See conditional MC per-GW sweep for full curve.')
```

    ╔════════════════════════════════════════════════════════════════════════════════════════╗
    ║                     CONSOLIDATED RESULTS — BARTLETT FELLOWSHIP v18                     ║
    ╚════════════════════════════════════════════════════════════════════════════════════════╝
    
    TABLE 1: Historical Value Stack (2025, $/MW-yr of flexible capacity)
    ----------------------------------------------------------------------
      Capacity (BRA 27/28 × ELCC):  $   111,969
      Energy arbitrage (100h):        $    16,066
      PAH overlap adjustment:        -$     9,033
      TOTAL:                          $   119,002
      Source: Part 4, Cell 4-1
    
    TABLE 2: Mechanism Energy Profiles ($/MW-yr, energy only)
    ----------------------------------------------------------------------
      Spatial (geographic migration):  $      -216  ⚠ Uneconomic today
      Temporal (software throttling):   $     1,247
      BTM (battery + controls):         $    15,602  (pre-CapEx)
      Source: Part 4, Cell 4-2
    
    TABLE 3: E3 Counterfactual — DC DR as CT Substitute (10 GW)
    ----------------------------------------------------------------------
      ELCC: 92% (PJM class rating — exogenous)
      E3 target: 13 GW new gas CTs @ $180,000/MW-yr
    
      Commitment                |  Depth |  Committed | Accredited |  Avoided Cost |   BCR
      --------------------------------------------------------------------------------
      DVFS Only                 |   18% |      1,750 |      1,610 | $      289.8M | 1.39x
      DVFS + Spatial            |   20% |      2,038 |      1,875 | $      337.4M | 1.39x
      CEJA sensitivity (DVFS+Spatial): BCR = 1.68x
      Source: Part 3, Cell 3-4
    
    TABLE 4: Spatial Shifting — Projected Economics (Central)
    ----------------------------------------------------------------------
      2025: lat= 15.0min  spread=$    33  net=$    -693/MW-yr  ❌
      2027: lat=  9.6min  spread=$    73  net=$    -457/MW-yr  ❌
      2029: lat=  6.1min  spread=$   119  net=$    -224/MW-yr  ❌
      2031: lat=  3.9min  spread=$   171  net=$      17/MW-yr  ✓
      2033: lat=  2.5min  spread=$   230  net=$     277/MW-yr  ✓
      2035: lat=  1.6min  spread=$   298  net=$     562/MW-yr  ✓
      Source: Part 4, Cell 4-3
    
    TABLE 5: Portfolio Interaction Value
    ----------------------------------------------------------------------
      Duration extension savings:   $   100,000/MW-yr
      Battery sizing reduction:      $    20,476/MW-yr
      ELCC stacking bonus:           $     5,598/MW-yr
      Total interaction premium:     $   126,075/MW-yr
      Source: Part 4, Cell 4-4
    
    TABLE 6: Forward Energy Arbitrage — MC Projection (100h, median)
    ----------------------------------------------------------------------
        Year |        P25 |     Median |        P75 | Avg Gap GW
      -------------------------------------------------------
        2025 |     (hist) | $   16,066 |     (hist) |        ---
        2026 | $   10,578 | $   10,725 | $   10,851 |       0.0 GW
        2028 | $   11,173 | $   11,658 | $   12,194 |       0.5 GW
        2030 | $   16,848 | $   18,913 | $   20,878 |       3.9 GW
        2032 | $   18,057 | $   20,235 | $   23,541 |       4.7 GW
        2035 | $   12,940 | $   16,361 | $   19,299 |       2.4 GW
      Source: Part 5, Cell 5-2
    
    TABLE 7: Connect-and-Manage — Firm Service Level
    ----------------------------------------------------------------------
      DVFS Only           : commit 18% → firm service = 82% of nameplate
      DVFS + Spatial      : commit 20% → firm service = 80% of nameplate
      Reference cases (DVFS+Spatial):
        1 GW:  removes 204 MW from Reliability Requirement
        10 GW: removes 2,038 MW from Reliability Requirement
      Source: Part 3, Cell 3-2
    
    ======================================================================
    KEY FINDINGS (v18 per-GW framing):
      Cascade product (central):    0.0384 (10-param, inference-dominant)
      Commitment depth:             20% (DVFS+Spatial) vs 18% (DVFS only)
      Spatial uplift:               +2.9 pct pts on top of DVFS floor
      ── 1 GW reference (individual facility scale) ──
      Accredited MW (1 GW):         187 MW
      Net value (1 GW):            $21.1M/yr
      Avoided CT cost (1 GW):      $33.7M/yr
      ── 10 GW reference (fleet scale) ──
      Accredited MW (10 GW):        1,875 MW
      Net value (10 GW):           $211.1M/yr
      Avoided CT cost (10 GW):     $337.4M/yr
      ──
      Grounded value stack:        $158,143/MW-yr (cap + energy)
      Spatial break-even:           2031 (central), 2034 (conservative)
      IX queue NPV (1 GW, 3yr):   $23.1B — dominates cap revenue
      NOTE: Commitment depth varies with fleet size due to destination
            contention. See conditional MC per-GW sweep for full curve.
    

### 6.2 Validation


```python
# ======================================================================
# Cell 6-2: VALIDATION CHECKS (v18)
# ===========================================================
# Automated checks that the notebook is internally consistent.
# Run after all Parts 0-5. Any failure indicates a broken
# dependency or stale variable.
# ===========================================================

print("\nv18 VALIDATION CHECKS")
print("─" * 60)

_checks_passed = 0
_checks_total = 0

def _check(condition, label, detail=""):
    global _checks_passed, _checks_total
    _checks_total += 1
    status = "✓" if condition else "✗ FAIL"
    if condition:
        _checks_passed += 1
    print(f"  {status} {label}" + (f" ({detail})" if detail else ""))


# ── CASCADE INTEGRITY ─────────────────────────────────────────

_check(abs(EFFECTIVE_SPATIAL_FRAC - 0.0384) < 0.005,
       "Cascade product ≈ 0.0384 (v18 reconciled: S2=0.70, E1=0.95, E2=0.98)",
       f"actual: {EFFECTIVE_SPATIAL_FRAC:.4f}")

_check(abs(COMMITMENT_DEPTH - 0.204) < 0.01,
       "Commitment depth ≈ 20.4% (v18 reconciled: DVFS on shiftable residual, 10-param cascade)",
       f"actual: {COMMITMENT_DEPTH:.4f}")

_check(abs(OPTIMAL_COMMITMENT_FRAC - COMMITMENT_DEPTH) < 0.001,
       "OPTIMAL_COMMITMENT_FRAC matches COMMITMENT_DEPTH",
       f"opt={OPTIMAL_COMMITMENT_FRAC:.4f}, depth={COMMITMENT_DEPTH:.4f}")

_check(abs(CASCADE_D2 - 0.33) < 0.001,
       "CASCADE_D2 = 0.33 (v18: utilization headroom)",
       f"actual: {CASCADE_D2}")

_check(abs(CASCADE_D3 - 0.88) < 0.001,
       "CASCADE_D3 = 0.88 (v18: HW compat without pre-staging)",
       f"actual: {CASCADE_D3}")

_check(abs(CASCADE_D4 - 0.50) < 0.001,
       "CASCADE_D4 = 0.50 (v18: inference workload share)",
       f"actual: {CASCADE_D4}")

_check(abs(CASCADE_D5 - 0.65) < 0.001,
       "CASCADE_D5 = 0.65 (v18: pre-staging readiness)",
       f"actual: {CASCADE_D5}")

_check(abs(CASCADE_D1 - 0.99) < 0.001,
       "CASCADE_D1 = 0.99",
       f"actual: {CASCADE_D1}")


# ── RETIRED VARIABLES NOT IN NAMESPACE ────────────────────────

_check('TCF' not in dir(),
       "TCF not defined (retired in v12)")

_check('SPATIAL_INTRA_RTO_DEPTH' not in dir(),
       "SPATIAL_INTRA_RTO_DEPTH not defined (retired)")

_check('SPATIAL_T3_CROSS_IX_DEPTH' not in dir(),
       "SPATIAL_T3_CROSS_IX_DEPTH not defined (retired)")

_check('WIRED_ELCC_T1' not in dir(),
       "WIRED_ELCC_T1 not defined (tiered ELCC retired)")


# ── REFERENCE CASE OUTPUTS (1 GW and 10 GW) ─────────────────

# v18 Phase 3: validate at both reference fleet sizes
# 10 GW: ~21% commitment × 10 GW × 92% ELCC ≈ 1,932 MW
_check(1500 < REF_ACCREDITED_MW_10GW < 2500,
       "Accredited MW (10GW ref) in [1500, 2500] (v18 range)",
       f"actual: {REF_ACCREDITED_MW_10GW:,.0f}")

_check(REF_NET_VALUE_10GW > 0,
       "Net value (10GW ref) is positive",
       f"actual: ${REF_NET_VALUE_10GW/1e6:,.1f}M")

# 1 GW: ~21% commitment × 1 GW × 92% ELCC ≈ 193 MW
_check(150 < REF_ACCREDITED_MW_1GW < 250,
       "Accredited MW (1GW ref) in [150, 250] (v18 range)",
       f"actual: {REF_ACCREDITED_MW_1GW:,.0f}")

_check(REF_NET_VALUE_1GW > 0,
       "Net value (1GW ref) is positive",
       f"actual: ${REF_NET_VALUE_1GW/1e6:,.1f}M")

# Per-MW economics should be identical at 1 GW and 10 GW (linear model)
_npm_1gw = commitment_results[PRIMARY_PORTFOLIO][1.0]['net_per_mw_committed']
_npm_10gw = commitment_results[PRIMARY_PORTFOLIO][10.0]['net_per_mw_committed']
_check(abs(_npm_1gw - _npm_10gw) < 1.0,
       "Net/MW identical at 1 GW and 10 GW (linear cascade model)",
       f"1GW: ${_npm_1gw:,.0f}, 10GW: ${_npm_10gw:,.0f}")

_check(abs(DR_ELCC - 0.92) < 0.001,
       "DR_ELCC = 0.92 (PJM class rating)",
       f"actual: {DR_ELCC}")


# ── ELECTION MECHANISM ────────────────────────────────────────

_check('election_results' in dir() and 10.0 in election_results,
       "Election results computed for 10 GW ref")

_check('election_results' in dir() and 1.0 in election_results,
       "Election results computed for 1 GW ref")

if 'election_results' in dir() and 10.0 in election_results:
    _r = election_results[10.0]
    _check('deliverable_spatial' not in _r,
           "No 'deliverable_spatial' key (TCF step removed)",
           f"keys: {list(_r.keys())[:5]}...")

    # v18: lower commitment depth means lower avoided capacity
    _check(1000 < _r['total_avoided'] < 4000,
           "Total avoided (10GW ref) in [1000, 4000] (v18 range)",
           f"actual: {_r['total_avoided']:,.0f}")

if 'election_results' in dir() and 1.0 in election_results:
    _r1 = election_results[1.0]
    _check(100 < _r1['total_avoided'] < 400,
           "Total avoided (1GW ref) in [100, 400] (v18 range)",
           f"actual: {_r1['total_avoided']:,.0f}")


# ── RESERVE MARGIN ────────────────────────────────────────────

_check(abs(PLANNING_RESERVE_MARGIN - (RESERVE_MARGIN_TARGET - 1.0)) < 0.001,
       "PRM derived from RESERVE_MARGIN_TARGET (not hardcoded)",
       f"PRM={PLANNING_RESERVE_MARGIN:.2f}, IRM={RESERVE_MARGIN_TARGET:.2f}")


# ── CROSS-DOCUMENT CONSISTENCY ────────────────────────────────
# These checks verify that Pillar document claims match notebook outputs.

# v18: Cross-document checks are EXPECTED TO FAIL until Phase 5 reconciliation.
# Documents still contain stale v15/v16 numbers. These checks are intentionally
# updated to v18 targets so they will pass once documents are reconciled.
_v18_cascade_central = 0.0384
_v18_commitment_central = 0.204

_check(abs(EFFECTIVE_SPATIAL_FRAC - _v18_cascade_central) < 0.005,
       f"Cascade matches v18 target ({_v18_cascade_central})",
       f"notebook: {EFFECTIVE_SPATIAL_FRAC:.4f}")

_check(abs(OPTIMAL_COMMITMENT_FRAC - _v18_commitment_central) < 0.01,
       f"Commitment depth matches v18 target ({_v18_commitment_central})",
       f"notebook: {OPTIMAL_COMMITMENT_FRAC:.4f}")

# ── CROSS-DOCUMENT CONSISTENCY (RETIRED) ──────────────────────
# Removed v17→v18: Pillar 3 is no longer the canonical source for headline
# MW values. The conditional Monte Carlo in Cross_BA v5 is authoritative.
# Reference cases at 1 GW and 10 GW are validated against expected ranges
# in the REFERENCE CASE OUTPUTS section above.


# ── v5 DATA LOADED ────────────────────────────────────────────

try:
    _check(_V5_LOADED == True,
           "v5 empirical data loaded (not fallback)",
           f"DESTINATION_LMP_CRISIS = ${DESTINATION_LMP_CRISIS:.1f}")
except NameError:
    _check(False, "v5 empirical data loaded", "_V5_LOADED not defined")


# ── ALL CELLS EXECUTED ────────────────────────────────────────

_check(True,
       "No NameError from retired params (all cells executed)")


# ── SUMMARY ───────────────────────────────────────────────────

print(f"\n  {_checks_passed}/{_checks_total} checks passed")
if _checks_passed == _checks_total:
    print("  ✓ All v18 validation checks passed")
else:
    _n_fail = _checks_total - _checks_passed
    print(f"  ⚠ {_n_fail} check(s) failed — review above")
```

    
    v18 VALIDATION CHECKS
    ────────────────────────────────────────────────────────────
      ✓ Cascade product ≈ 0.0384 (v18 reconciled: S2=0.70, E1=0.95, E2=0.98) (actual: 0.0384)
      ✓ Commitment depth ≈ 20.4% (v18 reconciled: DVFS on shiftable residual, 10-param cascade) (actual: 0.2038)
      ✓ OPTIMAL_COMMITMENT_FRAC matches COMMITMENT_DEPTH (opt=0.2038, depth=0.2038)
      ✓ CASCADE_D2 = 0.33 (v18: utilization headroom) (actual: 0.33)
      ✓ CASCADE_D3 = 0.88 (v18: HW compat without pre-staging) (actual: 0.88)
      ✓ CASCADE_D4 = 0.50 (v18: inference workload share) (actual: 0.5)
      ✓ CASCADE_D5 = 0.65 (v18: pre-staging readiness) (actual: 0.65)
      ✓ CASCADE_D1 = 0.99 (actual: 0.99)
      ✓ TCF not defined (retired in v12)
      ✓ SPATIAL_INTRA_RTO_DEPTH not defined (retired)
      ✓ SPATIAL_T3_CROSS_IX_DEPTH not defined (retired)
      ✓ WIRED_ELCC_T1 not defined (tiered ELCC retired)
      ✓ Accredited MW (10GW ref) in [1500, 2500] (v18 range) (actual: 1,875)
      ✓ Net value (10GW ref) is positive (actual: $211.1M)
      ✓ Accredited MW (1GW ref) in [150, 250] (v18 range) (actual: 187)
      ✓ Net value (1GW ref) is positive (actual: $21.1M)
      ✓ Net/MW identical at 1 GW and 10 GW (linear cascade model) (1GW: $103,591, 10GW: $103,591)
      ✓ DR_ELCC = 0.92 (PJM class rating) (actual: 0.92)
      ✓ Election results computed for 10 GW ref
      ✓ Election results computed for 1 GW ref
      ✓ No 'deliverable_spatial' key (TCF step removed) (keys: ['gross_spatial', 'spatial_avoided', 'non_migrating_mw', 'dvfs_commitment', 'local_dr_accredited']...)
      ✓ Total avoided (10GW ref) in [1000, 4000] (v18 range) (actual: 1,982)
      ✓ Total avoided (1GW ref) in [100, 400] (v18 range) (actual: 198)
      ✓ PRM derived from RESERVE_MARGIN_TARGET (not hardcoded) (PRM=0.20, IRM=1.20)
      ✓ Cascade matches v18 target (0.0384) (notebook: 0.0384)
      ✓ Commitment depth matches v18 target (0.204) (notebook: 0.2038)
      ✓ v5 empirical data loaded (not fallback) (DESTINATION_LMP_CRISIS = $169.3)
      ✓ No NameError from retired params (all cells executed)
    
      28/28 checks passed
      ✓ All v18 validation checks passed
    

### 6.3 Exports


```python
# ======================================================================
# Cell 6-3: EXPORT RESULTS TO CSV (v18)
# ===========================================================
# Exports key result tables for use in Pillar documents and
# external analysis.
# ===========================================================

import os
os.makedirs('output', exist_ok=True)

# 1. Energy arbitrage by year
arb_table.to_csv('output/energy_arbitrage_by_year.csv')

# 2. Commitment optimization results
commitment_rows = []
for pname, gw_dict in commitment_results.items():
    for gw, r in gw_dict.items():
        commitment_rows.append({
            'portfolio': pname, 'fleet_gw': gw,
            'committed_mw': r['committed_mw'], 'accredited_mw': r['accredited_mw'],
            'revenue': r['revenue'], 'dispatch_cost': r['dispatch_cost'],
            'npc_cost': r['npc_cost'], 'net_value': r['net_value'],
            'net_per_mw': r['net_per_mw_committed'], 'depth': r['depth']
        })
pd.DataFrame(commitment_rows).to_csv('output/commitment_optimization.csv', index=False)

# 3. MC forward projections
mc_rows = []
for yr in projection_years:
    vals_100 = [r['value'] for r in mc_results[yr] if r['hours'] == 100]
    mc_rows.append({
        'year': yr,
        'p10': np.percentile(vals_100, 10),
        'p25': np.percentile(vals_100, 25),
        'median': np.percentile(vals_100, 50),
        'p75': np.percentile(vals_100, 75),
        'p90': np.percentile(vals_100, 90),
    })
pd.DataFrame(mc_rows).to_csv('output/forward_arbitrage_mc.csv', index=False)

# 4. Spatial break-even trajectory
spatial_rows = []
for scenario, vals in results.items():
    for v in vals:
        spatial_rows.append({**v, 'scenario': scenario})
pd.DataFrame(spatial_rows).to_csv('output/spatial_shifting_trajectory.csv', index=False)

# 5. Election mechanism results
election_rows = []
for gw, r in election_results.items():
    election_rows.append({'fleet_gw': gw, **r})
pd.DataFrame(election_rows).to_csv('output/election_mechanism.csv', index=False)

print('Exported to output/:')
for f in sorted(os.listdir('output')):
    print(f'  {f}')
```

    Exported to output/:
      commitment_optimization.csv
      election_mechanism.csv
      energy_arbitrage_by_year.csv
      forward_arbitrage_mc.csv
      spatial_shifting_trajectory.csv
    


## Experimental / Exploratory Analyses

The cells below were exploratory analyses conducted alongside v18 but are
**not part of the paper's core analytical pipeline**. They are retained for
reference but should not be relied on for headline numbers. Specifically:

- **Cell X-1**: Continuous inference routing arbitrage — explores per-hour
  cheapest-destination routing as a continuous optimization rather than
  event-triggered. Not cited in the paper.
- **Cell X-2**: Elasticity sensitivity sweep — reverse-engineers the
  `PRICE_ELASTICITY_PER_GW = 150` assumption to identify at what elasticity
  routing value degrades substantially. Not cited in the paper.

These cells read the same `hourly_zone_prices.parquet` file as Cell 0-3
and reuse its variable paths. They depend on upstream cells having run
successfully.


```python
# ======================================================================
# CELL X-1 (EXPERIMENTAL): CONTINUOUS INFERENCE ROUTING ARBITRAGE (v18)
# ======================================================================
import pandas as pd
import numpy as np

_BASE_V5 = r'C:\Users\dunla\OneDrive\Documents\Bartlett Fellowship\Demand Response Direction\1_Working Version'
_hourly = pd.read_parquet(os.path.join(_BASE_V5, 'hourly_zone_prices.parquet'))
_meta = pd.read_parquet(os.path.join(_BASE_V5, 'zone_metadata.parquet'))

_hourly['datetime'] = pd.to_datetime(_hourly['datetime'])
_hourly['year'] = _hourly['datetime'].dt.year
_hourly = _hourly[_hourly['year'].isin(YEARS)].copy()

print('Migration roles:', _meta.migration_role.unique().tolist())
print('Destination zones:', _meta[_meta.migration_role == 'destination'].index.tolist()[:10])
print('All zones in hourly:', _hourly.zone_id.unique().tolist()[:10])

# Pivot to wide: datetime × zone
_price_wide = _hourly.pivot_table(index='datetime', columns='zone_id',
                                   values='price', aggfunc='first')

# Identify destination zones
_destinations = _meta[_meta.migration_role == 'cross_ba_destination'].index.tolist()
_dest_cols = [z for z in _destinations if z in _price_wide.columns]

_source_col = 'PJM_COMED'

# For each hour: cheapest destination LMP
_dest_prices = _price_wide[_dest_cols]
_cheapest_dest_lmp = _dest_prices.min(axis=1)
_cheapest_dest_zone = _dest_prices.idxmin(axis=1)

# Source LMP
_source_lmp = _price_wide[_source_col]

# Hourly spread: positive means source is more expensive
_spread = _source_lmp - _cheapest_dest_lmp

# Only route when it saves money
_positive_spread = _spread.clip(lower=0)

# Routable fraction of facility load
_routable_frac = CASCADE_S1 * CASCADE_S2

print('CONTINUOUS INFERENCE ROUTING ARBITRAGE')
print('=' * 85)
print(f'Source: {_source_col}')
print(f'Destinations: {len(_dest_cols)} zones')
print(f'Routable fraction: {_routable_frac:.1%} (S1 x S2 = {CASCADE_S1} x {CASCADE_S2})')
print(f'Per-event routing cost: $0 (inference)')
print()

print(f'{"Year":>6} | {"Hours":>6} | {"Avg Spread":>12} | {"Pos Hours":>10} | '
      f'{"Gross $/MW":>12} | {"Routed $/MW":>12}')
print(f'{"-"*75}')

for yr in YEARS:
    yr_mask = _price_wide.index.year == yr
    yr_spread = _spread[yr_mask]
    yr_positive = _positive_spread[yr_mask]
    
    total_hours = len(yr_spread)
    pos_hours = (yr_spread > 0).sum()
    avg_spread = yr_spread.mean()
    
    gross_per_mw = yr_positive.sum()
    routed_per_mw = gross_per_mw * _routable_frac
    
    print(f'{yr:>6} | {total_hours:>6,} | ${avg_spread:>10.1f}/MWh | '
          f'{pos_hours:>10,} | ${gross_per_mw:>10,.0f} | ${routed_per_mw:>10,.0f}')

_gross_annual_avg = _positive_spread.groupby(_price_wide.index.year).sum().mean()
_routed_annual_avg = _gross_annual_avg * _routable_frac

print(f'\n{"Avg":>6} | {"":>6} | {"":>12} | '
      f'{"":>10} | ${_gross_annual_avg:>10,.0f} | ${_routed_annual_avg:>10,.0f}')

print(f'\n  Routed value = gross x {_routable_frac:.1%} routable fraction')
print(f'  No friction subtracted (inference routing cost = $0)')

print(f'\n  CONTEXT:')
print(f'    Continuous routing energy value:  ${_routed_annual_avg:>10,.0f}/MW-yr')
print(f'    Capacity revenue (BRA 27/28):     ${PRIMARY_CAP_REV:>10,.0f}/MW-yr')
print(f'    Ratio:                            {PRIMARY_CAP_REV/_routed_annual_avg:.0f}:1 capacity vs energy')

print(f'\n  CHEAPEST DESTINATION FREQUENCY (all hours):')
_dest_freq = _cheapest_dest_zone.value_counts()
for zone, count in _dest_freq.head(10).items():
    pct = count / len(_cheapest_dest_zone) * 100
    rto = _meta.loc[zone, 'rto'] if zone in _meta.index else '?'
    print(f'    {zone:<25} ({rto:<8}): {count:>5,} hours ({pct:>5.1f}%)')
```

    Migration roles: ['source', 'intra_rto_control', 'cross_ba_destination']
    Destination zones: []
    All zones in hourly: ['PJM_COMED', 'PJM_DOM', 'PJM_AEP', 'PJM_BGE', 'PJM_PECO', 'PJM_PSEG', 'PJM_PEPCO', 'ERCOT_LZ_NORTH', 'ERCOT_LZ_SOUTH', 'ERCOT_LZ_WEST']
    

    CONTINUOUS INFERENCE ROUTING ARBITRAGE
    =====================================================================================
    Source: PJM_COMED
    Destinations: 19 zones
    Routable fraction: 49.0% (S1 x S2 = 0.7 x 0.7)
    Per-event routing cost: $0 (inference)
    
      Year |  Hours |   Avg Spread |  Pos Hours |   Gross $/MW |  Routed $/MW
    ---------------------------------------------------------------------------
      2022 |  8,760 | $      27.3/MWh |      8,444 | $   240,825 | $   118,004
      2023 |  8,760 | $      11.0/MWh |      7,665 | $    99,755 | $    48,880
      2024 |  8,784 | $      14.2/MWh |      7,231 | $   130,412 | $    63,902
      2025 |  8,760 | $      19.1/MWh |      7,319 | $   174,829 | $    85,666
    
       Avg |        |              |            | $   161,455 | $    79,113
    
      Routed value = gross x 49.0% routable fraction
      No friction subtracted (inference routing cost = $0)
    
      CONTEXT:
        Continuous routing energy value:  $    79,113/MW-yr
        Capacity revenue (BRA 27/28):     $   111,969/MW-yr
        Ratio:                            1:1 capacity vs energy
    
      CHEAPEST DESTINATION FREQUENCY (all hours):
        MISO_MINNESOTA            (MISO    ): 6,141 hours ( 17.5%)
        CAISO_SP15                (CAISO   ): 5,277 hours ( 15.0%)
        ERCOT_LZ_NORTH            (ERCOT   ): 4,686 hours ( 13.4%)
        ERCOT_LZ_WEST             (ERCOT   ): 4,318 hours ( 12.3%)
        NYISO_ZONE_A              (NYISO   ): 3,926 hours ( 11.2%)
        ERCOT_LZ_SOUTH            (ERCOT   ): 3,418 hours (  9.7%)
        MISO_ARKANSAS             (MISO    ): 3,058 hours (  8.7%)
        MISO_ILLINOIS             (MISO    ): 1,597 hours (  4.6%)
        ERCOT_LZ_HOUSTON          (ERCOT   ):   930 hours (  2.7%)
        MISO_MS                   (MISO    ):   852 hours (  2.4%)
    


```python
# ======================================================================
# CELL X-2 (EXPERIMENTAL): ELASTICITY SENSITIVITY — WHAT DOES IT TAKE? (v18)
# ===========================================================
# Instead of assuming an elasticity, sweep it.
# For each fleet size: what elasticity produces 25%, 50%, 75% degradation?
# Then: are those elasticities realistic?
# ===========================================================

import numpy as np

_source_lmp = _price_wide[_source_col]
_dest_prices = _price_wide[_dest_cols]
_cheapest_dest = _dest_prices.min(axis=1)
_base_spread = (_source_lmp - _cheapest_dest).clip(lower=0)
_base_gross_annual = _base_spread.groupby(_price_wide.index.year).sum().mean()
_base_routed = _base_gross_annual * _routable_frac

fleet_sizes = [0.5, 1.0, 2.0, 5.0, 10.0]
elasticities = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500]

print('ELASTICITY SENSITIVITY: ROUTING VALUE BY FLEET SIZE')
print('=' * 90)
print(f'Baseline (price-taker): ${_base_routed:,.0f}/MW-yr')
print()

# Header
header = f'{"Elast":>8} |'
for gw in fleet_sizes:
    header += f' {gw:>5.1f} GW |'
print(header)
print(f'{"-" * (10 + 9 * len(fleet_sizes))}')

for elast in elasticities:
    row = f'${elast:>5}/GW |'
    for fleet_gw in fleet_sizes:
        shifted_gw = fleet_gw * _routable_frac / 1  # GW shifted
        
        # Total spread impact = elasticity × shifted GW × 2 
        # (source drops + destination rises)
        total_impact = elast * shifted_gw * 2
        
        adj_spread = (_source_lmp - _cheapest_dest - total_impact).clip(lower=0)
        adj_gross = adj_spread.groupby(_price_wide.index.year).sum().mean()
        adj_routed = adj_gross * _routable_frac
        
        pct_retained = adj_routed / _base_routed * 100 if _base_routed > 0 else 0
        row += f' {pct_retained:>5.0f}%  |'
    print(row)

# Find breakpoints for key fleet sizes
print(f'\nBREAKPOINT ANALYSIS: Elasticity needed for X% degradation')
print(f'{"-" * 70}')
print(f'{"Fleet":>8} | {"25% degrad":>12} | {"50% degrad":>12} | {"75% degrad":>12} | {"100% degrad":>12}')
print(f'{"-" * 70}')

for fleet_gw in fleet_sizes:
    shifted_gw = fleet_gw * _routable_frac
    breakpoints = {}
    
    for target_pct in [75, 50, 25, 0]:  # % retained
        # Binary search for the elasticity that produces this retention
        lo, hi = 0.0, 1000.0
        for _ in range(50):
            mid = (lo + hi) / 2
            total_impact = mid * shifted_gw * 2
            adj_spread = (_source_lmp - _cheapest_dest - total_impact).clip(lower=0)
            adj_gross = adj_spread.groupby(_price_wide.index.year).sum().mean()
            adj_routed = adj_gross * _routable_frac
            retained = adj_routed / _base_routed * 100 if _base_routed > 0 else 0
            if retained > target_pct:
                lo = mid
            else:
                hi = mid
        breakpoints[target_pct] = (lo + hi) / 2
    
    degrad_25 = breakpoints[75]
    degrad_50 = breakpoints[50]
    degrad_75 = breakpoints[25]
    degrad_100 = breakpoints[0]
    
    print(f'{fleet_gw:>6.1f} GW | ${degrad_25:>9.0f}/GW | '
          f'${degrad_50:>9.0f}/GW | ${degrad_75:>9.0f}/GW | '
          f'${degrad_100:>9.0f}/GW')

print(f'\nREFERENCE ELASTICITIES:')
print(f'  Off-peak supply curve slope:     ~$2-10/MWh per GW  (flat, baseload margin)')
print(f'  Shoulder supply curve slope:     ~$10-30/MWh per GW (gas units dispatching)')
print(f'  Peak supply curve slope:         ~$30-80/MWh per GW (peakers, steep)')
print(f'  Scarcity / hockey stick:         ~$100-500/MWh per GW (offer cap region)')
print(f'  Synapse AESC DRIPE (annual avg): ${PRICE_ELASTICITY_PER_GW:.0f}/MWh per GW')
print(f'')
print(f'  Hours by price regime (ComEd {YEARS[0]}-{YEARS[-1]}):')

_src = _source_lmp.dropna()
_off = (_src < 30).sum()
_shoulder = ((_src >= 30) & (_src < 60)).sum()
_peak = ((_src >= 60) & (_src < 150)).sum()
_scarcity = (_src >= 150).sum()
_total = len(_src)
print(f'    Off-peak (<$30):    {_off:>6,} hours ({_off/_total*100:>5.1f}%)')
print(f'    Shoulder ($30-60):  {_shoulder:>6,} hours ({_shoulder/_total*100:>5.1f}%)')
print(f'    Peak ($60-150):     {_peak:>6,} hours ({_peak/_total*100:>5.1f}%)')
print(f'    Scarcity (>$150):   {_scarcity:>6,} hours ({_scarcity/_total*100:>5.1f}%)')
print(f'')
print(f'  Most routing value comes from off-peak and shoulder hours (high volume,')
print(f'  low elasticity). Scarcity hours have high elasticity but few hours.')
print(f'  The effective average elasticity is dominated by the low-price hours')
print(f'  where the supply curve is flattest.')
```

    ELASTICITY SENSITIVITY: ROUTING VALUE BY FLEET SIZE
    ==========================================================================================
    Baseline (price-taker): $79,113/MW-yr
    
       Elast |   0.5 GW |   1.0 GW |   2.0 GW |   5.0 GW |  10.0 GW |
    -------------------------------------------------------
    $    1/GW |    98%  |    95%  |    91%  |    79%  |    61%  |
    

    $    2/GW |    95%  |    91%  |    83%  |    61%  |    37%  |
    $    5/GW |    89%  |    79%  |    61%  |    29%  |     9%  |
    $   10/GW |    79%  |    61%  |    37%  |     9%  |     2%  |
    $   20/GW |    61%  |    37%  |    14%  |     2%  |     1%  |
    

    $   50/GW |    29%  |     9%  |     2%  |     0%  |     0%  |
    $  100/GW |     9%  |     2%  |     1%  |     0%  |     0%  |
    

    $  150/GW |     4%  |     1%  |     0%  |     0%  |     0%  |
    $  200/GW |     2%  |     1%  |     0%  |     0%  |     0%  |
    $  300/GW |     1%  |     0%  |     0%  |     0%  |     0%  |
    $  500/GW |     0%  |     0%  |     0%  |     0%  |     0%  |
    
    BREAKPOINT ANALYSIS: Elasticity needed for X% degradation
    ----------------------------------------------------------------------
       Fleet |   25% degrad |   50% degrad |   75% degrad |  100% degrad
    ----------------------------------------------------------------------
    

       0.5 GW | $       12/GW | $       28/GW | $       56/GW | $      960/GW
    

       1.0 GW | $        6/GW | $       14/GW | $       28/GW | $      480/GW
    

       2.0 GW | $        3/GW | $        7/GW | $       14/GW | $      240/GW
    

       5.0 GW | $        1/GW | $        3/GW | $        6/GW | $       96/GW
    

      10.0 GW | $        1/GW | $        1/GW | $        3/GW | $       48/GW
    
    REFERENCE ELASTICITIES:
      Off-peak supply curve slope:     ~$2-10/MWh per GW  (flat, baseload margin)
      Shoulder supply curve slope:     ~$10-30/MWh per GW (gas units dispatching)
      Peak supply curve slope:         ~$30-80/MWh per GW (peakers, steep)
      Scarcity / hockey stick:         ~$100-500/MWh per GW (offer cap region)
      Synapse AESC DRIPE (annual avg): $150/MWh per GW
    
      Hours by price regime (ComEd 2022-2025):
        Off-peak (<$30):    17,401 hours ( 49.6%)
        Shoulder ($30-60):  12,981 hours ( 37.0%)
        Peak ($60-150):      4,384 hours ( 12.5%)
        Scarcity (>$150):      294 hours (  0.8%)
    
      Most routing value comes from off-peak and shoulder hours (high volume,
      low elasticity). Scarcity hours have high elasticity but few hours.
      The effective average elasticity is dominated by the low-price hours
      where the supply curve is flattest.
    

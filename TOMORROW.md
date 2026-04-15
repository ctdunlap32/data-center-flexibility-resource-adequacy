# TOMORROW.md — pickup notes

Last session: 2026-04-15

## Current state

### Pipeline: COMPLETE AND VERIFIED ✓

Three-notebook pipeline built, end-to-end verified (outputs wiped, all
three notebooks rerun from fresh kernels, final_results.json matches).
Pushed to GitHub.

- **Notebook 01 — Empirical Evidence** (51 cells): Cross-BA stress
  decorrelation, workload characterization, three contract exports.
  Slow to run (~20 min, Azure traces peg CPU — caused laptop keyboard
  to lock up from thermal throttle on last run).
- **Notebook 02 — Cascade and Simulation** (38 cells): Ten-parameter
  cascade, variance decomposition, conditional MC (single facility,
  empirical fleet, per-GW sweep), sensitivity analysis (2D surface,
  tornado), two contract exports.
- **Notebook 03 — System Implications** (18 cells): Accredited MW,
  avoided capacity cost, IX acceleration NPV, final_results.json.

### Contract files in outputs/contracts/

| File | Producer | Consumer |
|---|---|---|
| stress_correlation_results.json | NB 01 | NB 02 |
| per_hour_destination_availability.parquet | NB 01 | NB 02 |
| workload_parameters.json | NB 01 | NB 02 |
| cascade_parameters.json | NB 02 | NB 03 |
| conditional_mc_results.json | NB 02 | NB 03 |
| final_results.json | NB 03 | — |

### Paper: v20 saved, not yet sent to Rosner

NE_Working_Draft_v20.docx has all corrected numbers from the clean
pipeline. v19 archived. Key changes from v19 → v20:

| Claim | v19 (stale) | v20 (corrected) |
|---|---|---|
| Single facility mean depth | 52.0% | 46.9% |
| Empirical fleet mean depth | 26.5% | 28.9% |
| 1 GW sweep depth | 48.7% | 46.4% |
| 10 GW sweep depth | 24.6% | 26.9% |
| 1 GW accredited MW | 448 | 427 |
| 10 GW accredited MW | 2,263 | 2,474 |
| 1 GW avoided cost | $81M | $77M |
| 10 GW avoided cost | $407M | $445M |
| Contention onset | placeholder | 3.0 GW |
| Fleet CVaR @ P5 | 18.0% | 17.5% (= DVFS floor) |
| Destination zones | 19 (MISO 8) | 17 (MISO 6) |
| BurstGPT requests | 5.3M | 5.2M |

S1 fixed-parameter treatment addressed in Methods (two sentences added).
Three typos fixed (Mthods, accomodates, plural agreement).

### Data fixes applied (three prior sessions)

1. Cell 48 variable shadowing: multi-source robustness loop overwrote
   source_stress_idx with CAISO_NP15 instead of ComEd. Fixed with
   loop-local variables + defensive recomputation.
2. MISO zone metadata: restructured from 8 LRZ-keyed to 6 hub-keyed
   zones. Fixed Iowa pricing (was MICHIGAN.HUB, now MINN.HUB), removed
   MISO_ILLINOIS double-count (was 1,646.7 MW = ComEd's value, real
   Ameren IL is 66.2 MW), folded Wisconsin into MINN_HUB.
3. ~30 county-level CSV corrections (IL PJM/MISO split, Montana,
   Missouri, Kentucky, El Paso).

## Remaining tasks (none are blocking)

### Before Rosner

- [ ] Read v20 end to end once more for prose flow — the number
      substitutions might have created awkward sentences
- [ ] Decide on filename for Rosner copy (drop version number,
      e.g. Dunlap_DataCenter_Flexibility_NE.docx)
- [ ] Verify the sensitivity surface range in §7 was updated
      ("20.7% to 30.3%" → "21.6% to 34.1%")

### Polish (can wait for post-Rosner)

- [ ] docs/contract_schemas.md — document all six contract file schemas
- [ ] Pillar 1 rewrite → docs/technical_appendix_cascade_derivation.md
- [ ] Pillar 3 rewrite → docs/technical_appendix_policy_framework.md
- [ ] BurstGPT auto-download URLs in notebook 01 Cell 2-3 (currently
      404ing, files cached locally, only matters for reviewer repro)
- [ ] Azure data: move out of OneDrive entirely, set AZURE_DATA_DIR
      env var (Option 3 from earlier conversation)
- [ ] Per-notebook polish pass (variable names, dead refs, defensive
      code) — budget 90 min each
- [ ] Notebook 01 performance: Azure CSV loading takes ~10 min,
      consider parquet conversion for the raw traces

### Contention onset finding (flagged for later thought)

Contention onset is now 3.0 GW (>50% of draws constrained). This is
a sharper, earlier transition than expected. Worth grappling with the
intuition before Rosner asks about it. The fleet is 100% constrained
at 10 GW. The paper's scale-dependence narrative is stronger, not
weaker, but the 3 GW threshold is a number Rosner will want to
understand.

### Marathon

Chicagoland Spring, May 3rd. 18 days out. Taper starts now. Don't
sacrifice sleep for code — the pipeline is done.

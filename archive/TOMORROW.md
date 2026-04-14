# Tomorrow's pickup — Phase 3 (build notebook 02)

## State at end of prior session

### Phase 1 (repo skeleton) — COMPLETE ✓
- All top-level directories exist: `notebooks/`, `data/raw/`, `data/processed/`, `outputs/contracts/`, `outputs/figures/`, `outputs/tables/`, `docs/`, `paper/`, `archive/`
- `README.md` cleaned up (removed stray code fences, updated cascade numbers to reconciled values 0.0384 / 20.4%, §6 numbers marked TODO)
- `requirements.txt` cleaned up (prose stripped, bare package list)
- `paper/` populated with NE_Working_Draft_v19.docx and Dunlap_EPIC_Article_Submission.docx

### Phase 2 (notebook 01 — Empirical Evidence) — COMPLETE ✓
- `notebooks/01_empirical_evidence.ipynb` built from Cross_BA v5 Parts 0-3 + new Part 2 (workload characterization) + new Part 4 (contract exports)
- 51 cells total, Parts 0-4 in clean structure
- REPO_ROOT resolver pattern at top of Cell 2-1 makes all paths launch-location-independent (works from repo root or notebooks/ dir)
- Cell 4-2 PJM zones bug fixed (was filtering dest_zones for 'PJM', now uses `intra_pjm` variable directly)
- Full regression passes against three baselines:
  - Part 1 (Cross_BA v5): capacity-weighted overlap 27.2%, dynamic availability 99.0%, all-stressed 1.0%, intra-PJM 49.0%, cross-BA 34.8%, all exact
  - Part 2 DynamoLLM: 44.1M requests, Coding P99 4.52s, Conv P99 11.57s, exact
  - Part 2 BurstGPT: P99 12.12s, exact (5.19M rows — uses without_fails version, 5.3M in paper should become 5.2M, see paper-update list)
  - Part 3: $169.3/MWh cap-weighted mean destination LMP, exact
- Three contract files written to `outputs/contracts/`:
  - `stress_correlation_results.json` (headline numbers + per_zone + yearly)
  - `per_hour_destination_availability.parquet` (200 rows × 21 cols, cross-BA + pjm_co_stressed_mw)
  - `workload_parameters.json` (DynamoLLM + BurstGPT P99s + S3 = 0.90)
- One figure: `outputs/figures/figure4_drain_time_cross_validation.png`
- Committed and pushed to GitHub

### Pre-Phase-2 baselines captured and committed ✓
- `archive/executed_bartlett_v18.ipynb/.md` — Bartlett v18 pre-reconciliation baseline
- `archive/executed_bartlett_v18_reconciled.ipynb/.md` — Bartlett v18 reconciled (S2: 0.80→0.70, E1: 0.997→0.95, E2: 0.995→0.98)
- `archive/executed_P1_Azure_Trace_Analysis.ipynb/.md` — Azure trace baseline
- `archive/regression_baseline_burstgpt.txt` — BurstGPT regression baseline
- `archive/executed_v5.md` — Cross_BA v5 regression baseline (from yesterday)
- `archive/burstgpt_validation.py` — script itself (was missing from repo, added mid-session)

## Next task: Phase 3 — Build notebook 02 (Cascade and Simulation)

Migration guide Part D.4 (note: guide calls this "Phase 3" while TOMORROW.md from yesterday called it "Phase 3" too, just for notebook 02 not 01).

### Source material for notebook 02
- **Cascade parameters (Cell 2-1):** Lifted from `archive/bartlett_analysis_v18.ipynb` Cell 0-3, post-reconciliation. Reads S3 from `outputs/contracts/workload_parameters.json` instead of hardcoding.
- **Variance decomposition (Cell 2-2):** From `archive/bartlett_analysis_v18.ipynb` Cell 2-1 (Sobol-style first-order variance attribution, N=50,000 MC draws)
- **Conditional Monte Carlo (Cell 3-X):** Lifted from `archive/Cross_BA_Stress_Correlation_v5.ipynb` Part 4 (conditional MC joint D1-D5 feasibility, single facility and fleet). Reads per-hour availability from `outputs/contracts/per_hour_destination_availability.parquet`.
- **Sensitivity analysis (Cell 4-X):** From `archive/Cross_BA_Stress_Correlation_v5.ipynb` Part 5 (2D surface, tornado chart)
- **Output contracts:** Two files for notebook 03 consumption:
  - `outputs/contracts/cascade_parameters.json` — ten parameters, central values, ranges
  - `outputs/contracts/conditional_mc_results.json` — commitment depth distributions (1 GW, 10 GW, per-GW sweep)

### Key architectural decisions already locked
- Notebook 02 reads its inputs from `outputs/contracts/` files produced by notebook 01 — never from raw data directly
- REPO_ROOT resolver pattern should be used for ALL path references in notebook 02 (copy-paste from notebook 01 Cell 2-1)
- Cascade parameters use reconciled values: S2=0.70, E1=0.95, E2=0.98 (NOT the old 0.80 / 0.997 / 0.995)
- Expected headline numbers: cascade product 0.0384, commitment depth 20.4%, accredited MW ~187 @ 1 GW and ~1,875 @ 10 GW

### The guide's missing content that needs mental translation
The migration guide's D.4 was written before the parameter reconciliation, so it references the OLD values (0.0467, 21.0%, 448 MW, 2,263 MW, $81M, $407M). When building notebook 02, these all need updating to:
- Cascade product: 0.0467 → 0.0384
- Commitment depth: 21.0% → 20.4%
- Accredited MW @ 1 GW: 448 → ~187 (actual value from reconciled Bartlett v18)
- Accredited MW @ 10 GW: 2,263 → ~1,875
- Avoided cost @ 1 GW: $81M → ~$67M (but DON'T update the paper until notebook 03 produces the final number through the new pipeline)
- Avoided cost @ 10 GW: $407M → ~$335M

### First concrete step for tomorrow
1. Verify the final clean nbconvert run of notebook 01 committed successfully (look for `per_hour_destination_availability.parquet` showing `pjm_co_stressed_mw` non-zero in sample rows)
2. Copy `archive/bartlett_analysis_v18.ipynb` → `notebooks/02_cascade_simulation.ipynb` as the starting point
3. Same pattern as Phase 2: replace title cell, restructure Parts, then deal with what needs to be added from Cross_BA v5 (conditional MC and sensitivity)
4. Unlike Phase 2, notebook 02 does NOT have a clean 1:1 source file — it's a merge of Bartlett v18 (cascade + variance decomp + capacity market integration) and Cross_BA v5 Parts 4-5 (conditional MC + sensitivity)

## Running paper-update list (accumulating items for final paper polish pass)

Do NOT edit the docx yet — wait until after notebook 03 is complete. These are items to apply in one pass later:

### Typos
- "see Mthods" → "see Methods" (Section 4 body)
- "accomodates" → "accommodates" (Section 4 body)
- "conversational session with persistent state" → "conversational sessions with persistent state" (plural agreement)

### Methodological precision
- "5.3 million" BurstGPT requests → "5.2 million" (without_fails version has 5,188,507 rows)
- "P99 drain times range from 4.5 to 12.12 seconds" → consider "4.5 to 12.1" for clean rounding
- Methods workload datasets paragraph: rewrite the "Diurnal usage patterns are confirmed in DynamoLLM (peak traffic approximately 2.5 times trough) but weaker in BurstGPT" sentence. Draft replacement:
  > "Diurnal usage patterns in DynamoLLM range from approximately 3× peak-to-trough for conversational workloads to 35× for coding workloads (ref. 15), while BurstGPT exhibits near-continuous 24-hour activity with minimal weekly periodicity, consistent with differing user populations (Azure OpenAI regional API traffic vs. Azure internal production workloads)."
  - Verify 3×/35× numbers against ref. 15 (DynamoLLM paper) before committing to them
  - Verify DynamoLLM dataset characterization matches ref. 15

### Cascade parameter reconciliation — numbers that need updating everywhere
- Abstract: cascade product 0.0467 → 0.0384 (if cited)
- §5 and Methods: commitment depth 21.0% → 20.4%
- §6: avoided cost @ 1 GW, @ 10 GW, accredited MW numbers
- Any figures that plot cascade product or commitment depth
- Methods §2 / Methods cascade parameter specifications (already correct in v19, the code just needed to catch up)

### Other Methods improvements (optional)
- Could add one sentence justifying the conservative E1/E2 values: "E1 and E2 are held at conservative central values reflecting typical operator-facing SLA envelopes... Variance decomposition confirms E1 and E2 together contribute less than 5% of cascade output variance."

## Other items to revisit (not paper-related)

### BurstGPT URL fix (polish item for notebook 01)
The auto-download URLs in Cell 2-3 are stale (404ing). Fix during end-of-phase polish:
- Check current layout at https://github.com/HPMLL/BurstGPT to find working paths
- Update `BURSTGPT_URLS` and `BURSTGPT_HF_URLS` dicts in notebook 01 Cell 2-3
- Not urgent — files are cached locally, only matters for reviewer reproducibility

### OneDrive / external drive question
Parked for post-migration cleanup. Current status: Azure traces (1.8 GB) live in OneDrive, notebook 01 uses the `AZURE_DATA_DIR` environment variable pattern to find them. No action needed during Phase 3. Revisit after Phase 5 (Option 3 from yesterday's conversation: move all large data out of OneDrive entirely, set env vars).

### Polish pass per notebook
Plan is one dedicated polish pass per notebook AFTER each notebook is construction-complete and regression-verified. Maybe 90 minutes each:
- Structural/readability: variable names, banner consistency, markdown cleanup, dead reference removal
- Correctness/robustness: what-if-input-is-weird defensive code
- Skip performance polish except for Azure CSV loading (10 min per run is painful)
- Regression-verify after each polish pass (output should match pre-polish byte-for-byte)

### Pillar1 and Pillar3 rewrite
Scheduled for Phase 5 Step 5.3 per migration guide. Pillar1 is currently inconsistent with both code and paper (7-parameter cascade, pre-reconciliation values). Don't touch until notebook 02 and 03 are done. Destination: `docs/technical_appendix_cascade_derivation.md` and `docs/technical_appendix_policy_framework.md`.

## Important reminders
- Use Anaconda Prompt, not PowerShell
- Always pass `--ExecutePreprocessor.kernel_name=python3` to nbconvert
- Always run nbconvert from the repo root with `notebooks\XX_name.ipynb` as the input path — this makes the REPO_ROOT resolver's `elif _cwd.name == "notebooks"` branch fire correctly
- Commit and push after each meaningful milestone — don't leave uncommitted work overnight
- `git` may not be in the Anaconda Prompt PATH; if `git` commands fail with "not recognized," use a regular Command Prompt or add git to the PATH
- Reviewer reproducibility depends on contract files in `outputs/contracts/` being correct and consistent with their schemas
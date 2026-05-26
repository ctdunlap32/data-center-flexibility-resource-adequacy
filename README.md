# Quantifying AI data center flexibility as a resource adequacy asset

This repository contains the analysis code and data supporting the manuscript "Quantifying AI data center flexibility as a resource adequacy asset" by Chris Dunlap (University of Chicago, Energy Policy Institute at Chicago / Booth / Harris).

## Overview

The analysis quantifies how deeply AI data center fleets can credibly commit as demand response resources. The framework decomposes accreditable system slack into three composable mechanisms — spatial migration across grid regions, temporal deferral of non-interactive workloads, and dynamic voltage-frequency scaling on residual compute — each itself a multiplicatively-composed group of parameters representing the conditions that must hold for one MW of that mechanism's contribution to realize during a dispatch event. The three mechanisms compose additively at the megawatt layer with sequential ordering and an explicit overlap correction. The framework is reported under two scope scenarios: mixed-use facilities (φ_inf = 0.43, anchored in McKinsey Dec 2025) and inference-dominant facilities (φ_inf = 0.70, anchored in IEA 2025 and Zhou et al. 2024).

Three notebooks reproduce the paper end to end:

- **Notebook 01 — Empirical Evidence.** Cross-region stress decorrelation analysis across five U.S. grid regions (§3) and workload characterization from 49.4 million production inference requests (§4).
- **Notebook 02 — Cascade and Simulation.** Eleven-parameter cascade framework, variance decomposition, conditional Monte Carlo, per-GW parametric sweep, two-dimensional sensitivity surface, and tornado chart (§2, §5, §7).
- **Notebook 03 — System Implications.** ELCC application, avoided capacity cost, and interconnection acceleration NPV at two reference scales (1 GW facility and 10 GW fleet) under each scope scenario (§6).

## Requirements

- Python 3.11+
- Packages: see `requirements.txt`
- Data: see `data/raw/README.md` for source locations. Most data downloads automatically on first notebook run; DA LMP data must be obtained separately from each ISO/RTO market portal.

## How to run

From the repository root:

1. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order: 01 → 02 → 03. Each notebook writes to `outputs/contracts/` and downstream notebooks read from there. Running out of order will fail cleanly with a missing-file error.

## Claim-to-cell mapping

Every numerical claim in the paper is produced by a specific cell. This table is the authoritative reference for reviewer verification. Cell numbers reflect the current state of each notebook.

| Paper claim | Section | Notebook | Cell |
|---|---|---|---|
| Eleven-parameter framework specification (Table M1) | §2, Methods | 02 | Cell 1-1 |
| Variance attribution (DEST_absorb 44%, DVFS_headroom 22%, MIG_elig 13%, NIW_share 10%, QUEUE_ok 8%) | Methods | 02 | Cell 2-1 |
| 99.0% dynamic availability (ComEd) | §3 | 01 | Cell 1-3 |
| Multi-source generalization: PJM-DOM 99.5%, ERCOT-LZ-NORTH 100%, CAISO-NP15 100% | §3 | 01 | Cell 3-7b |
| Stress overlap heatmap (Fig. 1) | §3 | 01 | Cell 1-2 |
| P99 drain time 4.5–12.12s across two production datasets (49.4M requests, 128 days) | §4 | 01 | Cells 2-2, 2-4 |
| Single-facility 500 MW: mean 40.0%, P5 38.5%, CVaR 33.1% (inference-dominant) | §5 | 02 | Cell 3-2 |
| Single-facility 500 MW: mean 24.6%, P5 23.7%, CVaR 20.6% (mixed-use) | §5 | 02 | Cell 3-2 |
| Empirical fleet (30 ComEd co-stress hours): mean 33.6% inf-dom / 21.1% mixed-use | §5 | 02 | Cell 3-3 |
| Per-GW sweep: 39.8% (1 GW) → 38.5% (10 GW) inference-dominant | §5 | 02 | Cell 3-4 |
| Per-GW sweep: 24.6% (1 GW) → 24.1% (10 GW) mixed-use | §5 | 02 | Cell 3-4 |
| Destination-constrained share: 4.9% (1 GW) → 25.7% (10 GW) inference-dominant | §5 | 02 | Cell 3-4 |
| Three-panel conditional MC figure (Fig. 2) | §5 | 02 | Cell 3-5 |
| 2D sensitivity surface 32.5%–40.0% across DEST_absorb × QUEUE_ok at 10 GW (Fig. 3a) | §7 | 02 | Cell 4-1 |
| Tornado chart at 10 GW (Fig. 3b) | §7 | 02 | Cell 4-2 |
| Source-zone robustness: PJM Dominion 40.2% vs ComEd 40.0% inference-dominant (Extended Data Table 3) | §7 | 02 | Cell 6-1 |
| 1 GW inference-dominant: 367 MW accredited (mean), 350 MW (P5) | §6 | 03 | Cell 1-1 |
| 10 GW inference-dominant: 3,546 MW accredited (mean), 2,322 MW (P5) | §6 | 03 | Cell 1-1 |
| 1 GW mixed-use: 226 MW accredited (mean), 217 MW (P5) | §6 | 03 | Cell 1-1 |
| 10 GW mixed-use: 2,216 MW accredited (mean), 1,483 MW (P5) | §6 | 03 | Cell 1-1 |
| Avoided capacity cost: $66M/yr (1 GW inf-dom), $638M/yr (10 GW inf-dom) | §6 | 03 | Cell 2-1 |
| Avoided capacity cost: $41M/yr (1 GW mixed-use), $399M/yr (10 GW mixed-use) | §6 | 03 | Cell 2-1 |
| IX acceleration NPV: $23.1B central (1 GW inf-dom, 3-year window); $16.1B / $29.5B conservative / optimistic (Extended Data Table 2) | §6 | 03 | Cell 3-1 |

## Repository structure

See directory tree in `docs/repo_layout.md`.

## Inter-notebook contracts

Notebooks communicate through JSON and parquet files in `outputs/contracts/`. Schema is documented in `docs/contract_schemas.md`. These files are regenerated each time the upstream notebook runs. Do not hand-edit.

## Data and code availability

DA LMP data: publicly available from PJM Data Miner 2 (https://dataminer2.pjm.com), ERCOT Market Information System (https://www.ercot.com/mp/data-products), CAISO OASIS (http://oasis.caiso.com), MISO Market Reports (https://www.misoenergy.org/markets-and-operations/real-time--market-data), and NYISO Custom Reports (https://www.nyiso.com/custom-reports).

Microsoft Azure LLM Inference Dataset 2024 (DynamoLLM): https://github.com/Azure/AzurePublicDataset under AzureLLMInferenceDataset2024.

BurstGPT: https://github.com/HPMLL/BurstGPT.

Data center location data: U.S. Department of Energy Accelerating Speed to Power Data Viewer (formerly hosted at maps.nrel.gov/speed-to-power, accessed via its public backend API in February 2026). The data viewer was retired in 2026; the extracted dataset is archived in data/raw/dc_capacity_mapped in this repository.

All analysis code in this repository is released under the MIT License.

## Contact

Christopher Dunlap — cdunlap@chicagobooth.edu
